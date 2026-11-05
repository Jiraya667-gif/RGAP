"""Run real baseline adapters or import their outputs; never substitute algorithms."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import shlex
import subprocess
import time
import numpy as np
import torch
from .data import image_metrics, load_case, sha256, synthetic_case
from .engine import METRIC_SPEC, job_path, make_model, prepare, resolve_device, seed_all
from .physics import channels, combine, fft2c, make_mask


def validate_output(folder, config, reference, provenance):
    """Require a full trace and paired best/final images, not hand-entered means."""
    folder = Path(folder)
    if not provenance.get("implementation") or not provenance.get("version"):
        raise ValueError("Baseline provenance requires implementation and version.")
    if config.method == "nila":
        image = np.load(folder/"prediction.npy", allow_pickle=False)
        psnr, ssim, image, _ = image_metrics(torch.as_tensor(image), reference.cpu())
        return {"contextual": {"psnr": psnr, "ssim": ssim},
                "prior_condition": "supervised_external_checkpoint"}
    with open(folder/"trajectory.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if [int(r["step"]) for r in rows] != list(range(1, config.steps+1)):
        raise ValueError("Baseline must return every step from 1 through the common budget.")
    values = np.array([[float(r["psnr"]), float(r["ssim"])] for r in rows])
    if not np.isfinite(values).all():
        raise ValueError("Non-finite baseline metrics.")
    best_index = int(values[:, 0].argmax())
    images = np.load(folder/"reconstructions.npz", allow_pickle=False)
    for key, index in (("best", best_index), ("final", len(rows)-1)):
        image = torch.as_tensor(images[key])
        p, s, _, _ = image_metrics(image, reference.cpu())
        if not np.allclose((p, s), values[index], atol=5e-5, rtol=0):
            raise ValueError(f"{key} image does not reproduce its PSNR/paired SSIM.")
    return {"best": {"step": best_index+1, "psnr": values[best_index, 0], "ssim": values[best_index, 1]},
            "final": {"step": config.steps, "psnr": values[-1, 0], "ssim": values[-1, 1]},
            "oracle_evaluated_every_iteration": True,
            "prior_condition": provenance.get("prior_condition", "not_documented")}


def run_external(config, case_path, output_root, adapter, overwrite=False):
    if config.steps < 1 or config.protocol not in ("source", "ortho"):
        raise ValueError("Invalid external experiment budget/protocol.")
    seed_all(config.seed)
    case = "synthetic" if config.synthetic else Path(case_path).stem
    folder = job_path(config, case, output_root)
    if (folder/"summary.json").exists() and not overwrite:
        raise RuntimeError(f"External run exists: {folder}. Reimport explicitly to verify provenance.")
    folder.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    if config.synthetic:
        ref, sens, data = synthetic_case(config.size, device=device)
    else:
        ref, sens, data = load_case(case_path, config.size, config.divisor, device, config.protocol)
    mask_np, mask_meta = make_mask(config.size, config.size, config.acceleration, config.mask_seed)
    mask = torch.as_tensor(mask_np, device=device)
    # The common preparation is explicit in the request. An official adapter
    # must implement the matched protocol and document any necessary departure.
    net = make_model(config.backbone).to(device)
    z0, target, norm, scale = prepare(ref, sens, mask, net, config)
    measurements = folder/"measurements.npz"
    np.savez_compressed(measurements, y=(fft2c(sens*ref, "ortho")*mask).cpu().numpy(),
        target=target.cpu().numpy(), sensitivities=sens.cpu().numpy(),
        z0=z0.cpu().numpy(), mask=mask_np)
    reference_path = folder/"reference_for_offline_evaluation.npy"
    np.save(reference_path, ref.abs().cpu().numpy())
    request = {"method": config.method, "config": asdict(config),
        "measurements": str(measurements.resolve()), "prediction_fft_norm": norm,
        "target_scale": scale, "metrics": METRIC_SPEC,
        "reference_for_offline_evaluation": str(reference_path.resolve()),
        "reference_policy": "offline_metrics_only;never_an_input_or_optimization_loss",
        "output_directory": str(folder.resolve())}
    request_path = folder/"request.json"
    request_path.write_text(json.dumps(request, indent=2)+"\n")
    if not isinstance(adapter.get("command"), list) or not adapter["command"]:
        raise ValueError(f"{config.method}: adapter command must be a nonempty argv list.")
    command = [part.format(request=str(request_path.resolve()), output=str(folder.resolve()),
                           python=__import__("sys").executable) for part in adapter["command"]]
    start = time.perf_counter()
    with open(folder/"adapter_stdout.log", "w") as out, open(folder/"adapter_stderr.log", "w") as err:
        subprocess.run(command, cwd=adapter.get("cwd"), check=True, stdout=out, stderr=err)
    provenance_path = folder/"provenance.json"
    if not provenance_path.exists():
        raise ValueError("Adapter must write provenance.json with actual implementation/version/weights.")
    provenance = json.loads(provenance_path.read_text())
    result = validate_output(folder, config, ref, provenance)
    result.update({"schema_version": 1, "config": asdict(config), "case": case,
        "data": data, "mask": mask_meta, "metrics": METRIC_SPEC,
        "external_provenance": provenance, "command": command,
        "external_process_elapsed_s": time.perf_counter()-start,
        "not_publication_run": config.synthetic or config.size != 320 or config.steps != 6000,
        "output_path": str(folder.resolve())})
    (folder/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result
