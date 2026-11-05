"""Shared controlled variants, with full per-step PSNR/paired-SSIM logging."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import csv
import hashlib
import json
import platform
import random
import time
import numpy as np
import torch
from .data import image_metrics, load_case, sha256, synthetic_case
from .models import make_model
from .physics import (channels, combine, complex_image, fft2c, magnitude01,
                      make_mask, residual_gate, tv_loss, update_gate)

NATIVE = ("rgap", "dip", "no_ema", "no_tv", "joint_opt", "rescat",
          "uniform_noise", "shuffled_gate", "global_gate")
METRIC_SPEC = {"image": "abs(sum(conj(s)*ifft2c(predicted_full_kspace)))",
    "normalization": "each_magnitude_image_divided_by_own_max",
    "psnr_data_range": 1.0, "ssim_data_range": 1.0, "ssim_win_size": 7,
    "ssim_gaussian_weights": False, "ssim_sample_covariance": True,
    "best_checkpoint": "argmax_per_run_PSNR;SSIM_at_the_same_iteration",
    "final_checkpoint": "last_training_forward_at_prespecified_iteration",
    "note": "Historical SSIM/checkpoint settings require original-log verification."}


@dataclass(frozen=True)
class Config:
    method: str = "rgap"
    dataset: str = "knee"
    acceleration: int = 8
    seed: int = 1234
    mask_seed: int = 0
    steps: int = 6000
    lr: float = 1e-4
    tv: float = 1e-3
    sigma: float = 1.0
    alpha: float = 0.05
    gmin: float = 0.05
    gamma: float = 0.8
    gate_eps: float = 1e-8
    size: int = 320
    divisor: float = 32767.0
    backbone: str = "medium"
    protocol: str = "source"
    device: str = "auto"
    eval_every: int = 1
    save_every: int = 500
    synthetic: bool = False
    timing: bool = False
    warmup: int = 10

    def validate(self):
        if self.method not in NATIVE:
            raise ValueError(f"Unknown native method {self.method}")
        if self.steps < 1 or self.eval_every < 1 or self.save_every < 0:
            raise ValueError("Invalid iteration/logging frequency.")
        if not 0 <= self.alpha <= 1 or not 0 <= self.gmin <= 1:
            raise ValueError("alpha and gmin must be in [0,1].")
        if self.gamma <= 0 or self.gate_eps <= 0 or self.sigma < 0 or self.tv < 0:
            raise ValueError("Invalid gate/noise/TV parameter.")
        if self.protocol not in ("source", "ortho"):
            raise ValueError("Choose source or ortho protocol.")
        if self.backbone not in ("medium", "tiny"):
            raise ValueError("Choose medium or test-only tiny backbone.")
        if self.size < 8 or (self.backbone == "medium" and self.size < 64):
            raise ValueError("MediumUNet needs size>=64 for training BatchNorm.")
        if self.timing and self.steps <= self.warmup:
            raise ValueError("Timing needs steps > warmup.")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_device(name):
    name = ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu.")
    return device


def source_digest():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def job_path(config, case, root):
    key = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()[:12]
    return (Path(root) / config.dataset / f"x{config.acceleration}" / config.method
            / case / f"seed{config.seed}_{key}")


def prepare(ref, sens, mask, model, config):
    """Reproduce the supplied solver's measurement resimulation/target scaling."""
    y = fft2c(sens * ref, "ortho") * mask
    z0 = channels(combine(y * mask, sens, "ortho")).detach()
    norm = "backward" if config.protocol == "source" else "ortho"
    base_input = torch.cat((z0, torch.zeros_like(z0)), 1) if config.method == "rescat" else z0
    with torch.no_grad():
        if config.protocol == "source":
            # Exactly the original initial-output norm matching. Keep this
            # explicit; it is not silently replaced by a new physical protocol.
            scale = torch.linalg.vector_norm(model(base_input)) / torch.linalg.vector_norm(
                combine(y, sens, "backward")).clamp_min(1e-12)
        else:
            scale = torch.ones((), device=y.device)
    return z0, y * scale, norm, float(scale)


def run(config, case_path, output_root, overwrite=False):
    config.validate()
    case = "synthetic" if config.synthetic else Path(case_path).stem
    folder = job_path(config, case, output_root)
    if (folder / "summary.json").exists() and not overwrite:
        previous = json.loads((folder / "summary.json").read_text())
        same_input = config.synthetic or previous["data"]["input_sha256"] == sha256(case_path)
        if previous.get("source_sha256") != source_digest() or not same_input:
            raise RuntimeError(f"Stale cached run: {folder}; choose a new output root.")
        print(f"cached {folder}", flush=True)
        return previous
    if folder.exists() and any(folder.iterdir()) and not overwrite:
        raise RuntimeError(f"Partial run exists: {folder}; use --overwrite or a new output root.")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(asdict(config), indent=2)+"\n")
    seed_all(config.seed)
    device = resolve_device(config.device)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    t0 = time.perf_counter()
    if config.synthetic:
        ref, sens, data_info = synthetic_case(config.size, device=device)
    else:
        ref, sens, data_info = load_case(case_path, config.size, config.divisor,
                                         device, config.protocol)
    m, mask_info = make_mask(config.size, config.size, config.acceleration, config.mask_seed)
    mask = torch.as_tensor(m, device=device)
    np.save(folder / "mask.npy", m)
    (folder / "mask.json").write_text(json.dumps(mask_info, indent=2)+"\n")
    net = make_model(config.backbone, config.method == "rescat").to(device)
    net.train()
    z0, target, norm, target_scale = prepare(ref, sens, mask, net, config)
    latent = torch.nn.Parameter(z0.clone()) if config.method == "joint_opt" else None
    params = list(net.parameters()) + ([] if latent is None else [latent])
    optimizer = torch.optim.Adam(params, lr=config.lr)
    gate = torch.ones_like(z0)
    res_input = torch.zeros_like(z0)
    sigma = config.sigma * z0.abs().amax().detach()
    # Separate RNGs preserve noise pairing when the gate shuffle changes.
    noise_rng = torch.Generator(device=device).manual_seed(config.seed + 100003)
    shuffle_rng = torch.Generator(device=device).manual_seed(config.seed + 200003)
    permutation = torch.randperm(config.size**2, generator=shuffle_rng, device=device)
    sync(device)
    preprocessing_s = time.perf_counter() - t0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows, snapshots = [], {}
    best_psnr, best_ssim, best_step = -float("inf"), None, None
    best_image, final_image, ref_image = None, None, None
    tv_weight = 0.0 if config.method == "no_tv" else config.tv
    noise_methods = {"rgap", "no_ema", "no_tv", "uniform_noise", "shuffled_gate", "global_gate"}
    stage_times = []
    measured_start = time.perf_counter()
    for step in range(1, config.steps+1):
        sync(device) if config.timing else None
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if config.method in noise_methods:
            xi = sigma * torch.randn(z0.shape, dtype=z0.dtype, device=device, generator=noise_rng)
            used_gate = gate
            if config.method in ("uniform_noise", "global_gate"):
                # Identical spatially averaged conditional noise variance.
                used_gate = gate.square().mean().sqrt()
            elif config.method == "shuffled_gate":
                g = gate[0, 0].flatten()[permutation].view(config.size, config.size)
                used_gate = g[None, None].expand_as(gate)
            network_input = z0 + used_gate * xi
        elif latent is not None:
            network_input = latent
        elif config.method == "rescat":
            network_input = torch.cat((z0, res_input), dim=1)
        else:
            network_input = z0
        out = net(network_input)
        predicted = fft2c(sens * complex_image(out), norm)
        residual = target - mask * predicted
        dc = torch.linalg.vector_norm(residual)
        tv = tv_loss(out)
        loss = dc + tv_weight * tv
        sync(device) if config.timing else None
        forward_end = time.perf_counter()
        loss.backward()
        optimizer.step()
        sync(device) if config.timing else None
        backward_end = time.perf_counter()
        with torch.no_grad():
            if config.method in noise_methods:
                current = residual_gate(residual.detach(), sens, config.gamma, config.gate_eps, norm)
                alpha = 1.0 if config.method == "no_ema" else config.alpha
                gate = update_gate(gate, current, alpha, config.gmin)
            elif config.method == "rescat":
                res_input = channels(combine(residual.detach(), sens, norm))
        sync(device) if config.timing else None
        gate_end = time.perf_counter()
        if config.timing:
            if step > config.warmup:
                stage_times.append((forward_end-step_start, backward_end-forward_end,
                                    gate_end-backward_end, gate_end-step_start))
            continue  # No PSNR, SSIM, plotting, or disk I/O in timed optimization.
        with torch.no_grad():
            if step % config.eval_every == 0 or step in (1, config.steps):
                psnr, ssim, image, ref_image = image_metrics(combine(predicted.detach(), sens, norm), ref)
                rows.append({"step": step, "loss": float(loss), "dc": float(dc),
                             "tv": float(tv), "psnr": psnr, "ssim": ssim})
                if psnr > best_psnr:
                    best_psnr, best_ssim, best_step = psnr, ssim, step
                    best_image = image.copy()
                if step == config.steps:
                    final_image = image.copy()
            if config.save_every and (step % config.save_every == 0 or step in (1, config.steps)):
                snapshots[f"iter_{step}"] = magnitude01(combine(predicted.detach(), sens, norm)).cpu().numpy()
                snapshots[f"gate_{step}"] = gate[0, 0].cpu().numpy()
        if step == 1 or step == config.steps or step % 500 == 0:
            print(f"{config.dataset} x{config.acceleration} {config.method} {case} "
                  f"seed={config.seed} {step}/{config.steps}", flush=True)
    sync(device)
    optimization_s = time.perf_counter() - measured_start
    metadata = {"schema_version": 1, "config": asdict(config), "case": case,
        "source_sha256": source_digest(), "data": data_info, "mask": mask_info,
        "metrics": METRIC_SPEC, "target_scale": target_scale,
        "noise_distribution": "standard_normal_randn" if config.method in noise_methods else "none",
        "sigma_absolute": float(sigma) if config.method in noise_methods else 0.0,
        "effective_alpha": (1.0 if config.method == "no_ema" else config.alpha)
                           if config.method in noise_methods else None,
        "effective_tv": tv_weight, "initial_rescat_residual": "zeros; lagged_after_each_forward",
        "joint_opt_lr": config.lr, "preprocessing_s": preprocessing_s,
        "device": str(device), "torch": torch.__version__, "python": platform.python_version(),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "not_publication_run": config.synthetic or config.backbone != "medium" or config.steps != 6000
                               or config.size != 320 or config.eval_every != 1,
        "output_path": str(folder.resolve())}
    if config.timing:
        a = np.array(stage_times)
        timing = {"steps": config.steps, "warmup_excluded_from_step_mean": config.warmup,
            "ms_per_iteration": float(a[:, 3].mean()*1000),
            "ms_per_iteration_sd": float(a[:, 3].std(ddof=1)*1000) if len(a)>1 else 0,
            "forward_loss_ms": float(a[:, 0].mean()*1000),
            "backward_adam_ms": float(a[:, 1].mean()*1000),
            "gate_update_ms": float(a[:, 2].mean()*1000),
            "optimization_s_including_warmup": optimization_s,
            "total_s_excluding_output_writes": preprocessing_s+optimization_s,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
            "protocol": "synchronized_stage_timing;no_metrics_or_saving_in_loop;setup_includes_NPZ_load",
            "note": "gate_update_ms excludes resampling and residual already in forward_loss_ms"}
        metadata["timing"] = timing
        (folder / "timing.json").write_text(json.dumps(timing, indent=2)+"\n")
    else:
        with open(folder / "trajectory.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        metadata["best"] = {"step": best_step, "psnr": best_psnr, "ssim": best_ssim}
        metadata["final"] = {"step": config.steps, "psnr": rows[-1]["psnr"], "ssim": rows[-1]["ssim"]}
        metadata["oracle_evaluated_every_iteration"] = config.eval_every == 1
        np.savez_compressed(folder / "reconstructions.npz", best=best_image, final=final_image,
                            reference=ref_image, **snapshots)
    (folder / "summary.json").write_text(json.dumps(metadata, indent=2)+"\n")
    manifest = Path(output_root) / "experiment_manifest.csv"
    fields = ("dataset", "acceleration", "method", "case", "seed", "mask_seed",
              "source_sha256", "input_sha256", "config_path", "output_path")
    record = {k: getattr(config, k) for k in fields[:3]}
    record.update(case=case, seed=config.seed, mask_seed=config.mask_seed,
                  source_sha256=metadata["source_sha256"], input_sha256=data_info["input_sha256"],
                  config_path=str((folder/"config.json").resolve()), output_path=str(folder.resolve()))
    new = not manifest.exists()
    with open(manifest, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(record)
    return metadata
