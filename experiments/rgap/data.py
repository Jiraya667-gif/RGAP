"""NPZ loading follows the supplied solver; sensitivity maps are fixed inputs."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from .physics import combine, magnitude01


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_case(path, size=320, divisor=32767.0, device="cpu", protocol="source"):
    with np.load(path, allow_pickle=False) as data:
        missing = {"k_r", "k_i", "s_r", "s_i"} - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        arrays = [np.asarray(data[k], dtype=np.float32) for k in
                  ("k_r", "k_i", "s_r", "s_i")]
    if len({a.shape for a in arrays}) != 1 or arrays[0].ndim != 3:
        raise ValueError("All four arrays must have equal shape [coils, H, W].")
    _, h, w = arrays[0].shape
    if size > min(h, w) or divisor <= 0:
        raise ValueError("Invalid crop or input divisor.")
    y, x = h // 2 - size // 2, w // 2 - size // 2
    arrays = [a[:, y:y+size, x:x+size] / divisor for a in arrays]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError(f"Non-finite input: {path}")
    k = torch.from_numpy(arrays[0] + 1j * arrays[1]).to(device)
    s = torch.from_numpy(arrays[2] + 1j * arrays[3]).to(device)
    if s.abs().amax() == 0:
        raise ValueError("Sensitivity maps are all zero.")
    norm = "backward" if protocol == "source" else "ortho"
    ref = combine(k, s, norm)
    ref = ref / (ref.abs().amax() + 1e-8)
    if ref.abs().amax() == 0:
        raise ValueError("Reference image is all zero.")
    return ref, s, {"input_path": str(Path(path).resolve()),
                    "input_sha256": sha256(path), "input_divisor": divisor,
                    "crop_size": size, "coils": len(s),
                    "measurement_mode": "resimulated_from_normalized_combined_reference",
                    "sensitivities": "precomputed_NPZ_fixed_not_estimated_by_runner"}


def synthetic_case(size=64, coils=3, device="cpu"):
    """Small analytic phantom, exclusively for execution tests (not fastMRI)."""
    y, x = torch.meshgrid(torch.linspace(-1, 1, size, device=device),
                          torch.linspace(-1, 1, size, device=device), indexing="ij")
    image = (torch.exp(-4*(x*x+1.7*y*y))
             + 0.35*torch.exp(-60*((x-.25)**2+(y+.18)**2)))
    image = image * torch.exp(0.15j*x)
    sens = []
    for c in range(coils):
        angle = 2*np.pi*c/coils
        sens.append((1+.2*(x*np.cos(angle)+y*np.sin(angle)))
                    * torch.exp(1j*(x*np.cos(angle)+y*np.sin(angle))))
    sens = torch.stack(sens)
    sens = sens / sens.abs().square().sum(dim=0).sqrt()
    return image/image.abs().amax(), sens, {"input_path": "synthetic_phantom",
        "input_sha256": "synthetic", "coils": coils, "crop_size": size,
        "measurement_mode": "synthetic_smoke_test", "not_paper_data": True}


def image_metrics(reconstruction, reference):
    from skimage.metrics import structural_similarity
    out = magnitude01(reconstruction).detach().cpu().numpy().astype(np.float64)
    ref = magnitude01(reference).detach().cpu().numpy().astype(np.float64)
    mse = np.mean((out-ref)**2)
    psnr = float(-10*np.log10(max(mse, 1e-16)))
    # Explicit skimage defaults; historical SSIM settings were not supplied.
    ssim = float(structural_similarity(ref, out, data_range=1.0,
                                       win_size=7, gaussian_weights=False,
                                       use_sample_covariance=True))
    return psnr, ssim, out, ref
