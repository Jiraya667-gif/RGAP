"""Explicit spatial FFTs and the uploaded solver's random Cartesian mask.

The default `source` protocol preserves the two FFT scaling paths and evaluation
object of the supplied Residual_Guided.py. `ortho` is a separate, opt-in protocol;
its results must never be mixed with source-protocol results.
"""
import numpy as np
import torch


def fft2c(x, norm="backward"):
    return torch.fft.fftshift(torch.fft.fft2(
        torch.fft.ifftshift(x, dim=(-2, -1)), norm=norm), dim=(-2, -1))


def ifft2c(x, norm="backward"):
    return torch.fft.fftshift(torch.fft.ifft2(
        torch.fft.ifftshift(x, dim=(-2, -1)), norm=norm), dim=(-2, -1))


def combine(kspace, sensitivities, norm="backward"):
    return (sensitivities.conj() * ifft2c(kspace, norm)).sum(dim=-3)


def channels(x):
    return torch.stack((x.real, x.imag), dim=-3).unsqueeze(0)


def complex_image(x):
    return torch.complex(x[0, 0], x[0, 1])


def residual_gate(residual, sensitivities, gamma=0.8, eps=1e-8,
                  norm="backward"):
    magnitude = combine(residual, sensitivities, norm).abs()
    gate = (magnitude / (magnitude.amax() + eps)).clamp(0, 1).pow(gamma)
    return gate[None, None].repeat(1, 2, 1, 1)


def update_gate(old, current, alpha=0.05, floor=0.05):
    return (alpha * current + (1 - alpha) * old).clamp(floor, 1.0)


def make_mask(height, width, acceleration=8, seed=0):
    """Keep the uploaded make_vdrs_mask selection, including its two gaps.

    At width=320: x4 has 25 central + 27+27 outer = 79 lines (R=320/79);
    x8 has 12 central + 14+14 outer = 40 lines. The columns immediately
    outside the center are not candidates, exactly as in the uploaded code.
    No claim of a strict variable-density sampling law is made.
    """
    nlines = int(width / acceleration)
    init_lines = int(width * 0.32 / acceleration)
    low1, low2 = (width - init_lines) // 2, (width + init_lines) // 2
    nout = (nlines - init_lines) // 2
    if init_lines < 1 or nout < 0 or nout > min(low1 - 1, width - low2 - 1):
        raise ValueError("Resolution/acceleration is too small for the source mask.")
    rng = np.random.default_rng(seed)
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:, low1:low2] = 1
    mask[:, rng.choice(low1 - 1, nout, replace=False)] = 1
    mask[:, rng.choice(np.arange(low2 + 1, width), nout, replace=False)] = 1
    return mask, {"nominal_acceleration": acceleration, "seed": seed,
                  "center_start": low1, "center_stop": low2,
                  "center_lines": low2 - low1,
                  "sampled_lines": int(mask[0].sum()),
                  "actual_acceleration": width / mask[0].sum().item(),
                  "phase_encode_axis": "last dimension / columns"}


def tv_loss(real_channels):
    return ((real_channels[..., 1:, :] - real_channels[..., :-1, :]).abs().sum()
            + (real_channels[..., :, 1:] - real_channels[..., :, :-1]).abs().sum())


def magnitude01(x):
    x = x.abs()
    return x / x.amax().clamp_min(1e-12)
