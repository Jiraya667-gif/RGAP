import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft as fft
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from tqdm.auto import tqdm

from unet import MediumUNet
from physics.ct import CT
from physics.mri import SinglecoilMRI_comp, MulticoilMRI
from utils_ import clear


def fft_with_shifts(img):
    return fft.fftshift(fft.fft2(fft.ifftshift(img)))


def ifft_with_shifts(ksp):
    return fft.fftshift(fft.ifft2(fft.ifftshift(ksp)))


def ksp_and_mps_to_gt(ksp, mps):
    gt = mps.conj() * ifft_with_shifts(ksp)
    gt = torch.sum(gt.squeeze(), axis=0)
    return gt


def residual_to_gate(k_residual, mps_complex, eps=1e-8, gamma=0.8):
    """
    Build an image-domain gate in [0, 1] from sampled k-space residual.
    k_residual: (1, Coil, H, W) or (Coil, H, W) complex
    mps_complex: (1, Coil, H, W) complex
    returns: (1, 2, H, W) real-valued gate
    """
    if k_residual.dim() == 3:
        k_residual = k_residual.unsqueeze(0)

    r_img = ksp_and_mps_to_gt(k_residual, mps_complex)
    r_abs = torch.abs(r_img)
    r_abs = r_abs / (r_abs.amax() + eps)
    r_abs = torch.clamp(r_abs, 0.0, 1.0)
    r_abs = r_abs ** gamma
    return r_abs.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)


def make_vdrs_mask(n1, n2, nlines, init_lines, seed=0):
    mask_vdrs = np.zeros((n1, n2), dtype="bool")
    low1 = (n2 - init_lines) // 2
    low2 = (n2 + init_lines) // 2
    mask_vdrs[:, low1:low2] = True
    nlinesout = (nlines - init_lines) // 2
    rng = np.random.default_rng(seed)
    t1 = rng.choice(low1 - 1, size=nlinesout, replace=False)
    t2 = rng.choice(np.arange(low2 + 1, n2), size=nlinesout, replace=False)
    mask_vdrs[:, t1] = True
    mask_vdrs[:, t2] = True
    return mask_vdrs


class ResidualGuidedZeroShotDIP:
    """
    Residual-guided zero-shot DIP reconstruction.

    This class keeps the same external calling pattern as the original file:
        runner = ResidualGuidedZeroShotDIP(args, config)
        runner.sample()
    """

    def __init__(self, args, config, device=None):
        self.args = args
        self.args.image_folder = Path(self.args.image_folder)
        for folder_name in ["input", "recon", "label"]:
            if folder_name == "recon":
                (self.args.image_folder / folder_name / "progress").mkdir(
                    exist_ok=True, parents=True
                )
            else:
                (self.args.image_folder / folder_name).mkdir(
                    exist_ok=True, parents=True
                )

        self.config = config
        if device is None:
            device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

    def sample(self):
        """Compatibility entrypoint used by the existing main script."""
        self.run()

    def run(self):
        args, config = self.args, self.config
        print(f"Dataset path: {self.args.dataset_path}")
        root = Path(self.args.dataset_path)

        if self.args.deg == "SV-CT":
            a_funcs = CT(
                img_width=256,
                radon_view=self.args.Nview,
                uniform=True,
                circle=False,
                device=config.device,
            )
        elif self.args.deg == "LA-CT":
            a_funcs = CT(
                img_width=256,
                radon_view=self.args.Nview,
                uniform=False,
                circle=False,
                device=config.device,
            )
        else:
            a_funcs = None

        npz_files = sorted(root.glob("*.npz"))
        print(npz_files)

        basename = os.path.basename(self.args.dataset_path)
        save_name = f"test/{basename}_{self.args.acc_factor}"
        os.makedirs(save_name, exist_ok=True)

        source_file = Path(__file__).resolve()
        target_file = Path(save_name) / source_file.name
        shutil.copy2(source_file, target_file)

        for fname in npz_files:
            if "MRI" in args.deg:
                filename_mps = fname
                print(filename_mps.stem)

                a_temp = np.load(f"{fname}")
                for name in ["k_r", "k_i", "s_r", "s_i"]:
                    arr = a_temp[name]
                    print(
                        name,
                        "dtype:",
                        arr.dtype,
                        "min:",
                        arr.min(),
                        "max:",
                        arr.max(),
                    )

                k_r = a_temp["k_r"] / 32767.0
                k_i = a_temp["k_i"] / 32767.0
                s_r = a_temp["s_r"] / 32767.0
                s_i = a_temp["s_i"] / 32767.0

                _, nx, ny = k_r.shape
                k_np_ = np.stack(
                    (
                        k_r[
                            :,
                            nx // 2 - 160 : nx // 2 + 160,
                            ny // 2 - 160 : ny // 2 + 160,
                        ],
                        k_i[
                            :,
                            nx // 2 - 160 : nx // 2 + 160,
                            ny // 2 - 160 : ny // 2 + 160,
                        ],
                    ),
                    axis=0,
                )
                s_np = np.stack(
                    (
                        s_r[
                            :,
                            nx // 2 - 160 : nx // 2 + 160,
                            ny // 2 - 160 : ny // 2 + 160,
                        ],
                        s_i[
                            :,
                            nx // 2 - 160 : nx // 2 + 160,
                            ny // 2 - 160 : ny // 2 + 160,
                        ],
                    ),
                    axis=0,
                )

                k_tor = torch.tensor(k_np_, dtype=torch.float32).permute(1, 0, 2, 3)
                k_tor = k_tor[:, 0, :, :] + 1j * k_tor[:, 1, :, :]
                s_tor = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)
                s_tor = s_tor[:, 0, :, :] + 1j * s_tor[:, 1, :, :]

                x_orig = ksp_and_mps_to_gt(k_tor, s_tor)
                print(x_orig.shape, ifft_with_shifts(k_tor).shape)

                scale = x_orig.abs().max()
                print("before scale, abs max =", scale.item())
                x_orig = x_orig / (scale + 1e-8)
            else:
                x_orig = torch.from_numpy(
                    np.load(os.path.join(root, fname), allow_pickle=True)
                )

            if args.deg == "MRI-single":
                a_funcs = SinglecoilMRI_comp(config.data.image_size, mask)
            elif args.deg == "MRI-multi":
                nx, ny = 320, 320
                mask = make_vdrs_mask(
                    nx,
                    ny,
                    int(ny * 1 / self.args.acc_factor),
                    int(ny * 0.32 / self.args.acc_factor),
                ).astype(np.float32)

                mask = torch.from_numpy(mask).to(self.device)
                mps = s_tor
                n_coil, _, _ = mps.shape
                mps = mps.view(
                    1, n_coil, config.data.image_size, config.data.image_size
                ).to(self.device)
                a_funcs = MulticoilMRI(config.data.image_size, mask, mps)

            A = lambda z: a_funcs.A(z)
            Ap = lambda z: a_funcs.A_dagger(z)

            h, w = x_orig.shape
            x_orig = x_orig.view(1, 1, h, w).to(self.device)
            y = A(x_orig)
            apy = Ap(y)
            mask_from_file = mask
            ksp1 = y
            mps1 = mps
            gt1 = x_orig.squeeze()

            device = self.device

            z0 = torch.zeros(1, 2, 320, 320, device=device)
            z0[:, 0, :, :] = apy.real
            z0[:, 1, :, :] = apy.imag
            z0 = z0.detach()

            net = MediumUNet(n_channels=2, n_classes=2).to(device)
            learning_rate = 1e-4
            mps_fixed = mps1.detach()

            with torch.no_grad():
                scale_factor = torch.linalg.norm(net(z0)) / torch.linalg.norm(
                    ksp_and_mps_to_gt(ksp1, mps1).to(device)
                )
                target_ksp = scale_factor * ksp1.to(device)
                uns_target_ksp = mask_from_file * target_ksp
                print("K-space scaled by: ", scale_factor)

            show_every = 500
            avg_psnrs = []
            losses = []

            def forward_model(img_complex, mps_complex):
                """
                Physics forward model: S * x -> FFT -> y
                img_complex: (Batch, H, W) complex
                mps_complex: (Batch, Coil, H, W) complex
                Returns: (Batch, Coil, H, W) complex k-space
                """
                coil_imgs = mps_complex * img_complex
                pred_ksp = fft_with_shifts(coil_imgs)
                return pred_ksp

            def tv_loss(img):
                h_tv = torch.abs(img[..., 1:, :] - img[..., :-1, :]).sum()
                w_tv = torch.abs(img[..., :, 1:] - img[..., :, :-1]).sum()
                return h_tv + w_tv

            residual_gate = torch.ones_like(z0, device=device)
            gate_alpha = 0.1
            min_gate = 0.05
            xi_scale = z0.abs().amax().detach() / 2.0

            total_steps = 6000
            optimizer_img = optim.Adam(net.parameters(), lr=learning_rate)

            pbar = tqdm(range(total_steps))
            for step in pbar:
                optimizer_img.zero_grad()

                xi_t = xi_scale * torch.randn_like(z0)
                z_t = z0 + residual_gate * xi_t

                img_out = net(z_t)
                img_complex = torch.view_as_complex(
                    img_out.permute(0, 2, 3, 1).contiguous()
                ).squeeze(0)

                pred_ksp_complex = forward_model(img_complex, mps_fixed)
                sampled_residual = (
                    uns_target_ksp - mask_from_file * pred_ksp_complex.squeeze()
                )

                loss_tv = tv_loss(img_out)
                dc_loss = torch.linalg.norm(sampled_residual)
                loss = dc_loss + 0.001 * loss_tv

                loss.backward()
                optimizer_img.step()

                with torch.no_grad():
                    g_t = residual_to_gate(sampled_residual.unsqueeze(0), mps_fixed)
                    residual_gate = gate_alpha * g_t + (1.0 - gate_alpha) * residual_gate
                    residual_gate = torch.clamp(residual_gate, min_gate, 1.0)

                    out = torch.abs(
                        ksp_and_mps_to_gt(pred_ksp_complex.detach(), mps_fixed)
                    ).squeeze().cpu()
                    out /= torch.max(out)
                    out = out.detach().squeeze()
                    gt1 = torch.abs(gt1) / torch.max(torch.abs(gt1))

                    losses.append(loss.item())
                    avg_psnr = compute_psnr(
                        np.array(torch.abs(gt1.cpu())),
                        np.array(out.cpu()) / float(out.max().item()),
                    )
                    avg_psnrs.append(avg_psnr)

                epoch = step
                save_dir = f"{save_name}/test_{filename_mps.stem}"
                os.makedirs(save_dir, exist_ok=True)

                if step % show_every == 0:
                    out_avg_np = out.cpu().numpy()
                    gt1_np = gt1.cpu().numpy()
                    mask_from_file_np = mask_from_file.cpu().numpy()

                    out_avg_np = out_avg_np / out_avg_np.max()
                    gt1_np = gt1_np / gt1_np.max()

                    plt.imsave(
                        os.path.join(save_dir, f"out_avg_epoch{epoch}.png"),
                        out_avg_np,
                        cmap="gray",
                    )

                    if epoch == 0:
                        lr_psnr = compute_psnr(
                            np.array(torch.abs(gt1.cpu())),
                            np.array(np.abs(clear(apy)))
                            / float(np.abs(clear(apy)).max().item()),
                        )
                        save_path = os.path.join(save_dir, "input.png")

                        plt.figure()
                        plt.imshow(np.abs(clear(apy)), cmap="gray")
                        plt.title(f"PSNR: {lr_psnr:.2f} dB")
                        plt.axis("off")
                        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

                    if epoch == 0:
                        plt.imsave(
                            os.path.join(save_dir, f"gt1_epoch{epoch}.png"),
                            np.abs(gt1_np),
                            cmap="gray",
                        )
                        plt.imsave(
                            os.path.join(save_dir, f"mask_epoch{epoch}.png"),
                            np.abs(mask_from_file_np.squeeze()),
                            cmap="gray",
                        )

                    plt.figure(figsize=(12, 12))

                    plt.subplot(131)
                    plt.imshow(out_avg_np, cmap="gray")
                    plt.title(f"Sliding Average\\nPSNR = {round(avg_psnr, 2)}")

                    plt.subplot(132)
                    plt.imshow(np.abs(gt1_np), cmap="gray")
                    plt.title("Ground Truth")

                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(save_dir, f"comparison_epoch{epoch}.png"),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()

                    max_idx = max(range(len(avg_psnrs)), key=lambda i: avg_psnrs[i])

                    with open(f"{save_dir}/A_psnr_results.txt", "w") as f:
                        f.write(f"max:{avg_psnrs[max_idx]:.4f}\\n")
                        f.write(f"input:{lr_psnr:.4f}\\n")
                        for i, value in enumerate(avg_psnrs):
                            if i != max_idx:
                                f.write(f"{value:.4f}\\n")


