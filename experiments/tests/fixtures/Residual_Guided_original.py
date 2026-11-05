import os
import json
import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from unet import UNet, FullUNet, MediumUNet, OneLayerUNet, ExtraDeepUNet, SuperDeepUNet  # SmoothCoilGen disabled for fixed-CSM MRKC
import torch.optim as optim
import torch.fft as fft
import torch.nn as nn
from tqdm.auto import tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import glob
# import two_channel_dataset_DIP_github
from torch.nn import init
from torch.autograd import Variable
from guided_diffusion.script_util import create_model

from skimage.metrics import peak_signal_noise_ratio
from pathlib import Path

from physics.ct import CT
from physics.mri import SinglecoilMRI_comp, MulticoilMRI
#from utils_ import CG, clear, get_mask, get_mask_fix, nchw_comp_to_real, real_to_nchw_comp, normalize_np, get_beta_schedule
from utils_ import CG, clear, get_mask, nchw_comp_to_real, real_to_nchw_comp, normalize_np, get_beta_schedule
from functools import partial
import math
import shutil

def to_vis_2d(x, eps=1e-8):
    """
    x: torch tensor, shape can be
       (1,2,H,W), (2,H,W), (1,1,H,W), (H,W)
    return: normalized 2D numpy array
    """
    if torch.is_tensor(x):
        x = x.detach().float().cpu()

    if x.ndim == 4:
        x = x[0]   # -> (C,H,W)
    if x.ndim == 3:
        if x.shape[0] == 2:
            # 两通道 prior / noise: 合成单张强度图
            x = torch.sqrt((x ** 2).sum(dim=0))
        else:
            x = x[0]
    elif x.ndim != 2:
        raise ValueError(f"Unsupported shape for visualization: {x.shape}")

    x = x.numpy()
    x = x - x.min()
    x = x / (x.max() + eps)
    return x


def save_gray_map(x, save_path, cmap="magma"):
    arr = to_vis_2d(x)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, arr, cmap=cmap)


def sos(x):
    """Sum of Squares（线圈域平方和合并）
    输入：实虚部分开的张量 → 输出：单通道幅度图像
    Args:
        x: 张量，维度为 [B, 2*N_coil, H, W] 或 [N_coil, 2, H, W]（核心：2个通道对应实/虚部）
    Returns:
        sos_img: 单通道幅度图像，维度 [B, 1, H, W] 或 [1, H, W]
    """
    x_mag_sq = torch.abs(x) ** 2  # [15,320,320] → 每个位置的幅度平方
    # Step2: 对线圈维度（dim=0）求和（SOS核心）
    x_sos = torch.sum(x_mag_sq, dim=0)  # [320,320] → 合并所有线圈
    # Step3: 开平方 + 增加通道维度（兼容网络输入）
    x_sos = torch.sqrt(x_sos)  # [320,320]
    x_sos = torch.unsqueeze(x_sos, 0)  # [1,320,320]
    return x_sos

def fft_with_shifts(img):
    return fft.fftshift(fft.fft2(fft.ifftshift(img)))


def ifft_with_shifts(ksp):
    return fft.fftshift(fft.ifft2(fft.ifftshift(ksp)))


def ksp_and_mps_to_gt(ksp, mps):
    gt = mps.conj() * ifft_with_shifts(ksp)
    gt = torch.sum(gt.squeeze(), axis=0)
    return gt


def complex_to_2ch(x_complex):
    """Convert complex-valued (H, W) image to (1, 2, H, W) real tensor."""
    return torch.stack([x_complex.real, x_complex.imag], dim=0).unsqueeze(0)


def residual_to_gate(k_residual, mps_complex, eps=1e-8, gamma=0.8):
    """
    Build an image-domain residual gate in [0, 1] from sampled k-space residual.
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

def make_vdrs_mask(N1, N2, nlines, init_lines, seed=0):
    mask_vdrs = np.zeros((N1, N2), dtype='bool')
    low1 = (N2 - init_lines) // 2
    low2 = (N2 + init_lines) // 2
    mask_vdrs[:, low1:low2] = True
    nlinesout = (nlines - init_lines) // 2
    rng = np.random.default_rng(seed)
    t1 = rng.choice(low1 - 1, size=nlinesout, replace=False)
    t2 = rng.choice(np.arange(low2 + 1, N2), size=nlinesout, replace=False)
    mask_vdrs[:, t1] = True;
    mask_vdrs[:, t2] = True
    return mask_vdrs

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.args.image_folder = Path(self.args.image_folder)
        for t in ["input", "recon", "label"]:
            if t == "recon":
                (self.args.image_folder / t / "progress").mkdir(exist_ok=True, parents=True)
            else:
                (self.args.image_folder / t).mkdir(exist_ok=True, parents=True)
        self.config = config
        if device is None:
            device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        config_dict = vars(self.config.model)
        model = create_model(**config_dict)

        self.dds(model)

    def dds(self, model):
        args, config = self.args, self.config
        print(f"Dataset path: {self.args.dataset_path}")
        root = Path(self.args.dataset_path)

        if self.args.deg == "SV-CT":
            A_funcs = CT(img_width=256, radon_view=self.args.Nview, uniform=True, circle=False, device=config.device)
        elif self.args.deg == "LA-CT":
            A_funcs = CT(img_width=256, radon_view=self.args.Nview, uniform=False, circle=False, device=config.device)

        # A = lambda z: A_funcs.A(z)
        # Ap = lambda z: A_funcs.A_dagger(z)
        npz_files = sorted(root.glob("*.npz"))
        print(npz_files)


        #save_name = "test"  # 保存目录，可改为你想要的路径
        basename = os.path.basename(self.args.dataset_path)
        save_name = f"test/{basename}_{self.args.acc_factor}"
        os.makedirs(save_name, exist_ok=True)

        source_file = os.path.join(os.getcwd(), "Residual_Guided.py")
        # 目标文件：save_dir 下的 solver_2d.py
        target_file = os.path.join(save_name, "Residual_Guided.py")

            # 3. 确保目标文件夹存在（不存在则创建）
        os.makedirs(save_name, exist_ok=True)
                    
            # 4. 复制文件（copy2 会保留文件的元数据，比 copy 更推荐）
        shutil.copy2(source_file, target_file)

        for idx, fname in enumerate(npz_files):
            if "MRI" in args.deg:
                filename_mps = fname
                print(filename_mps.stem)
                #filename = Path(root)

                #filename_mps = Path(root) / fname
                A_temp = np.load(f"{fname}")
                for name in ['k_r', 'k_i', 's_r', 's_i']:
                    arr = A_temp[name]
                    print(name, "dtype:", arr.dtype,
                          "min:", arr.min(), "max:", arr.max())
                k_r = A_temp['k_r'] / 32767.0
                k_i = A_temp['k_i'] / 32767.0
                s_r = A_temp['s_r'] / 32767.0
                s_i = A_temp['s_i'] / 32767.0
                _, nx, ny = k_r.shape
                k_np_ = np.stack((k_r[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160],
                                  k_i[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]), axis=0)
                s_np = np.stack((s_r[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160],
                                 s_i[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]), axis=0)

                k_tor = torch.tensor(k_np_, dtype=torch.float32).permute(1, 0, 2, 3)
                k_tor = k_tor[:, 0, :, :] + 1j * k_tor[:, 1, :, :]
                s_tor = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)
                s_tor = s_tor[:, 0, :, :] + 1j * s_tor[:, 1, :, :]
                # # print(k_tor.shape,s_tor.shape)
                # x_orig = sos(ifft_with_shifts(k_tor)).squeeze()
                x_orig = ksp_and_mps_to_gt(k_tor, s_tor)
                coil_count, _, _ = ifft_with_shifts(k_tor).shape
                print(x_orig.shape,ifft_with_shifts(k_tor).shape)


                # x_orig = ksp_and_mps_to_gt(k_tor, s_tor) #不用线圈敏感图，用sos
                scale = x_orig.abs().max()
                print("before scale, abs max =", scale.item())

                # 避免除 0
                eps = 1e-8
                x_orig = x_orig / (scale + eps)


                # x_orig = torch.from_numpy(k_r)
            else:
                x_orig = torch.from_numpy(np.load(os.path.join(root, fname), allow_pickle=True))

            # For MRI, we have to redefine the forward operator for every slice,
            # as we have different sensitivity maps for multi-coil MRI.
            if args.deg == "MRI-single":
                # mask = get_mask(torch.zeros([1, 1, config.data.image_size, config.data.image_size]),
                #                 config.data.image_size,
                #                 config.sampling.batch_size, type=args.mask_type,
                #                 acc_factor=args.acc_factor, center_fraction=args.center_fraction)
                # mask = mask.to(self.device)
                A_funcs = SinglecoilMRI_comp(config.data.image_size, mask)
            elif args.deg == "MRI-multi":
                # mask = get_mask(torch.zeros([1, 1, config.data.image_size, config.data.image_size]),
                #                 config.data.image_size,
                #                 config.sampling.batch_size, type=args.mask_type,
                # #                 acc_factor=args.acc_factor, center_fraction=args.center_fraction)
                # mask = get_mask_fix(torch.zeros([1, 1, config.data.image_size, config.data.image_size]),
                #                     config.data.image_size,
                #                     config.sampling.batch_size, type="gaussian1d",
                #                     acc_factor=16, center_fraction=0.02)
                nx, ny = 320, 320
                #mask = make_vdrs_mask(nx, ny, int(ny * 0.125), int(ny * 0.04)).astype(np.float32) #G8
                mask = make_vdrs_mask(nx, ny, int(ny * 1 / self.args.acc_factor), int(ny * 0.32 / self.args.acc_factor)).astype(np.float32) 

                mask = torch.from_numpy(mask)
                mask = mask.to(self.device)
                mps = s_tor
                Ncoil, _, _ = mps.shape
                mps = mps.view(1, Ncoil, config.data.image_size, config.data.image_size).to(self.device)
                A_funcs = MulticoilMRI(config.data.image_size, mask, mps)
            A = lambda z: A_funcs.A(z)
            #A1 = lambda z: A_funcs.A1(z)
            AT = lambda z: A_funcs.AT(z)
            Ap = lambda z: A_funcs.A_dagger(z)

            h, w = x_orig.shape
            x_orig = x_orig.view(1, 1, h, w)
            x_orig = x_orig.to(self.device)
            #y = A1(x_orig)
            y = A(x_orig)
            Apy = Ap(y)
            mask_from_file = mask
            ksp1 = y
            mps1 = mps
            gt1 = x_orig.squeeze()


            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = self.device
            
            z0 = torch.zeros(1, 2, 320, 320, device=device)
            z0[:, 0, :, :] = Apy.real
            z0[:, 1, :, :] = Apy.imag
            z0 = z0.detach()   # fixed base prior, no gradient update


            net = MediumUNet(n_channels=2, n_classes=2).to(device)
            # NOTE: sensitivity-map estimation branch (SmoothCoilGen / optimizer_coil / MCMC) is disabled.
            # We use the pre-estimated CSM read from the input .npz file (MRKC / fixed-CSM setting).
            learning_rate = 1e-4
            mps_fixed = mps1.detach()

            with torch.no_grad():
                scale_factor = torch.linalg.norm(net(z0)) / torch.linalg.norm(
                    ksp_and_mps_to_gt(ksp1, mps1).to(device))
                target_ksp = scale_factor * ksp1.to(device)
                uns_target_ksp = mask_from_file * target_ksp
                print('K-space scaled by: ', scale_factor)

            num_epochs = 50
            show_every = 500
            avg_psnrs = []
            losses = []

            def forward_model(img_complex, mps_complex):
                """
                物理前向模型: S * x -> FFT -> y
                img_complex: (Batch, H, W) Complex
                mps_complex: (Batch, Coil, H, W) Complex
                Returns: (Batch, Coil, H, W) Complex K-space
                """
                coil_imgs = mps_complex * img_complex
                pred_ksp = fft_with_shifts(coil_imgs)
                return pred_ksp

            def mps_and_gt_to_ksp(mps, gt):
                ksp = fft_with_shifts(mps * gt)
                return ksp

            total_steps = 6000

            optimizer_img = optim.Adam(net.parameters(), lr=learning_rate)

            def tv_loss(img):
                """计算图像平滑度，越小越平滑"""
                h_tv = torch.abs(img[..., 1:, :] - img[..., :-1, :]).sum()
                w_tv = torch.abs(img[..., :, 1:] - img[..., :, :-1]).sum()
                return h_tv + w_tv

            # NOTE: the original sensitivity-map robustness utilities are disabled together with the
            # unknown-CSM branch. Keep only the adaptive image prior variables below.
            self.ksp_loss_base = None
            
            # fixed residual-guided prior state
            self.residual_gate = torch.ones_like(z0, device=device)
            gate_alpha = 0.1   # G_m^t = alpha * G^t + (1-alpha) * G^{t-1}
            min_gate = 0.05
            
            # fixed noise scale for xi^t
            xi_scale = z0.abs().amax().detach() / 2.0


            pbar = tqdm(range(total_steps))
            for step in pbar:
                optimizer_img.zero_grad()
            
                xi_t = xi_scale * torch.rand_like(z0)   # or torch.randn_like(z0)
                z_t = z0 + self.residual_gate * xi_t
            
                img_out = net(z_t)


                img_complex = torch.view_as_complex(img_out.permute(0, 2, 3, 1).contiguous()).squeeze(0)

                pred_ksp_complex = forward_model(img_complex, mps_fixed)
                sampled_residual = uns_target_ksp - mask_from_file * pred_ksp_complex.squeeze()

                loss_tv = tv_loss(img_out)
                dc_loss = torch.linalg.norm(sampled_residual)
                loss = dc_loss + 0.001 * loss_tv
                
                loss.backward()
                optimizer_img.step()


                with torch.no_grad():
                    G_t = residual_to_gate(sampled_residual.unsqueeze(0), mps_fixed)
                    self.residual_gate = gate_alpha * G_t + (1.0 - gate_alpha) * self.residual_gate
                    self.residual_gate = torch.clamp(self.residual_gate, min_gate, 1.0)


                    out = torch.abs(ksp_and_mps_to_gt(pred_ksp_complex.detach(), mps_fixed)).squeeze().cpu()
                    out /= torch.max(out)
                    out = out.detach().squeeze()
                    gt1 = torch.abs(gt1) / torch.max(torch.abs(gt1))

                    losses.append(loss.item())
                    avg_psnr = compute_psnr(np.array(torch.abs(gt1.cpu())),
                                            np.array(out.cpu()) / float(out.max().item()))
                    avg_psnrs.append(avg_psnr)
                epoch = step
                save_dir = f"{save_name}/test_{filename_mps.stem}"
                os.makedirs(save_dir, exist_ok=True)

                if step % show_every == 0:
                    '''save_gray_map(
                        self.residual_gate[0, 0],
                        os.path.join(save_dir, f"gate_epoch{epoch}.png"),
                        cmap="viridis"
                    )
            
                    save_gray_map(
                        noise_prior_before_gate,
                        os.path.join(save_dir, f"noise_prior_before_rgap_epoch{epoch}.png"),
                        cmap="magma"
                    )
            
                    save_gray_map(
                        noise_prior_after_gate,
                        os.path.join(save_dir, f"noise_prior_after_rgap_epoch{epoch}.png"),
                        cmap="magma"
                    )'''
                    out_avg_np = out.cpu().numpy()
                    gt1_np = gt1.cpu().numpy()

                    mask_from_file_np = mask_from_file.cpu().numpy()

                    # 归一化到 [0,1]，避免保存时亮度不对
                    out_avg_np = out_avg_np / out_avg_np.max()
                    gt1_np = gt1_np / gt1_np.max()

                    # 保存单张图像（独立保存）
                    plt.imsave(os.path.join(save_dir, f"out_avg_epoch{epoch}.png"), out_avg_np, cmap='gray')

                    if epoch == 0:
                        lr_psnr = compute_psnr(np.array(torch.abs(gt1.cpu())),
                                                np.array(np.abs(clear(Apy))) / float(np.abs(clear(Apy)).max().item()))
                        save_path = os.path.join(save_dir, "input.png")

                        plt.figure()
                        plt.imshow(np.abs(clear(Apy)), cmap='gray')
                        plt.title(f"PSNR: {lr_psnr:.2f} dB")  # 把 psnr 作为标题
                        plt.axis('off')  # 去掉坐标轴（可选）
                        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

                    if epoch == 0:
                        plt.imsave(os.path.join(save_dir, f"gt1_epoch{epoch}.png"), np.abs(gt1_np), cmap='gray')
                        plt.imsave(os.path.join(save_dir, f"mask_epoch{epoch}.png"), np.abs(mask_from_file_np.squeeze()),
                                   cmap='gray')

                    # 同时保存对比可视化图
                    plt.figure(figsize=(12, 12))

                    plt.subplot(131)
                    plt.imshow(out_avg_np, cmap='gray')
                    plt.title(f'Sliding Average\nPSNR = {round(avg_psnr, 2)}')

                    plt.subplot(132)
                    plt.imshow(np.abs(gt1_np), cmap='gray')
                    plt.title('Ground Truth')

                    plt.tight_layout()

                    # 保存整个可视化窗口
                    plt.savefig(os.path.join(save_dir, f"comparison_epoch{epoch}.png"), dpi=300, bbox_inches='tight')

                    plt.close()  # 不再弹窗显示，提高训练速度

                    max_idx = max(range(len(avg_psnrs)), key=lambda i: avg_psnrs[i])

                    with open(f'{save_dir}/A_psnr_results.txt', 'w') as f:
                        # 先写最大值
                        f.write(f'max:{avg_psnrs[max_idx]:.4f}\n')
                        f.write(f'input:{lr_psnr:.4f}\n')
                        # 再写剩下的（保持原来的顺序）
                        for i, v in enumerate(avg_psnrs):
                            if i != max_idx:
                                f.write(f'{v:.4f}\n')



def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
