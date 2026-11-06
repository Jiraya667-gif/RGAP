# RGAP: Residual-Guided Adaptive Prior for Zero-Shot Multi-Coil MRI Reconstruction

This repository contains the reference implementation of **RGAP** (Residual-Guided Adaptive Prior), a scan-specific and training-free method for multi-coil MRI reconstruction.

RGAP keeps the simplicity of DIP-style optimization, but replaces the fixed stochastic input with a **residual-guided adaptive prior**. At each iteration, the sampled k-space residual is backprojected to the image domain, converted into a spatial gate, smoothed over time, and used to modulate the next perturbation of the base prior.

## Highlights

- **Training-free / scan-specific** reconstruction
- **No extra trainable parameters**
- **Residual-guided input adaptation**
- Works with **known sensitivity maps** stored together with the data
- Reproducible evaluation with a **fixed selected-case list**

## Repository layout

A typical minimal layout is:

```text
.
├── main.py
├── Residual_Guided.py
├── unet.py
├── utils_.py
├── physics/
│   ├── ct.py
│   └── mri.py
├── configs/
│   └── vp/
│       └── *.yml
└── data/
    ├── knee/
    ├── Brain_data/T1/
    └── Brain_data/T2/
```

`main.py` is the entry point.  
`Residual_guided.py` contains the main reconstruction class `ResidualGuidedZeroShotDIP`.

## Installation

Create a Python environment and install the basic dependencies:

```bash
pip install torch numpy matplotlib scikit-image tqdm pyyaml
```

If your local environment already contains PyTorch with CUDA support, you can reuse it directly.

## Expected data format

Each sample is stored as a single `.npz` file. The current multi-coil MRI pipeline expects the following arrays:

- `k_r`: real part of multi-coil k-space
- `k_i`: imaginary part of multi-coil k-space
- `s_r`: real part of the coil sensitivity maps
- `s_i`: imaginary part of the coil sensitivity maps

In the current implementation these arrays are loaded and rescaled by `32767.0`, so the released `.npz` files should follow the same convention as the files used in our experiments.

Each `.npz` file corresponds to **one 2D multi-coil slice/case** used for evaluation.

## Evaluation protocol used in the paper

- Datasets: **fastMRI Knee**, **Brain(T1)**, and **Brain(T2)**
- Acceleration factors: **x4** and **x8**
- Sampling pattern: **random masks**
- Coil sensitivity maps are treated as **fixed inputs**
- For each subset, **15 cases were randomly selected once and then kept fixed**
- All compared scan-specific methods use the **same case list**
- Metrics are computed **per reconstructed slice/case**, and then averaged over the 15 fixed cases

## Fixed selected cases

The following 15 cases per subset are the fixed evaluation list used in our experiments.

### Knee (15 cases)

```text
Kpsace_smap0351.npz
Kpsace_smap0371.npz
Kpsace_smap0411.npz
Kpsace_smap0438.npz
Kpsace_smap0471.npz
Kpsace_smap0537.npz
Kpsace_smap0575.npz
Kpsace_smap0617.npz
Kpsace_smap0655.npz
Kpsace_smap0700.npz
Kpsace_smap0730.npz
Kpsace_smap0754.npz
Kpsace_smap0800.npz
Kpsace_smap0840.npz
Kpsace_smap0862.npz
```

### Brain(T1) (15 cases)

```text
file_brain_AXT1POST_203_6000630_07.npz
file_brain_AXT1POST_203_6000630_10.npz
file_brain_AXT1POST_207_2070506_01.npz
file_brain_AXT1POST_210_6001591_02.npz
file_brain_AXT1POST_210_6001714_07.npz
file_brain_AXT1PRE_201_6002726_01.npz
file_brain_AXT1PRE_201_6002726_08.npz
file_brain_AXT1PRE_203_6000622_09.npz
file_brain_AXT1PRE_203_6000758_07.npz
file_brain_AXT1PRE_209_6001430_05.npz
file_brain_AXT1PRE_209_6001430_09.npz
file_brain_AXT1PRE_210_6001689_05.npz
file_brain_AXT1PRE_210_6001695_09.npz
file_brain_AXT1PRE_210_6001791_06.npz
file_brain_AXT1PRE_210_6001791_10.npz
```

### Brain(T2) (15 cases)

```text
file_brain_AXT2_200_2000205_08.npz
file_brain_AXT2_200_2000271_05.npz
file_brain_AXT2_200_2000341_08.npz
file_brain_AXT2_200_2000534_09.npz
file_brain_AXT2_200_2000596_03.npz
file_brain_AXT2_200_6002153_03.npz
file_brain_AXT2_200_6002153_09.npz
file_brain_AXT2_200_6002174_06.npz
file_brain_AXT2_200_6002283_09.npz
file_brain_AXT2_200_6002398_10.npz
file_brain_AXT2_200_6002495_10.npz
file_brain_AXT2_200_6002602_05.npz
file_brain_AXT2_200_6002623_08.npz
file_brain_AXT2_201_2010103_08.npz
file_brain_AXT2_201_2010190_05.npz
```

## Running reconstruction

### Generic command

```bash
python main.py \
  --type 2d \
  --config <config_name>.yml \
  --dataset_path <path_to_subset_folder> \
  --deg MRI-multi \
  --acc_factor 8 \
  --device cuda \
  -i ./results
```

### Example: Knee, x8

```bash
python main.py \
  --type 2d \
  --config fastmri_knee_320_complex.yml \
  --dataset_path data/knee \
  --deg MRI-multi \
  --acc_factor 8 \
  --device cuda \
  -i ./results
```

For `Brain(T1)` and `Brain(T2)`, replace `--dataset_path` with the corresponding folder and use the matching YAML config in `configs/vp/`.

## Default settings in the current release

The current implementation uses the following defaults in the MRI-multi pipeline:

- random seed: `1234`
- optimization steps: `6000`
- optimizer: `Adam`
- learning rate: `1e-4`
- TV weight: `1e-3`
- gate EMA factor: `0.05`
- minimum gate value: `0.05`
- perturbation scale: `0.5 * max(|z0|)`

The residual gate is built from the backprojected sampled residual and updated at every iteration.

## Reproducibility notes

- The selected case list above is **fixed**.
- The current code sets the global random seed with `--seed` (default: `1234`).
- In the current implementation, the undersampling mask is generated deterministically by `make_vdrs_mask(..., seed=0)`.
- To reproduce a multi-run average, repeat the same command with different `--seed` values and average the reported PSNR numbers across runs.

## Output files

In the current implementation, the per-case outputs are written under:

```text
test_randn/<subset_name>_<acc_factor>/test_<case_id>/
```

Typical saved files include:

- `A_psnr_results.txt`: best PSNR / input PSNR / logged PSNR values
- `input.png`: zero-filled or adjoint initialization
- `out_avg_epoch*.png`: reconstruction snapshots
- `comparison_epoch*.png`: qualitative comparison snapshots
- `gt1_epoch0.png`: ground truth image
- `mask_epoch0.png`: undersampling mask

## Notes

- The current release assumes that **sensitivity maps are already available** in each `.npz` file.
- The code path above focuses on the **MRI-multi** setting used in the paper.
- If you want all outputs to be redirected to a custom folder, you may additionally adjust the `save_name` logic in `Residual_Guided.py`.

## Citation

If you use this code, please cite:

```bibtex
@article{rgap2025,
  title={Residual-Guided Adaptive Prior for Zero-Shot Multi-Coil MRI Reconstruction},
  author={Anonymous},
  journal={Under review},
  year={2025}
}
```
