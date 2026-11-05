"""Trajectory, reconstruction/MAE, component ablation, and gate visualizations."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summaries(root):
    return [(p, json.loads(p.read_text())) for p in Path(root).rglob("summary.json")]


def trajectories(items, output):
    groups = {}
    for path, s in items:
        if not (path.parent/"trajectory.csv").exists():
            continue
        c = s["config"]
        # Keep sensitivities and protocols apart instead of pooling them.
        key = tuple(c[k] for k in ("dataset", "acceleration", "method", "sigma", "alpha", "gmin",
                                   "gamma", "tv", "lr", "protocol", "backbone", "steps"))
        with open(path.parent/"trajectory.csv") as f:
            rows = list(csv.DictReader(f))
        groups.setdefault(key, []).append(np.array([[float(r["step"]),float(r["psnr"])] for r in rows]))
    for i, (key, curves) in enumerate(groups.items()):
        if any(not np.array_equal(c[:, 0], curves[0][:, 0]) for c in curves):
            raise ValueError("Trajectories use different logged iterations.")
        a = np.stack(curves)
        mean, sd = a[:, :, 1].mean(0), a[:, :, 1].std(0)
        fig, ax = plt.subplots(figsize=(5.3, 3.4))
        ax.plot(a[0, :, 0], mean)
        ax.fill_between(a[0, :, 0], mean-sd, mean+sd, alpha=.2)
        ax.set(xlabel="Iteration", ylabel="PSNR (dB)",
               title=f"{key[0]} x{key[1]} {key[2]} | mean +/- SD over case/run pairs")
        ax.grid(alpha=.2)
        fig.tight_layout()
        name = f"trajectory_{key[0]}_x{key[1]}_{key[2]}_{i}"
        fig.savefig(output/f"{name}.pdf"); fig.savefig(output/f"{name}.png", dpi=180)
        plt.close(fig)
        np.savetxt(output/f"{name}.csv", np.column_stack((a[0, :, 0],mean,sd)),
                   delimiter=",", header="step,mean_psnr,sd_across_case_runs", comments="")


def qualitative(items, output, args):
    selected = {}
    references = []
    for path, s in items:
        c = s["config"]
        if (s["case"] != args.case or c["dataset"] != args.dataset or c["acceleration"] != args.acceleration
                or c["seed"] != args.seed or c["method"] not in args.methods):
            continue
        if (c["sigma"], c["alpha"], c["gmin"], c["gamma"]) != (1.0,.05,.05,.8):
            continue
        if c["method"] in selected:
            raise ValueError("Multiple eligible runs for one method; use a single protocol/output root.")
        arrays = np.load(path.parent/"reconstructions.npz", allow_pickle=False)
        selected[c["method"]] = (arrays[args.checkpoint], s[args.checkpoint])
        references.append(arrays["reference"] if "reference" in arrays else None)
    if not selected or any(r is None for r in references):
        raise ValueError("No matching per-case images/reference; run native experiments first.")
    reference = references[0]
    if not all(np.allclose(r, reference) for r in references):
        raise ValueError("Methods do not share the same reference image.")
    missing = set(args.methods)-set(selected)
    if missing:
        raise ValueError(f"Missing methods in qualitative panel: {sorted(missing)}")
    if args.crop:
        x, y, w, h = args.crop
        if min(x, y) < 0 or min(w, h) < 1 or x+w>reference.shape[1] or y+h>reference.shape[0]:
            raise ValueError("Crop exceeds image bounds.")
        view = lambda a: a[y:y+h,x:x+w]
    else:
        view = lambda a: a
    names = ["Reference"]+args.methods
    images = [reference]+[selected[m][0] for m in args.methods]
    errors = [np.abs(im-reference) for im in images]
    vmax = args.error_max or max(float(e.max()) for e in errors)
    fig, axs = plt.subplots(2, len(names), figsize=(3*len(names),6), squeeze=False)
    for j, (name, image, error) in enumerate(zip(names, images, errors)):
        title = name if j == 0 else name+f"\n{selected[name][1]['psnr']:.2f} / {selected[name][1]['ssim']:.4f}"
        axs[0,j].imshow(view(image), cmap="gray", vmin=0,vmax=1)
        axs[0,j].set_title(title)
        im = axs[1,j].imshow(view(error), cmap="magma",vmin=0,vmax=max(vmax,1e-8))
        axs[1,j].set_title("MAE (shared scale)")
        axs[0,j].axis("off"); axs[1,j].axis("off")
    fig.colorbar(im, ax=axs[1,:].tolist(), shrink=.6)
    fig.suptitle(f"{args.dataset} x{args.acceleration} | {args.case} | {args.checkpoint}")
    fig.subplots_adjust(top=.85, bottom=.05, wspace=.08)
    name = f"qualitative_{args.dataset}_x{args.acceleration}_{args.case}_{args.checkpoint}"
    fig.savefig(output/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(output/f"{name}.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("results"))
    p.add_argument("--output",type=Path,default=Path("results/figures"))
    p.add_argument("--kind",choices=["trajectories","qualitative"],default="trajectories")
    p.add_argument("--dataset",default="knee")
    p.add_argument("--acceleration",type=int,default=8)
    p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--case",default="synthetic")
    p.add_argument("--methods",nargs="+",default=["dip","no_ema","no_tv","rgap"])
    p.add_argument("--checkpoint",choices=["best","final"],default="final")
    p.add_argument("--crop",nargs=4,type=int,metavar=("X","Y","W","H"))
    p.add_argument("--error-max",type=float)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    items=summaries(args.root)
    if not items:
        p.error("No real run summaries found; figures are never drawn from reported means.")
    if args.kind=="trajectories":
        trajectories(items,args.output)
    else:
        qualitative(items,args.output,args)


if __name__=="__main__":
    main()
