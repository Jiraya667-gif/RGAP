"""Common CLI and a complete experiment planner, with no silent baseline skipping."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
from .engine import Config, NATIVE, run

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ("sg_dip", "udig_dip", "aseqdip", "dktm", "cd", "dd", "nila")
SUITES = {
    "main": ("rgap", "dip", "sg_dip", "udig_dip", "aseqdip", "dktm", "cd", "dd"),
    "ablation": ("dip", "no_ema", "no_tv", "rgap"),
    "joint": ("dip", "joint_opt", "rgap"),
    "rescat": ("dip", "rescat", "rgap"),
    "stability": ("rgap", "sg_dip", "udig_dip", "aseqdip", "dktm"),
    "mechanism": ("dip", "uniform_noise", "shuffled_gate", "rgap"),
    "nila": ("nila",),
}


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--datasets", nargs="+", choices=["knee", "t1", "t2"], default=["knee", "t1", "t2"])
    p.add_argument("--accelerations", nargs="+", type=int, choices=[4, 8], default=[4, 8])
    p.add_argument("--seeds", nargs="+", type=int, default=[1234, 1235, 1236],
                   help="New reproducible seed choice; historical three seeds were not supplied.")
    p.add_argument("--case-lists", type=Path, default=ROOT/"configs/cases.json")
    p.add_argument("--paths", type=Path, default=ROOT/"configs/data_paths.json")
    p.add_argument("--limit-cases", type=int, help="Smoke/debug subset; not a 15-case table.")
    p.add_argument("--case", help="Select an explicit filename from the fixed case list.")
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tv", type=float, default=1e-3)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--gmin", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--mask-seed", type=int, default=0)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--input-divisor", type=float, default=32767.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--backbone", choices=["medium", "tiny"], default="medium")
    p.add_argument("--protocol", choices=["source", "ortho"], default="source")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--synthetic", action="store_true", help="Analytic phantom, never paper data.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--plan", action="store_true", help="Write job plan without requiring data/GPU.")
    p.add_argument("--adapters", type=Path, default=ROOT/"configs/external_methods.json")
    p.add_argument("--allow-missing-baselines", action="store_true",
                   help="Run native jobs and write missing_baselines.json; not a complete main comparison.")
    return p


def base_config(args):
    return Config(steps=args.steps, lr=args.lr, tv=args.tv, sigma=args.sigma,
        alpha=args.alpha, gmin=args.gmin, gamma=args.gamma, size=args.size,
        divisor=args.input_divisor, device=args.device, backbone=args.backbone,
        protocol=args.protocol, mask_seed=args.mask_seed, eval_every=args.eval_every,
        save_every=args.save_every, synthetic=args.synthetic)


def cases_for(args):
    if args.synthetic:
        return {d: [None] for d in args.datasets}
    lists = json.loads(args.case_lists.read_text())
    paths = json.loads(args.paths.read_text())
    result = {}
    for ds in args.datasets:
        names = lists[ds]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate case names in {ds}.")
        if args.case:
            names = [n for n in names if n == args.case]
            if not names:
                raise ValueError(f"{args.case} is not in the {ds} case list.")
        if args.limit_cases is not None:
            if args.limit_cases < 1:
                raise ValueError("--limit-cases must be positive.")
            names = names[:args.limit_cases]
        result[ds] = [args.data_root / paths[ds] / n for n in names]
    return result


def jobs_for(args, methods):
    base = base_config(args)
    return [(replace(base, dataset=ds, acceleration=acc, seed=seed, method=method), case)
        for ds, cases in cases_for(args).items() for acc in args.accelerations
        for case in cases for seed in args.seeds for method in methods]


def dispatch(args, jobs, description):
    # De-duplicate a shared default used by multiple sweeps/suites.
    unique = {}
    for cfg, case in jobs:
        unique[(json.dumps(asdict(cfg), sort_keys=True), str(case))] = (cfg, case)
    jobs = list(unique.values())
    args.output.mkdir(parents=True, exist_ok=True)
    adapters = json.loads(args.adapters.read_text()) if args.adapters.exists() else {}
    missing = sorted({c.method for c, _ in jobs if c.method in EXTERNAL and not adapters.get(c.method)})
    plan = {"description": description, "job_count": len(jobs),
        "missing_external_implementations": missing,
        "historical_run_seeds_supplied": False,
        "jobs": [{"config": asdict(c), "case_path": str(p) if p else None,
                  "implementation": "native" if c.method in NATIVE else "external_adapter"}
                 for c, p in jobs]}
    plan_path = args.output/"experiment_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2)+"\n")
    print(f"{len(jobs)} jobs; plan: {plan_path}", flush=True)
    if args.plan:
        if missing:
            print("External implementations required: "+", ".join(missing))
        return
    if missing:
        (args.output/"missing_baselines.json").write_text(json.dumps(missing, indent=2)+"\n")
        if not args.allow_missing_baselines:
            raise SystemExit("Missing original baseline adapters: "+", ".join(missing)
                +". See experiments/EXTERNAL_BASELINES.md. No baseline is substituted.")
        print("INCOMPLETE comparison: external methods explicitly skipped: "+", ".join(missing))
    absent = sorted({str(p) for _, p in jobs if p is not None and not p.is_file()})
    if absent:
        raise SystemExit(f"Missing {len(absent)} data files; first: {absent[0]}")
    # Preflight completes before a long experiment starts.
    for cfg, case in jobs:
        if cfg.method in NATIVE:
            run(cfg, case, args.output, args.overwrite)
        elif cfg.method not in missing:
            from .external import run_external
            run_external(cfg, case, args.output, adapters[cfg.method], args.overwrite)
