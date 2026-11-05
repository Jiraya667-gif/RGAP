"""One-factor-at-a-time sweeps corresponding to response R1C6/R2C6."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
from experiments.rgap.cli import dispatch, jobs_for, parser


def main():
    p = parser(__doc__)
    p.set_defaults(datasets=["knee"], accelerations=[8])
    p.add_argument("--parameter", choices=["all", "sigma", "alpha", "gmin"], default="all")
    p.add_argument("--sigma-values", nargs="+", type=float, default=[.25, .5, .75, 1.0])
    p.add_argument("--alpha-values", nargs="+", type=float, default=[.05, .1, .2])
    p.add_argument("--gmin-values", nargs="+", type=float, default=[.01, .05, .1])
    args = p.parse_args()
    base = jobs_for(args, ["rgap"])
    grids = {"sigma": args.sigma_values, "alpha": args.alpha_values, "gmin": args.gmin_values}
    if args.parameter != "all":
        grids = {args.parameter: grids[args.parameter]}
    jobs = [(replace(cfg, **{parameter: value}), case)
            for cfg, case in base for parameter, values in grids.items() for value in values]
    dispatch(args, jobs, "OFAT: non-varied parameters held at the supplied defaults")


if __name__ == "__main__":
    main()
