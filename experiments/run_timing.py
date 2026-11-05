"""Optional new measurements; no manuscript timing numbers are invented."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
from experiments.rgap.cli import dispatch, jobs_for, parser
from experiments.rgap.engine import NATIVE


def main():
    p = parser(__doc__)
    p.set_defaults(datasets=["knee"], accelerations=[8], save_every=0)
    p.add_argument("--methods", nargs="+", choices=NATIVE, default=["dip", "rescat", "rgap"])
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()
    jobs = [(replace(c, timing=True, warmup=args.warmup), path)
            for c, path in jobs_for(args, args.methods)]
    dispatch(args, jobs, "Same-device synchronized timing with evaluation disabled")


if __name__ == "__main__":
    main()
