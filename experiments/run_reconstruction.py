"""Main, component, Joint-Opt, ResCat, and fixed-budget experiment suites."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.rgap.cli import SUITES, dispatch, jobs_for, parser


def main():
    p = parser(__doc__)
    p.add_argument("--suite", choices=[*SUITES, "all"], default="ablation")
    p.add_argument("--methods", nargs="+", help="Explicit methods override suite selection.")
    args = p.parse_args()
    methods = args.methods or (tuple(dict.fromkeys(m for s in SUITES.values() for m in s))
                              if args.suite == "all" else SUITES[args.suite])
    from experiments.rgap.engine import NATIVE
    from experiments.rgap.cli import EXTERNAL
    unknown = set(methods)-set(NATIVE)-set(EXTERNAL)
    if unknown:
        p.error(f"Unknown methods: {sorted(unknown)}")
    dispatch(args, jobs_for(args, methods), f"{args.suite}: oracle + final + full trajectories")


if __name__ == "__main__":
    main()
