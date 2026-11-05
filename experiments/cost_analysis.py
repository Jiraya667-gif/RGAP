"""Reproduce the *assumed* arithmetic ratio used in R2C4, not a latency claim."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network-forward-macs", type=float, default=5.5e9)
    p.add_argument("--backward-factor", type=float, default=2.0)
    p.add_argument("--mac-to-flop", type=float, default=2.0)
    p.add_argument("--rgap-flops", type=float, default=6.9e7)
    p.add_argument("--output", type=Path, default=Path("results/cost_analysis.json"))
    args = p.parse_args()
    train_macs = args.network_forward_macs * (1+args.backward_factor)
    if train_macs <= 0 or args.mac_to_flop <= 0 or args.rgap_flops < 0:
        p.error("Operation counts and conversion must be non-negative, denominator positive.")
    result = {"kind": "illustrative_response_assumptions_not_profiled_operations",
        "coils": 15, "height": 320, "width": 320,
        "network_forward_macs": args.network_forward_macs,
        "backward_factor": args.backward_factor,
        "training_macs_exact_under_assumptions": train_macs,
        "response_rounded_training_macs": 1.7e10,
        "mac_to_flop": args.mac_to_flop, "rgap_flops": args.rgap_flops,
        "ratio_percent_exact_assumptions": 100*args.rgap_flops/(train_macs*args.mac_to_flop),
        "ratio_percent_using_response_rounded_denominator": 100*args.rgap_flops/(1.7e10*2),
        "elapsed_time": "not inferred", "warning":
        "The supplied sources do not establish the 5.5e9/6.9e7 counts by profiler or detailed operator enumeration."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
