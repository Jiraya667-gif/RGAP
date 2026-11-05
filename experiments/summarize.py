"""Aggregate real runs, make table CSV/LaTeX, and compare to uploaded numbers.

Average per-run PSNR-selected peaks, not the peak of an averaged trajectory.
SSIM always comes from the same checkpoint as the associated PSNR.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

PARAMETERS = ("sigma", "alpha", "gmin", "gamma", "tv", "lr", "protocol", "backbone", "steps")
DEFAULTS = (1.0, .05, .05, .8, .001, .0001, "source", "medium", 6000)


def write_csv(path, rows, fields=None):
    if not rows and fields is None:
        return
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("results"))
    p.add_argument("--output",type=Path,default=Path("results/tables"))
    p.add_argument("--reported",type=Path,default=Path(__file__).parent/"configs/reported_values.csv")
    p.add_argument("--expect-cases",type=int,default=15)
    p.add_argument("--expect-seeds",type=int,default=3)
    p.add_argument("--allow-incomplete",action="store_true",help="Write clearly labeled partial/debug summaries.")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    items=[]; times=[]; seen=set()
    for path in args.root.rglob("summary.json"):
        s=json.loads(path.read_text()); c=s["config"]
        if "timing" in s:
            times.append({"dataset":c["dataset"],"acceleration":c["acceleration"],"method":c["method"],
                          "case":s["case"],"seed":c["seed"],"device":s.get("device"),
                          **s["timing"]})
            continue
        key=(c["dataset"],c["acceleration"],c["method"],s["case"],c["seed"],
             *(c[k] for k in PARAMETERS))
        if key in seen:
            raise ValueError(f"Duplicate/conflicting run in {path}; select one output tree.")
        seen.add(key); items.append((path,s))
    if times:
        write_csv(args.output/"timing_per_run.csv",times)
    groups=defaultdict(list)
    flat=[]
    for path,s in items:
        c=s["config"]
        for checkpoint in ("best","final","contextual"):
            if checkpoint not in s:
                continue
            result=s[checkpoint]
            row={"dataset":c["dataset"],"acceleration":c["acceleration"],"method":c["method"],
                 "case":s["case"],"seed":c["seed"],"checkpoint":checkpoint,
                 "step":result.get("step"),"psnr":result["psnr"],"ssim":result["ssim"],
                 **{k:c[k] for k in PARAMETERS},"not_publication_run":s["not_publication_run"],
                 "source":str(path)}
            flat.append(row)
            groups[(c["dataset"],c["acceleration"],c["method"],checkpoint,
                    *(c[k] for k in PARAMETERS))].append((row,s))
    if not flat and not times:
        p.error("No evaluated runs. Reported manuscript numbers are never used as measured outputs.")
    write_csv(args.output/"per_run_metrics.csv",flat)
    aggregate=[]; incomplete=[]
    for key,pairs in groups.items():
        rows=[r for r,_ in pairs]; cases=defaultdict(list)
        sources={s.get("source_sha256",json.dumps(s.get("external_provenance",{}),sort_keys=True)) for _,s in pairs}
        # External provenance may have per-case output filenames; implementers
        # should retain a stable implementation/version across the dataset.
        native_sources={s["source_sha256"] for _,s in pairs if "source_sha256" in s}
        if len(native_sources)>1:
            raise ValueError("Native code versions differ inside one aggregate group.")
        for r in rows:
            cases[r["case"]].append(r)
        seed_sets=[{r["seed"] for r in rr} for rr in cases.values()]
        matched_seeds=all(ss==seed_sets[0] for ss in seed_sets)
        complete=(len(cases)==args.expect_cases and all(len(rr)==args.expect_seeds for rr in cases.values())
                  and matched_seeds and not any(r["not_publication_run"] for r in rows))
        # First average seeds within each case; then average the 15 cases.
        means=np.array([[np.mean([r[m] for r in rr]) for m in ("psnr","ssim")] for rr in cases.values()])
        row={"dataset":key[0],"acceleration":key[1],"method":key[2],"checkpoint":key[3],
             **dict(zip(PARAMETERS,key[4:])),"cases":len(cases),"runs":len(rows),
             "complete":complete,"psnr":means[:,0].mean(),"ssim":means[:,1].mean(),
             "psnr_sd_case_means":means[:,0].std(ddof=1) if len(means)>1 else 0,
             "ssim_sd_case_means":means[:,1].std(ddof=1) if len(means)>1 else 0,
             "psnr_mean_within_case_seed_sd":np.mean([np.std([r['psnr'] for r in rr],ddof=1)
                                                  if len(rr)>1 else 0 for rr in cases.values()])}
        aggregate.append(row)
        if not complete:
            incomplete.append({k:row[k] for k in ("dataset","acceleration","method","checkpoint","cases","runs")})
    write_csv(args.output/"group_metrics.csv",aggregate)
    with open(args.reported,newline="") as f:
        reported=list(csv.DictReader(f))
    comparisons=[]
    for r in reported:
        params=list(DEFAULTS)
        if r.get("parameter"):
            params[PARAMETERS.index(r["parameter"])]=float(r["value"])
        match=[g for g in aggregate if (g["dataset"],str(g["acceleration"]),g["method"],g["checkpoint"])
                 ==(r["dataset"],r["acceleration"],r["method"],r["checkpoint"])
                 and tuple(g[k] for k in PARAMETERS)==tuple(params)]
        g=match[0] if len(match)==1 else None
        value=g[r["metric"]] if g else None
        comparisons.append({**r,"recomputed":value,"delta":value-float(r["reported"]) if g else None,
                            "status":"complete_run_group" if g and g["complete"] else
                                     "partial_run_group" if g else "missing_run_group"})
    write_csv(args.output/"reported_vs_recomputed.csv",comparisons)
    for table in sorted({r["table"] for r in comparisons}):
        selected=[r for r in comparisons if r["table"]==table]
        write_csv(args.output/f"{table}.csv",selected)
        lines=[r"% Generated from actual outputs; empty entries are missing, never copied from reported means.",
               r"\begin{tabular}{lllrr}\hline",r"Data & Acc. & Method/metric & Recomputed & Reported\\\hline"]
        for r in selected:
            measured="--" if r["recomputed"] is None else f"{r['recomputed']:.4f}"
            label=(r["method"]+"/"+r["metric"]).replace("_",r"\_")
            lines.append(f"{r['dataset']} & {r['acceleration']} & {label} & {measured} & {r['reported']} "+r"\\")
        lines.extend([r"\hline\end{tabular}"])
        (args.output/f"{table}.tex").write_text("\n".join(lines)+"\n")
    status={"incomplete_groups":incomplete,"missing_reported_entries":sum(r["status"]=="missing_run_group" for r in comparisons),
            "aggregation":"per-run PSNR oracle, SSIM paired; seed means within case then case mean",
            "historical_seed_and_SSIM_provenance":"not supplied; requires author verification"}
    (args.output/"coverage.json").write_text(json.dumps(status,indent=2)+"\n")
    if (incomplete or status["missing_reported_entries"]) and not args.allow_incomplete:
        raise SystemExit("Partial/missing experiments: see coverage.json. No complete reproduction claimed.")
    print(f"Wrote {len(aggregate)} measured groups; {status['missing_reported_entries']} reported entries lack runs.")


if __name__=="__main__":
    main()
