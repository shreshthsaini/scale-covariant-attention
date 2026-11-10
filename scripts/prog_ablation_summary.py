#!/usr/bin/env python3
"""Readout of experiment B (HANDOVER Section 40): per-head table inside the progressive schedule.
Reads results/prog195_ablation (written by 000b_0a0_eval_prog195_ablation and 000b_0a1_latency_prog_ablation) and
prints, for prog_headmean, prog_layermean and prog_win against sca_v2_prog (and against HRDiT), every per-image metric
delta with its 95% interval (paired bootstrap, 195 prompts) and a sig mark, the corrected set-level point deltas
(FID, KID, patch FID, patch KID; intervals need the CPU bootstrap afterwards), the detector people/animal rates and the
clean batch-1 seconds. Usage: prog_ablation_summary.py [--out summary.txt]
"""

import argparse
import csv
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results/prog195_ablation"
LABELS = ("prog_headmean", "prog_layermean", "prog_win")
LOWER = {"fid", "kid", "patch_fid", "patch_kid"}


def rows(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    lines = []
    for ref in ("sca_v2_prog", "hrdit_dev_author"):
        for lab in LABELS:
            d = rows(R / f"{lab}_vs_{ref}.csv")
            if not d:
                lines.append(f"{lab} vs {ref}: no paired file yet")
                continue
            sig = []
            for r in d:
                lo, hi, m = float(r["ci_low"]), float(r["ci_high"]), r["metric"]
                better = (hi < 0) if m in LOWER else (lo > 0)
                worse = (lo > 0) if m in LOWER else (hi < 0)
                sig.append(f"{m} {float(r['delta']):+.4g} [{lo:+.4g},{hi:+.4g}]{' WIN' if better else ' LOSS' if worse else ''}")
            lines.append(f"{lab} vs {ref}: " + "; ".join(sig))
    for r in rows(R / "corrected_fidp/paired_deltas.csv"):
        if r.get("reference_method") == "sca_v2_prog" and r.get("method") in LABELS + ("hrdit_dev_author",):
            lines.append(f"set-level {r['method']} minus sca_v2_prog {r['metric']} {float(r['value']):+.4g} (point; CPU bootstrap for the interval)")
    for r in rows(R / "dup_rates_people.csv"):
        lines.append("detector people/animal: " + ", ".join(f"{k}={v}" for k, v in r.items() if k in ("method", "n", "dup", "rate", "diff_vs_ref", "mcnemar_p")))
    for r in rows(R / "latency.csv"):
        lines.append(f"clean batch-1 {r['method']}: median {r['median_s']} s over {r['n_timed']} prompts (clean={r['clean']})")
    t = R / "timing.txt"
    if t.exists():
        lines.append("campaign medians: " + " | ".join(ln.strip() for ln in t.read_text().splitlines() if ln.strip()))
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
