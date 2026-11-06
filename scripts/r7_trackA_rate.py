#!/usr/bin/env python3
"""Human-corrected people/animal duplication rate of one method on the 95 eligible prompts with a stratified
unflagged sample (round 7, Track A).

rate = (confirmed flags + known unflagged duplicates + sum_s U_s * k_s / n_s) / P, where for each stratum s, U_s is
the number of unflagged prompts without a label outside the sample, n_s the sampled unflagged prompts and k_s the
duplicates among them (human_final == "yes"); "borderline" counts as no in the primary rate and as yes in the
with-borderline rate. 95% interval: Wilson on the pooled sample proportion carried through (as Step 1 did with k/30).
Flags: the per-image people CSV of the official (GPU) detector. Labels: human_labels_r6.csv and human_labels_r7.csv.
Usage: r7_trackA_rate.py --label r6_ffl --flags eval/r7_195_r6_ffl/dup_per_image_people.csv --strata A.json B.json
       (each strata JSON: {"prompts": [...], "sample": [...]}); prints the primary and with-borderline rates.
"""

import argparse
import csv
import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results/dup_audit_20261027"


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--flags", type=Path, required=True)
    ap.add_argument("--strata", nargs="+", type=Path, required=True)
    a = ap.parse_args()
    lab = {}
    for name in ("human_labels_r6.csv", "human_labels_r7.csv"):
        for r in csv.DictReader(open(A / name)):
            if r["label"] == a.label:
                lab[r["prompt_id"]] = r["human_final"]
    fl = {r["prompt_id"]: int(r["dup"]) for r in csv.DictReader(open(a.flags)) if r["method"] == a.label}
    P = len(fl)
    strata = [json.loads(Path(s).read_text()) for s in a.strata]
    sampled = {p for s in strata for p in s["sample"]}
    for border in (False, True):
        yes = ("yes", "borderline") if border else ("yes",)
        flags = [p for p, d in fl.items() if d]
        unl = [p for p, d in fl.items() if not d]
        tp = sum(lab.get(p, "yes") in yes for p in flags)
        unlabeled_flags = [p for p in flags if p not in lab]
        known = [p for p in unl if p in lab and p not in sampled]
        k_known = sum(lab[p] in yes for p in known)
        est, kk, nn = tp + k_known, 0, 0
        for s in strata:
            ids = [p for p in s["prompts"] if p in fl and not fl[p]]
            smp = [p for p in s["sample"] if p in fl and not fl[p] and p in lab]
            k = sum(lab[p] in yes for p in smp)
            U = len([p for p in ids if p not in known])
            est += U * (k / len(smp) if smp else 0.0)
            kk, nn = kk + k, nn + len(smp)
        lo, hi = wilson(kk, nn)
        U_all = len([p for p in unl if p not in known])
        print(f"{a.label} {'with_borderline' if border else 'primary'}: prompts {P}, flags {len(flags)} "
              f"(confirmed {tp}, unlabeled {len(unlabeled_flags)}), known unflagged {len(known)} ({k_known} dup), "
              f"sample {kk}/{nn} -> human-corrected {est / P:.1%} "
              f"[{(tp + k_known + U_all * lo) / P:.1%}, {(tp + k_known + U_all * hi) / P:.1%}]")


if __name__ == "__main__":
    main()
