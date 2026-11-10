#!/usr/bin/env python3
"""Crop sharpness of the reach-annealed variants on the 46-prompt subset (HANDOVER Section 40): paired MUSIQ on the
four 1024^2 crops (musiq_crop1024) and on the whole image (musiq) of LABEL against sca_v2 and against progressive SCA
(sca_v2_prog, whose per-image rows come from eval/prog195_sca_v2, the same evaluator), plus the set-level patch FID /
patch KID deltas against sca_v2 that the 46-prompt evaluation wrote (base/paired_deltas.csv). CPU only.
Usage: ra_sharpness.py LABEL [LABEL ...] [--out FILE.csv]
"""

import argparse
import csv
from pathlib import Path
import random

S = Path("/scratch/09032/saini_2/sca")
PROG = S / "eval/prog195_sca_v2/base/per_image.csv"
METRICS = ("musiq_crop1024", "musiq")


def rows(path, methods):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["method"] in methods:
            for m in METRICS:
                try:
                    out[(r["method"], r["prompt_id"], m)] = float(r[m])
                except (KeyError, ValueError):
                    pass
    return out


def boot(d, draws=2000, seed=0):
    rng = random.Random(seed)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return sum(d) / n, means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("labels", nargs="+")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    prog = rows(PROG, {"sca_v2_prog"})
    table = []
    for lab in a.labels:
        E = S / f"eval/r7_46_{lab}"
        if not (E / "DONE").exists():
            print(f"{lab}: evaluation not finished ({E})")
            continue
        per = rows(E / "base/per_image.csv", {lab, "sca_v2", "area"})
        per.update(prog)
        ids = sorted({k[1] for k in per if k[0] == lab})
        for ref in ("sca_v2", "sca_v2_prog", "area"):
            for m in METRICS:
                d = [per[(lab, i, m)] - per[(ref, i, m)] for i in ids if (lab, i, m) in per and (ref, i, m) in per]
                if not d:
                    continue
                mean, lo, hi = boot(d)
                table.append({"label": lab, "reference": ref, "metric": m, "n": len(d), "delta": round(mean, 3),
                              "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                              "sig": "loss" if hi < 0 else "gain" if lo > 0 else ""})
        pd = E / "base/paired_deltas.csv"
        if pd.exists():
            for r in csv.DictReader(open(pd)):
                if r.get("method") == lab and r.get("reference_method") == "sca_v2" and r["metric"] in ("patch_fid", "patch_kid"):
                    table.append({"label": lab, "reference": "sca_v2", "metric": r["metric"], "n": "set",
                                  "delta": r["value"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                                  "sig": "worse" if float(r["ci_low"]) > 0 else "better" if float(r["ci_high"]) < 0 else ""})
    for r in table:
        print(f"{r['label']:>9} vs {r['reference']:<11} {r['metric']:<15} n={r['n']:<4} {r['delta']:>8} "
              f"[{r['ci_low']}, {r['ci_high']}] {r['sig']}")
    if a.out and table:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        print("->", a.out)


if __name__ == "__main__":
    main()
