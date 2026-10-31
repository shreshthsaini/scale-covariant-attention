"""Paired bootstrap of per-prompt metric differences (method minus reference) from per-image CSVs.

Usage: paired_bootstrap.py --method M --reference R --out deltas.csv per_image.csv [per_image.csv ...]
Inputs are eval.py per_image.csv and eval_extra.py extra_per_image.csv files (columns method, prompt_id,
and one column per metric). Resamples prompts with replacement (2000 draws, seed 0), 95% percentile CI.
Each resolution and run seed is analyzed separately; repeated prompts are never silently overwritten.
"""
import argparse, csv
import numpy as np

SKIP = {"method", "res", "prompt_id", "seed", "path", "seconds"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--method", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    vals = {}
    for f in a.files:
        for r in csv.DictReader(open(f)):
            for k, v in r.items():
                if k in SKIP or k.startswith("eval_seconds") or v in ("", None):
                    continue
                try:
                    value = float(v)
                except ValueError:
                    continue
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite metric {k} in {f}: {r['prompt_id']}")
                group = (r.get("res", ""), r.get("seed", ""))
                dest = vals.setdefault(group, {}).setdefault(k, {}).setdefault(r["method"], {})
                if r["prompt_id"] in dest:
                    raise ValueError(f"Duplicate metric {k} in group {group}: {r['method']}/{r['prompt_id']}")
                dest[r["prompt_id"]] = value
    rng = np.random.default_rng(0)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "reference", "metric", "delta", "ci_low", "ci_high", "n_prompts", "res", "seed"])
        for (res, seed), metrics in sorted(vals.items()):
            for m, d in sorted(metrics.items()):
                x, y = d.get(a.method, {}), d.get(a.reference, {})
                if set(x) != set(y):
                    raise ValueError(f"Unpaired metric {m} at {res}, seed {seed}: "
                                     f"method-only={len(set(x)-set(y))}, reference-only={len(set(y)-set(x))}")
                ids = sorted(x)
                if len(ids) < 2:
                    continue
                diff = np.array([x[i] - y[i] for i in ids])
                boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(a.draws)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                w.writerow([a.method, a.reference, m, f"{diff.mean():.6g}", f"{lo:.6g}", f"{hi:.6g}", len(ids), res, seed])


if __name__ == "__main__":
    main()
