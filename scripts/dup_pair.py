#!/usr/bin/env python3
"""Paired comparison of two methods from existing outputs (no GPU): per-image metric deltas A - B with 95% bootstrap
intervals over prompts, and an exact McNemar test on the detector's duplication flags (people/animals and any subject).
Methods are LABEL@EVALDIR pairs; per-image metrics come from EVALDIR/{extra/extra_per_image.csv, base/per_image.csv},
detections from the shared 195-prompt run plus EVALDIR/det. Usage:
  dup_pair.py --set 46|195 A@/path/to/eval B@/path/to/eval [--out CSV]"""

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dup_rate as D  # noqa: E402

S = Path("/scratch/09032/saini_2/sca")
ROOT = Path(__file__).resolve().parents[1]
MET = ["ir", "aesthetic", "hpsv2", "hpsv21", "pickscore", "clip", "clipiqa", "arniqa", "musiq", "musiq_crop1024"]


def per_image(edir, label):
    out = {}
    for f in (Path(edir) / "extra/extra_per_image.csv", Path(edir) / "base/per_image.csv"):
        if f.exists():
            for r in csv.DictReader(open(f)):
                if r["method"] == label:
                    for m in MET:
                        if r.get(m) not in (None, ""):
                            out[(r["prompt_id"], m)] = float(r[m])
    return out


def boot(d, draws=4000):
    rng = random.Random(0)
    b = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(draws))
    return sum(d) / len(d), b[int(0.025 * draws)], b[int(0.975 * draws) - 1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", choices=("46", "195"), required=True)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    ids = [json.loads(l)["id"] for l in open("/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl")]
    if args.set == "46":
        keep = {json.loads(l)["id"] for l in open(S / "abl50.jsonl")} - {json.loads(l)["id"] for l in open(S / "dev4.jsonl")}
        ids = [i for i in ids if i in keep]
    (la, ea), (lb, eb) = (x.split("@", 1) for x in (args.a, args.b))
    rows_out = []
    pa, pb = per_image(ea, la), per_image(eb, lb)
    for m in MET:
        d = [pa[(i, m)] - pb[(i, m)] for i in ids if (i, m) in pa and (i, m) in pb]
        if d:
            mu, lo, hi = boot(d)
            rows_out.append({"comparison": f"{la} - {lb}", "set": args.set, "metric": m, "n": len(d), "delta": mu,
                             "ci_low": lo, "ci_high": hi, "mcnemar_p": ""})
    subj = {r["id"]: r for r in json.load(open(ROOT / "results/dup_audit_20261027/subjects_a4k195.json")) if r["id"] in set(ids)}
    dets = [S / "eval/dup_detect_20261027"] + [Path(e) / "det" for e in (ea, eb) if (Path(e) / "det").exists()]
    rows = D.load(dets)
    for kind, kinds in (("dup_people", {"person", "animal"}), ("dup_any", None)):
        a = SimpleNamespace(sides="800,1333", thr=0.35, nms=0.5, contain=0.9, contain_area=0.3, max_frac=0.9, kinds=kinds)
        fl = D.flags(rows, subj, a)
        common = [i for i in ids if (la, i) in fl and (lb, i) in fl]
        x = [fl[(la, i)][0] for i in common]
        y = [fl[(lb, i)][0] for i in common]
        b_ = sum(1 for u, v in zip(x, y) if u and not v)
        c_ = sum(1 for u, v in zip(x, y) if v and not u)
        rows_out.append({"comparison": f"{la} - {lb}", "set": args.set, "metric": kind, "n": len(common),
                         "delta": (sum(x) - sum(y)) / max(1, len(common)), "ci_low": f"{sum(x)}/{len(common)}",
                         "ci_high": f"{sum(y)}/{len(common)}", "mcnemar_p": D.mcnemar(b_, c_)})
    for r in rows_out:
        print(f"{r['comparison']:34s} {r['metric']:15s} n={r['n']:3d} delta={r['delta']:+.4g} [{r['ci_low']}, {r['ci_high']}]"
              + (f" McNemar p={r['mcnemar_p']:.3g}" if r["mcnemar_p"] != "" else ""))
    if args.out:
        new = not args.out.exists()
        with open(args.out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            if new:
                w.writeheader()
            w.writerows(rows_out)


if __name__ == "__main__":
    main()
