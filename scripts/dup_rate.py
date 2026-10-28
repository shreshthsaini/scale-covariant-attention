#!/usr/bin/env python3
"""Duplication flags and rates from scripts/dup_detect.py output.

Per image and subject: boxes with score >= --thr (from the chosen detection sides), merged across sides and
de-duplicated by IoU (--nms) and containment (--contain: a box at least that fraction inside a larger box of the
same query, with at least --contain-area of its area, is the same object); count = surviving boxes. An image is
flagged when any subject with a numeric implied count has count > implied. Reports per-method rates with Wilson
95% intervals, paired differences to --ref (bootstrap over prompts, exact McNemar p), and, with --labels,
agreement with hand labels.
"""

import argparse
import csv
import glob
import json
import math
from pathlib import Path
import random


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def inside(a, b):
    """Fraction of box a inside box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def area(a):
    return max(0, a[2] - a[0]) * max(0, a[3] - a[1])


def count_boxes(dets, query, thr, sides, nms, contain, contain_area, max_frac, img_area):
    cand = sorted((d for d in dets if d["q"] == query and d["s"] >= thr and d["side"] in sides
                   and area(d["b"]) < max_frac * img_area), key=lambda d: -d["s"])
    kept = []
    for d in cand:
        b = d["b"]
        dup = False
        for k in kept:
            kb = k["b"]
            if iou(b, kb) >= nms:
                dup = True
            else:
                small, big = (b, kb) if area(b) <= area(kb) else (kb, b)
                if inside(small, big) >= contain and area(small) >= contain_area * area(big):
                    dup = True
            if dup:
                break
        if not dup:
            kept.append(d)
    return kept


def load(det_dirs):
    rows = {}
    for d in det_dirs:
        for f in sorted(glob.glob(str(Path(d) / "dets_*.jsonl"))):
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows[(r["method"], r["prompt_id"])] = r
    return rows


def flags(rows, subjects, args):
    sides = set(int(s) for s in args.sides.split(","))
    out = {}
    for (m, pid), r in rows.items():
        subj = subjects.get(pid)
        if subj is None:
            continue
        img_area = r["size"][0] * r["size"][1]
        detail, flag, eligible = [], False, False
        for s in subj["subjects"]:
            if s.get("implied") is None:
                continue
            eligible = True
            n = len(count_boxes(r["dets"], s["query"], args.thr, sides, args.nms, args.contain,
                                args.contain_area, args.max_frac, img_area))
            detail.append(f"{s['query']}:{n}/{s['implied']}")
            flag |= n > s["implied"]
        if eligible:
            out[(m, pid)] = (int(flag), ";".join(detail))
    return out


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dets", nargs="+", required=True)
    p.add_argument("--subjects", type=Path, required=True)
    p.add_argument("--thr", type=float, default=0.35)
    p.add_argument("--sides", default="800,1333")
    p.add_argument("--nms", type=float, default=0.5)
    p.add_argument("--contain", type=float, default=0.9)
    p.add_argument("--contain-area", type=float, default=0.3)
    p.add_argument("--max-frac", type=float, default=0.9, help="drop boxes covering this fraction of the image")
    p.add_argument("--ids", type=Path, help="JSONL whose ids restrict the prompt set (e.g. abl50 minus dev4)")
    p.add_argument("--exclude-ids", type=Path, help="JSONL whose ids are removed (e.g. dev4)")
    p.add_argument("--methods", help="comma-separated method order (default: all, sorted)")
    p.add_argument("--ref", default="sca_nb")
    p.add_argument("--labels", type=Path, help="CSV method,prompt_id,dup for validation")
    p.add_argument("--per-image", type=Path, help="write per-image flags CSV")
    p.add_argument("--out", type=Path, help="write per-method rates CSV")
    p.add_argument("--boot", type=int, default=10000)
    args = p.parse_args(argv)
    subjects = {r["id"]: r for r in json.loads(args.subjects.read_text())}
    if args.ids:
        keep = {json.loads(l)["id"] for l in open(args.ids)}
        subjects = {k: v for k, v in subjects.items() if k in keep}
    if args.exclude_ids:
        drop = {json.loads(l)["id"] for l in open(args.exclude_ids)}
        subjects = {k: v for k, v in subjects.items() if k not in drop}
    rows = load(args.dets)
    fl = flags(rows, subjects, args)
    methods = args.methods.split(",") if args.methods else sorted({m for m, _ in fl})
    if args.per_image:
        with open(args.per_image, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "prompt_id", "dup", "counts"])
            for (m, pid), (f, det) in sorted(fl.items()):
                w.writerow([m, pid, f, det])
    if args.labels:
        lab = [(r["method"], r["prompt_id"], int(r["dup"])) for r in csv.DictReader(open(args.labels))]
        tp = fp = tn = fn = miss = 0
        wrong = []
        for m, pid, y in lab:
            if (m, pid) not in fl:
                miss += 1
                wrong.append(f"{m}:{pid}:not-scored")
                continue
            f = fl[(m, pid)][0]
            tp += f and y; fp += f and not y; tn += (not f) and (not y); fn += (not f) and y
            if f != y:
                wrong.append(f"{m}:{pid}:label{y}:{fl[(m, pid)][1]}")
        n = tp + fp + tn + fn
        print(f"validation n={n} acc={(tp + tn) / max(1, n):.3f} TP={tp} FP={fp} TN={tn} FN={fn} unscored={miss}")
        for x in wrong:
            print("  disagree", x)
    res = []
    ref = {pid: f for (m, pid), (f, _) in fl.items() if m == args.ref}
    rng = random.Random(0)
    for m in methods:
        mine = {pid: f for (mm, pid), (f, _) in fl.items() if mm == m}
        k, n = sum(mine.values()), len(mine)
        lo, hi = wilson(k, n)
        row = {"method": m, "n": n, "dup": k, "rate": k / n if n else float("nan"), "lo95": lo, "hi95": hi}
        common = sorted(set(mine) & set(ref))
        if m != args.ref and common:
            d = [mine[i] - ref[i] for i in common]
            diffs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(args.boot))
            row.update({"n_paired": len(common), "diff_vs_ref": sum(d) / len(d),
                        "diff_lo95": diffs[int(0.025 * args.boot)], "diff_hi95": diffs[int(0.975 * args.boot) - 1],
                        "mcnemar_p": mcnemar(sum(x == 1 for x in d), sum(x == -1 for x in d))})
        res.append(row)
        extra = (f" vs {args.ref}: {row['diff_vs_ref']:+.3f} [{row['diff_lo95']:+.3f},{row['diff_hi95']:+.3f}] "
                 f"p={row['mcnemar_p']:.3g} (n={row['n_paired']})") if "diff_vs_ref" in row else ""
        print(f"{m:28s} {k:3d}/{n:3d} = {row['rate']:.3f} [{lo:.3f},{hi:.3f}]{extra}")
    if args.out:
        keys = ["method", "n", "dup", "rate", "lo95", "hi95", "n_paired", "diff_vs_ref", "diff_lo95", "diff_hi95", "mcnemar_p"]
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in res:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
