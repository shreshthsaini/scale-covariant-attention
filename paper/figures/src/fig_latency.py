"""Seconds per image against resolution for full attention, AReA and SCA (batch 1, one GH200).

Input, one of:
  --csv PATH   columns res,method,seconds; res like 4096x4096, one row per image (the median
               over rows of a (res, method) pair is plotted). Default:
               /scratch/09032/saini_2/sca/results_latency_curve.csv
  results/latency_4096_batch1.csv (repo), used when the curve CSV does not exist or with
               --source 4096. Adapter: full and AReA from run "latency", SCA final (no band
               guard, code 2d11740) from run "latency3"; the band-guard SCA runs are dropped.
With several resolutions the figure is a log-log curve; with one resolution it is a bar chart.
Speedups of SCA over full attention and AReA (median seconds) are printed and annotated.

Regenerate: <sca env python> paper/figures/src/fig_latency.py [--csv PATH] [--out STEM]
"""

import argparse
import csv
import hashlib
import json
import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical PDF/SVG on rerun
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1data import FIGDIR  # noqa: E402
from orx_figstyle import BASELINE, COLUMN, PALETTE, figure, label_ends, save, use_style  # noqa: E402

REPO = os.path.normpath(os.path.join(FIGDIR, "..", ".."))
CURVE_CSV = os.path.join(REPO, "results", "latency_curve.csv")
CSV_4096 = os.path.join(REPO, "results", "latency_4096_batch1.csv")

METHODS = {  # canonical key -> (label, color, marker)
    "full": ("full attention", BASELINE, "s"),
    "area": ("AReA", PALETTE["orange"], "^"),
    "sca": ("SCA (ours)", PALETTE["blue"], "o"),
}
ALIASES = {"full": "full", "full_attention": "full", "sdpa": "full", "area": "area",
           "sca": "sca", "sca_nb": "sca", "sca_final": "sca"}


def read_curve(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("warmup_verified", "True").lower() != "true":
                raise ValueError("Latency figure requires verified warm-up ordering")
            m = ALIASES.get(r["method"].strip().lower())
            if m is None:
                print(f"skipping unknown method {r['method']!r}")
                continue
            rows.append((r["res"].strip(), m, float(r["seconds"])))
    return rows


def read_4096(path):
    """Adapter: latency_4096_batch1.csv (run,code,host,method,image,seconds) -> res,method,seconds."""
    keep = {("latency", "full"): "full", ("latency", "area"): "area", ("latency3", "sca"): "sca"}
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            m = keep.get((r["run"], r["method"]))
            if m is not None:
                rows.append(("4096x4096", m, float(r["seconds"])))
    return rows


def medians(rows):
    acc = defaultdict(list)
    for res, m, s in rows:
        acc[res, m].append(s)
    if any(len(v) != 3 for v in acc.values()):
        raise ValueError("Each reported latency group must contain three timed images")
    return {k: (float(np.median(v)), len(v)) for k, v in acc.items()}


def parse_res(res):
    w, h = (int(v) for v in res.lower().split("x"))
    return w, h


def res_label(res):
    w, h = parse_res(res)
    return f"${w}^2$" if w == h else f"{w}x{h}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CURVE_CSV)
    ap.add_argument("--source", choices=("auto", "curve", "4096"), default="auto")
    ap.add_argument("--out", default=os.path.join(FIGDIR, "latency"))
    args = ap.parse_args()

    if args.source == "4096" or (args.source == "auto" and not os.path.exists(args.csv)):
        src, rows = CSV_4096, read_4096(CSV_4096)
    else:
        src, rows = args.csv, read_curve(args.csv)
    print(f"latency source: {src}")
    med = medians(rows)
    ranges = {}
    for res, method, seconds in rows:
        ranges.setdefault((res, method), []).append(seconds)
    resolutions = sorted({r for r, _ in med}, key=lambda r: np.prod(parse_res(r)))
    for (res, m), (s, n) in sorted(med.items()):
        print(f"  {res} {m}: median {s:.2f} s over {n} images")

    use_style()
    plt.rcParams["svg.hashsalt"] = "sca"
    big = resolutions[-1]
    speed = {b: med[big, b][0] / med[big, "sca"][0]
             for b in ("full", "area") if (big, b) in med and (big, "sca") in med}
    for b, v in speed.items():
        print(f"speedup of SCA over {b} at {big}: {v:.2f}x")

    if len(resolutions) == 1:
        fig, ax = figure(width=COLUMN, ratio=0.36)
        order = [m for m in ("sca", "full", "area") if (big, m) in med]
        y = np.arange(len(order))[::-1]
        vals = [med[big, m][0] for m in order]
        ax.barh(y, vals, color=[METHODS[m][1] for m in order], height=0.62, edgecolor="none")
        for yi, m, v in zip(y, order, vals):
            note = f"{v:.0f} s"
            if m == "sca":
                note += "  (" + ", ".join(f"{speed[b]:.1f}$\\times$ vs {METHODS[b][0]}" for b in ("full", "area")
                                          if b in speed) + ")"
            ax.text(v + 3, yi, note, va="center", ha="left", fontsize=6.5)
        ax.set_yticks(y, [METHODS[m][0] for m in order])
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_xlabel(f"seconds per {res_label(big)} image (batch 1)")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    else:
        fig, ax = figure(width=COLUMN, ratio=0.72)
        mp = {r: np.prod(parse_res(r)) / 1e6 for r in resolutions}
        lines, labels = [], []
        for m, (lbl, col, mk) in METHODS.items():
            rs = [r for r in resolutions if (r, m) in med]
            if not rs:
                continue
            (ln,) = ax.plot([mp[r] for r in rs], [med[r, m][0] for r in rs], color=col, marker=mk,
                            ms=3.2, lw=1.3)
            # These are observed ranges over three timed prompts, not confidence intervals.
            lower = [med[r, m][0] - min(ranges[r, m]) for r in rs]
            upper = [max(ranges[r, m]) - med[r, m][0] for r in rs]
            ax.errorbar([mp[r] for r in rs], [med[r, m][0] for r in rs],
                        yerr=[lower, upper], fmt="none", ecolor=col, capsize=2, elinewidth=0.7)
            lines.append(ln)
            labels.append(lbl)
        ax.set_xscale("log")
        ax.set_yscale("log")
        # label every resolution, dropping a label that would crowd the previous one on log x
        kept, last = [], -np.inf
        for r in resolutions:
            if np.log10(mp[r]) - last >= 0.1:
                kept.append(r)
                last = np.log10(mp[r])
        ax.xaxis.set_major_locator(FixedLocator([mp[r] for r in kept]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([res_label(r) for r in kept])
        ys = [v for v, _ in med.values()]
        lo, hi = np.floor(np.log10(min(ys))), np.ceil(np.log10(max(ys)))
        yt = [m * 10 ** e for e in np.arange(lo, hi + 1) for m in (1, 2, 5)]
        ax.yaxis.set_major_locator(FixedLocator([t for t in yt if min(ys) / 1.3 <= t <= max(ys) * 1.3]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("output resolution (log megapixels)")
        ax.set_ylabel("seconds per image (log)")
        ax.grid(axis="both", which="major")
        label_ends(ax, lines, labels)
        if speed:
            x0, y0 = mp[big], med[big, "sca"][0]
            txt = "\n".join(f"{speed[b]:.1f}$\\times$ vs {METHODS[b][0]}" for b in ("full", "area") if b in speed)
            ax.annotate(txt, (x0, y0), xytext=(5, -7), textcoords="offset points", ha="left", va="top",
                        fontsize=6, color=METHODS["sca"][1])
    save(fig, args.out)
    with open(src, "rb") as handle:
        source_hash = hashlib.sha256(handle.read()).hexdigest()
    with open(args.out + "_provenance.json", "w") as handle:
        json.dump({"source": src, "sha256": source_hash, "statistic": "median of three timed prompts",
                   "whiskers": "observed minimum and maximum, not a confidence interval",
                   "groups": [{"res": r, "method": m, "median_seconds": value, "n": n}
                              for (r, m), (value, n) in sorted(med.items())]}, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
