"""Seconds per image against resolution for full attention, AReA and SCA (batch 1, one GH200).

Input, one of:
  --csv PATH   columns res,method,seconds; res like 4096x4096, one row per image (the median
               over rows of a (res, method) pair is plotted). Default:
               results/latency_curve.csv (the verified clean batch-1 sweep)
  results/latency_4096_batch1.csv (repo), used when the curve CSV does not exist or with
               --source 4096. Adapter: full and AReA from run "latency", SCA final (no band
               guard, code 2d11740) from run "latency3"; the band-guard SCA runs are dropped.
Only clean batch-1 measurements are plotted, never campaign means.

With several resolutions the figure has two panels:
  (a) median seconds per image against resolution (log-log), whiskers = observed min and max,
      a light fill between AReA and SCA, and the least-squares exponent b of seconds ~ N^b
      (N = image tokens, (W/16)(H/16)) fitted to each method's medians;
  (b) SCA's speedup (ratio of medians) over AReA and over full attention at every resolution.
With one resolution it is a bar chart. Speedups and exponents are printed for the caption.

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
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scafig as sf  # noqa: E402
from k1data import FIGDIR  # noqa: E402

REPO = os.path.normpath(os.path.join(FIGDIR, "..", ".."))
CURVE_CSV = os.path.join(REPO, "results", "latency_curve.csv")
CSV_4096 = os.path.join(REPO, "results", "latency_4096_batch1.csv")

METHODS = {  # canonical key -> (label, color, marker); drawn in this order (SCA on top)
    "full": ("Full attention", sf.METHOD_COLOR["full"], "^"),
    "area": ("AReA", sf.METHOD_COLOR["area"], "s"),
    "sca": ("SCA (ours)", sf.SCA, "o"),
}
ALIASES = {"full": "full", "full_attention": "full", "sdpa": "full", "area": "area",
           "sca": "sca", "sca_nb": "sca", "sca_final": "sca"}
PATCH = 16  # pixels per image token side: VAE factor 8 times 2x2 latent packing
W, H = sf.TEXT, 2.05  # inches; the main-text figure must not grow


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
    return f"{w}\u00b2" if w == h else f"{w}\u00d7{h}"  # plain text keeps the sans tick font


def tokens(res):
    w, h = parse_res(res)
    return (w // PATCH) * (h // PATCH)


def box(x, y, w, h):
    return [x / W, y / H, w / W, h / H]


def bar_chart(med, speed, big):
    """Single-resolution fallback: horizontal bars of median seconds."""
    fig = plt.figure(figsize=(sf.base.COLUMN, 1.2))
    ax = fig.add_axes([0.28, 0.32, 0.62, 0.62])
    order = [m for m in ("sca", "full", "area") if (big, m) in med]
    y = np.arange(len(order))[::-1]
    vals = [med[big, m][0] for m in order]
    ax.barh(y, vals, color=[METHODS[m][1] for m in order], height=0.62, edgecolor="none")
    for yi, m, v in zip(y, order, vals):
        note = f"{v:.0f} s"
        if m == "sca":
            note += "  (" + ", ".join(f"{speed[big][b]:.2f}$\\times$ vs {METHODS[b][0]}"
                                      for b in ("full", "area") if b in speed[big]) + ")"
        ax.text(v + 3, yi, note, va="center", ha="left", fontsize=sf.FS_SMALL)
    ax.set_yticks(y, [METHODS[m][0] for m in order])
    ax.set_xlim(0, max(vals) * 2.5)
    ax.set_xlabel(f"seconds per {res_label(big)} image (batch 1, one GH200)")
    ax.grid(visible=False)
    return fig


def curve_figure(med, ranges, resolutions, speed, expo):
    """Seconds per image (log-log) with SCA's speedup over AReA printed above each resolution."""
    fig = plt.figure(figsize=(W, H))
    left, right, bottom, top = 0.47, 0.04, 0.37, 0.36
    ax = fig.add_axes(box(left, bottom, W - left - right, H - bottom - top))
    n = {r: tokens(r) for r in resolutions}
    xt = [n[r] for r in resolutions]

    both = [r for r in resolutions if (r, "area") in med and (r, "sca") in med]
    ax.fill_between([n[r] for r in both], [med[r, "sca"][0] for r in both],
                    [med[r, "area"][0] for r in both], color=sf.SCA_LIGHT, lw=0, zorder=0)
    ends = {}
    for m, (lbl, col, mk) in METHODS.items():
        rs = [r for r in resolutions if (r, m) in med]
        if not rs:
            continue
        xs, ys = [n[r] for r in rs], [med[r, m][0] for r in rs]
        # whiskers: observed minimum and maximum over the three timed prompts, not intervals
        ax.vlines(xs, [min(ranges[r, m]) for r in rs], [max(ranges[r, m]) for r in rs], color=col,
                  lw=0.8, zorder=2)
        ax.plot(xs, ys, color=col, lw=1.6 if m == "sca" else 1.1, marker=mk,
                ms=3.8 if m == "sca" else 3.3, mec="white", mew=0.4, zorder=3 if m == "sca" else 2.5,
                solid_capstyle="round")
        ends[m] = (xs[-1], ys[-1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([res_label(r) for r in resolutions])
    ys = [v for v, _ in med.values()]
    lo, hi = np.floor(np.log10(min(ys))), np.ceil(np.log10(max(ys)))
    yt = [k * 10 ** e for e in np.arange(lo, hi + 1) for k in (1, 2, 5)]
    yt = [t for t in yt if min(ys) / 1.5 <= t <= max(ys) * 1.5]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{t:g}" for t in yt])
    ax.set_xlim(xt[0] / 1.2, xt[-1] * 2.3)
    ax.set_ylim(min(ys) / 1.35, max(ys) * 1.3)
    ax.grid(visible=False)
    ax.set_xlabel("output resolution (log scale)")
    ax.set_ylabel("seconds per image")

    # direct labels: name and fitted exponent at each line's end
    for m, (x, y) in ends.items():
        lbl, col, _ = METHODS[m]
        if m == "full":
            # The line ends inside the shaded AReA-SCA gap, so it gets a key in the empty lower
            # right instead of a direct label; its exponent is in the caption.
            _, _, mk = METHODS[m]
            kx, ky, kw = 0.70, 0.11, 0.045
            ax.plot([kx, kx + kw], [ky, ky], transform=ax.transAxes, color=col, lw=1.1,
                    solid_capstyle="round", clip_on=False, zorder=4)
            ax.plot([kx + kw / 2], [ky], transform=ax.transAxes, color=col, marker=mk, ms=3.3,
                    mec="white", mew=0.4, clip_on=False, zorder=4)
            ax.text(kx + kw + 0.012, ky, lbl, transform=ax.transAxes, ha="left", va="center",
                    fontsize=sf.FS, color=sf.MID)
            continue
        ax.annotate(lbl, (x, y), xytext=(7, 2.5), textcoords="offset points", ha="left", va="center",
                    fontsize=sf.FS, color=col, fontweight="bold" if m == "sca" else "normal")
        # "proportional to N^b": the exponent is set as its own 7 pt text so it stays legible
        base = ax.annotate(r"$\propto N$", (x, y), xytext=(7, -3.5), textcoords="offset points",
                           ha="left", va="top", fontsize=sf.FS, color=col)
        fig.canvas.draw()
        bb = base.get_window_extent()
        px = 72 / fig.dpi
        x_disp, y_disp = ax.transData.transform((x, y))
        ax.annotate(f"{expo[m]:.2f}", (x, y), xytext=((bb.x1 - x_disp) * px + 0.6, (bb.y1 - y_disp) * px + 2.2),
                    textcoords="offset points", ha="left", va="top", fontsize=sf.FS_SMALL, color=col)
    ax.text(0.012, 0.97, "batch 1, one GH200,\nmedian of three prompts", transform=ax.transAxes,
            ha="left", va="top", fontsize=sf.FS_TICK, color=sf.MID, linespacing=1.15)

    # speedup row: SCA over AReA at every resolution, as tick labels on a top axis
    top_ax = ax.secondary_xaxis("top")
    top_ax.xaxis.set_major_locator(FixedLocator([n[r] for r in both]))
    top_ax.xaxis.set_minor_locator(NullLocator())
    top_ax.set_xticklabels([f"{speed[r]['area']:.2f}\u00d7" for r in both])
    top_ax.tick_params(axis="x", length=0, pad=2, labelsize=sf.FS_TICK, labelcolor=sf.SCA)
    top_ax.spines["top"].set_visible(False)
    labels = top_ax.get_xticklabels()
    labels[-1].set_fontweight("bold")
    labels[-1].set_fontsize(sf.FS)
    top_ax.set_xlabel("speedup of SCA over AReA", fontsize=sf.FS_TICK, color=sf.MID, labelpad=3)
    return fig, ax, top_ax


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

    speed = {r: {b: med[r, b][0] / med[r, "sca"][0] for b in ("full", "area")
                 if (r, b) in med and (r, "sca") in med} for r in resolutions}
    for r in resolutions:
        for b, v in speed[r].items():
            print(f"speedup of SCA over {b} at {r}: {v:.2f}x")
    expo = {}
    for m in METHODS:
        rs = [r for r in resolutions if (r, m) in med]
        if len(rs) >= 2:
            expo[m] = float(np.polyfit(np.log([tokens(r) for r in rs]), np.log([med[r, m][0] for r in rs]), 1)[0])
            print(f"least-squares exponent of seconds ~ N^b for {m} over {len(rs)} resolutions: b = {expo[m]:.3f}")

    sf.use_style()
    plt.rcParams["svg.hashsalt"] = "sca"
    if len(resolutions) == 1:
        fig = bar_chart(med, speed, resolutions[0])
    else:
        fig, _, _ = curve_figure(med, ranges, resolutions, speed, expo)
    sf.save(fig, args.out)
    with open(src, "rb") as handle:
        source_hash = hashlib.sha256(handle.read()).hexdigest()
    with open(args.out + "_provenance.json", "w") as handle:
        json.dump({"source": src, "sha256": source_hash, "statistic": "median of three timed prompts",
                   "whiskers": "observed minimum and maximum, not a confidence interval",
                   "exponent_fit": "least squares of log median seconds on log N, N = (W/16)(H/16) tokens",
                   "exponents": {m: round(v, 6) for m, v in sorted(expo.items())},
                   "groups": [{"res": r, "method": m, "median_seconds": value, "n": n}
                              for (r, m), (value, n) in sorted(med.items())]}, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
