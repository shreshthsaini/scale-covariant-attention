"""Seconds per image against resolution for full attention, AReA and SCA (batch 1, one GH200).

Switch. Until the final operator's latency curve lands (v2swap.landed("latency_curve_v2"), that is,
results/latency_curve_v2/summary.csv exists) the figures are the published ones, SCA = sca_nb. Once
it lands, the "SCA (ours)" curve is that file's sca_v2 rows read through v2swap.latency() (median,
minimum and maximum of the timed images), published SCA (sca_nb, now SCA without the schedule) is
not plotted, and the speedups, exponents, headline ratio and caption snippet are recomputed from the
plotted cells. Full attention and AReA keep the per-image cells described below, and each must equal
results/latency_curve_summary.csv wherever that file holds it (it lacks full attention at 6144^2, so
it cannot replace them). Every run first prints one status line, "fig_latency.py: pending
(latency_curve_v2)" or "fig_latency.py: landed (latency_curve_v2)". With SCA_V2_MOCK=DIR
(v2swap.py --make-mock) the outputs go to DIR/figures/ and the snippet to
DIR/figures/snippets/latency.tex, never into paper/figures.

Input, one of:
  --csv PATH [PATH ...]   columns res,method,seconds plus an image id (image or prompt_id); res
               like 4096x4096, one row per image (the median over rows of a (res, method) cell is
               plotted). Rows whose id starts with sca_latency_warmup are the excluded warm-up
               images and are dropped. A file with a warmup_verified column must have it True on
               every row; a file without one must hold exactly one warm-up row per cell. A cell
               comes from the first file that has it; a later file holding the same cell must agree
               with it image for image to the coarser file's rounding. Default, in this order:
               results/latency_curve.csv (the audited clean batch-1 sweep, full precision), then
               results/latency_curve_batch1.csv (the same sweep rounded to 0.01 s with its warm-up
               rows, plus full attention at 6144^2, which only this file holds)
  results/latency_4096_batch1.csv (repo), used when the curve CSV does not exist or with
               --source 4096. Adapter: full and AReA from run "latency", SCA final (no band
               guard, code 2d11740) from run "latency3"; the band-guard SCA runs are dropped.
The lines and filled markers are clean batch-1 measurements only.

Campaign means (results/baseline46/base_summary.csv, 46 prompts on shared nodes), UltraImage's
191-prompt campaign mean (results/e1_ultraimage_base_summary.csv) and HRDiT's clean batch-1 time
(median of the three hrdit_dev_author rows of results/latency_curve_batch1.csv) are read, checked
and recorded in the provenance file, but not plotted: the figure follows the house style of
figures/REFERENCE_STYLE.md (AReA Fig 6) and shows the clean sweep only.

With several resolutions the figure is one log-log panel:
  median seconds per image against resolution (units in the ticks), whiskers = observed min and
  max, a light fill between AReA and SCA, the least-squares exponent b of seconds ~ N^b (N = image
  tokens, (W/16)(H/16)) as a direct label at each line's end, SCA's speedup over AReA as small grey
  ratios inside the gap, one bold headline ratio at the largest size, and the model name in a
  corner. No legend, no grid, no top or right spine.
  Every curve run also writes STEM_wrap.pdf/.svg (default figures/latency_wrap.pdf), the
  main-text wrapfigure at about 0.47 text width: 186 x 135 pt at natural size, the same data and
  encodings thinned to fit (labels 7 pt, the per-size ratios and exponents 6.5 and 6 pt, the
  per-size ratios just under the SCA line, the headline in the empty lower right, "SCA" without
  "(ours)", no whiskers), with thin light-grey horizontal gridlines behind the data (plot standard
  of 2026-10-19). The 5.5 in figure stays as it was.
With one resolution it is a bar chart. Speedups and exponents are printed, and the caption
snippet figures/snippets/latency.tex is written from the same numbers (trial runs with --out
write it beside the output instead).

Regenerate: <sca env python> paper/figures/src/fig_latency.py [--csv PATH [PATH ...]] [--out STEM]
"""

import argparse
import csv
import hashlib
import json
import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical PDF/SVG on rerun
import sys
from collections import defaultdict

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scafig as sf  # noqa: E402
import v2swap  # noqa: E402  (the sca_v2 switch: landed runs and their loaders)
from k1data import FIGDIR  # noqa: E402

REPO = os.path.normpath(os.path.join(FIGDIR, "..", ".."))
CURVE_CSV = os.path.join(REPO, "results", "latency_curve.csv")  # audited sweep, full precision
BATCH1_CSV = os.path.join(REPO, "results", "latency_curve_batch1.csv")  # 0.01 s, adds full at 6144^2
CURVE_CSVS = (CURVE_CSV, BATCH1_CSV)  # a cell comes from the first file that has it
WARMUP_PREFIX = "sca_latency_warmup"  # id of the excluded first image of every timed process
CSV_4096 = os.path.join(REPO, "results", "latency_4096_batch1.csv")
CAMPAIGN_CSV = os.path.join(REPO, "results", "baseline46", "base_summary.csv")
CAMPAIGN_191_CSV = os.path.join(REPO, "results", "e1_ultraimage_base_summary.csv")
OUTDIR = str(v2swap.MOCK / "figures") if v2swap.MOCK else FIGDIR  # a mock run never writes into paper/figures
SNIPPET = os.path.join(OUTDIR, "snippets", "latency.tex")
V2_METHOD = "sca_v2"  # the final operator's rows in results/latency_curve_v2/summary.csv

METHODS = {  # canonical key -> (label, color, marker); drawn in this order (SCA on top)
    "full": ("Full attention", sf.METHOD_COLOR["full"], "^"),
    "area": ("AReA", sf.METHOD_COLOR["area"], "s"),
    "sca": ("SCA (ours)", sf.SCA, "o"),
}
ALIASES = {"full": "full", "full_attention": "full", "sdpa": "full", "area": "area",
           "sca": "sca", "sca_nb": "sca", "sca_final": "sca"}
CAMPAIGN_RES = "4096x4096"
CAMPAIGN = {  # base_summary.csv method -> (label, color, marker); baselines in neutral greys
    "sca_nb": ("SCA (ours)", sf.SCA, "o"),
    "hrdit_dev_author": ("HRDiT", "#6F7C89", "D"),
    "ultraimage": ("UltraImage", sf.METHOD_COLOR["ultraimage"], "D"),
}
PATCH = 16  # pixels per image token side: VAE factor 8 times 2x2 latent packing
W, H = sf.TEXT, 1.90  # inches; the main-text figure must not grow


def rel(path):
    """Repository-relative name for messages and the snippet header; other paths stay absolute."""
    path = os.path.abspath(path)
    return os.path.relpath(path, REPO) if path.startswith(REPO + os.sep) else path


def read_curve(path):
    """Rows (res, method, seconds) of one curve CSV without its warm-up images, and the file's
    rounding: half a unit in the last decimal it prints."""
    rows, warm, skipped, decimals = [], defaultdict(int), defaultdict(int), 0
    with open(path) as fh:
        reader = csv.DictReader(fh)
        verified = "warmup_verified" in reader.fieldnames
        id_col = next((c for c in ("image", "prompt_id") if c in reader.fieldnames), None)
        for r in reader:
            if verified and r["warmup_verified"].strip().lower() != "true":
                raise ValueError("Latency figure requires verified warm-up ordering")
            m = ALIASES.get(r["method"].strip().lower())
            if m is None:
                skipped[r["method"]] += 1
                continue
            res = r["res"].strip()
            if id_col and r[id_col].strip().startswith(WARMUP_PREFIX):
                warm[res, m] += 1
                continue
            seconds = r["seconds"].strip()
            decimals = max(decimals, len(seconds.partition(".")[2]))
            rows.append((res, m, float(seconds)))
    name = rel(path)
    for method, k in sorted(skipped.items()):
        print(f"skipping {k} rows of unplotted method {method!r} in {name}")
    print(f"dropped {sum(warm.values())} warm-up rows ({WARMUP_PREFIX}*) from {name}")
    if not verified:  # the warm-up row is this file's only record that the warm-up ran
        cells = {(res, m) for res, m, _ in rows}
        bad = sorted(c for c in cells | set(warm) if warm[c] != 1)
        if bad:
            raise ValueError(f"{name} lacks warmup_verified and does not hold exactly one warm-up "
                             f"row per cell: {bad}")
    return rows, 0.5 * 10.0 ** -decimals


def read_curves(paths):
    """Merge curve CSVs: each (res, method) cell comes from the first file that has it, and a later
    file holding the same cell must match it image for image within the coarser file's rounding."""
    cells, origin, tol = {}, {}, {}
    for path in paths:
        rows, tol[path] = read_curve(path)
        acc = defaultdict(list)
        for res, m, s in rows:
            acc[res, m].append(s)
        n_new, worst, limit = 0, 0.0, 0.0
        for key, vals in acc.items():
            if key not in cells:
                cells[key], origin[key], n_new = vals, path, n_new + 1
                continue
            a, b = sorted(cells[key]), sorted(vals)
            limit = max(limit, tol[path], tol[origin[key]])
            gap = max(abs(x - y) for x, y in zip(a, b)) if len(a) == len(b) else float("inf")
            if gap > max(tol[path], tol[origin[key]]) + 1e-9:
                raise ValueError(f"{key} differs between {origin[key]} and {path} beyond rounding")
            worst = max(worst, gap)
        msg = f"{rel(path)}: {len(acc)} cells, {n_new} taken from this file"
        if len(acc) > n_new:
            msg += (f"; {len(acc) - n_new} already read, matched image for image within {worst:.6f} s "
                    f"(rounding bound {limit:g} s)")
        print(msg)
    rows = [(res, m, s) for (res, m), vals in cells.items() for s in vals]
    return rows, origin


def src_name(path):
    """Basename for the curve CSVs in results/, the repository-relative (or absolute) path otherwise."""
    in_results = os.path.dirname(os.path.abspath(path)) == os.path.join(REPO, "results")
    return os.path.basename(path) if in_results else rel(path)


def switch_sca(med, ranges, origin):
    """Landed: SCA's cells become the sca_v2 rows of v2swap.latency() and published SCA (sca_nb) is dropped.

    Full attention and AReA keep their per-image cells; each must equal results/latency_curve_summary.csv
    (median, minimum, maximum, n) wherever that file holds it. Edits the three dicts in place and returns the
    sca_v2 source path."""
    published = v2swap.latency(v2=False)
    for key in [k for k in med if k[1] != "sca"]:
        if key in published:
            got = (med[key][0], min(ranges[key]), max(ranges[key]))
            if max(abs(a - b) for a, b in zip(got, published[key][:3])) > 1e-6 or med[key][1] != published[key][3]:
                raise ValueError(f"{key}: per-image cell {got} disagrees with "
                                 f"{rel(v2swap.LATENCY_PUBLISHED)} {published[key]}")
    for key in [k for k in med if k[1] == "sca"]:
        del med[key], ranges[key]
        origin.pop(key, None)
    src = str(v2swap.LATENCY_V2)
    for (res, m), (s, lo, hi, n) in v2swap.latency(v2=True).items():
        if m != V2_METHOD:
            continue
        if n != 3:
            raise ValueError(f"{res} {m}: {n} timed images in {rel(src)}; the captions report the median of three")
        med[res, "sca"] = (s, n)
        ranges[res, "sca"] = [lo, hi]
        origin[res, "sca"] = src
    if not any(k[1] == "sca" for k in med):
        raise ValueError(f"no {V2_METHOD} rows in {rel(src)}")
    print(f"landed: SCA (ours) = {V2_METHOD} rows of {rel(src)}; published SCA (sca_nb) not plotted")
    return src


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


def read_campaign(path, methods, res=CAMPAIGN_RES):
    """Campaign mean seconds per image (and bootstrap 95% interval) from a *_base_summary.csv."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["metric"] == "seconds" and r["res"] == res and r["method"] in methods:
                if r["method"] in out:
                    raise ValueError(f"duplicate seconds row for {r['method']} in {path}")
                out[r["method"]] = dict(mean=float(r["value"]), ci_low=float(r["ci_low"]),
                                        ci_high=float(r["ci_high"]), n_prompts=int(r["n_prompts"]))
    missing = set(methods) - set(out)
    if missing:
        raise ValueError(f"no campaign seconds for {sorted(missing)} in {path}")
    return out


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


def exponent_label(fig, ax, x, y, name, col, b, side, bold=False):
    """Two-line direct label at a line's end: the method name over "proportional to N^b".

    The exponent is set as its own small text so it stays legible (a mathtext superscript at 8 pt
    would print below 6 pt). `side` "right" hangs the label right of the point, "left" sets it
    above and to the left, where a line that stops short of the largest size has free space.
    """
    px = 72 / fig.dpi
    x_disp, y_disp = ax.transData.transform((x, y))
    if side == "right":
        ax.annotate(name, (x, y), xytext=(7, 2.5), textcoords="offset points", ha="left", va="center",
                    fontsize=sf.FS, color=col, fontweight="bold" if bold else "normal")
        base = ax.annotate(r"$\propto N$", (x, y), xytext=(7, -3.5), textcoords="offset points",
                           ha="left", va="top", fontsize=sf.FS, color=col)
        fig.canvas.draw()
        bb = base.get_window_extent()
        ax.annotate(f"{b:.2f}", (x, y), xytext=((bb.x1 - x_disp) * px + 0.6, (bb.y1 - y_disp) * px + 2.2),
                    textcoords="offset points", ha="left", va="top", fontsize=sf.FS_SMALL, color=col)
        return
    ax.annotate(name, (x, y), xytext=(-4, 10.5), textcoords="offset points", ha="right", va="bottom",
                fontsize=sf.FS, color=col, fontweight="bold" if bold else "normal")
    ex = ax.annotate(f"{b:.2f}", (x, y), xytext=(-4, 3.2), textcoords="offset points", ha="right",
                     va="bottom", fontsize=sf.FS_SMALL, color=col)
    fig.canvas.draw()
    bb = ex.get_window_extent()
    ax.annotate(r"$\propto N$", (x, y), xytext=((bb.x0 - x_disp) * px - 0.6, 1.0), textcoords="offset points",
                ha="right", va="bottom", fontsize=sf.FS, color=col)


def curve_figure(med, ranges, resolutions, speed, expo):
    """Seconds per image (log-log): baseline lines against SCA, the gap to AReA shaded, one bold
    headline ratio, the per-size ratios small and grey inside the gap, exponents as direct labels."""
    fig = plt.figure(figsize=(W, H))
    left, right, bottom, top = 0.56, 0.04, 0.37, 0.08
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
        ax.plot(xs, ys, color=col, lw=1.5 if m == "sca" else 1.1, marker=mk,
                ms=3.8 if m == "sca" else 3.3, mec="white", mew=0.4, zorder=3 if m == "sca" else 2.5,
                solid_capstyle="round")
        ends[m] = (rs[-1], xs[-1], ys[-1])
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
    ax.set_yticklabels([f"{t:g} s" for t in yt])  # units in the ticks
    ax.set_xlim(xt[0] / 1.2, xt[-1] * 2.3)
    ax.set_ylim(min(ys) / 1.35, max(ys) * 1.75)  # headroom for the full-attention label
    ax.grid(visible=False)
    ax.set_xlabel("output resolution")
    ax.set_ylabel("time per image")

    # direct labels: name and fitted exponent at each line's end
    for m, (res, x, y) in ends.items():
        lbl, col, _ = METHODS[m]
        side = "left" if m == "full" and res != resolutions[-1] else "right"
        exponent_label(fig, ax, x, y, lbl, col, expo[m], side, bold=(m == "sca"))

    # per-size ratios: small and grey, inside the shaded gap (a halo in the gap colour keeps them
    # legible where the full-attention line crosses); the largest size is the headline
    for r in both[:-1]:
        xr = n[r] * (1.14 if r == both[0] else 1.0)  # the gap is narrowest at the smallest size
        ax.text(xr, np.sqrt(med[r, "sca"][0] * med[r, "area"][0]), f"{speed[r]['area']:.2f}\u00d7",
                ha="center", va="center", fontsize=sf.FS_SMALL, color=sf.MID, zorder=4,
                path_effects=[pe.withStroke(linewidth=2.2, foreground=sf.SCA_LIGHT)])
    big = both[-1]
    ax.text(0.02, 0.96, f"{speed[big]['area']:.2f}\u00d7 faster at {res_label(big)}", transform=ax.transAxes,
            ha="left", va="top", fontsize=sf.FS, fontweight="bold", color=sf.SCA)
    ax.text(0.985, 0.03, "FLUX.1-dev", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=sf.FS_TICK, color=sf.MID)
    return fig, ax


# Main-text wrap version: 186 x 135 pt (about 0.47 of the 396 pt text width), natural size.
WRAP_WPT, WRAP_HPT = 186, 135  # points
WRAP_MARGIN = dict(left=37, right=1, bottom=24, top=3)  # points
WRAP_FS, WRAP_FS_EXP, WRAP_FS_HEAD = 7.0, 6.0, 7.5  # labels and ticks; exponents; the headline
WRAP_NAME = {"full": "Full attention", "area": "AReA", "sca": "SCA"}
WRAP_XPAD_LEFT, WRAP_XPAD = 1.15, 2.2  # left margin of the data; right room for the end labels
WRAP_YLIM_PAD = (1.45, 1.3)  # below the fastest time (room for the 2048^2 ratio); above the slowest
# Each per-size ratio sits just under the SCA line, right of SCA's point at that size, offset
# (dx, dy) points (text top-left corner); the line rises to the right, so the label stays clear of
# it and of every marker. In the band the full-attention line leaves no line-free room at 2048^2
# and 3072^2, so all four follow this one rule.
WRAP_RATIO_OFFSET = (3.0, 0.5)
WRAP_FULL_OFFSET = (-4.5, 0.0)  # one-line full-attention label, left of its last point (baseline)
WRAP_HEAD_Y = 0.13  # headline bottom, axes fraction, above the model name


def chain_label(fig, ax, x, y, parts, dx, dy, color, ha="left"):
    """Text pieces set side by side on one baseline at offset (dx, dy) points from data point (x, y).

    parts: (text, fontsize, bold, raise_pt, gap_before_pt). The pieces are measured first, so a
    raised exponent keeps its own size (a mathtext superscript would print below 6 pt). `ha`
    "right" ends the run at dx instead of starting it there.
    """
    px = 72 / fig.dpi
    renderer = fig.canvas.get_renderer()
    widths = []
    for text, fs, bold, _, _ in parts:
        probe = ax.text(0, 0, text, fontsize=fs, fontweight="bold" if bold else "normal")
        widths.append(probe.get_window_extent(renderer).width * px)
        probe.remove()
    total = sum(w + gap for w, (*_, gap) in zip(widths, parts))
    ox = dx - total if ha == "right" else dx
    out = []
    for (text, fs, bold, lift, gap), w in zip(parts, widths):
        ox += gap
        out.append(ax.annotate(text, (x, y), xytext=(ox, dy + lift), textcoords="offset points", ha="left",
                               va="baseline", fontsize=fs, color=color, clip_on=False,
                               fontweight="bold" if bold else "normal"))
        ox += w
    return out


def curve_figure_wrap(med, resolutions, speed, expo):
    """curve_figure on the main-text wrap canvas: same lines, markers, shaded gap, per-size ratios,
    exponents and bold headline, without whiskers, laid out for 186 x 135 pt with horizontal
    gridlines."""
    w, h, m = WRAP_WPT, WRAP_HPT, WRAP_MARGIN
    fig = plt.figure(figsize=(w / 72, h / 72))
    ax = fig.add_axes([m["left"] / w, m["bottom"] / h, (w - m["left"] - m["right"]) / w,
                       (h - m["bottom"] - m["top"]) / h])
    n = {r: tokens(r) for r in resolutions}
    xt = [n[r] for r in resolutions]

    both = [r for r in resolutions if (r, "area") in med and (r, "sca") in med]
    # above the gridlines (zorder 0.5), below every line
    ax.fill_between([n[r] for r in both], [med[r, "sca"][0] for r in both],
                    [med[r, "area"][0] for r in both], color=sf.SCA_LIGHT, lw=0, zorder=0.8)
    ends = {}
    for k, (lbl, col, mk) in METHODS.items():
        rs = [r for r in resolutions if (r, k) in med]
        if not rs:
            continue
        xs, ys = [n[r] for r in rs], [med[r, k][0] for r in rs]
        # no whiskers here (the main-text figure carries no error bars); the 5.5 in figure keeps them
        ax.plot(xs, ys, color=col, lw=1.5 if k == "sca" else 1.1, marker=mk,
                ms=3.8 if k == "sca" else 3.3, mec="white", mew=0.4, zorder=3 if k == "sca" else 2.5,
                solid_capstyle="round")
        ends[k] = (rs[-1], xs[-1], ys[-1])
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
    ax.set_yticklabels([f"{t:g} s" for t in yt])
    ax.set_xlim(xt[0] / WRAP_XPAD_LEFT, xt[-1] * WRAP_XPAD)
    ax.set_ylim(min(ys) / WRAP_YLIM_PAD[0], max(ys) * WRAP_YLIM_PAD[1])
    sf.value_grid(ax, "y")
    ax.set_xlabel("output resolution")
    ax.set_ylabel("time per image")
    fig.canvas.draw()

    # direct labels: AReA and SCA hang right of their last point (name over the exponent); full
    # attention, which stops short of the largest size, reads on one line left of its end
    for k, (res, x, y) in ends.items():
        col = METHODS[k][1]
        exp = [(r"$\propto N$", WRAP_FS, False, 0, 0), (f"{expo[k]:.2f}", WRAP_FS_EXP, False, 2.6, 0.5)]
        if k == "full" and res != resolutions[-1]:
            exp[0] = (r"$\propto N$", WRAP_FS, False, 0, 2.5)  # a word space after the name
            chain_label(fig, ax, x, y, [(WRAP_NAME[k], WRAP_FS, False, 0, 0)] + exp, *WRAP_FULL_OFFSET, col,
                        ha="right")
        else:
            chain_label(fig, ax, x, y, [(WRAP_NAME[k], WRAP_FS, k == "sca", 0, 0)], 5, 3.0, col)
            chain_label(fig, ax, x, y, exp, 5, -7.2, col)

    # per-size ratios: small and grey, just under the SCA line at each size; the largest size is the
    # bold headline in the empty lower right
    for r in both[:-1]:
        ax.annotate(f"{speed[r]['area']:.2f}\u00d7", (n[r], med[r, "sca"][0]), xytext=WRAP_RATIO_OFFSET,
                    textcoords="offset points", ha="left", va="top", fontsize=sf.FS_SMALL, color=sf.MID,
                    zorder=4)
    big = both[-1]
    ax.text(0.985, WRAP_HEAD_Y, f"{speed[big]['area']:.2f}\u00d7 faster at {res_label(big)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=WRAP_FS_HEAD, fontweight="bold",
            color=sf.SCA)
    ax.text(0.985, 0.03, "FLUX.1-dev", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=sf.FS_SMALL, color=sf.MID)
    return fig, ax


def tex_res(res):
    w, h = parse_res(res)
    return f"${w}^2$" if w == h else rf"${w}\times{h}$"


HRDIT_SIDECARS = "/scratch/09032/saini_2/sca/out/latency_hrdit/hrdit_dev_author/4096x4096"  # per image


def hrdit_clean_seconds(path=BATCH1_CSV, res=CAMPAIGN_RES):
    """HRDiT's three clean batch-1 timings at 4096^2: the hrdit_dev_author rows of the batch-1 CSV.
    The run timed three images with no excluded warm-up (timing_scope in the sidecars)."""
    with open(path) as fh:
        vals = [float(r["seconds"]) for r in csv.DictReader(fh)
                if r["method"].strip() == "hrdit_dev_author" and r["res"].strip() == res]
    if len(vals) != 3:
        raise ValueError(f"expected three HRDiT batch-1 rows at {res} in {path}, found {len(vals)}")
    return vals


def write_snippet(path, resolutions, med, speed, expo, sources):
    """Caption numbers come from the same CSVs as the plot. At most 50 words: a bold claim, then the
    measurement conditions in two sentences (HRDiT's clean time is quoted in the main text)."""
    big = max((r for r in resolutions if "area" in speed[r]), key=tokens)  # the largest size SCA and AReA share
    full_last = max((r for r in resolutions if (r, "full") in med), key=tokens, default=None)
    stops = f" full attention was not run at {tex_res(big)}." if full_last not in (None, big) else ""
    cap = (
        rf"\textbf{{\sca{{}} is fastest at every size; its lead over AReA grows to "
        rf"${speed[big]['area']:.2f}\times$ at {tex_res(big)}.}}" "\n"
        rf"  Batch 1 on one GH200, median of three prompts, whiskers min to max;{stops}" "\n"
        r"  Grey ratios: speedup over AReA per size; exponents fit seconds $\propto N^b$ ($N$ image tokens)."
    )
    curves = " and ".join(rel(p) for p in sources)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"% Generated by paper/figures/src/fig_latency.py from {curves}\n"
                 f"% (clean batch-1 sweep) and {rel(BATCH1_CSV)} (HRDiT's clean time); the\n"
                 "% results/baseline46/base_summary.csv campaign means are recorded in the provenance only.\n")
        fh.write("\\begin{figure}[tp]\n  \\centering\n  \\includegraphics{figures/latency.pdf}\n")
        fh.write("  \\caption{" + cap + "}\n  \\label{fig:latency}\n\\end{figure}\n")
    return path


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=list(CURVE_CSVS),
                    help="curve CSVs; a (res, method) cell comes from the first file that has it")
    ap.add_argument("--source", choices=("auto", "curve", "4096"), default="auto")
    default_out = os.path.join(OUTDIR, "latency")
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()
    landed = v2swap.landed(v2swap.LATENCY)
    print(f"fig_latency.py: {'landed' if landed else 'pending'} ({v2swap.LATENCY})"
          + (f" [mock {v2swap.MOCK}]" if v2swap.MOCK else ""), flush=True)
    # The paper snippet is written only for the default output; trial runs put it beside --out.
    snippet = SNIPPET if os.path.abspath(args.out) == default_out else args.out + "_snippet.tex"

    if args.source == "4096" or (args.source == "auto" and not any(os.path.exists(p) for p in args.csv)):
        rows = read_4096(CSV_4096)
        origin = {(r, m): CSV_4096 for r, m, _ in rows}
    else:
        rows, origin = read_curves(args.csv)
    srcs = [p for p in dict.fromkeys([*args.csv, CSV_4096]) if p in origin.values()]
    print("latency sources: " + ", ".join(srcs))
    med = medians(rows)
    ranges = {}
    for res, method, seconds in rows:
        ranges.setdefault((res, method), []).append(seconds)
    v2_src = switch_sca(med, ranges, origin) if landed else None
    if v2_src:
        srcs.append(v2_src)
    resolutions = sorted({r for r, _ in med}, key=lambda r: np.prod(parse_res(r)))
    for (res, m), (s, n) in sorted(med.items(), key=lambda kv: (tokens(kv[0][0]), kv[0][1])):
        print(f"  {res} {m}: median {s:.2f} s over {n} images ({src_name(origin[res, m])})")
    no_sca = [r for r in resolutions if (r, "sca") not in med]
    if no_sca:
        print(f"WARNING: no SCA cell at {', '.join(no_sca)}; the fastest-at-every-size check skips them")

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

    camp = read_campaign(CAMPAIGN_CSV, CAMPAIGN)
    if len({v["n_prompts"] for v in camp.values()}) != 1:
        raise ValueError("campaign means must come from one prompt set")
    ultra191 = read_campaign(CAMPAIGN_191_CSV, {"ultraimage": None})["ultraimage"]
    for k, v in camp.items():
        print(f"campaign mean at {CAMPAIGN_RES}, {k}: {v['mean']:.2f} s "
              f"[{v['ci_low']:.2f}, {v['ci_high']:.2f}] over {v['n_prompts']} prompts")
    print(f"campaign mean at {CAMPAIGN_RES}, ultraimage: {ultra191['mean']:.2f} s over "
          f"{ultra191['n_prompts']} prompts (provenance only)")
    fastest = all(med[r, "sca"][0] == min(v for (rr, _), (v, _) in med.items() if rr == r)
                  for r in resolutions if (r, "sca") in med)
    fastest &= camp["sca_nb"]["mean"] == min(v["mean"] for v in camp.values())
    if not fastest:
        raise ValueError("caption lead no longer holds: SCA is not the fastest measured method at some size")
    sf.use_style()
    plt.rcParams["svg.hashsalt"] = "sca"
    hrdit = None  # HRDiT's clean time is quoted in the caption, which only the curve figure has
    if len(resolutions) == 1:
        fig = bar_chart(med, speed, resolutions[0])
    else:
        hrdit = hrdit_clean_seconds()
        hrdit_med = float(np.median(hrdit))
        print(f"HRDiT clean batch-1 at {CAMPAIGN_RES}: median {hrdit_med:.2f} s of {hrdit} (caption only)")
        fig, _ = curve_figure(med, ranges, resolutions, speed, expo)
        print("wrote", write_snippet(snippet, resolutions, med, speed, expo, srcs))
        # the main-text wrapfigure, 186 x 135 pt, from the same numbers
        sf.save(curve_figure_wrap(med, resolutions, speed, expo)[0], args.out + "_wrap")
    sf.save(fig, args.out)
    full_res = sorted((r for r in resolutions if (r, "full") in med), key=tokens)
    hrdit_prov = {} if hrdit is None else {"hrdit_clean_batch1": {
        "source": BATCH1_CSV, "sha256": sha256(BATCH1_CSV), "res": CAMPAIGN_RES,
        "method": "hrdit_dev_author", "seconds": hrdit, "median_seconds": hrdit_med,
        "n": len(hrdit), "sidecars": HRDIT_SIDECARS,
        "timing_scope": "pipeline including text encoding and decoding, not model load; three "
                        "timed images with no excluded warm-up",
        "use": "quoted in the caption; not plotted"}}
    v2_prov = {} if v2_src is None else {"sca_v2_switch": {
        "state": "landed" + (f" (mock tree {v2swap.MOCK})" if v2swap.MOCK else ""),
        "source": v2_src, "sha256": sha256(v2_src), "method": V2_METHOD, "plotted_as": METHODS["sca"][0],
        "not_plotted": "sca_nb (published SCA, now SCA without the schedule)",
        "full_and_area": "per-image cells as in the published figure; each equals "
                         "results/latency_curve_summary.csv wherever that file holds it"}}
    with open(args.out + "_provenance.json", "w") as handle:
        json.dump({"sources": [{"path": p, "sha256": sha256(p), "cells": sum(o == p for o in origin.values())}
                               for p in srcs],
                   "cell_rule": "each (res, method) cell comes from the first source that holds it; a later "
                                "source holding the same cell matched it image for image within the coarser "
                                "file's rounding (half a unit in its last printed decimal)",
                   "warmup": f"rows whose image id starts with {WARMUP_PREFIX} (the one excluded warm-up "
                             "image per process) are dropped; latency_curve.csv rows carry warmup_verified "
                             "from scripts/audit_latency.py",
                   "statistic": "median of three timed prompts",
                   "whiskers": "observed minimum and maximum, not a confidence interval",
                   "exponent_fit": "least squares of log median seconds on log N, N = (W/16)(H/16) tokens",
                   "exponents": {m: round(v, 6) for m, v in sorted(expo.items())},
                   "groups": [{"res": r, "method": m, "median_seconds": value, "n": n,
                               "source": src_name(origin[r, m])}
                              for (r, m), (value, n) in sorted(med.items())],
                   "speedup_sca_over_area": {r: round(v["area"], 4) for r, v in speed.items() if "area" in v},
                   "full_attention": {"measured": full_res,
                                      "not_plotted": "8192x8192 not run by design (quadratic attention cost)"},
                   **hrdit_prov,
                   **v2_prov,
                   "campaign_means": {
                       "source": CAMPAIGN_CSV, "sha256": sha256(CAMPAIGN_CSV), "res": CAMPAIGN_RES,
                       "statistic": "mean seconds per image over the campaign's prompts on shared nodes, "
                                    "bootstrap 95% interval; not clean batch-1 latency",
                       "drawn_as": "not plotted (house style: clean sweep only); recorded for reconciliation",
                       "methods": {k: {**v, "label": CAMPAIGN[k][0]} for k, v in camp.items()},
                       "protocols": {"hrdit_dev_author": "progressive 30 + 17 + 10 steps at 1024^2, 2048^2, "
                                                         "4096^2 (scripts/sample_progressive.py HRDIT)",
                                     "ultraimage": "released two-stage recipe, 30 steps per stage",
                                     "sca_nb": "28 steps at the target size"},
                       "ultraimage_191_prompts": {"source": CAMPAIGN_191_CSV, "sha256": sha256(CAMPAIGN_191_CSV),
                                                  **ultra191, "use": "not plotted; recorded to reconcile with the 191-prompt tables"}}},
                  handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
