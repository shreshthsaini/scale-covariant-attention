"""Seconds per image against resolution for full attention, AReA and SCA (batch 1, one GH200).

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

Campaign means are a separate, marked layer: hollow markers at 4096^2 for SCA, HRDiT and
UltraImage, read from results/baseline46/base_summary.csv (metric "seconds", the same 46 prompts
for every method, run on shared nodes). They are never joined to the clean curves. UltraImage's
191-prompt campaign mean (results/e1_ultraimage_base_summary.csv) is only recorded in the provenance.
HRDiT's clean batch-1 time at 4096^2 (median of the three hrdit_dev_author rows of
results/latency_curve_batch1.csv; that run has no excluded warm-up) is quoted in the caption and
not plotted.

With several resolutions the figure is one log-log panel:
  median seconds per image against resolution, whiskers = observed min and max, a light fill
  between AReA and SCA, the least-squares exponent b of seconds ~ N^b (N = image tokens,
  (W/16)(H/16)) fitted to each method's medians, SCA's speedup over AReA as a top row, the
  campaign means as hollow markers, and a note that full attention stops at 6144^2 because
  8192^2 was not run by design.
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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator
from matplotlib.transforms import offset_copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scafig as sf  # noqa: E402
from k1data import FIGDIR  # noqa: E402

REPO = os.path.normpath(os.path.join(FIGDIR, "..", ".."))
CURVE_CSV = os.path.join(REPO, "results", "latency_curve.csv")  # audited sweep, full precision
BATCH1_CSV = os.path.join(REPO, "results", "latency_curve_batch1.csv")  # 0.01 s, adds full at 6144^2
CURVE_CSVS = (CURVE_CSV, BATCH1_CSV)  # a cell comes from the first file that has it
WARMUP_PREFIX = "sca_latency_warmup"  # id of the excluded first image of every timed process
CSV_4096 = os.path.join(REPO, "results", "latency_4096_batch1.csv")
CAMPAIGN_CSV = os.path.join(REPO, "results", "baseline46", "base_summary.csv")
CAMPAIGN_191_CSV = os.path.join(REPO, "results", "e1_ultraimage_base_summary.csv")
SNIPPET = os.path.join(FIGDIR, "snippets", "latency.tex")

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
FULL_STOP_NOTE = ("8192\u00b2 not run", "by design (quadratic cost)")  # two lines of the in-plot note
PATCH = 16  # pixels per image token side: VAE factor 8 times 2x2 latent packing
W, H = sf.TEXT, 2.05  # inches; the main-text figure must not grow


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


def curve_figure(med, ranges, resolutions, speed, expo, camp):
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
    ax.set_yticklabels([f"{t:g}" for t in yt])
    ax.set_xlim(xt[0] / 1.2, xt[-1] * 2.3)
    ax.set_ylim(min(ys) / 1.35, max(ys) * 1.3)
    ax.grid(visible=False)
    ax.set_xlabel("output resolution (log scale)")
    ax.set_ylabel("seconds per image")

    # direct labels: name and fitted exponent at each line's end (full attention, when it stops
    # short of the largest size, is named by a keyed note instead)
    full_stops = "full" in ends and ends["full"][0] != resolutions[-1]
    for m, (_, x, y) in ends.items():
        lbl, col, _ = METHODS[m]
        if m == "full" and full_stops:
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

    # Why the full-attention curve stops: a note keyed by the line's own marker, in the free space
    # below the SCA line, with the marker in the column of the line's last point. No leader: from
    # there it would cross the AReA and SCA lines, and a label above the point would sit under the
    # speedup row and in line with the key.
    if full_stops:
        _, fx, _ = ends["full"]
        lbl, col, mk = METHODS["full"]
        ny = 0.36  # top of the note, axes fraction
        at = ax.get_xaxis_transform()  # x in data, y in axes fraction
        ax.plot([fx], [ny], transform=offset_copy(at, fig=fig, y=-4.6, units="points"), color=col,
                marker=mk, ms=3.3, mec="white", mew=0.4, ls="none", zorder=4)
        ax.annotate(f"{lbl}: {FULL_STOP_NOTE[0]}\n{FULL_STOP_NOTE[1]}", (fx, ny), xycoords=at,
                    xytext=(5.5, 0), textcoords="offset points", ha="left", va="top",
                    fontsize=sf.FS_TICK, color=sf.MID, linespacing=1.15)

    # Campaign means at 4096^2 (shared nodes): hollow markers on the clean column, never joined.
    xc = n[CAMPAIGN_RES]
    for key, (lbl, col, mk) in CAMPAIGN.items():
        v = camp[key]["mean"]
        if key == "sca_nb":  # a ring around SCA's clean point, which it nearly equals
            ax.plot([xc], [v], ls="none", marker="o", ms=7.4, mfc="none", mec=col, mew=0.85, zorder=3.6)
            continue
        ax.plot([xc], [v], ls="none", marker=mk, ms=4.0, mfc="white", mec=col, mew=0.9, zorder=3.6)
        dy = -3.0 if key == "hrdit_dev_author" else 0  # clear the full-attention line just above it
        ax.annotate(lbl, (xc, v), xytext=(-5.5, dy), textcoords="offset points", ha="right", va="center",
                    fontsize=sf.FS_TICK, color=col)

    # key: filled = clean batch-1 sweep, hollow = campaign mean (grey baselines, green SCA)
    kx, ky, dy, dx = 0.020, 0.955, 0.13, 0.019  # axes fractions
    base_col = METHODS["area"][1]
    glyphs = [  # (row, slot, marker, size, face, edge, edge width)
        (0, 0, "s", 3.3, base_col, "white", 0.4), (0, 1, "o", 3.8, sf.SCA, "white", 0.4),
        (1, 0, "D", 3.6, "white", base_col, 0.85), (1, 1, "o", 5.0, "none", sf.SCA, 0.85)]
    for row, slot, mk, ms, face, edge, mew in glyphs:
        ax.plot([kx + slot * dx], [ky - row * dy], transform=ax.transAxes, ls="none", marker=mk, ms=ms,
                mfc=face, mec=edge, mew=mew, clip_on=False, zorder=4)
    texts = ["batch 1, one GH200, median of three prompts",
             f"campaign mean at {res_label(CAMPAIGN_RES)}, shared nodes, {camp['sca_nb']['n_prompts']} prompts"]
    for row, t in enumerate(texts):
        ax.text(kx + dx + 0.017, ky - row * dy, t, transform=ax.transAxes, ha="left", va="center",
                fontsize=sf.FS_TICK, color=sf.MID)

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


def write_snippet(path, resolutions, speed, expo, camp, hrdit_clean, sources):
    """Caption numbers come from the same CSVs as the plot (HRDiT's clean time from the batch-1 CSV)."""
    big = resolutions[-1]
    n_camp = camp["sca_nb"]["n_prompts"]
    cap = (
        rf"\textbf{{\sca{{}} is the fastest of the methods measured at each size, and its lead over AReA "
        rf"grows to ${speed[big]['area']:.2f}\times$ at {tex_res(big)}.}}" "\n"
        r"  Filled markers: batch-1 seconds per image on one GH200 (median and range of three prompts; speedup labels use unrounded medians); "
        rf"full attention scales as $N^{{{expo['full']:.2f}}}$ ($N$: image tokens)." "\n"
        rf"  Hollow markers: \mbox{{{n_camp}-prompt}} campaign means at {tex_res(CAMPAIGN_RES)} on shared nodes; HRDiT "
        r"runs 30, 17, 10 steps at $1024^2$, $2048^2$, $4096^2$, UltraImage two 30-step stages. "
        rf"HRDiT's clean batch-1 time is {hrdit_clean:.1f}\,s (median of three)."
    )
    curves = " and ".join(rel(p) for p in sources)
    with open(path, "w") as fh:
        fh.write(f"% Generated by paper/figures/src/fig_latency.py from {curves}\n"
                 f"% (clean batch-1 sweep), {rel(BATCH1_CSV)} (HRDiT's clean time) and\n"
                 "% results/baseline46/base_summary.csv (46-prompt campaign means).\n")
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
    default_out = os.path.join(FIGDIR, "latency")
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()
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
    resolutions = sorted({r for r, _ in med}, key=lambda r: np.prod(parse_res(r)))
    for (res, m), (s, n) in sorted(med.items(), key=lambda kv: (tokens(kv[0][0]), kv[0][1])):
        print(f"  {res} {m}: median {s:.2f} s over {n} images ({os.path.basename(origin[res, m])})")

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
    fastest = all(med[r, "sca"][0] == min(v for (rr, _), (v, _) in med.items() if rr == r) for r in resolutions)
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
        fig, _, _ = curve_figure(med, ranges, resolutions, speed, expo, camp)
        print("wrote", write_snippet(snippet, resolutions, speed, expo, camp, hrdit_med, srcs))
    sf.save(fig, args.out)
    full_res = sorted((r for r in resolutions if (r, "full") in med), key=tokens)
    hrdit_prov = {} if hrdit is None else {"hrdit_clean_batch1": {
        "source": BATCH1_CSV, "sha256": sha256(BATCH1_CSV), "res": CAMPAIGN_RES,
        "method": "hrdit_dev_author", "seconds": hrdit, "median_seconds": hrdit_med,
        "n": len(hrdit), "sidecars": HRDIT_SIDECARS,
        "timing_scope": "pipeline including text encoding and decoding, not model load; three "
                        "timed images with no excluded warm-up",
        "use": "quoted in the caption; not plotted"}}
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
                               "source": os.path.basename(origin[r, m])}
                              for (r, m), (value, n) in sorted(med.items())],
                   "speedup_sca_over_area": {r: round(v["area"], 4) for r, v in speed.items() if "area" in v},
                   "full_attention": {"measured": full_res,
                                      "not_plotted": "8192x8192 not run by design (quadratic attention cost)"},
                   **hrdit_prov,
                   "campaign_means": {
                       "source": CAMPAIGN_CSV, "sha256": sha256(CAMPAIGN_CSV), "res": CAMPAIGN_RES,
                       "statistic": "mean seconds per image over the campaign's prompts on shared nodes, "
                                    "bootstrap 95% interval; not clean batch-1 latency",
                       "drawn_as": "hollow markers at 4096^2, not joined to the clean curves",
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
