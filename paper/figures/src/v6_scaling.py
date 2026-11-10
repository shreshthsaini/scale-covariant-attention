"""Figure 7 (paper/figures/v6/scaling.pdf): clean batch-1 seconds per image against image side, every method timed.

Sources (results/, read only; every plotted value is read here, none typed):
  results/latency_curve_v2/summary.csv      sweep 2, single-pass SCA (method sca_v2) at 2048^2 to 8192^2: median,
                                            minimum and maximum of three timed images (sca_nb, SCA without the
                                            schedule, is in the file and is not plotted)
  results/latency_curve_summary.csv         sweep 1, AReA and full attention: the same statistics; it lacks full
                                            attention at 6144^2. Each of its AReA and full-attention cells must equal
                                            the per-image rows of results/latency_curve.csv (audited, full precision,
                                            warmup_verified True).
  results/latency_curve_batch1.csv          sweep 1 per image, rounded to 0.01 s: full attention at 6144^2, the median
                                            (and range) of its three timed rows after dropping the one warm-up row
                                            (image id sca_latency_warmup...); full attention was not run at 8192^2
  results/prog_latency/latency_summary.csv  sweep 3, 4096^2 only: progressive SCA (sca_v2_prog) and HRDiT
                                            (hrdit_dev_author); its AReA cell must agree with sweep 1 within 0.1 s
One GH200, batch 1, 28 steps (HRDiT and progressive SCA: their own schedules), text encoding and VAE decoding
included, one excluded warm-up image per process. Fitted exponent: least squares of log median seconds on log N over
all sizes of a method, N = (W/16)(H/16) image tokens (as fig_latency.py). Ratios divide two clean times (AReA's
median over single-pass SCA's median at the same size).

Checks (the run stops on any failure): every cell has three timed images; the plotted medians round (0.1 s, half
up) to Table tab:latency (snippets/app_tables/latency.tex) and to headline.tex (\\hlSecSca, \\hlSecArea, \\hlSecProg,
\\hlSecHrditThree, \\hlSecAreaThree); the ratios round to \\hlSpeedArea (2.63 at 4096^2), \\hlSpeedProgArea,
\\hlSpeedProgHrdit and to speed_numbers.tex (\\hlSpeedScaAreaLo 2.1 at 2048^2, \\hlSpeedScaAreaHi 3.1 at 8192^2);
the exponents print as 0.99 (single-pass SCA) and 1.12 (AReA) and equal headline.tex's \\hlLatExpSca and \\hlLatExpArea,
the 8192^2 times and ratio \\hlLatEightSca, \\hlLatEightArea, \\hlLatEightRatio; single-pass SCA is the fastest method of
the sweep at every size, progressive SCA the fastest at 4096^2, and the lead over AReA grows with size.

Candidates (both rendered and scored; WINNER goes to v6/scaling.pdf, the other to v6/candidates/):
  A  one log-log panel, 0.48 text width: lines with direct labels and exponents, the per-size ratio as a row of
     numbers along the bottom of the panel.
  B  the same panel over a short bar panel of the per-size ratio, values printed on the bars.

Run on a compute node: <sca env python> -I paper/figures/src/v6_scaling.py [--layout A|B|both] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

import scafig as sf  # noqa: E402
import v6_plotkit as pk  # noqa: E402

R = pk.ROOT / "results"
SIDES = (2048, 3072, 4096, 6144, 8192)
PATCH = 16  # pixels per image token side (VAE factor 8, 2x2 latent packing)
WARMUP = "sca_latency_warmup"
STEM = "scaling"
WINNER = "A"
# Exponents as the paper prints them (main.tex Section 4 "Other sizes" and the method-comparison table of the
# appendix); a check on the fit, never plotted from here.
PRINTED_EXPONENT = {"sca_v2": "0.99", "area": "1.12"}
NAME = {"sca_v2": "Single-pass SCA", "area": "AReA", "full": "Full attention", "sca_v2_prog": "Progressive SCA",
        "hrdit_dev_author": "HRDiT"}
COLOR = {"sca_v2": sf.SCA, "area": sf.AREA, "full": sf.METHOD_COLOR["full"], "sca_v2_prog": sf.SCA,
         "hrdit_dev_author": sf.METHOD_COLOR["ultraimage"]}  # HRDiT, the strongest grey baseline: darkest grey
LINES = ("full", "area", "sca_v2")  # drawn in this order, single-pass SCA on top
MARK = {"sca_v2": dict(marker="o", ms=3.8), "area": dict(marker="o", ms=3.4), "full": dict(marker="^", ms=3.6),
        "sca_v2_prog": dict(marker="*", ms=9.0), "hrdit_dev_author": dict(marker="D", ms=3.9)}


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def side(res):
    w, h = (int(v) for v in res.split("x"))
    assert w == h, res
    return w


def tokens(s):
    return (s // PATCH) ** 2


def half_up(x, d=1):
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def load():
    """{(method, side): dict(med, lo, hi, n, src)} for the five lines' cells and the two 4096^2-only methods."""
    cells = {}
    for r in rows(R / "latency_curve_v2" / "summary.csv"):
        if r["method"] == "sca_v2":
            cells["sca_v2", side(r["res"])] = dict(med=float(r["median_seconds"]), lo=float(r["minimum_seconds"]),
                                                   hi=float(r["maximum_seconds"]), n=int(r["n"]),
                                                   src="results/latency_curve_v2/summary.csv")
    per_image = {}
    for r in rows(R / "latency_curve.csv"):
        assert r["warmup_verified"].strip().lower() == "true", r
        per_image.setdefault((r["method"], side(r["res"])), []).append(float(r["seconds"]))
    for r in rows(R / "latency_curve_summary.csv"):
        if r["method"] not in ("area", "full"):
            continue
        key = (r["method"], side(r["res"]))
        c = dict(med=float(r["median_seconds"]), lo=float(r["minimum_seconds"]), hi=float(r["maximum_seconds"]),
                 n=int(r["n"]), src="results/latency_curve_summary.csv")
        v = per_image[key]  # the audited per-image file holds every cell of this summary
        assert len(v) == c["n"] == 3 and abs(statistics.median(v) - c["med"]) < 1e-9 and \
            abs(min(v) - c["lo"]) < 1e-9 and abs(max(v) - c["hi"]) < 1e-9, (key, v, c)
        cells[key] = c
    timed, warm = [], 0
    for r in rows(R / "latency_curve_batch1.csv"):
        if r["method"] == "full" and r["res"] == "6144x6144":
            if r["image"].startswith(WARMUP):
                warm += 1
            else:
                timed.append(float(r["seconds"]))
    assert warm == 1 and len(timed) == 3 and ("full", 6144) not in cells, (warm, timed)
    cells["full", 6144] = dict(med=statistics.median(timed), lo=min(timed), hi=max(timed), n=3,
                               src="results/latency_curve_batch1.csv (three timed rows, warm-up dropped)")
    s3 = {r["method"]: r for r in rows(R / "prog_latency" / "latency_summary.csv") if r["res"] == "4096x4096"}
    for m in ("sca_v2_prog", "hrdit_dev_author"):
        r = s3[m]
        cells[m, 4096] = dict(med=float(r["median_s"]), lo=float(r["min_s"]), hi=float(r["max_s"]),
                              n=int(r["n_timed"]), src="results/prog_latency/latency_summary.csv (sweep 3)")
    area3 = float(s3["area"]["median_s"])
    assert abs(area3 - cells["area", 4096]["med"]) <= 0.1, (area3, cells["area", 4096])
    assert all(c["n"] == 3 for c in cells.values())
    assert {s for (m, s) in cells if m == "sca_v2"} == set(SIDES) == {s for (m, s) in cells if m == "area"}
    assert {s for (m, s) in cells if m == "full"} == set(SIDES) - {8192}
    return cells, area3


def derived(cells):
    expo = {}
    for m in LINES:
        ss = sorted(s for (k, s) in cells if k == m)
        expo[m] = float(np.polyfit(np.log([tokens(s) for s in ss]), np.log([cells[m, s]["med"] for s in ss]), 1)[0])
    ratio = {s: cells["area", s]["med"] / cells["sca_v2", s]["med"] for s in SIDES}
    return expo, ratio


def latency_table():
    """{(method, side): printed seconds} from Table tab:latency (columns full attention, AReA, single-pass SCA)."""
    text = (pk.SNIPPETS / "app_tables" / "latency.tex").read_text()
    head = re.search(r"Size.*?&\s*Full attention\s*&\s*AReA\s*&\s*Single-pass", text)
    assert head, "tab:latency columns are no longer full attention, AReA, single-pass SCA"
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*\$(\d+)\^2\$", line)
        if not m:
            continue
        cols = line.split("&")
        assert len(cols) == 4, line
        for key, cell in zip(("full", "area", "sca_v2"), cols[1:]):
            cell = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell.replace("\\\\", "")).strip()
            if cell != "not run":
                out[key, int(m.group(1))] = float(cell)
    assert out, "no rows parsed from tab:latency"
    return out


def checks(cells, area3, expo, ratio):
    bad = []
    printed = latency_table()
    for (m, s), v in printed.items():
        pk.check(half_up(cells[m, s]["med"]) == v, f"tab:latency {m} {s}: {cells[m, s]['med']:.3f} vs {v}", bad)
    pk.check(set(printed) == {k for k in cells if k[0] in LINES}, "tab:latency holds every line cell", bad)
    for name, value in (("hlSecSca", cells["sca_v2", 4096]["med"]), ("hlSecArea", cells["area", 4096]["med"]),
                        ("hlSecProg", cells["sca_v2_prog", 4096]["med"]),
                        ("hlSecHrditThree", cells["hrdit_dev_author", 4096]["med"]), ("hlSecAreaThree", area3)):
        pk.check(f"{half_up(value):.1f}" == pk.macro(name), f"\\{name} {pk.macro(name)} vs {value:.3f}", bad)
    pk.check(f"{ratio[4096]:.2f}" == pk.macro("hlSpeedArea"), f"\\hlSpeedArea vs {ratio[4096]:.4f}", bad)
    prog = cells["sca_v2_prog", 4096]["med"]
    pk.check(f"{area3 / prog:.1f}" == pk.macro("hlSpeedProgArea"), "\\hlSpeedProgArea", bad)
    pk.check(f"{cells['hrdit_dev_author', 4096]['med'] / prog:.1f}" == pk.macro("hlSpeedProgHrdit"),
             "\\hlSpeedProgHrdit", bad)
    sn = pk.SNIPPETS / "speed_numbers.tex"
    pk.check(f"{ratio[2048]:.1f}" == pk.macro("hlSpeedScaAreaLo", sn), f"\\hlSpeedScaAreaLo vs {ratio[2048]:.3f}", bad)
    pk.check(f"{ratio[8192]:.1f}" == pk.macro("hlSpeedScaAreaHi", sn), f"\\hlSpeedScaAreaHi vs {ratio[8192]:.3f}", bad)
    for m, v in PRINTED_EXPONENT.items():
        pk.check(f"{expo[m]:.2f}" == v, f"exponent {m}: {expo[m]:.4f}, printed {v}", bad)
    for name, value in (("hlLatExpSca", f"{expo['sca_v2']:.2f}"), ("hlLatExpArea", f"{expo['area']:.2f}"),
                        ("hlLatEightSca", f"{half_up(cells['sca_v2', 8192]['med']):.1f}"),
                        ("hlLatEightArea", f"{half_up(cells['area', 8192]['med']):.1f}"),
                        ("hlLatEightRatio", f"{ratio[8192]:.2f}")):
        pk.check(pk.macro(name) == value, f"headline.tex \\{name} {pk.macro(name)} vs {value}", bad)
    for s in SIDES:  # single-pass SCA is the fastest method of the sweep at every size
        others = [cells[m, s]["med"] for m in ("area", "full") if (m, s) in cells]
        pk.check(cells["sca_v2", s]["med"] < min(others), f"single-pass SCA not fastest at {s}", bad)
    pk.check(prog < min(c["med"] for (m, s), c in cells.items() if s == 4096 and m != "sca_v2_prog"),
             "progressive SCA not the fastest at 4096^2", bad)
    pk.check(all(ratio[a] <= ratio[b] for a, b in zip(SIDES, SIDES[1:])), "the lead over AReA grows with size", bad)
    pk.fail_if(bad, "v6_scaling.py")


# ------------------------------------------------------------------------------------------------ drawing
YLIM = (11.0, 1500.0)
YTICKS = (20, 50, 100, 200, 500, 1000)
XLIM = (2048 / 1.13, 8192 * 1.09)
HRDIT_LABEL_AT = (4750, 222)  # data coordinates: inside the gap, clear of the lines (checked on the render)


def side_axis(ax, labels=True):
    ax.set_xscale("log", base=2)
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(FixedLocator(SIDES))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}" if labels else ""))


def draw_main(fig, ax, cells, expo, fs=sf.FS_TICK):
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.yaxis.set_major_locator(FixedLocator(YTICKS))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    side_axis(ax)
    pk.clean_axes(ax, grid="y")
    for m in LINES:
        ss = sorted(s for (k, s) in cells if k == m)
        xs, ys = ss, [cells[m, s]["med"] for s in ss]
        ax.vlines(xs, [cells[m, s]["lo"] for s in ss], [cells[m, s]["hi"] for s in ss], color=COLOR[m], lw=0.8,
                  zorder=2)
        ax.plot(xs, ys, color=COLOR[m], lw={"sca_v2": 1.6, "area": 1.3, "full": 1.1}[m],
                ls=(0, (3.2, 1.6)) if m == "full" else "-", mec="white", mew=0.4, zorder=4 if m == "sca_v2" else 3,
                **MARK[m])
    for m in ("hrdit_dev_author", "sca_v2_prog"):
        c = cells[m, 4096]
        ax.plot([4096], [c["med"]], ls="none", color=COLOR[m], mec="white", mew=0.5 if m != "sca_v2_prog" else 0.0,
                zorder=6, **MARK[m])
    # direct labels: the two lines that reach 8192^2 at their right end (name over exponent), full attention up
    # left of its last point, the two 4096^2-only methods beside their markers
    lab = dict(fontsize=fs, textcoords="offset points", annotation_clip=False)
    for m, dy in (("area", 3.0), ("sca_v2", 1.0)):
        x, y = 8192, cells[m, 8192]["med"]
        nm = "Single pass" if m == "sca_v2" else NAME[m]
        # regular weight: as in the hero figure, only progressive SCA's name is bold
        ax.annotate(nm, (x, y), xytext=(5, dy + 1.0), ha="left", va="bottom", color=COLOR[m], **lab)
        ax.annotate(rf"$\propto N^{{{expo[m]:.2f}}}$", (x, y), xytext=(5, dy - 1.5), ha="left", va="top",
                    color=COLOR[m], **lab)
    c = cells["full", 6144]
    ax.annotate(rf"{NAME['full']} $\propto N^{{{expo['full']:.2f}}}$", (6144, c["med"]), xytext=(-5, 1.5),
                ha="right", va="bottom", color=sf.MID, **lab)
    c = cells["sca_v2_prog", 4096]
    ax.annotate(NAME["sca_v2_prog"], (4096, c["med"]), xytext=(6.5, -0.5), ha="left", va="center", color=sf.SCA,
                fontweight="bold", **lab)
    # HRDiT sits between the single-pass SCA line and the AReA and full-attention lines, which leave no room beside
    # it; its name goes up and to the right into the gap between single-pass SCA and AReA, with a short leader
    c = cells["hrdit_dev_author", 4096]
    ax.annotate(NAME["hrdit_dev_author"], (4096, c["med"]), xytext=HRDIT_LABEL_AT, textcoords="data", ha="left",
                va="center", color=COLOR["hrdit_dev_author"], fontsize=fs, annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=COLOR["hrdit_dev_author"], lw=0.5, shrinkA=0.5, shrinkB=2.8,
                                relpos=(0.0, 0.3)))
    ax.set_ylabel(f"Seconds per image {pk.DOWN}")
    return ax


def build(cells, expo, ratio, layout):
    pk.use_style(script_pt=sf.FS_TICK)
    W = pk.WRAP
    left, right, top = 31.0, 47.0, 4.0  # points
    if layout == "A":
        H_pt = 158.0
        bottom = 25.0
        fig = plt.figure(figsize=(W, H_pt / 72))
        ax = fig.add_axes([left / (W * 72), bottom / H_pt, 1 - (left + right) / (W * 72), 1 - (bottom + top) / H_pt])
        draw_main(fig, ax, cells, expo)
        ax.set_xlabel("Image side (px)")
        # per-size ratio, AReA over single-pass SCA, as a row of numbers along the bottom of the panel
        for s in SIDES:
            big = s == SIDES[-1]
            ax.annotate(f"{ratio[s]:.1f}{pk.TIMES}", (s, YLIM[0]), xytext=(0, 2.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=sf.FS_TICK, color=sf.SCA if big else sf.MID,
                        fontweight="bold" if big else "normal")
        ax.annotate("vs AReA", (SIDES[-1], YLIM[0]), xytext=(11.0, 2.5), textcoords="offset points",
                    ha="left", va="bottom", fontsize=sf.FS_TICK, color=sf.SCA, annotation_clip=False)
        return fig
    H_pt = 178.0
    bottom, gap, rh = 25.0, 7.0, 34.0  # bottom margin, gap between panels, ratio panel height (points)
    fig = plt.figure(figsize=(W, H_pt / 72))
    wfrac = 1 - (left + right) / (W * 72)
    axr = fig.add_axes([left / (W * 72), bottom / H_pt, wfrac, rh / H_pt])
    ax = fig.add_axes([left / (W * 72), (bottom + rh + gap) / H_pt, wfrac, 1 - (bottom + rh + gap + top) / H_pt])
    draw_main(fig, ax, cells, expo)
    side_axis(ax, labels=False)
    ax.set_xlabel("Image side (px)")
    ax.xaxis.label.set_visible(False)  # the ratio panel below carries the shared axis title
    # ratio panel: AReA's time over single-pass SCA's at each size, the value on each bar
    side_axis(axr)
    lo_x = [s / 2 ** 0.11 for s in SIDES]
    hi_x = [s * 2 ** 0.11 for s in SIDES]
    for s, a, b in zip(SIDES, lo_x, hi_x):
        big = s == SIDES[-1]
        axr.fill_between([a, b], 0, ratio[s], color=sf.SCA if big else sf.SCA_LIGHT, lw=0, zorder=2)
        axr.annotate(f"{ratio[s]:.1f}{pk.TIMES}", (s, ratio[s]), xytext=(0, 1.0), textcoords="offset points",
                     ha="center", va="bottom", fontsize=sf.FS_TICK, color=sf.SCA if big else sf.INK,
                     fontweight="bold" if big else "normal")
    axr.set_ylim(0, 4.3)
    axr.yaxis.set_major_locator(FixedLocator([0, 1, 2, 3]))
    axr.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    pk.clean_axes(axr, grid="y")
    axr.set_ylabel(f"Speed-up {pk.UP}", labelpad=9.0)
    axr.set_xlabel("Image side (px)")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=("A", "B", "both"), default="both")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    cells, area3 = load()
    expo, ratio = derived(cells)
    checks(cells, area3, expo, ratio)
    for (m, s), c in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"  {s}^2 {m:17s} median {c['med']:8.2f} s  [{c['lo']:.2f}, {c['hi']:.2f}]  ({c['src']})")
    print("exponents", {m: round(v, 4) for m, v in expo.items()})
    print("AReA / single-pass SCA", {s: round(v, 3) for s, v in ratio.items()})
    layouts = ("A", "B") if args.layout == "both" else (args.layout,)
    written = {}
    for lay in layouts:
        fig = build(cells, expo, ratio, lay)
        if args.out_dir is not None:
            stem = args.out_dir / f"{STEM}_{lay}"
        else:
            stem = pk.V6 / STEM if lay == WINNER else pk.CANDIDATES / f"{STEM}_{lay}"
        print(f"min text size {pk.min_text_pt(fig):.1f} pt")
        written[lay] = pk.save(fig, stem, printed_width=pk.WRAP)
    if args.out_dir is None:
        prov = {"figure": "Figure 7, paper/figures/v6/scaling.pdf", "script": "paper/figures/src/v6_scaling.py",
                "winner": WINNER, "outputs": {k: [os.path.relpath(p, pk.ROOT) for p in v] for k, v in written.items()},
                "statistic": "median of three timed images after one excluded warm-up; whiskers minimum to maximum",
                "exponent_fit": "least squares of log median seconds on log N, N = (W/16)(H/16), all sizes of a method",
                "exponents": {m: round(v, 6) for m, v in expo.items()},
                "ratio_area_over_single_pass": {str(s): round(v, 6) for s, v in ratio.items()},
                "cells": [dict(method=m, side=s, **{k: (round(v, 4) if isinstance(v, float) else v)
                                                   for k, v in c.items()})
                          for (m, s), c in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0]))],
                "area_sweep3_4096": area3}
        path = pk.V6 / f"{STEM}_provenance.json"
        path.write_text(json.dumps(prov, indent=1) + "\n")
        print("wrote", os.path.relpath(path, pk.ROOT))


if __name__ == "__main__":
    main()
