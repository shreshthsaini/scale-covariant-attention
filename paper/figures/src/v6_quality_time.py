"""Appendix quality-time figure (paper/figures/v6/quality_time.pdf, label fig:quality_time): ImageReward against
campaign seconds per image on the 195 Aesthetic-4K prompts at 4096^2.

The v6 restyle of fig_quality_time.py's wrap figure: the same values and the same claim (at 46% of HRDiT's time,
progressive SCA's ImageReward is indistinguishable from HRDiT's: difference -0.02 [-0.09, 0.03]). Every value comes
from fig_quality_time.load() and read_paired(), which read the 195-prompt summaries through data195.py and v2swap.py
(results/a4k195_main, results/a4k195_panel, results/prog195_sca_v2, results/a4k195_main_sca_v2) and the paired file
results/prog195_sca_v2/sca_v2_prog_vs_hrdit_dev_author.csv, and stop unless every value rounds to the number the
paper prints (seconds to 0.1, ImageReward to 0.01). This script adds the checks of the claim against headline.tex
(\\hlCampProgHrditPct, \\hlProgHrditIr, \\hlProgHrditIrLo, \\hlProgHrditIrHi) and that the methods below the plotted
range are exactly full attention, DyPE and SigMa (named in the caption).

Marks (the paper's): progressive SCA green star, single-pass SCA filled green circle, SCA without the schedule at 28,
42 and 56 steps and at guidance 7 open green circles (named in the caption, not labelled), AReA purple circle, HRDiT
dark grey diamond, UltraImage dark grey square, SEGA grey circle. A dashed guide joins HRDiT to progressive SCA and
carries the ratio of their campaign times (campaign over campaign). Light horizontal gridlines, direction on both
axis titles. Printed width: half the text width (2.75 in), included with width=0.5\\linewidth.

Run on a compute node: <sca env python> -I paper/figures/src/v6_quality_time.py [--out STEM]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

import fig_quality_time as fq  # noqa: E402  (load, read_paired: reads and checks only)
import scafig as sf  # noqa: E402
import v6_plotkit as pk  # noqa: E402

STEM = pk.V6 / "quality_time"
W, H = pk.HALF, 1.95
XLIM, YLIM = (40.0, 400.0), (0.35, 1.58)
XTICKS, YTICKS = (50, 70, 100, 150, 200, 300), (0.4, 0.6, 0.8, 1.0, 1.2, 1.4)
OFF_AXIS = {"full", "dype", "sigma_dev_matched"}
GREY, DARK = sf.METHOD_COLOR["sega"], sf.METHOD_COLOR["ultraimage"]
KIND = {
    "sega_official": dict(marker="o", ms=4.4, mfc=GREY, mec="white", mew=0.5, zorder=3),
    "area": dict(marker="o", ms=4.4, mfc=sf.AREA, mec="white", mew=0.5, zorder=3),
    "hrdit_dev_author": dict(marker="D", ms=4.2, mfc=DARK, mec="white", mew=0.5, zorder=3),
    "ultraimage": dict(marker="s", ms=4.0, mfc=DARK, mec="white", mew=0.5, zorder=3),
    "sca_v2": dict(marker="o", ms=4.8, mfc=sf.SCA, mec="white", mew=0.5, zorder=5),
    "sca_nb": dict(marker="o", ms=4.2, mfc="white", mec=sf.SCA, mew=0.9, zorder=4),
    "sca_g7": dict(marker="o", ms=4.2, mfc="white", mec=sf.SCA, mew=0.9, zorder=4),
    "sca_s42": dict(marker="o", ms=4.2, mfc="white", mec=sf.SCA, mew=0.9, zorder=4),
    "sca_s56": dict(marker="o", ms=4.2, mfc="white", mec=sf.SCA, mew=0.9, zorder=4),
    "sca_v2_prog": dict(marker="*", ms=10.0, mfc=sf.SCA, mec="white", mew=0.4, zorder=6),
}
# direct labels: (text, offset in points, ha, va, colour, bold); the open circles carry none (caption)
LABEL = {
    "sca_v2_prog": ("Progressive SCA", (-7.0, 15.5), "left", "bottom", sf.SCA, True),
    "sca_v2": ("Single-pass SCA", (2.5, -5.5), "left", "top", sf.SCA, False),
    "hrdit_dev_author": ("HRDiT", (5.0, 1.0), "left", "center", DARK, False),
    "ultraimage": ("UltraImage", (-5.0, 0.0), "right", "center", DARK, False),
    "area": ("AReA", (5.0, 0.0), "left", "center", sf.AREA, False),
    "sega_official": ("SEGA", (5.0, 0.0), "left", "center", sf.MID, False),
}
GUIDE_T = 0.5  # where the ratio sits along the guide (0 = progressive SCA, 1 = HRDiT), in log time


def checks(values, paired):
    bad = []
    h, p = values["hrdit_dev_author"], values["sca_v2_prog"]
    pk.check(f"{100 * p['seconds'] / h['seconds']:.0f}" == pk.macro("hlCampProgHrditPct"),
             f"\\hlCampProgHrditPct {pk.macro('hlCampProgHrditPct')} vs {100 * p['seconds'] / h['seconds']:.2f}", bad)
    for name, v, signed in (("hlProgHrditIr", paired["delta"], True), ("hlProgHrditIrLo", paired["ci_low"], False),
                            ("hlProgHrditIrHi", paired["ci_high"], False)):
        pk.check(pk.fmt(v, 2, signed) == pk.macro(name), f"\\{name} {pk.macro(name)} vs {v:+.4f}", bad)
    pk.check(paired["ci_low"] < 0 < paired["ci_high"], "claim: the paired ImageReward interval includes zero", bad)
    pk.check(p["seconds"] < 0.5 * h["seconds"], "claim: progressive SCA under half of HRDiT's campaign time", bad)
    below = {m for m, v in values.items() if v["ir"] < YLIM[0]}
    pk.check(below == OFF_AXIS, f"methods below the range {sorted(below)}, caption names {sorted(OFF_AXIS)}", bad)
    inside = all(XLIM[0] <= v["seconds"] <= XLIM[1] and YLIM[0] <= v["ir"] <= YLIM[1]
                 for m, v in values.items() if m not in OFF_AXIS)
    pk.check(inside, "a plotted method falls outside the axes", bad)
    pk.fail_if(bad, "v6_quality_time.py")


def build(values, speedup):
    pk.use_style()
    fig = plt.figure(figsize=(W, H))
    Wp, Hp = W * 72, H * 72
    left, right, bottom, top = 31.0, 4.0, 25.0, 3.0
    ax = fig.add_axes([left / Wp, bottom / Hp, 1 - (left + right) / Wp, 1 - (bottom + top) / Hp])
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_locator(FixedLocator(YTICKS))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    pk.clean_axes(ax, grid="y")
    ax.set_xlabel(f"Seconds per image, campaign mean {pk.DOWN}")
    ax.set_ylabel(f"ImageReward {pk.UP}")
    h, p = values["hrdit_dev_author"], values["sca_v2_prog"]
    ax.plot([h["seconds"], p["seconds"]], [h["ir"], p["ir"]], ls=(0, (2.5, 2.0)), lw=0.7, color=sf.MID, zorder=1)
    xm = h["seconds"] ** GUIDE_T * p["seconds"] ** (1 - GUIDE_T)
    ym = GUIDE_T * h["ir"] + (1 - GUIDE_T) * p["ir"]
    ax.annotate(f"{speedup:.1f}{pk.TIMES} faster", (xm, ym), xytext=(0, 2.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=sf.FS_TICK, fontweight="bold", color=sf.SCA)
    for m, kw in KIND.items():
        v = values[m]
        ax.plot([v["seconds"]], [v["ir"]], ls="none", **kw)
        if m in LABEL:
            text, off, ha, va, col, bold = LABEL[m]
            ax.annotate(text, (v["seconds"], v["ir"]), xytext=off, textcoords="offset points", ha=ha, va=va,
                        fontsize=sf.FS_TICK, color=col, fontweight="bold" if bold else "normal")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=STEM)
    args = ap.parse_args()
    values, _ = fq.load()
    paired = fq.read_paired()
    assert set(KIND) | OFF_AXIS == set(values), sorted(set(values) ^ (set(KIND) | OFF_AXIS))
    checks(values, paired)
    h, p = values["hrdit_dev_author"], values["sca_v2_prog"]
    speedup = h["seconds"] / p["seconds"]  # campaign over campaign
    print(f"HRDiT / progressive SCA campaign seconds: {speedup:.3f}; progressive minus HRDiT ImageReward "
          f"{paired['delta']:+.4f} [{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]")
    fig = build(values, speedup)
    print(f"min text size {pk.min_text_pt(fig):.1f} pt")
    pk.save(fig, args.out, printed_width=pk.HALF)


if __name__ == "__main__":
    main()
