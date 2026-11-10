"""Appendix latency figure (paper/figures/v6/latency.pdf, label fig:latency_full): seconds per image at every size.

The v6 restyle of fig_latency.py's full-width figure: the same clean batch-1 sweep and the same claims (single-pass
SCA is the fastest method at every size; its lead over AReA grows to 3.07x at 8192^2; fitted exponents of seconds
proportional to N^b). Data, checks and fits come from v6_scaling.py (load, derived, checks), which reads
results/latency_curve_v2/summary.csv (single-pass SCA), results/latency_curve_summary.csv and the audited
results/latency_curve.csv (AReA and full attention), and results/latency_curve_batch1.csv (full attention at 6144^2,
three timed rows after the warm-up row is dropped); see that docstring for every check. Progressive SCA and HRDiT,
timed at 4096^2 only, are in Figure 7 and Table 1 and are not drawn here.

(a) Median seconds per image of three timed images against image side, log-log; whiskers run from the fastest to the
    slowest of the three (with three images, the whisker ends and the marker are the three timings). Direct labels
    carry each method's fitted exponent b.
(b) Single-pass SCA's speed-up over AReA and over full attention at each size: the baseline's median time over
    single-pass SCA's, both clean. AReA's is printed at every size to two decimals (the text quotes 2.63 and 3.07),
    full attention's at its largest size (the text quotes 4.4). The dotted line marks parity.

Printed width: full text width (5.5 in), included with width=\\linewidth.
Run on a compute node: <sca env python> -I paper/figures/src/v6_latency.py [--out STEM]
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

import scafig as sf  # noqa: E402
import v6_plotkit as pk  # noqa: E402
import v6_scaling as vs  # noqa: E402

STEM = pk.V6 / "latency"
W, H = pk.TEXT, 1.95
LW = {"sca_v2": 1.6, "area": 1.3, "full": 1.1}
RATIO_OFFSET = {4096: (-2.5, 4.5)}  # points; keeps the 4096^2 value clear of the rising full-attention line


def build(cells, expo, ratio):
    pk.use_style(script_pt=sf.FS_TICK)
    fig = plt.figure(figsize=(W, H))
    Wp, Hp = W * 72, H * 72
    bottom, top = 26.0, 15.0
    ax = fig.add_axes([34 / Wp, bottom / Hp, 140 / Wp, 1 - (bottom + top) / Hp])    # labels run to about 244 pt
    axr = fig.add_axes([272 / Wp, bottom / Hp, 120 / Wp, 1 - (bottom + top) / Hp])

    # (a) seconds per image
    ax.set_yscale("log")
    ax.set_ylim(*vs.YLIM)
    ax.yaxis.set_major_locator(FixedLocator(vs.YTICKS))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    vs.side_axis(ax)
    pk.clean_axes(ax, grid="y")
    for m in vs.LINES:
        ss = sorted(s for (k, s) in cells if k == m)
        ax.vlines(ss, [cells[m, s]["lo"] for s in ss], [cells[m, s]["hi"] for s in ss], color=vs.COLOR[m], lw=0.8,
                  zorder=2)
        ax.plot(ss, [cells[m, s]["med"] for s in ss], color=vs.COLOR[m], lw=LW[m],
                ls=(0, (3.2, 1.6)) if m == "full" else "-", mec="white", mew=0.4, zorder=4 if m == "sca_v2" else 3,
                **vs.MARK[m])
    lab = dict(fontsize=sf.FS_TICK, textcoords="offset points", annotation_clip=False)
    for m in ("area", "sca_v2"):
        y = cells[m, 8192]["med"]
        ax.annotate(vs.NAME[m], (8192, y), xytext=(6, 1.5), ha="left", va="bottom", color=vs.COLOR[m],
                    fontweight="bold" if m == "sca_v2" else "normal", **lab)
        ax.annotate(rf"$\propto N^{{{expo[m]:.2f}}}$", (8192, y), xytext=(6, -1.0), ha="left", va="top",
                    color=vs.COLOR[m], **lab)
    ax.annotate(rf"{vs.NAME['full']} $\propto N^{{{expo['full']:.2f}}}$", (6144, cells["full", 6144]["med"]),
                xytext=(-6, 1.0), ha="right", va="bottom", color=sf.MID, **lab)
    ax.set_xlabel("Image side (px)")
    ax.set_ylabel(f"Seconds per image {pk.DOWN}")

    # (b) speed-up of single-pass SCA over each baseline, same sizes
    vs.side_axis(axr)
    axr.set_ylim(0.0, 5.6)
    axr.yaxis.set_major_locator(FixedLocator([0, 1, 2, 3, 4, 5]))
    axr.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    pk.clean_axes(axr, grid="y")
    axr.axhline(1.0, color=sf.MID, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    full_ratio = {s: cells["full", s]["med"] / cells["sca_v2", s]["med"] for s in vs.SIDES if ("full", s) in cells}
    for key, r, col, mk, ls in (("full", full_ratio, vs.COLOR["full"], "^", (0, (3.2, 1.6))),
                                ("area", ratio, sf.AREA, "o", "-")):
        ss = sorted(r)
        axr.plot(ss, [r[s] for s in ss], color=col, lw=LW[key], ls=ls, marker=mk, ms=vs.MARK[key]["ms"],
                 mec="white", mew=0.4, zorder=3)
    for s in vs.SIDES:  # AReA's ratio above each point (the claim); full attention's only where the text quotes it
        big = s == vs.SIDES[-1]
        axr.annotate(f"{ratio[s]:.2f}", (s, ratio[s]), xytext=RATIO_OFFSET.get(s, (0, 3.0)), ha="center", va="bottom",
                     color=sf.SCA if big else sf.AREA, fontweight="bold" if big else "normal", **lab)
    s_full = max(full_ratio)
    axr.annotate(f"{full_ratio[s_full]:.2f}", (s_full, full_ratio[s_full]), xytext=(-4.0, 0.0), ha="right",
                 va="center", color=sf.MID, **lab)
    handles = [plt.Line2D([], [], color=sf.AREA, lw=LW["area"], marker="o", ms=vs.MARK["area"]["ms"], mec="white",
                          mew=0.4),
               plt.Line2D([], [], color=vs.COLOR["full"], lw=LW["full"], ls=(0, (3.2, 1.6)), marker="^",
                          ms=vs.MARK["full"]["ms"], mec="white", mew=0.4)]
    axr.legend(handles, ["over AReA", "over full attention"], loc="upper left", frameon=False, fontsize=sf.FS_TICK,
               handlelength=1.8, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25)
    axr.set_xlabel("Image side (px)")
    axr.set_ylabel(f"Speed-up {pk.UP}")
    fig.canvas.draw()
    pk.panel_tag(fig, ax, "a", dy_pt=5.0, dx_pt=-30.0)
    pk.panel_tag(fig, axr, "b", dy_pt=5.0, dx_pt=-26.0)
    return fig, full_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=STEM)
    args = ap.parse_args()
    cells, area3 = vs.load()
    expo, ratio = vs.derived(cells)
    vs.checks(cells, area3, expo, ratio)
    fig, full_ratio = build(cells, expo, ratio)
    print("speed-up over AReA", {s: round(v, 3) for s, v in ratio.items()})
    print("speed-up over full attention", {s: round(v, 3) for s, v in full_ratio.items()})
    print("exponents", {m: round(v, 4) for m, v in expo.items()})
    print(f"min text size {pk.min_text_pt(fig):.1f} pt")
    pk.save(fig, args.out, printed_width=pk.TEXT)


if __name__ == "__main__":
    main()
