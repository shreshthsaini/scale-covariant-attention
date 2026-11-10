"""Appendix baseline figure (paper/figures/v6/baseline_deltas.pdf, label fig:baseline_quality): single-pass SCA minus
each baseline on four per-image metrics at 4096^2, with 95% paired intervals.

The v6 restyle of fig_baseline_deltas.py: the same statistics and the same claim (single-pass SCA is significantly
better than every shared-sampler baseline, full attention, AReA, DyPE and SEGA, on all four metrics; against
UltraImage, which runs its own two-stage recipe, the result is mixed). Every value comes from
fig_baseline_deltas.paired_statistics() and its checks: single-pass SCA (sca_v2 of the a4k195_main_sca_v2 run) minus
each baseline of results/a4k195_main on the same 195 prompts, seed 0, through v2swap.derived_pairs
(scripts/paired_bootstrap.py on the two runs' per-image evaluation files, 2000 prompt draws, seed 0, cached under
/scratch/09032/saini_2/sca/eval/app_tables_pairs/); each paired mean must equal the difference of the two summary
means. fig_baseline_deltas.claims() re-checks the caption's claim. Nothing is written under results/ (the old script's
results CSV is written only by its own main(), which this script never calls).

Encoding as in Figure 6, without a multiplicity correction (the appendix tables report these intervals unadjusted):
filled SCA green, the interval excludes zero in SCA's favour; filled LOSS orange, against SCA; open grey, it includes
zero. Single-pass SCA's mark is the circle. One panel per metric, each on its own scale in the metric's units
(HPSv2.1 x 100), zero line, light vertical gridlines, the direction of "better" on the shared axis title.

Also writes the caption snippet paper/figures/snippets/v6_baseline_deltas.tex (the v5.1 snippet baseline_deltas.tex
stays untouched). Printed width: full text width (5.5 in).
Run on a compute node: <sca env python> -I paper/figures/src/v6_baseline_deltas.py [--out STEM] [--no-snippet]
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
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402

import fig_baseline_deltas as fb  # noqa: E402  (paired_statistics, claims: reads and checks only)
import scafig as sf  # noqa: E402
import v2swap  # noqa: E402
import v6_plotkit as pk  # noqa: E402

STEM = pk.V6 / "baseline_deltas"
SNIPPET = pk.SNIPPETS / "v6_baseline_deltas.tex"
W = pk.TEXT
ROWS = ("full", "area", "dype", "sega_official", "ultraimage")
ROW_NAME = {"full": "Full attention", "area": "AReA", "dype": "DyPE", "sega_official": "SEGA",
            "ultraimage": "UltraImage\u2020"}
PANEL = {"ir": "ImageReward", "pickscore": "PickScore", "hpsv21": f"HPSv2.1 {pk.TIMES}100",
         "musiq_crop1024": r"MUSIQ$_{\mathregular{c}}$"}
STEP = {"ir": 1.0, "pickscore": 2.0, "hpsv21": 10.0, "musiq_crop1024": 10.0}  # tick spacing in metric units
GAP = 0.6  # extra row space before UltraImage (own recipe)


def colour(r):
    return pk.GAIN if r["ci_low"] > 0 else (pk.LOSS if r["ci_high"] < 0 else pk.NS)


def build(records):
    pk.use_style(script_pt=sf.FS)
    pos, y = {}, 0.0
    for k in ROWS:
        if k == "ultraimage":
            y += GAP
        pos[k] = y
        y += 1.0
    ymax = y - 1.0
    pitch, left, gap, right, top, bottom = 12.0, 62.0, 13.0, 6.0, 15.0, 26.0  # points
    H = (top + (ymax + 1.0) * pitch + bottom) / 72.0
    Wp, Hp = W * 72, H * 72
    pw = (Wp - left - right - 3 * gap) / 4
    fig = plt.figure(figsize=(W, H))
    axes = []
    for i, metric in enumerate(PANEL):
        ax = fig.add_axes([(left + i * (pw + gap)) / Wp, bottom / Hp, pw / Wp, (ymax + 1.0) * pitch / Hp])
        rows = {r["baseline"]: r for r in records if r["metric"] == metric}
        assert set(rows) == set(ROWS), (metric, sorted(rows))
        lo = min(0.0, min(r["ci_low"] for r in rows.values()))
        hi = max(0.0, max(r["ci_high"] for r in rows.values()))
        span = hi - lo
        ax.set_xlim(lo - 0.08 * span, hi + 0.06 * span)
        ax.set_ylim(ymax + 0.6, -0.6)
        ax.axvline(0, color=sf.INK, lw=0.7, zorder=1)
        for k, r in rows.items():
            c = colour(r)
            ax.plot([r["ci_low"], r["ci_high"]], [pos[k]] * 2, color=c, lw=1.0, solid_capstyle="butt", zorder=2)
            ax.plot([r["delta"]], [pos[k]], ls="none", marker="o", ms=3.9, mec=c, mew=0.9,
                    mfc="white" if c == pk.NS else c, zorder=3)
        step = STEP[metric]
        ticks = np.arange(np.ceil((lo - 0.08 * span) / step) * step, hi + 0.06 * span + 1e-9, step)
        ax.xaxis.set_major_locator(FixedLocator([float(t) for t in ticks]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:g}".replace("-", pk.MINUS)))
        pk.clean_axes(ax, grid="x", left=False)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_yticks([pos[k] for k in ROWS])
        ax.set_yticklabels([ROW_NAME[k] for k in ROWS] if i == 0 else [])
        ax.set_ylabel("baseline")
        ax.yaxis.label.set_visible(False)
        ax.set_xlabel(PANEL[metric])
        ax.xaxis.label.set_visible(False)
        axes.append(ax)
    title = f"Single-pass SCA {pk.MINUS} baseline, better {pk.ARROW}"
    x0, x1 = axes[0].get_position().x0, axes[-1].get_position().x1
    fig.text((x0 + x1) / 2, 3.0 / Hp, title, ha="center", va="bottom", fontsize=sf.FS, color=sf.INK)
    fig.canvas.draw()
    for ax, letter, metric in zip(axes, "abcd", PANEL):
        pk.panel_tag(fig, ax, letter, PANEL[metric], dy_pt=4.0)
    return fig


def write_snippet(path, claims):
    gains = claims["ultraimage_gains"]
    losses = claims["ultraimage_losses"]
    name = {"ir": "ImageReward", "pickscore": "PickScore", "hpsv21": "HPSv2.1", "musiq_crop1024": r"MUSIQ$_\text{c}$"}

    def words(keys):
        items = [name[k] for k in keys]
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    cap = (r"\textbf{Single-pass \sca{} beats every shared-sampler baseline on all four metrics; against UltraImage "
           r"the result is mixed.}" "\n"
           rf"  Single-pass \sca{{}} minus each baseline on the same {fb.N_PROMPTS} prompts at $4096^2$, seed 0; "
           rf"whiskers are 95\% paired intervals over prompts ({fb.DRAWS} bootstrap draws), not adjusted for "
           r"multiplicity. Filled: the interval excludes zero (green: \sca{} better, orange: worse); open grey: it "
           r"includes zero." "\n"
           rf"  Against UltraImage, \sca{{}} is better on {words(gains)} and worse on {words(losses)}; "
           r"$^\dagger$UltraImage runs its own two-stage recipe, so this is not an equal-cost comparison. "
           r"Table~\ref{tab:panel46_deltas} gives all metrics.")
    path.write_text("% Generated by paper/figures/src/v6_baseline_deltas.py from results/a4k195_main_sca_v2 and "
                    "results/a4k195_main\n% (paired per-image differences through v2swap.derived_pairs, checked "
                    "against the summaries); do not edit by hand.\n"
                    "\\begin{figure}[!htbp]\n  \\centering\n"
                    "  \\includegraphics[width=\\linewidth]{figures/v6/baseline_deltas.pdf}\n"
                    "  \\caption{" + cap + "}\n  \\label{fig:baseline_quality}\n\\end{figure}\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=STEM)
    ap.add_argument("--no-snippet", action="store_true")
    args = ap.parse_args()
    landed = v2swap.landed(fb.V2_RUN)
    if not landed:
        raise SystemExit(f"{fb.V2_RUN} has not landed; the v6 figure plots single-pass SCA (sca_v2) only")
    records, _ = fb.paired_statistics(landed)
    claims = fb.claims(records)  # raises unless the caption's claim holds
    # the AReA row is the sca_area pair of Figure 6: the same difference on the same prompts
    area_ir = next(r for r in records if r["baseline"] == "area" and r["metric"] == "ir")
    if pk.fmt(area_ir["delta"], 2, signed=True) != pk.macro("hlIrDelta"):
        raise SystemExit(f"AReA ImageReward difference {area_ir['delta']:+.4f} vs \\hlIrDelta {pk.macro('hlIrDelta')}")
    for r in records:
        print(f"  {r['baseline']:14s} {r['metric']:15s} {r['delta']:+9.4f} [{r['ci_low']:+9.4f}, {r['ci_high']:+9.4f}]")
    print("claims", claims)
    fig = build(records)
    print(f"min text size {pk.min_text_pt(fig):.1f} pt")
    pk.save(fig, args.out, printed_width=pk.TEXT)
    if not args.no_snippet and args.out == STEM:
        print("wrote", os.path.relpath(write_snippet(SNIPPET, claims), pk.ROOT))


if __name__ == "__main__":
    main()
