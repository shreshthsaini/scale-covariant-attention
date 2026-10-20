"""ImageReward against campaign seconds per image on the 46-prompt panel at 4096^2 (fig:quality_time).

Every plotted value is read from the panel's summary CSVs (columns method, metric, value, res,
n_prompts), paths relative to the repository root:
  results/baseline46/base_summary.csv         seconds  full, dype, sigma_dev_matched, sega_official,
  results/baseline46/extra_extra_summary.csv  ir       area, sca_nb, ultraimage, hrdit_dev_author
  results/budget/budget_base_summary.csv      seconds  sca_s56 (56 steps), sca_g7 (guidance 7)
  results/budget/budget_extra_summary.csv     ir
  results/progressive46/base_summary.csv      seconds  sca_s42 (42 steps), sca_prog (progressive)
  results/progressive46/extra_summary.csv     ir
The paired ImageReward difference progressive minus HRDiT in the caption comes from
results/progressive46/sca_prog_vs_hrdit_dev_author_paired.csv. The script stops without plotting
unless every value rounds to the number printed in the paper (PRINTED: seconds to 0.1, ImageReward
to 0.01), every row covers the same 46 prompts at 4096^2, and the caption's claims still hold.

Encoding (house style, figures/REFERENCE_STYLE.md): AReA purple circle, SEGA grey circle; dark grey
diamonds, multi-stage baselines with their own recipes (UltraImage, HRDiT); filled green circle,
single-pass SCA at 28 steps; hollow green circles, its budget variants (42 and 56 steps, guidance 7),
named only in the caption; filled green star, progressive SCA. Methods whose ImageReward falls below
the plotted range (full attention, SigMa, DyPE) are named in the caption, not plotted. A dashed grey
guide joins HRDiT to progressive SCA and carries the one bold ratio of their campaign times. The
figure is 3.6 x 1.8 in (2:1) and is included at natural size.

Every run also writes STEM_wrap.pdf/.svg (default figures/quality_time_wrap.pdf), the main-text
wrapfigure at 0.42 text width: 166 x 112 pt at natural size, the same data, markers and direct
labels re-placed for the smaller canvas, tick labels 7 pt, direct labels 7 pt, axis titles 8 pt,
and thin light-grey horizontal gridlines behind the data (plot standard of 2026-10-19). The
2:1 figure above stays as it was, without gridlines.

Regenerate on a compute node: <sca env python> paper/figures/src/fig_quality_time.py [--out STEM]
(a trial --out writes the snippet and provenance beside STEM instead of into the paper tree).
"""

import argparse
import csv
import hashlib
import json
import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical PDF/SVG on rerun
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scafig as sf  # noqa: E402
from k1data import FIGDIR  # noqa: E402

REPO = os.path.normpath(os.path.join(FIGDIR, "..", ".."))
SNIPPET = os.path.join(FIGDIR, "snippets", "quality_time.tex")
RES, N_PROMPTS = "4096x4096", 46
SOURCES = {  # campaign -> (seconds CSV, ImageReward CSV, methods read from it)
    "baseline46": ("results/baseline46/base_summary.csv", "results/baseline46/extra_extra_summary.csv",
                   ["full", "dype", "sigma_dev_matched", "sega_official", "area", "sca_nb", "ultraimage",
                    "hrdit_dev_author"]),
    "budget": ("results/budget/budget_base_summary.csv", "results/budget/budget_extra_summary.csv",
               ["sca_s56", "sca_g7"]),
    "progressive46": ("results/progressive46/base_summary.csv", "results/progressive46/extra_summary.csv",
                      ["sca_s42", "sca_prog"]),
}
PAIRED = "results/progressive46/sca_prog_vs_hrdit_dev_author_paired.csv"
# The numbers printed in the paper (Table panel46 and the text): a check on the CSVs, never plotted.
PRINTED = {"full": (191.6, -2.21), "dype": (204.2, -0.23), "sigma_dev_matched": (185.5, -0.77),
           "sega_official": (189.7, 0.47), "area": (241.0, 0.83), "sca_nb": (71.6, 1.12),
           "ultraimage": (322.0, 1.26), "hrdit_dev_author": (106.5, 1.37), "sca_s56": (140.1, 1.22),
           "sca_g7": (71.9, 1.14), "sca_s42": (110.4, 1.19), "sca_prog": (49.0, 1.30)}
NAME = {"full": "full attention", "dype": "DyPE", "sigma_dev_matched": "SigMa", "sega_official": "SEGA",
        "area": "AReA", "sca_nb": "SCA, 28 steps", "ultraimage": "UltraImage", "hrdit_dev_author": "HRDiT",
        "sca_s56": "56 steps", "sca_g7": "guidance 7", "sca_s42": "42 steps", "sca_prog": "Progressive SCA"}
OFF_AXIS = {"full", "dype", "sigma_dev_matched"}  # below YLIM; the caption names them

W, H = 3.6, 1.8  # 2:1, included at natural size; the height must not grow
LEFT, RIGHT, BOTTOM, TOP = 0.40, 0.05, 0.36, 0.05  # axes margins (in)
XLIM, YLIM = (40.0, 380.0), (0.35, 1.45)
XTICKS, YTICKS = [50, 70, 100, 150, 200, 300], [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
GREY, DARK = sf.METHOD_COLOR["sega"], sf.METHOD_COLOR["ultraimage"]  # circles, darker diamonds
FS_LABEL = sf.FS_TICK  # direct labels (7 pt)

# marker style per class, and for each method its class and label placement:
# (dx, dy) offset in points from the marker, horizontal and vertical alignment
KIND = {
    "single": dict(marker="o", ms=4.6, mfc=GREY, mec="white", mew=0.5, zorder=3),
    "area": dict(marker="o", ms=4.6, mfc=sf.METHOD_COLOR["area"], mec="white", mew=0.5, zorder=3),
    "multi": dict(marker="D", ms=4.2, mfc=DARK, mec="white", mew=0.5, zorder=3),
    "sca": dict(marker="o", ms=5.0, mfc=sf.SCA, mec="white", mew=0.5, zorder=4),
    "budget": dict(marker="o", ms=4.6, mfc="none", mec=sf.SCA, mew=0.9, zorder=5),
    "prog": dict(marker="*", ms=10.5, mfc=sf.SCA, mec="white", mew=0.5, zorder=6),
}
PLOT = {
    "sega_official": ("single", (5, 0), "left", "center"),
    "area": ("area", (5, 0), "left", "center"),
    "ultraimage": ("multi", (-5.5, 0), "right", "center"),
    "hrdit_dev_author": ("multi", (5, 0), "left", "center"),
    "sca_nb": ("sca", (4.5, -3.5), "left", "top"),
    "sca_g7": ("budget", None, "left", "bottom"),     # budget variants: markers only, the
    "sca_s42": ("budget", None, "center", "top"),     # caption names them (house rule 9)
    "sca_s56": ("budget", None, "left", "center"),
    "sca_prog": ("prog", (-9, 6.5), "left", "bottom"),
}

# Main-text wrap version: 166 x 112 pt (0.42 of the 396 pt text width), included at natural size.
WRAP_W, WRAP_H = 166 / 72, 112 / 72
WRAP_LEFT, WRAP_RIGHT, WRAP_BOTTOM, WRAP_TOP = 29 / 72, 3 / 72, 24 / 72, 2 / 72  # axes margins (in)
WRAP_YLIM = (0.35, 1.55)  # the band above 1.4 holds the progressive label clear of HRDiT
WRAP_FS_LABEL, WRAP_FS_RATIO = 7.0, 7.0  # direct labels; the bold ratio on the guide
WRAP_GUIDE_T = 0.55  # where the ratio sits along the guide (0 = progressive SCA, 1 = HRDiT)
WRAP_PLOT = {  # as PLOT, offsets re-placed for the smaller canvas
    "sega_official": ("single", (4.5, 0), "left", "center"),
    "area": ("area", (4.5, 0), "left", "center"),
    "ultraimage": ("multi", (-5, 0), "right", "center"),
    "hrdit_dev_author": ("multi", (4.5, 0), "left", "center"),
    "sca_nb": ("sca", (4, -3), "left", "top"),
    "sca_g7": ("budget", None, "left", "bottom"),
    "sca_s42": ("budget", None, "center", "top"),
    "sca_s56": ("budget", None, "left", "center"),
    "sca_prog": ("prog", (-12, 10), "left", "bottom"),
}


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def read_metric(path, metric, methods):
    rows = {}
    with open(os.path.join(REPO, path)) as handle:
        for row in csv.DictReader(handle):
            if row["metric"] == metric and row["method"] in methods:
                if row["method"] in rows:
                    raise ValueError(f"{path}: duplicate {metric} row for {row['method']}")
                if row["res"] != RES or int(row["n_prompts"]) != N_PROMPTS:
                    raise ValueError(f"{path}: {row['method']} {metric} is {row['res']} over "
                                     f"{row['n_prompts']} prompts, not {RES} over {N_PROMPTS}")
                rows[row["method"]] = float(row["value"])
    missing = set(methods) - set(rows)
    if missing:
        raise ValueError(f"{path}: no {metric} row for {sorted(missing)}")
    return rows


def load():
    values = {}
    for campaign, (sec_csv, ir_csv, methods) in SOURCES.items():
        sec, ir = read_metric(sec_csv, "seconds", methods), read_metric(ir_csv, "ir", methods)
        for m in methods:
            values[m] = dict(seconds=sec[m], ir=ir[m], campaign=campaign, seconds_csv=sec_csv, ir_csv=ir_csv)
    bad = [f"{m}: CSV {v['seconds']:.1f} s / {v['ir']:.2f}, printed {PRINTED[m][0]:.1f} s / {PRINTED[m][1]:.2f}"
           for m, v in values.items()
           if f"{v['seconds']:.1f}" != f"{PRINTED[m][0]:.1f}" or f"{v['ir']:.2f}" != f"{PRINTED[m][1]:.2f}"]
    if set(values) != set(PRINTED) or bad:
        sys.exit("values differ from the printed tables, not plotting:\n  " + "\n  ".join(bad or ["method sets differ"]))
    return values


def read_paired():
    with open(os.path.join(REPO, PAIRED)) as handle:
        rows = [r for r in csv.DictReader(handle) if r["metric"] == "ir"]
    if len(rows) != 1:
        raise ValueError(f"{PAIRED}: expected one ImageReward row, found {len(rows)}")
    r = rows[0]
    if (r["method"], r["reference"], r["res"], int(r["n_prompts"])) != ("sca_prog", "hrdit_dev_author", RES,
                                                                       N_PROMPTS):
        raise ValueError(f"{PAIRED}: unexpected row {r}")
    return dict(delta=float(r["delta"]), ci_low=float(r["ci_low"]), ci_high=float(r["ci_high"]), seed=r["seed"])


def signed(v):
    return f"{v:.2f}".replace("-", "\u2212")


def figure(values, speedup):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([LEFT / W, BOTTOM / H, (W - LEFT - RIGHT) / W, (H - BOTTOM - TOP) / H])
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in XTICKS])
    ax.yaxis.set_major_locator(FixedLocator(YTICKS))
    ax.set_yticklabels([f"{t:.1f}" for t in YTICKS])
    ax.set_xlabel("seconds per image (campaign mean)")
    ax.set_ylabel("ImageReward")

    # guide: HRDiT to progressive SCA, labelled with the ratio of campaign times
    h, p = values["hrdit_dev_author"], values["sca_prog"]
    ax.plot([h["seconds"], p["seconds"]], [h["ir"], p["ir"]], ls=(0, (2.5, 2)), lw=0.7, color=sf.MID, zorder=1)
    t = 0.68  # the ratio sits on the HRDiT side of the guide, clear of the progressive label
    xm = h["seconds"] ** t * p["seconds"] ** (1 - t)
    ym = t * h["ir"] + (1 - t) * p["ir"]
    # the panel's one bold speed ratio (house rule 8)
    ax.annotate(f"{speedup:.1f}\u00d7 faster", (xm, ym), xytext=(0, -3.5), textcoords="offset points",
                ha="center", va="top", fontsize=sf.FS, fontweight="bold", color=sf.SCA)

    for m, (kind, offset, ha, va) in PLOT.items():
        v = values[m]
        ax.plot([v["seconds"]], [v["ir"]], ls="none", **KIND[kind])
        if offset is None:
            continue
        green = kind in ("sca", "budget", "prog")
        color = sf.SCA if green else sf.METHOD_COLOR["area"] if kind == "area" else sf.MID
        ax.annotate(NAME[m], (v["seconds"], v["ir"]), xytext=offset, textcoords="offset points", ha=ha, va=va,
                    fontsize=FS_LABEL, color=color, fontweight="bold" if kind == "prog" else "normal")
    return fig


def figure_wrap(values, speedup):
    """The main-text wrapfigure: the same encoding as figure() on a 166 x 112 pt canvas, with
    light-grey horizontal gridlines at the ImageReward ticks."""
    fig = plt.figure(figsize=(WRAP_W, WRAP_H))
    ax = fig.add_axes([WRAP_LEFT / WRAP_W, WRAP_BOTTOM / WRAP_H, (WRAP_W - WRAP_LEFT - WRAP_RIGHT) / WRAP_W,
                       (WRAP_H - WRAP_BOTTOM - WRAP_TOP) / WRAP_H])
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*WRAP_YLIM)
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in XTICKS])
    ax.yaxis.set_major_locator(FixedLocator(YTICKS))
    ax.set_yticklabels([f"{t:.1f}" for t in YTICKS])
    ax.set_xlabel("seconds per image (campaign mean)")
    ax.set_ylabel("ImageReward")
    sf.value_grid(ax, "y")

    h, p = values["hrdit_dev_author"], values["sca_prog"]
    ax.plot([h["seconds"], p["seconds"]], [h["ir"], p["ir"]], ls=(0, (2.5, 2)), lw=0.7, color=sf.MID, zorder=1)
    t = WRAP_GUIDE_T
    xm = h["seconds"] ** t * p["seconds"] ** (1 - t)
    ym = t * h["ir"] + (1 - t) * p["ir"]
    ax.annotate(f"{speedup:.1f}\u00d7 faster", (xm, ym), xytext=(0, -3.5), textcoords="offset points",
                ha="center", va="top", fontsize=WRAP_FS_RATIO, fontweight="bold", color=sf.SCA)

    for m, (kind, offset, ha, va) in WRAP_PLOT.items():
        v = values[m]
        ax.plot([v["seconds"]], [v["ir"]], ls="none", **KIND[kind])
        if offset is None:
            continue
        green = kind in ("sca", "budget", "prog")
        color = sf.SCA if green else sf.METHOD_COLOR["area"] if kind == "area" else sf.MID
        ax.annotate(NAME[m], (v["seconds"], v["ir"]), xytext=offset, textcoords="offset points", ha=ha, va=va,
                    fontsize=WRAP_FS_LABEL, color=color, fontweight="bold" if kind == "prog" else "normal")
    return fig


def write_snippet(path, values, paired):
    """At most 50 words: a bold claim, then two sentences of conditions. The paired interval stays
    in the provenance and Table prog_deltas; it no longer fits the caption budget."""
    below = sorted(OFF_AXIS, key=lambda m: -values[m]["ir"])
    cap = (  # at most 50 words: a bold claim, then two sentences of measurement conditions
        r"\textbf{Progressive \sca{} reaches HRDiT's ImageReward in less than half its time; extra "
        r"single-pass steps gain less per second.}" "\n"
        rf"  {N_PROMPTS} prompts at $4096^2$, campaign means on shared nodes; diamonds: multi-stage recipes; "
        r"open circles: \sca{} at guidance 7, 42 and 56 steps." "\n"
        r"  Below the range: " + ", ".join(NAME[m] for m in below[:-1]) + f" and {NAME[below[-1]]} "
        r"(Table~\ref{tab:panel46})."
    )
    with open(path, "w") as fh:
        fh.write("% Generated by paper/figures/src/fig_quality_time.py from the 46-prompt panel summaries\n"
                 "% (results/baseline46, results/budget, results/progressive46) and " + PAIRED + ".\n")
        fh.write("\\begin{figure}[t]\n  \\centering\n  \\includegraphics{figures/quality_time.pdf}\n")
        fh.write("  \\caption{" + cap + "}\n  \\label{fig:quality_time}\n\\end{figure}\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    default_out = os.path.join(FIGDIR, "quality_time")
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()
    trial = os.path.abspath(args.out) != default_out
    snippet = args.out + "_snippet.tex" if trial else SNIPPET

    values = load()
    paired = read_paired()
    for m, v in sorted(values.items(), key=lambda kv: kv[1]["seconds"]):
        print(f"  {m:<18} {v['seconds']:7.1f} s  IR {v['ir']:+.2f}  ({v['campaign']})")
    # claims the caption makes, re-checked on every run
    h, p = values["hrdit_dev_author"], values["sca_prog"]
    speedup = h["seconds"] / p["seconds"]
    below = {m for m, v in values.items() if v["ir"] < YLIM[0]}
    if below != OFF_AXIS or any(not (YLIM[0] <= v["ir"] <= YLIM[1] and XLIM[0] <= v["seconds"] <= XLIM[1])
                                for m, v in values.items() if m not in OFF_AXIS):
        raise ValueError(f"plotted range no longer fits: below range {sorted(below)}")
    if not p["seconds"] < 0.5 * h["seconds"]:
        raise ValueError("caption lead no longer holds: progressive SCA is not under half of HRDiT's time")
    steps = [values[m] for m in ("sca_nb", "sca_s42", "sca_s56")]
    gain_steps = (steps[2]["ir"] - steps[0]["ir"]) / (steps[2]["seconds"] - steps[0]["seconds"])
    gain_prog = (p["ir"] - steps[0]["ir"]) / (p["seconds"] - steps[0]["seconds"])  # time falls, IR rises
    if not (steps[0]["ir"] < steps[1]["ir"] < steps[2]["ir"] < p["ir"] and gain_prog < 0 < gain_steps):
        raise ValueError("caption lead no longer holds: more single-pass steps do not trail progressive SCA")
    if not paired["ci_low"] < 0 < paired["ci_high"]:
        raise ValueError("the claim 'reaches HRDiT's ImageReward' assumes the paired interval includes zero")
    print(f"speedup HRDiT / progressive SCA: {speedup:.3f}x; progressive minus HRDiT IR {paired['delta']:+.4f} "
          f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]")

    sf.use_style()
    fig = figure(values, speedup)
    sf.save(fig, args.out)
    sf.save(figure_wrap(values, speedup), args.out + "_wrap")  # main-text wrapfigure, 166 x 112 pt
    print("wrote", write_snippet(snippet, values, paired))
    prov = os.path.join(os.path.dirname(args.out), "quality_time_provenance.json") if not trial \
        else args.out + "_provenance.json"
    with open(prov, "w") as handle:
        json.dump({
            "figure": "fig:quality_time", "script": "paper/figures/src/fig_quality_time.py",
            "panel": dict(res=RES, n_prompts=N_PROMPTS, seed=0,
                          seconds="campaign mean seconds per image on shared GH200 nodes",
                          ir="mean ImageReward over the panel"),
            "sources": {c: dict(seconds_csv=s, seconds_sha256=sha256(os.path.join(REPO, s)), ir_csv=i,
                                ir_sha256=sha256(os.path.join(REPO, i)), methods=ms)
                        for c, (s, i, ms) in SOURCES.items()},
            "values": values,
            "printed_check": {m: dict(seconds=PRINTED[m][0], ir=PRINTED[m][1], matches=True) for m in PRINTED},
            "plotted": sorted(set(values) - OFF_AXIS), "off_axis_named_in_caption": sorted(OFF_AXIS),
            "unlabelled_markers": [m for m, (_, off, *_r) in PLOT.items() if off is None],
            "figure_size_in": [W, H],
            "axes": dict(xlim=XLIM, xscale="log", ylim=YLIM, xticks=XTICKS, yticks=YTICKS),
            "encoding": {m: dict(kind=k, label=NAME[m]) for m, (k, *_rest) in PLOT.items()},
            "guide": dict(from_method="hrdit_dev_author", to_method="sca_prog", speedup=round(speedup, 4),
                          label=f"{speedup:.1f}x faster", note="a visual guide between two measured points, "
                                                                "not an interpolation"),
            "paired_ir_prog_minus_hrdit": dict(source=PAIRED, sha256=sha256(os.path.join(REPO, PAIRED)), **paired,
                                               interval="95% paired interval as recorded in the CSV"),
            "outputs": dict(pdf="paper/figures/quality_time.pdf", svg="paper/figures/quality_time.svg",
                            snippet="paper/figures/snippets/quality_time.tex"),
        }, handle, indent=2)
        handle.write("\n")
    print("wrote", prov)


if __name__ == "__main__":
    main()
