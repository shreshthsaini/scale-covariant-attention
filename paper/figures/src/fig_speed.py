"""Speed headline (fig:speed): ImageReward against seconds per image at 4096^2, and seconds against resolution.

(a) Every 4096^2 method of Table 1 at seed 0 over the 195 Aesthetic-4K prompts: ImageReward (y) against seconds per
    image (x, log). Filled markers sit at the clean batch-1 median on an idle GH200; hollow markers at the campaign mean
    on shared nodes, for the three methods that have no clean time (UltraImage, DyPE, SEGA). Bold green arrows run from
    AReA, HRDiT and UltraImage to progressive SCA and carry progressive SCA's speedup over each: clean over clean
    (sweep 3, the back-to-back timing of AReA, HRDiT and progressive SCA) or campaign over campaign (UltraImage),
    never mixed. Full attention, DyPE and SEGA carry the same ratio as a grey number beside their name. The y axis is
    broken: the lower strip holds DyPE and full attention, whose ImageReward falls far below the rest.
(b) Clean batch-1 seconds per image against image side for single-pass SCA (latency_curve_v2), AReA and full attention
    (latency_curve_summary, plus full attention at 6144^2 from latency_curve_batch1), log-log; progressive SCA and
    HRDiT at 4096^2 from sweep 3. Green numbers above AReA: single-pass SCA's speedup over AReA at that size.

Inputs (paths relative to the repository root; every value is read here, none typed):
  results/a4k195_main/{base,extra}_summary.csv        campaign seconds and ImageReward: full, dype, sega_official,
                                                       area, ultraimage (data195.load)
  results/a4k195_main_sca_v2/{base,extra}_summary.csv  sca_v2 (single-pass SCA; v2swap.load)
  results/prog195_sca_v2/{base,extra}_summary.csv     sca_v2_prog (progressive SCA), hrdit_dev_author (v2swap.load)
  results/prog_latency/latency_summary.csv            clean sweep 3 at 4096^2: area, hrdit_dev_author, sca_v2_prog
  results/latency_curve_v2/summary.csv                clean sweep 2: sca_v2 at 2048^2 to 8192^2
  results/latency_curve_summary.csv                   clean sweep 1: area and full at every size but full at 6144^2
  results/latency_curve_batch1.csv                    clean sweep 1 per image: full attention at 6144^2 (median of
                                                       the three timed images; warm-up rows dropped)
The script stops unless every value rounds to the number the paper prints (Table 1, Section 4.2 and Appendix E) and
the two headline ratios equal headline.tex's \\hlSpeedProgArea and \\hlSpeedProgHrdit.

Outputs:
  paper/figures/speed.pdf, .svg                  the paper figure, 5.5 in wide (TEXT), included at natural size
  paper/figures/snippets/speed_numbers.tex      the ratios the text quotes (\\hlSpeedProgUltraCamp and others)
  paper/figures/speed_provenance.json           inputs with sha256, every plotted value, outputs with sha256
  web/assets/img/figs/speed_{light,dark}.{svg,png,webp}           the same figure for the project page
  web/assets/img/figs/speed_time_{light,dark}.{svg,png,webp}      panel (a) alone, 4.6 x 2.9 in
  web/assets/img/figs/speed_latency_{light,dark}.{svg,png,webp}   panel (b) alone, 4.6 x 2.9 in
Web files have transparent backgrounds; the dark variant uses the page's dark palette (web/assets/css).

Regenerate on a compute node: <sca env python> paper/figures/src/fig_speed.py
"""
from __future__ import annotations

import csv
from decimal import ROUND_HALF_UP, Decimal
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical PDF and SVG on rerun
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data195  # noqa: E402
import scafig as sf  # noqa: E402
import v2swap  # noqa: E402

ROOT = HERE.parents[2]
FIGS = ROOT / "paper" / "figures"
WEB = ROOT / "web" / "assets" / "img" / "figs"
HEADLINE = FIGS / "snippets" / "main_tables" / "headline.tex"
RES = "4096x4096"

# Numbers printed in the paper (Table 1 clean and campaign seconds, ImageReward; Appendix E latency); a check only.
PRINTED_CLEAN = {"full": 186.5, "area": 202.4, "sca_v2": 77.1, "hrdit_dev_author": 107.0, "sca_v2_prog": 49.2}
PRINTED_CAMP = {"full": 196.2, "dype": 195.9, "sega_official": 190.4, "area": 264.4, "sca_v2": 78.4,
                "ultraimage": 308.5, "hrdit_dev_author": 110.4, "sca_v2_prog": 50.7}
PRINTED_IR = {"full": -2.18, "dype": -0.10, "sega_official": 0.56, "area": 0.92, "sca_v2": 1.25, "ultraimage": 1.28,
              "hrdit_dev_author": 1.34, "sca_v2_prog": 1.32}
PRINTED_CURVE = {("sca_v2", 2048): 19.8, ("sca_v2", 3072): 44.3, ("sca_v2", 4096): 77.1, ("sca_v2", 6144): 174.7,
                 ("sca_v2", 8192): 309.7, ("area", 2048): 41.0, ("area", 3072): 114.3, ("area", 4096): 202.4,
                 ("area", 6144): 482.3, ("area", 8192): 952.0, ("full", 2048): 22.9, ("full", 3072): 73.9,
                 ("full", 4096): 186.5, ("full", 6144): 762.7}
SIDES = (2048, 3072, 4096, 6144, 8192)
NAME = {"full": "Full attention", "dype": "DyPE", "sega_official": "SEGA", "area": "AReA", "sca_v2": "Single-pass SCA",
        "ultraimage": "UltraImage", "hrdit_dev_author": "HRDiT", "sca_v2_prog": "Progressive SCA"}
CAMPAIGN_ONLY = ("ultraimage", "dype", "sega_official")

# Paper palette (scafig) and the project page's dark palette (web/assets/css: --accent, --area, --ink, --muted, ...).
THEMES = {
    "light": dict(ink=sf.INK, mid=sf.MID, grid=sf.GRIDLINE, sca=sf.SCA, sca_fill=sf.SCA_LIGHT, area=sf.AREA,
                  strong="#56626F", weak="#828E9A", full="#9AA3AC", face="white", mask="white"),
    "dark": dict(ink="#E7EAE8", mid="#A3ACB3", grid="#232927", sca="#3DBB8C", sca_fill="#10271D", area="#B794E0",
                 strong="#C3C9CF", weak="#8A939C", full="#6F7880", face="#0D100F", mask="#0D100F"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path: Path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def r1(x: float) -> float:
    """Half-up rounding to 0.1 as the paper prints (gen_main_tables.py: 49.15 s, raw 49.153 s, prints 49.2)."""
    return float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def r2(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def headline_macro(name: str) -> str:
    m = re.search(r"\\newcommand\{\\%s\}\{([^}]*)\}" % name, HEADLINE.read_text())
    if m is None:
        raise SystemExit(f"{HEADLINE}: no \\{name}")
    return m.group(1)


def load() -> dict:
    """Every plotted number, read from results/ and checked against the paper."""
    main = data195.load("a4k195_main")[0]
    v2 = v2swap.load("a4k195_main_sca_v2")[0]
    prog = v2swap.load("prog195_sca_v2")[0]
    src = {m: main for m in ("full", "dype", "sega_official", "area", "ultraimage")}
    src.update(sca_v2=v2, sca_v2_prog=prog, hrdit_dev_author=prog)
    ir = {m: float(S[(m, "ir")]) for m, S in src.items()}
    camp = {m: float(S[(m, "seconds")]) for m, S in src.items()}

    sweep3 = {r["method"]: float(r["median_s"]) for r in rows(ROOT / "results/prog_latency/latency_summary.csv")
              if r["res"] == RES}
    sweep2 = {(r["method"], int(r["res"].split("x")[0])): float(r["median_seconds"])
              for r in rows(ROOT / "results/latency_curve_v2/summary.csv") if r["res"].split("x")[0] == r["res"].split("x")[1]}
    sweep1 = {(r["method"], int(r["res"].split("x")[0])): float(r["median_seconds"])
              for r in rows(ROOT / "results/latency_curve_summary.csv")}
    full6k = [float(r["seconds"]) for r in rows(ROOT / "results/latency_curve_batch1.csv")
              if r["res"] == "6144x6144" and r["method"] == "full" and not r["image"].startswith("sca_latency_warmup")]
    if len(full6k) != 3:
        raise SystemExit(f"full attention at 6144^2: expected 3 timed images, found {len(full6k)}")
    sweep1[("full", 6144)] = statistics.median(full6k)

    clean = {"area": sweep3["area"], "hrdit_dev_author": sweep3["hrdit_dev_author"],
             "sca_v2_prog": sweep3["sca_v2_prog"], "sca_v2": sweep2[("sca_v2", 4096)], "full": sweep1[("full", 4096)]}
    curve = {("sca_v2", s): sweep2[("sca_v2", s)] for s in SIDES}
    curve.update({(m, s): sweep1[(m, s)] for m in ("area", "full") for s in SIDES if (m, s) in sweep1})

    bad = [f"clean {m} {r1(v)} != {PRINTED_CLEAN[m]}" for m, v in clean.items() if r1(v) != PRINTED_CLEAN[m]]
    bad += [f"campaign {m} {r1(v)} != {PRINTED_CAMP[m]}" for m, v in camp.items() if r1(v) != PRINTED_CAMP[m]]
    bad += [f"IR {m} {r2(v)} != {PRINTED_IR[m]}" for m, v in ir.items() if r2(v) != PRINTED_IR[m]]
    bad += [f"curve {k} {r1(v)} != {PRINTED_CURVE[k]}" for k, v in curve.items() if r1(v) != PRINTED_CURVE[k]]
    if set(curve) != set(PRINTED_CURVE):
        bad.append(f"curve cells {sorted(set(curve) ^ set(PRINTED_CURVE))}")
    # Speedups of progressive SCA: clean over clean where both are clean, campaign over campaign otherwise.
    speed = {}
    for m in ("area", "hrdit_dev_author", "full"):
        speed[m] = (clean[m] / clean["sca_v2_prog"], "clean")
    for m in CAMPAIGN_ONLY:
        speed[m] = (camp[m] / camp["sca_v2_prog"], "campaign")
    if f"{speed['area'][0]:.1f}" != headline_macro("hlSpeedProgArea"):
        bad.append(f"AReA ratio {speed['area'][0]:.3f} vs \\hlSpeedProgArea {headline_macro('hlSpeedProgArea')}")
    if f"{speed['hrdit_dev_author'][0]:.1f}" != headline_macro("hlSpeedProgHrdit"):
        bad.append(f"HRDiT ratio {speed['hrdit_dev_author'][0]:.3f} vs \\hlSpeedProgHrdit")
    curve_ratio = {s: curve[("area", s)] / curve[("sca_v2", s)] for s in SIDES}
    if f"{curve_ratio[4096]:.2f}" != headline_macro("hlSpeedArea") or f"{curve_ratio[8192]:.2f}" != "3.07":
        bad.append(f"single-pass ratios {curve_ratio[4096]:.3f}, {curve_ratio[8192]:.3f} vs the text's 2.63 and 3.07")
    if bad:
        raise SystemExit("fig_speed.py: values differ from the paper:\n  " + "\n  ".join(bad))
    return dict(ir=ir, camp=camp, clean=clean, curve=curve, speed=speed, curve_ratio=curve_ratio)


def times(d) -> dict:
    """x position of each method in panel (a): clean where it exists, campaign otherwise."""
    return {m: (d["camp"][m] if m in CAMPAIGN_ONLY else d["clean"][m]) for m in NAME}


def arrow(ax, p0, p1, color, lw, rad=0.0, shrink=(4.0, 7.0), z=4):
    a = FancyArrowPatch(p0, p1, transform=ax.transData, arrowstyle="-|>,head_length=3.6,head_width=2.1",
                        connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, shrinkA=shrink[0], shrinkB=shrink[1],
                        zorder=z, capstyle="round", joinstyle="round")
    ax.add_patch(a)
    return a


def style_axes(ax, th, fsl):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["ink"])
    ax.tick_params(colors=th["ink"], labelsize=fsl, length=2.5, width=0.6)
    ax.grid(visible=True, axis="y", which="major", color=th["grid"], linewidth=sf.GRIDLINE_LW)
    ax.set_axisbelow(True)


def draw_time(fig, box, d, th, fs_scale=1.0):
    """Panel (a) in the figure-fraction rectangle box = (x0, y0, w, h): the upper axis holds every method from AReA up;
    a lower strip on its own scale holds SEGA, DyPE and full attention, whose ImageReward is far lower."""
    x0, y0, w, h = box
    gap, lo_h = 0.045 * h, 0.235 * h
    ax_lo = fig.add_axes([x0, y0, w, lo_h])
    ax_hi = fig.add_axes([x0, y0 + lo_h + gap, w, h - lo_h - gap], sharex=ax_lo)
    t, ir, sp = times(d), d["ir"], d["speed"]
    fs, fsl, fss = sf.FS * fs_scale, sf.FS_TICK * fs_scale, sf.FS_SMALL * fs_scale
    ms = 4.6 * fs_scale

    def off(dx, dy):  # label offsets in points, scaled with the text
        return (dx * fs_scale, dy * fs_scale)
    for ax in (ax_hi, ax_lo):
        ax.set_xscale("log")
        ax.set_xlim(26, 480)
        ax.xaxis.set_major_locator(FixedLocator([50, 100, 200, 400]))
        ax.xaxis.set_minor_locator(NullLocator())
        style_axes(ax, th, fsl)
    ax_hi.set_ylim(0.80, 1.585)
    ax_hi.yaxis.set_major_locator(FixedLocator([0.9, 1.1, 1.3, 1.5]))
    ax_hi.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax_lo.set_ylim(-2.6, 1.05)
    ax_lo.yaxis.set_major_locator(FixedLocator([-2.0, 0.0]))
    ax_lo.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: "−2" if v < 0 else "0"))
    ax_hi.spines["bottom"].set_visible(False)
    ax_hi.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_lo.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    # Break marks where the scale changes (matplotlib's broken-axis idiom: one slanted tick on each side of the gap).
    brk = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=5.5, linestyle="none", color=th["ink"], mec=th["ink"],
               mew=0.6, clip_on=False, zorder=10)
    ax_hi.plot([0], [0], transform=ax_hi.transAxes, **brk)
    ax_lo.plot([0], [1], transform=ax_lo.transAxes, **brk)

    color = {"sca_v2_prog": th["sca"], "sca_v2": th["sca"], "area": th["area"], "hrdit_dev_author": th["strong"],
             "ultraimage": th["strong"], "dype": th["weak"], "sega_official": th["weak"], "full": th["full"]}
    marker = {"sca_v2_prog": "*", "sca_v2": "o", "area": "o", "hrdit_dev_author": "D", "ultraimage": "s",
              "dype": "o", "sega_official": "o", "full": "o"}
    size = {"sca_v2_prog": ms * 2.4, "hrdit_dev_author": ms * 0.85, "ultraimage": ms * 0.92}
    low = ("sega_official", "dype", "full")
    for m in NAME:
        ax = ax_lo if m in low else ax_hi
        hollow = m in CAMPAIGN_ONLY
        ax.plot([t[m]], [ir[m]], marker=marker[m], ms=size.get(m, ms), color=color[m], ls="none",
                zorder=8 if m == "sca_v2_prog" else 6, mfc=th["mask"] if hollow else color[m], mec=color[m],
                mew=1.0 if hollow else (0.6 if m != "sca_v2_prog" else 0.0))

    # Bold arrows from AReA, HRDiT and UltraImage to progressive SCA.
    P = (t["sca_v2_prog"], ir["sca_v2_prog"])
    lw = 1.6 * fs_scale
    arrow(ax_hi, (t["hrdit_dev_author"], ir["hrdit_dev_author"]), P, th["sca"], lw, rad=0.0)
    arrow(ax_hi, (t["ultraimage"], ir["ultraimage"]), P, th["sca"], lw, rad=0.19)
    arrow(ax_hi, (t["area"], ir["area"]), P, th["sca"], lw, rad=0.0)

    # Direct labels: each method's name, then progressive SCA's speedup over it (bold green where an arrow is drawn,
    # grey otherwise). The arrows themselves carry no text, so no label sits on a line.
    nk = dict(fontsize=fss, zorder=9)
    rk = dict(fontsize=fss + 0.5, fontweight="bold", color=th["sca"], zorder=9)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    def name_ratio(ax, m, dx, dy, name_kw, ratio_kw, ratio_text, ha="left"):
        """Name at (dx, dy) points from the marker, ratio right after it; ha='right' puts the pair left of it."""
        xy = (t[m], ir[m])
        if ha == "left":
            tn = ax.annotate(NAME[m], xy, xytext=off(dx, dy), textcoords="offset points", ha="left", va="center",
                             **name_kw)
            wpt = tn.get_window_extent(rend).width * 72 / fig.dpi
            ax.annotate(ratio_text, xy, xytext=off(dx, dy) if False else (off(dx, dy)[0] + wpt + 2.5 * fs_scale,
                                                                          off(dx, dy)[1]),
                        textcoords="offset points", ha="left", va="center", **ratio_kw)
        else:
            tr = ax.annotate(ratio_text, xy, xytext=off(dx, dy), textcoords="offset points", ha="right", va="center",
                             **ratio_kw)
            wpt = tr.get_window_extent(rend).width * 72 / fig.dpi
            ax.annotate(NAME[m], xy, xytext=(off(dx, dy)[0] - wpt - 2.5 * fs_scale, off(dx, dy)[1]),
                        textcoords="offset points", ha="right", va="center", **name_kw)

    # Progressive SCA: to the left of the star.
    ax_hi.annotate("Progressive\nSCA", P, xytext=off(-7.5, 0), textcoords="offset points", ha="right", va="center",
                   color=th["sca"], fontweight="bold", linespacing=1.0, **nk)
    ax_hi.annotate("Single-pass SCA", (t["sca_v2"], ir["sca_v2"]), xytext=off(4.5, -0.5), textcoords="offset points",
                   ha="left", va="center", color=th["sca"], **nk)
    name_ratio(ax_hi, "hrdit_dev_author", 4.5, 0, dict(color=th["ink"], **nk), rk,
               f"{sp['hrdit_dev_author'][0]:.1f}\u00d7")
    name_ratio(ax_hi, "area", 4.5, 0, dict(color=th["area"], fontweight="bold", **nk), rk, f"{sp['area'][0]:.1f}\u00d7")
    # UltraImage: name below the marker, its campaign-time ratio under the name.
    ax_hi.annotate("UltraImage", (t["ultraimage"], ir["ultraimage"]), xytext=off(0, -5.0), textcoords="offset points",
                   ha="center", va="top", color=th["ink"], **nk)
    ax_hi.annotate(f"{sp['ultraimage'][0]:.1f}\u00d7", (t["ultraimage"], ir["ultraimage"]), xytext=off(-1.5, -13.5),
                   textcoords="offset points", ha="right", va="top", **rk)
    ax_hi.annotate("campaign", (t["ultraimage"], ir["ultraimage"]), xytext=off(1.0, -13.8), textcoords="offset points",
                   ha="left", va="top", color=th["sca"], **nk)
    # Lower strip: names and ratios in grey (SEGA to the left, so that it clears DyPE).
    gk = dict(color=th["mid"], **nk)
    name_ratio(ax_lo, "sega_official", -5.0, 0, gk, gk, f"{sp['sega_official'][0]:.1f}\u00d7", ha="right")
    name_ratio(ax_lo, "dype", 5.0, 0, gk, gk, f"{sp['dype'][0]:.1f}\u00d7")
    name_ratio(ax_lo, "full", 5.0, 0, gk, gk, f"{sp['full'][0]:.1f}\u00d7")
    # Marker key in the strip's empty left side: filled = clean batch-1 median, hollow = campaign mean.
    handles = [Line2D([], [], marker="o", ls="none", ms=ms * 0.85, color=th["mid"], mew=0.6),
               Line2D([], [], marker="o", ls="none", ms=ms * 0.85, color=th["mid"], mfc=th["mask"], mew=1.0)]
    leg = ax_lo.legend(handles, ["clean batch-1 median", "campaign mean"], loc="center left",
                       bbox_to_anchor=(0.0, 0.5), frameon=False, fontsize=fss, handlelength=0.7, handletextpad=0.35,
                       borderaxespad=0.2, borderpad=0.0, labelspacing=0.2, labelcolor=th["mid"])
    leg.set_zorder(9)
    # One y label for both strips, centred on their combined height; the strip below carries the same text hidden,
    # so that each axes reports its label to the figure audit.
    hi_h = h - lo_h - gap
    ax_hi.set_ylabel("ImageReward ↑", fontsize=fs, color=th["ink"])
    ax_hi.yaxis.set_label_coords(-0.34 * fs_scale / (w * fig.get_figwidth()), 0.5 - (lo_h + gap) / (2 * hi_h))
    ax_lo.set_ylabel("ImageReward ↑", fontsize=fs)
    ax_lo.yaxis.label.set_visible(False)
    ax_lo.set_xlabel(r"Seconds per image at $4096^2$, log scale $\downarrow$", fontsize=fs, color=th["ink"],
                     labelpad=2)
    return ax_hi, ax_lo


def draw_latency(fig, box, d, th, fs_scale=1.0):
    x0, y0, w, h = box
    ax = fig.add_axes([x0, y0, w, h])
    fs, fsl, fss = sf.FS * fs_scale, sf.FS_TICK * fs_scale, sf.FS_SMALL * fs_scale
    ms = 4.0 * fs_scale

    def off(dx, dy):  # label offsets in points, scaled with the text
        return (dx * fs_scale, dy * fs_scale)
    cv = d["curve"]
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(1800, 9400)
    ax.set_ylim(8, 4200)
    ax.xaxis.set_major_locator(FixedLocator(list(SIDES)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.yaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    style_axes(ax, th, fsl)
    xs = list(SIDES)
    sca = [cv[("sca_v2", s)] for s in xs]
    area = [cv[("area", s)] for s in xs]
    fx = [s for s in xs if ("full", s) in cv]
    full = [cv[("full", s)] for s in fx]
    ax.fill_between(xs, sca, area, color=th["sca_fill"], lw=0, zorder=1)
    ax.plot(fx, full, color=th["full"], lw=1.1 * fs_scale, marker="s", ms=ms * 0.8, zorder=3, ls=(0, (3, 1.6)))
    ax.plot(xs, area, color=th["area"], lw=1.3 * fs_scale, marker="o", ms=ms, zorder=4)
    ax.plot(xs, sca, color=th["sca"], lw=1.5 * fs_scale, marker="o", ms=ms, zorder=5)
    ax.plot([4096], [d["clean"]["sca_v2_prog"]], marker="*", ms=ms * 2.3, color=th["sca"], ls="none", zorder=7,
            mew=0.0)
    # Single-pass SCA's speedup over AReA, one number per size along the top.
    for s in xs:
        ax.text(s, 0.975, f"{d['curve_ratio'][s]:.1f}×", transform=ax.get_xaxis_transform(), ha="center",
                va="top", color=th["sca"], fontsize=fsl, fontweight="bold", zorder=7)
    nk = dict(fontsize=fss, zorder=8)
    # Line labels.
    ax.annotate("AReA", (3072, cv[("area", 3072)]), xytext=off(-3, 4.5), textcoords="offset points", ha="right",
                va="bottom", color=th["area"], fontweight="bold", **nk)
    ax.annotate("Full attention", (6144, cv[("full", 6144)]), xytext=off(-5, 1), textcoords="offset points",
                ha="right", va="bottom", color=th["mid"], **nk)
    ax.annotate("Single-pass SCA", (2048, cv[("sca_v2", 2048)]), xytext=off(3, -4.5), textcoords="offset points",
                ha="left", va="top", color=th["sca"], **nk)
    ax.annotate("Progressive\nSCA", (4096, d["clean"]["sca_v2_prog"]), xytext=off(6.5, -1), textcoords="offset points",
                ha="left", va="center", color=th["sca"], fontweight="bold", linespacing=1.0, **nk)
    ax.text(1.0, 1.01, "single-pass SCA over AReA", transform=ax.transAxes, ha="right", va="bottom",
            color=th["sca"], fontsize=fss, style="italic", zorder=8)
    ax.set_xlabel("Image side (px), log scale", fontsize=fs, color=th["ink"], labelpad=2)
    ax.set_ylabel("Seconds per image ↓", fontsize=fs, color=th["ink"], labelpad=2)
    return ax


LAYOUT = dict(W=sf.TEXT, H=1.52, bottom=0.33, top=0.165, a=(0.46, 3.00), b=(4.02, 1.43))


def build(d, th, panel_letter_color):
    L = LAYOUT
    W, H = L["W"], L["H"]
    fig = plt.figure(figsize=(W, H))
    bot, hh = L["bottom"] / H, 1 - (L["bottom"] + L["top"]) / H
    ax_hi, _ = draw_time(fig, (L["a"][0] / W, bot, L["a"][1] / W, hh), d, th)
    ax_b = draw_latency(fig, (L["b"][0] / W, bot, L["b"][1] / W, hh), d, th)
    for ax, letter, dx in ((ax_hi, "a", -0.46), (ax_b, "b", -0.44)):
        bb = ax.get_position()
        fig.text(bb.x0 + dx / W, bb.y1 + 0.03 / H, letter, fontsize=sf.FS_PANEL, fontweight="bold",
                 color=panel_letter_color, ha="left", va="bottom")
    return fig


def paper_figure(d):
    sf.use_style()
    sf.script_floor(sf.FS_TICK)
    return build(d, THEMES["light"], sf.INK)


def web_figures(d, out: Path):
    """Transparent SVG, PNG (300 dpi) and WebP of the figure and of each panel, light and dark."""
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def emit(fig, stem):
        for ext in ("svg", "png"):
            path = out / f"{stem}.{ext}"
            fig.savefig(path, format=ext, dpi=300 if ext == "png" else None, transparent=True)
            written.append(path)
        Image.open(out / f"{stem}.png").save(out / f"{stem}.webp", "WEBP", quality=92, method=6)
        written.append(out / f"{stem}.webp")
        plt.close(fig)

    for name, th in THEMES.items():
        sf.use_style()
        sf.script_floor(sf.FS_TICK)
        mpl.rcParams.update({"text.color": th["ink"], "axes.labelcolor": th["ink"]})
        emit(build(d, th, th["ink"]), f"speed_{name}")
        W, H = 4.6, 2.9
        fig = plt.figure(figsize=(W, H))
        draw_time(fig, (0.62 / W, 0.48 / H, 3.85 / W, 2.18 / H), d, th, fs_scale=1.15)
        emit(fig, f"speed_time_{name}")
        fig = plt.figure(figsize=(W, H))
        draw_latency(fig, (0.72 / W, 0.48 / H, 3.75 / W, 2.18 / H), d, th, fs_scale=1.15)
        emit(fig, f"speed_latency_{name}")
    return written


def snippet(d) -> str:
    sp, cr = d["speed"], d["curve_ratio"]
    lines = ["% Written by paper/figures/src/fig_speed.py from results/; do not edit by hand.",
             "% Speedups of progressive SCA at 4096^2: clean over clean (sweep 3) or campaign over campaign.",
             "\\newcommand{\\hlSpeedProgUltraCamp}{%.1f}" % sp["ultraimage"][0],
             "\\newcommand{\\hlSpeedProgFull}{%.1f}" % sp["full"][0],
             "\\newcommand{\\hlSpeedProgDypeCamp}{%.1f}" % sp["dype"][0],
             "\\newcommand{\\hlSpeedProgSegaCamp}{%.1f}" % sp["sega_official"][0],
             "\\newcommand{\\hlSecUltraCamp}{%.1f}" % d["camp"]["ultraimage"],
             "\\newcommand{\\hlSecProgCamp}{%.1f}" % d["camp"]["sca_v2_prog"],
             "% Single-pass SCA over AReA per side (clean sweeps 1 and 2).",
             "\\newcommand{\\hlSpeedScaAreaLo}{%.1f}" % min(cr.values()),
             "\\newcommand{\\hlSpeedScaAreaHi}{%.1f}" % max(cr.values())]
    return "\n".join(lines) + "\n"


def main():
    d = load()
    stem = FIGS / "speed"
    fig = paper_figure(d)
    sf.save(fig, str(stem))
    snip = FIGS / "snippets" / "speed_numbers.tex"
    snip.write_text(snippet(d))
    web = web_figures(d, WEB)
    inputs = ["results/a4k195_main/base_summary.csv", "results/a4k195_main/extra_summary.csv",
              "results/a4k195_main_sca_v2/base_summary.csv", "results/a4k195_main_sca_v2/extra_summary.csv",
              "results/prog195_sca_v2/base_summary.csv", "results/prog195_sca_v2/extra_summary.csv",
              "results/prog_latency/latency_summary.csv", "results/latency_curve_v2/summary.csv",
              "results/latency_curve_summary.csv", "results/latency_curve_batch1.csv",
              "paper/figures/snippets/main_tables/headline.tex"]
    prov = {
        "figure": "fig:speed", "script": "paper/figures/src/fig_speed.py",
        "inputs": {p: sha256(ROOT / p) for p in inputs},
        "values": {
            "imagereward_seed0_195": d["ir"], "campaign_seconds": d["camp"], "clean_seconds_4096": d["clean"],
            "clean_curve": {f"{m}@{s}": v for (m, s), v in sorted(d["curve"].items())},
            "speedup_of_progressive": {m: {"ratio": v, "basis": b} for m, (v, b) in d["speed"].items()},
            "single_pass_over_area": {str(s): v for s, v in d["curve_ratio"].items()},
        },
        "rules": ["x of panel (a): clean batch-1 median where measured, else campaign mean (hollow marker)",
                  "ratios: clean over clean (sweep 3 for AReA and HRDiT, sweeps 1 and 3 for full attention) or "
                  "campaign over campaign (UltraImage, DyPE, SEGA)"],
        "outputs": {str(p.relative_to(ROOT)): sha256(p) for p in
                    [stem.with_suffix(".pdf"), stem.with_suffix(".svg"), snip] + web},
    }
    (FIGS / "speed_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print({m: round(v, 3) for m, (v, _) in d["speed"].items()})


if __name__ == "__main__":
    main()
