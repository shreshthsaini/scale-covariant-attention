"""Appendix figure heads_view (v6, paper/figures/v6/heads_view.pdf): the keys SCA reads, and each head's measured reach.

(a) The final operator's 4096^2 output of the Figure 1 strip prompt (minus8343371271909676658, seed 0, sca_v2): one
    8x8-token query tile (dark outline), its near view, the 64x64-token window (amber), and the far view's pooled keys
    (blue, every fourth per axis), the geometry of fig_heads_view.py.
(b) Measured reach of three heads at the noise bin centred at sigma = 0.62: full-attention mass density in square rings
    of Chebyshev radius 2, 4, 8, 16 tokens around a centred query (log scale, 0.1 to 100 times uniform), over the same
    image as a stand-in for the 1024^2 view; each tile carries the head's near share (share within 16 tokens).
(c) The fitted weight pi_h(sigma) of those heads in the ratio-2 table, with the median and interquartile range of all
    1,368 heads; sigma > sigma_g = 0.89 runs the far view only (grey band); the noise level falls from left to right as
    in Figure 2.
(d) pi_h against the near share for all 1,368 heads at the same bin, Pearson rho.
Heads: fig_heads_view.py's rank rule (heads with at least half of their mass on image keys; mid-ranks of pi and share;
the head nearest (q, q) in rank space for q = 0.90, 0.20, 0.05), labelled near, mid and far.

Data: fig_heads_view.compute() runs into a scratch folder (BUILD/heads_view_csv), and its four CSVs must equal
results/heads_view byte for byte, as fig_heads_view.py --no-results-csv requires; its SHA256SUMS lines must appear in
results/heads_view/SHA256SUMS (that file now also lists two CSVs other scripts add). Then fig_heads_view.load() draws
every number from results/heads_view, and fig_heads_view.source() checks the image. Nothing is written to results/;
the two thumbnails go to BUILD/heads_view_cache. Only the presentation follows the v6 house style (v6_emulation.py).

  cd paper/figures/src && /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I v6_app_heads_view.py
Writes paper/figures/v6/heads_view.pdf and paper/figures/v6/heads_view_provenance.json.
"""
from __future__ import annotations

import filecmp
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_heads_view as base  # noqa: E402
import scafig as sf  # noqa: E402

ROOT = HERE.parents[2]
V6 = ROOT / "paper" / "figures" / "v6"
STEM = "heads_view"
BUILD = Path(os.environ.get("V6_BUILD", "/scratch/09032/saini_2/sca/paper_builds/v6_methodfigs_2"))
W, H = sf.TEXT, 1.62
FS_T, FS_L, FS_S, FS_V = 8.5, 7.5, 7.0, 6.5
FS_SUP = 6.5
ROLE = {"local": "near", "mid": "mid", "global": "far"}         # the three heads, in the paper's words
COL = base.COL                                                  # amber, ink, blue
TEXT_COL = {"local": base.LOCAL_TEXT, "mid": sf.INK, "global": base.GLOBAL_TEXT}
CSVS = ["pi_sigma.csv", "pi_vs_mass.csv", "rings.csv", "summary.csv"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


# ------------------------------------------------------------------------------------------------------------ data
def data() -> dict:
    """fig_heads_view.py main() with --no-results-csv, the temporary folder on scratch."""
    kept = base.RES
    fresh = BUILD / "heads_view_csv"
    fresh.mkdir(parents=True, exist_ok=True)
    base.RES = fresh
    try:
        sel = base.compute()
    finally:
        base.RES = kept
    same = {k: filecmp.cmp(fresh / k, kept / k, shallow=False) for k in CSVS}
    assert all(same.values()), f"results/heads_view differs from a fresh computation: {same}"
    lines = set((fresh / "SHA256SUMS").read_text().splitlines())
    assert lines <= set((kept / "SHA256SUMS").read_text().splitlines()), "SHA256SUMS lines differ"
    d = base.load()
    png, meta = base.source()
    img = Image.open(png).convert("RGB")
    assert img.size == (base.RES_PX, base.RES_PX)
    base.CACHE = BUILD / "heads_view_cache"                    # regenerable thumbnails, not the shared figcache
    img_a, path_a = base.thumb(img, 768, base.SAT_A, base.VEIL_A, "a_church")
    img_b, path_b = base.thumb(img, 256, base.SAT_B, base.VEIL_B, "b_church")
    return dict(d, sel=sel, png=png, meta=meta, img_a=img_a, img_b=img_b, cache=[path_a, path_b])


# ------------------------------------------------------------------------------------------------------------ draw
def axes_in(fig, x, y, w, h, **kw):
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def title(fig, x, y, letter, name, sup=None):
    t = fig.text(x / W, y / H, f"({letter}) {name}", fontsize=FS_T, fontweight="bold", ha="left", va="baseline")
    if sup:  # a superscript at 6.5 pt, as the v5.1 figure set it (a mathtext script would print in another face)
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        fig.text(bb.x1 / fig.dpi / W + 0.004 / W, (y + 0.042) / H, sup, fontsize=FS_SUP, fontweight="bold",
                 ha="left", va="baseline")


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=FS_S, length=2.2, width=0.5, pad=1.6)


def chip(ax, x, y, text, color, **kw):
    return ax.text(x, y, text, fontsize=FS_S, color=color, zorder=8,
                   bbox=dict(boxstyle="round,pad=0.16,rounding_size=0.18", fc="white", ec="none", alpha=0.9), **kw)


def panel_keys(fig, d, x, y, s):
    """(a) the near window and the far view's pooled keys on the 4096^2 output (fig_heads_view.py geometry)."""
    ax = axes_in(fig, x, y, s, s)
    R, PX = base.RES_PX, base.PX
    ax.imshow(d["img_a"], extent=(0, R, R, 0), interpolation="none", zorder=0)
    ax.set_xlim(0, R)
    ax.set_ylim(R, 0)
    ax.axis("off")
    centres = (base.POOL * np.arange(base.KEY_OFFSET, R // (base.POOL * PX), base.KEY_STEP) + base.POOL // 2) * PX \
        + PX / 2
    gx, gy = np.meshgrid(centres, centres)
    ax.scatter(gx.ravel(), gy.ravel(), s=2.6, color=sf.GLOBAL, edgecolors="white", linewidths=0.3, zorder=3)
    tr, tc = base.TILE_RC
    wr, wc = base.window_start(tr), base.window_start(tc)
    ax.add_patch(Rectangle((wc * PX, wr * PX), base.WINDOW * PX, base.WINDOW * PX, fill=False, ec=sf.LOCAL, lw=1.0,
                           zorder=4))
    halo = [patheffects.withStroke(linewidth=1.3, foreground="white")]
    ax.add_patch(Rectangle((tc * base.TILE * PX, tr * base.TILE * PX), base.TILE * PX, base.TILE * PX, fill=False,
                           ec=sf.INK, lw=0.65, zorder=5, path_effects=halo))
    chip(ax, wc * PX + 20, wr * PX - 70, "near", TEXT_COL["local"], ha="left", va="bottom")
    chip(ax, centres[0] + 110, centres[1], "far", TEXT_COL["global"], ha="left", va="center")
    return centres


def panel_reach(fig, d, xb, ya, sa):
    """(b) 2 x 2 grid: the three heads as ring-density heatmaps around a centred query, then the colour scale."""
    gap = 0.05
    tb = (sa - gap) / 2
    q = base.NATIVE // 2
    lo, hi = q - base.ZOOM_R, q + base.ZOOM_R + 1
    norm = LogNorm(vmin=base.DMIN, vmax=base.DMAX)
    cells = {"local": (0, 0), "mid": (0, 1), "global": (1, 0)}
    for role, _ in base.TARGETS:
        rr, cc = cells[role]
        x0, y0 = xb + cc * (tb + gap), ya + (1 - rr) * (tb + gap)
        tx = axes_in(fig, x0, y0, tb, tb)
        tx.imshow(d["img_b"], extent=(0, base.NATIVE, base.NATIVE, 0), interpolation="none", zorder=0)
        tx.set_xlim(lo, hi)
        tx.set_ylim(hi, lo)
        tx.axis("off")
        rg = d["rings"][role]
        tx.add_patch(PathPatch(base.annulus(q, base.NATIVE, base.RADII[-1]), fc=base.MASS_CMAP(norm(rg["outside"])),
                               alpha=base.HEAT_A, lw=0, zorder=2))
        for j, r in enumerate(base.RADII):
            tx.add_patch(PathPatch(base.annulus(q, r, base.RADII[j - 1] if j else None),
                                   fc=base.MASS_CMAP(norm(rg["density"][j])), alpha=base.HEAT_A, lw=0, zorder=2))
            tx.add_patch(Rectangle((q - r, q - r), 2 * r + 1, 2 * r + 1, fill=False, ec="white", lw=0.45, zorder=3))
        tx.plot([q + 0.5], [q + 0.5], "o", ms=2.0, mfc=sf.INK, mec="white", mew=0.4, zorder=4)
        lx, ly = x0 + 0.07, y0 + tb - 0.075
        fig.patches.append(FancyBboxPatch(((lx - 0.045) / W, (ly - 0.05) / H), 0.32 / W, 0.1 / H,
                                          boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                                          fc="white", ec="none", alpha=0.92, zorder=5))
        fig.add_artist(Line2D([lx / W], [ly / H], marker="o", ms=4.0, mfc=COL[role], mec="white", mew=0.5, ls="none",
                              transform=fig.transFigure, zorder=6))
        fig.text((lx + 0.05) / W, ly / H, f"{d['heads'][role]['share_r16']:.2f}", fontsize=FS_S, va="center",
                 ha="left", color=sf.INK, zorder=6)
    # colour scale in the fourth cell
    x0, y0 = xb + tb + gap, ya
    cb_w, cb_h = tb - 0.14, 0.075
    cbx = axes_in(fig, x0 + 0.07, y0 + 0.22, cb_w, cb_h)
    grad = np.logspace(np.log10(base.DMIN), np.log10(base.DMAX), 256)[None, :]
    cbx.imshow(grad, aspect="auto", cmap=base.MASS_CMAP, norm=norm, alpha=base.HEAT_A,
               extent=(np.log10(base.DMIN), np.log10(base.DMAX), 0, 1), interpolation="bilinear")
    cbx.set_yticks([])
    cbx.set_xticks([-1, 0, 1, 2], ["0.1", "1", "10", "100"])
    cbx.tick_params(axis="x", length=2.0, width=0.5, labelsize=FS_S, pad=1.5)
    for sp in cbx.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color(sf.MID)
    fig.text((x0 + tb / 2) / W, (y0 + 0.22 + cb_h + 0.06) / H, "mass density\nvs. uniform", fontsize=FS_S,
             ha="center", va="bottom", color=sf.INK, linespacing=1.1)


def panel_weights(fig, d, x, y, w, h):
    """(c) pi_h(sigma) of the three heads, median and interquartile range of all heads; noise level falling."""
    ax = axes_in(fig, x, y, w, h)
    cv = d["curves"]
    s = cv["sigma"]
    ax.axvspan(base.SIGMA_G, 1.0, color=base.BAND_GREY, lw=0, zorder=0)
    ax.axvline(base.SIGMA_G, color=sf.INK, lw=0.7, ls=(0, (2.4, 1.6)), zorder=2)
    ax.annotate(r"$\sigma_g$", (base.SIGMA_G, 0.0), xytext=(2, 0), textcoords="offset points", ha="left",
                va="bottom", fontsize=FS_L, color=sf.INK)
    ax.fill_between(s, cv["q25_all"], cv["q75_all"], color=sf.SCA_LIGHT, lw=0, zorder=1)
    ax.plot(s, cv["median_all"], color=sf.SCA, lw=1.4, zorder=3, solid_capstyle="round")
    i8 = list(cv["bin"]).index(base.BIN)
    for role, _ in base.TARGETS:
        ax.plot(s, cv[f"pi_{role}"], color=COL[role], lw=1.1, zorder=4, solid_capstyle="round")
        ax.plot([s[i8]], [cv[f"pi_{role}"][i8]], "o", ms=3.4, mfc=COL[role], mec="white", mew=0.5, zorder=5)
    ax.set_xlim(1.0, 0.05)
    ax.set_ylim(-0.03, 1.09)
    ax.set_xticks([0.8, 0.5, 0.2], ["0.8", "0.5", "0.2"])  # no end ticks: (d) starts at 0 right beside
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    clean(ax)
    ax.set_xlabel(r"noise level $\sigma$", fontsize=FS_L, labelpad=1.5)
    ax.set_ylabel(r"fitted weight $\pi_h$", fontsize=FS_L, labelpad=1.5)
    lab = dict(fontsize=FS_S, ha="left")
    ax.text(0.62, 0.975, ROLE["local"], color=TEXT_COL["local"], va="bottom", **lab)
    ax.text(0.62, float(np.interp(0.45, s, cv["median_all"])) - 0.035, "median", color=sf.SCA, va="top", **lab)
    ax.text(0.40, float(np.interp(0.25, s, cv["pi_mid"])) - 0.04, ROLE["mid"], color=sf.INK, va="top", **lab)
    ax.text(0.40, float(np.interp(0.30, s, cv["pi_global"])) + 0.05, ROLE["global"], color=TEXT_COL["global"],
            va="bottom", **lab)
    return ax


def panel_scatter(fig, d, x, y, w, h, sharey):
    ax = axes_in(fig, x, y, w, h, sharey=sharey)
    ax.scatter(d["x"], d["y"], s=2.2, color=base.DOT_GREY, lw=0, alpha=0.75, zorder=2, rasterized=True)
    for role, _ in base.TARGETS:
        hd = d["heads"][role]
        ax.plot([hd["share_r16"]], [hd[f"pi_s{base.SCALE}"]], "o", ms=4.0, mfc=COL[role], mec="white", mew=0.6,
                zorder=4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    clean(ax)
    ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("near share", fontsize=FS_L, labelpad=1.5)
    ax.text(0.97, 0.03, rf"$\rho={d['r']:.2f}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_L)
    return ax


def render(d):
    sf.use_style()
    sf.script_floor(FS_S)
    fig = plt.figure(figsize=(W, H))
    ty, top = 1.50, 1.42
    sa = 1.38
    ya = top - sa
    xb = sa + 0.08
    xc, wc = 3.24, 1.04
    xd = xc + wc + 0.10
    wd = W - 0.04 - xd
    yc = 0.30
    title(fig, 0.0, ty, "a", "Keys at 4096", sup="2")
    centres = panel_keys(fig, d, 0.0, ya, sa)
    title(fig, xb, ty, "b", "Measured reach")
    panel_reach(fig, d, xb, ya, sa)
    title(fig, xc - 0.34, ty, "c", "Across noise levels")
    cx = panel_weights(fig, d, xc, yc, wc, top - yc)
    title(fig, xd, ty, "d", f"All {len(d['x']):,} heads")
    panel_scatter(fig, d, xd, yc, wd, top - yc, sharey=cx)
    return fig, centres


def main() -> None:
    d = data()
    V6.mkdir(parents=True, exist_ok=True)
    fig, centres = render(d)
    out = V6 / f"{STEM}.pdf"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    png = d["png"]
    ins = [base.NPZ] + [base.RES / k for k in CSVS] + [png, png.with_suffix(".json")]
    cv = d["curves"]
    prov = {"figure": "Appendix figure heads_view (fig:heads), v6", "script": rel(Path(__file__)),
            "data_code": "paper/figures/src/fig_heads_view.py compute() into BUILD/heads_view_csv (four CSVs equal to "
                         "results/heads_view byte for byte), then load() and source() (imported)",
            "size_in": [W, H], "inputs": {rel(p): sha256(p) for p in ins},
            "values": {"pearson_rho_pi_vs_near_share": d["r"], "eligible_heads": d["sel"]["eligible"],
                       "heads": {ROLE[r]: d["heads"][r] for r in d["heads"]},
                       "sigma_bin_centre": float(cv["sigma"][list(cv["bin"]).index(base.BIN)]),
                       "pooled_keys_drawn": int(centres.size ** 2)},
            "thumbnails": {rel(p): sha256(p) for p in d["cache"]},
            "output_sha256": sha256(out)}
    (V6 / f"{STEM}_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    for role, hd in d["heads"].items():
        print(f"{ROLE[role]:4s} block {hd['block'] + 1:2d} head {hd['head'] + 1:2d}  pi_s2 {hd['pi_s2']:.3f}  "
              f"near share {hd['share_r16']:.3f}")
    print(f"rho = {d['r']:.4f}")


if __name__ == "__main__":
    main()
