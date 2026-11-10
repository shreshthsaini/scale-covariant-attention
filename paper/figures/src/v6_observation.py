"""Figure 2 (v6, paper/figures/v6/observation.pdf): reach belongs to the head, not to the denoising step.

FLUX.1-dev at 1024^2, where full attention is still valid. A head's near share at a query is the share of its image
attention within Chebyshev distance 16 tokens of the query.
(a) One query (the face of the leftmost sheep, token row 43, column 11) of one 1024^2 image (the native stage of
    progressive SCA for prompt a4k_eval_4096_minus8343371271909676658, seed 0), block 12, two heads, three noise
    levels (sigma 0.90, 0.61, 0.33: the captured steps 8, 20, 26 of 30). Heads 4 and 7 are the highest and lowest
    share within 16 tokens among block 12's 24 heads in the reach statistics (32 prompts, noise bin centred at 0.62),
    the rule of fig_attn_focus.py. Full attention over the 64 x 64 image keys: amber inside the 16-token
    neighbourhood (dashed), blue outside; the number is the near share of that map
    (paper/figures/src/data/v6_attn_block12.npz, v6_attn_extract.py).
(b) Candidate A: the near share of all 1,368 heads (57 blocks x 24 heads) against the noise level, one line per head
    (32 prompts, 256 seeded queries per block and step, the 11 populated noise bins after the gate, sigma < 0.89),
    coloured by the head's mean share; heads 4 and 7 drawn heavy; the head-averaged curve in black.
    Candidate B: the same numbers as a matrix (rows = heads sorted by mean share, columns = noise bins) with the
    head-averaged curve as a strip under it.
(c) Candidate A: the variance split of the near share (FLUX.1-dev, 1024^2, after the gate): head 97.5%, noise level
    2.5%; the zoom shows the part of the noise level that the head-averaged curve carries, 0.7% of the total.
    Candidate B: head and noise-level shares for FLUX.1-dev, FLUX.1-Krea-dev and FLUX.2-dev (Table B.1 rows).
Numbers: results/heads_view/variance_split.csv (checked against the recomputation from calib/k1_1024.npz and against
headline.tex \\hlVarHead, \\hlVarStep, \\hlVarAvg). Nothing is typed.

  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I paper/figures/src/v6_observation.py [--keep A|B]
Writes both candidates to BUILD/observation_{A,B}.pdf, the kept one to paper/figures/v6/observation.pdf, and
paper/figures/v6/observation_provenance.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig as sf  # noqa: E402

ROOT = HERE.parents[2]
FIGS = ROOT / "paper" / "figures"
V6 = FIGS / "v6"
CALIB = ROOT / "calib" / "k1_1024.npz"
VARSPLIT = ROOT / "results" / "heads_view" / "variance_split.csv"
HEADLINE = FIGS / "snippets" / "main_tables" / "headline.tex"
ATTN = HERE / "data" / "v6_attn_block12.npz"
IMG = FIGS / "attn_focus" / "image_1024.png"
BUILD = Path(os.environ.get("V6_BUILD", "/scratch/09032/saini_2/sca/paper_builds/v6_methodfigs_1")) / "obs"
GATE, RADIUS = 0.89, 16
BLOCK, HEADS = 11, (3, 6)          # zero-based: block 12, heads 4 and 7
W, H = sf.TEXT, 1.90
FS_T, FS_L, FS_S = 8.5, 7.5, 7.0   # panel titles, labels, ticks and small labels
DMAX = 100.0
HEAD_C, STEP_C, AVG_C = "#3B4249", "#C9CED3", sf.INK
LINE_CMAP = LinearSegmentedColormap.from_list("farnear", [(0, sf.GLOBAL), (0.5, "#9AA3AC"), (1, sf.LOCAL)])
NEAR = np.array([int(sf.LOCAL[i:i + 2], 16) / 255 for i in (1, 3, 5)])
FAR = np.array([int(sf.GLOBAL[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def half_up(x: Decimal) -> str:
    return str(x.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def split(X: np.ndarray):
    """scripts/variance_split.py arithmetic: shares of the total variance of X[layer, head, bin]."""
    v = X.reshape(-1)
    tot = v.var()
    lm = X.mean(axis=(1, 2))
    hm = X.mean(axis=2)
    bl = ((lm[:, None, None] - v.mean()) ** 2 * np.ones_like(X)).mean()
    hl = ((hm[:, :, None] - lm[:, None, None]) ** 2 * np.ones_like(X)).mean()
    wh = ((X - hm[:, :, None]) ** 2).mean()
    t = X.mean(axis=(0, 1))
    te = ((t - t.mean()) ** 2).mean()
    return bl / tot, hl / tot, wh / tot, te / tot


# ------------------------------------------------------------------------------------------------------------ data
def load() -> dict:
    z = np.load(CALIB, allow_pickle=False)
    sig = z["sigma_centers"].astype(float)
    pop = z["populated"].astype(bool)
    radii = [int(r) for r in z["radii"]]
    text = z["mass_mean"][0]
    share = z["mass_mean"][radii.index(RADIUS) + 1] / np.clip(1 - text, 1e-6, None)        # [57, 24, 14]
    post = np.where(pop & (sig < GATE))[0]
    cols = post[np.argsort(-sig[post], kind="stable")]                                     # sigma falls
    X = share[:, :, cols]
    nl, nh, nb = X.shape
    flat = X.reshape(nl * nh, nb)
    layer, head_in, noise, curve = split(share[:, :, post])

    rows = {}
    with open(VARSPLIT, newline="") as fh:
        for r in csv.DictReader(fh):
            if (r["res"] == "1024x1024" and r["sigma_range"] == "post_gate" and r["quantity"] == "image_share"
                    and r["radius"] == str(RADIUS)):
                rows[r["checkpoint"]] = r
    main = rows["flux1_dev"]
    for key, mine in (("layer", layer), ("head_within_layer", head_in), ("noise_within_head", noise),
                      ("head_avg_curve", curve)):
        if abs(float(main[key]) - mine) > 1e-6:
            raise SystemExit(f"{key}: archive gives {mine:.6f}, variance_split.csv {main[key]}")
    if int(main["n_heads"]) != nl * nh or int(main["n_bins"]) != nb:
        raise SystemExit("shape differs from variance_split.csv")
    pct = {}
    for ck, r in rows.items():
        n = Decimal(r["noise_within_head"])
        pct[ck] = {"head": half_up(100 * (1 - n)), "step": half_up(100 * n),
                   "avg": half_up(100 * Decimal(r["head_avg_curve"])),
                   "head_f": 1 - float(n), "step_f": float(n), "avg_f": float(r["head_avg_curve"])}
    tex = HEADLINE.read_text()
    for key, macro in (("head", "hlVarHead"), ("step", "hlVarStep"), ("avg", "hlVarAvg")):
        found = re.search(r"\\newcommand\{\\%s\}\{([^}]*)\}" % macro, tex)
        if found is None or found.group(1) != pct["flux1_dev"][key]:
            raise SystemExit(f"\\{macro}: headline.tex has {found and found.group(1)}, CSV rounds to "
                             f"{pct['flux1_dev'][key]}")
    for ck, macro in (("flux1_krea_dev", "hlVarHeadKrea"), ("flux2_dev", "hlVarHeadFluxTwo")):
        found = re.search(r"\\newcommand\{\\%s\}\{([^}]*)\}" % macro, tex)
        if found is None or found.group(1) != pct[ck]["head"]:
            raise SystemExit(f"\\{macro} differs from the CSV")
    curve_vals = X.mean(axis=(0, 1))
    assert abs(curve_vals.min() - float(main["curve_min"])) < 1e-6
    assert abs(curve_vals.max() - float(main["curve_max"])) < 1e-6

    a = np.load(ATTN)
    assert int(a["block"]) == BLOCK and tuple(int(v) for v in a["query"]) == (43, 11)
    # heads 4 and 7: highest and lowest share within 16 tokens in block 12 at the bin centred at 0.62
    s16 = share[BLOCK, :, 8]
    assert (int(np.argmax(s16)), int(np.argmin(s16))) == HEADS, "head rule"
    order = np.argsort(-flat.mean(axis=1), kind="stable")
    return dict(sig_cols=sig[cols], flat=flat, order=order, curve=curve_vals, pct=pct,
                parts={"layer": layer, "head_in": head_in, "noise": noise, "curve": curve},
                maps=a["full"], map_sigma=a["sigma"], map_share=a["share16"], query=(43, 11),
                head_rows={h: flat[BLOCK * nh + h] for h in HEADS},
                ranks={h: int(np.where(order == BLOCK * nh + h)[0][0]) for h in HEADS})


# ------------------------------------------------------------------------------------------------------------ draw
def light_base(px: int, lo: float = 0.60, hi: float = 0.97) -> np.ndarray:
    L = np.asarray(Image.open(IMG).convert("L").resize((px, px), Image.LANCZOS), dtype=np.float64) / 255
    return np.repeat((lo + (hi - lo) * L)[..., None], 3, -1)


def heat(base, dens, color, mask):
    v = np.clip(np.log10(np.maximum(dens, 1.0)) / np.log10(DMAX), 0, 1)
    v = np.where(mask, v, 0.0)
    a = np.minimum(1.0, v ** 0.6)
    k = base.shape[0] // dens.shape[0]
    a = np.kron(a, np.ones((k, k)))[..., None]
    return base * (1 - a) + color[None, None, :] * a


def inch_axes(fig, x, y, w, h, **kw):
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def ftext(fig, x, y, s, **kw):
    return fig.text(x / W, y / H, s, **kw)


def title(fig, x, y, letter, name):
    ftext(fig, x, y, f"({letter}) {name}", fontsize=FS_T, fontweight="bold", ha="left", va="baseline")


def panel_tiles(fig, d, x0, y_top, T, gap):
    """(a) rows: heads 4 and 7; columns: the three captured noise levels."""
    px = 448
    base = light_base(px)
    n = 64
    qr, qc = d["query"]
    rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    near = np.maximum(np.abs(rr - qr), np.abs(cc - qc)) <= RADIUS
    r0, r1 = max(0, qr - RADIUS), min(n, qr + RADIUS + 1)
    c0, c1 = max(0, qc - RADIUS), min(n, qc + RADIUS + 1)
    s = px / n
    for ri, h in enumerate(HEADS):
        y0 = y_top - (ri + 1) * T - ri * gap
        for ci in range(3):
            x = x0 + ci * (T + gap)
            ax = inch_axes(fig, x, y0, T, T)
            full = d["maps"][ci, h].astype(np.float64)
            dens = full / full.sum() * n * n
            im = heat(heat(base, dens, NEAR, near), dens, FAR, ~near)
            ax.imshow(np.clip(im, 0, 1), extent=(0, px, px, 0), interpolation="none")
            ax.add_patch(Rectangle((c0 * s, r0 * s), (c1 - c0) * s, (r1 - r0) * s, fill=False, ec=sf.LOCAL,
                                   lw=0.8, ls=(0, (2.2, 1.4)), zorder=5))
            ax.add_patch(Rectangle(((qc + 0.5) * s - 13, (qr + 0.5) * s - 13), 26, 26, fc="white", ec="none",
                                   zorder=6))
            ax.add_patch(Rectangle(((qc + 0.5) * s - 9.5, (qr + 0.5) * s - 9.5), 19, 19, fc=sf.INK, ec="none",
                                   zorder=7))
            ax.set_xlim(0, px)
            ax.set_ylim(px, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(sf.METHOD_COLOR["full"])
                sp.set_linewidth(0.5)
            val = float(d["map_share"][ci, h])
            ax.text(0.95, 0.94, f"{val:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=FS_S,
                    color=sf.INK, zorder=9, bbox=dict(boxstyle="round,pad=0.16,rounding_size=0.18", fc="white",
                                                       ec="none", alpha=0.88))
            if ri == 0:
                ftext(fig, x + T / 2, y_top + 0.045, rf"$\sigma={d['map_sigma'][ci]:.2f}$", fontsize=FS_S,
                      ha="center", va="bottom")
        col = sf.LOCAL if ri == 0 else sf.GLOBAL
        ftext(fig, x0 - 0.06, y0 + T / 2, f"head {h + 1}", fontsize=FS_L, rotation=90, ha="right", va="center",
              color=col, fontweight="bold")


def panel_lines(fig, d, x, y, w, h):
    ax = inch_axes(fig, x, y, w, h)
    sig = d["sig_cols"]
    flat = d["flat"]
    means = flat.mean(1)
    for i in np.argsort(np.abs(means - 0.5))[::-1]:          # extremes first, the grey middle on top
        ax.plot(sig, flat[i], color=LINE_CMAP(float(np.clip(means[i], 0, 1))), lw=0.32, alpha=0.22,
                solid_joinstyle="round", zorder=2)
    for hh, col in zip(HEADS, (sf.LOCAL, sf.GLOBAL)):
        ax.plot(sig, d["head_rows"][hh], color="white", lw=2.6, zorder=4, solid_capstyle="round")
        ax.plot(sig, d["head_rows"][hh], color=col, lw=1.3, zorder=5, solid_capstyle="round")
    ax.plot(sig, d["curve"], color="white", lw=2.8, zorder=6, solid_capstyle="round")
    ax.plot(sig, d["curve"], color=AVG_C, lw=1.4, zorder=7, solid_capstyle="round")
    ax.set_xlim(sig[0] + 0.02, sig[-1] - 0.02)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xticks([0.8, 0.6, 0.4, 0.2])
    sf.value_grid(ax, "y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=FS_S, length=2.2, width=0.5, pad=1.6)
    ax.set_ylabel("near share", fontsize=FS_L, labelpad=2)
    ax.set_xlabel(r"noise level $\sigma$  $\rightarrow$ denoising", fontsize=FS_L, labelpad=1.5)
    ends = [(d["head_rows"][HEADS[0]][-1], f"head {HEADS[0] + 1}", sf.LOCAL),
            (d["curve"][-1], "head average", AVG_C),
            (d["head_rows"][HEADS[1]][-1], f"head {HEADS[1] + 1}", sf.GLOBAL)]
    for yv, lab, col in ends:
        ax.annotate(lab, (sig[-1], yv), xytext=(3, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=FS_S, color=col, annotation_clip=False)
    return ax


def panel_matrix(fig, d, x, y, w, h, strip_h=0.075):
    flat, order = d["flat"], d["order"]
    M = flat[order]
    n, nb = M.shape
    ax = inch_axes(fig, x, y + strip_h + 0.05, w, h - strip_h - 0.05)
    rgb = np.repeat(sf.PI_CMAP(np.clip(M, 0, 1))[..., :3], 24, axis=1)
    ax.imshow(rgb, extent=(0, nb, n, 0), interpolation="none", aspect="auto")
    ax.set_xlim(0, nb)
    ax.set_ylim(n, 0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
    ax.set_yticks([0, n], ["1", f"{n:,}"])
    ax.tick_params(labelsize=FS_S, length=0, pad=1.5)
    ax.set_ylabel("heads, by mean", fontsize=FS_L, labelpad=1)
    for hh, col in zip(HEADS, (sf.LOCAL, sf.GLOBAL)):
        yy = d["ranks"][hh] + 0.5
        ax.annotate(f"head {hh + 1}", (nb, yy), xytext=(4, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=FS_S, color=col, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7, shrinkA=0, shrinkB=0))
    sx = inch_axes(fig, x, y, w, strip_h, sharex=ax)
    rgb = np.repeat(sf.PI_CMAP(np.clip(d["curve"][None, :], 0, 1))[..., :3], 24, axis=1)
    sx.imshow(rgb, extent=(0, nb, 1, 0), interpolation="none", aspect="auto")
    for sp in sx.spines.values():
        sp.set_visible(False)
    sx.set_yticks([])
    ticks = [1, 4, 7, 10]
    sx.set_xticks([t + 0.5 for t in ticks], [f"{d['sig_cols'][t]:.2f}" for t in ticks])
    sx.tick_params(labelsize=FS_S, length=2, width=0.5, pad=1.5)
    sx.set_xlabel(r"noise level $\sigma$  $\rightarrow$ denoising", fontsize=FS_L, labelpad=1.5)
    sx.annotate("head average", (nb, 0.5), xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                fontsize=FS_S, color=AVG_C, annotation_clip=False)
    return ax


def panel_zoom(fig, d, x, y, w, h):
    """(c) A: the variance split drawn large (FLUX.1-dev), a zoom into the noise-level share, and the head's share
    in the two other checkpoints as slim bars on the same scale."""
    ax = inch_axes(fig, x, y, w, h)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, h)              # y in inches inside the panel
    ax.set_axis_off()
    p = d["pct"]["flux1_dev"]
    head, step, avg = 100 * p["head_f"], 100 * p["step_f"], 100 * p["avg_f"]
    bt, bh = h - 0.38, 0.25        # top bar: all of the variance, FLUX.1-dev
    zt, zh = h - 0.86, 0.25        # zoom bar: the noise-level share
    ax.text(0, bt + bh + 0.035, "FLUX.1-dev", ha="left", va="bottom", fontsize=FS_S, color=sf.MID)
    ax.add_patch(Rectangle((0, bt), head, bh, fc=HEAD_C, ec="none"))
    ax.add_patch(Rectangle((head, bt), step, bh, fc=STEP_C, ec="none"))
    ax.text(4, bt + bh / 2, f"head {p['head']}%", ha="left", va="center", fontsize=FS_L, color="white",
            fontweight="bold")
    ax.add_patch(Polygon([(head, bt), (100, bt), (100, zt + zh), (0, zt + zh)], closed=True, fc="#EDEFF1",
                         ec="none", zorder=0))
    ax.add_patch(Rectangle((0, zt), 100, zh, fc=STEP_C, ec="none"))
    ax.add_patch(Rectangle((0, zt), 100 * avg / step, zh, fc=AVG_C, ec="none"))
    ax.text(97, zt + zh / 2, f"noise level {p['step']}%", ha="right", va="center", fontsize=FS_S, color=sf.INK)
    ax.plot([100 * avg / step / 2] * 2, [zt, zt - 0.06], color=AVG_C, lw=0.6)
    ax.text(0, zt - 0.075, f"head average {p['avg']}%", ha="left", va="top", fontsize=FS_S, color=AVG_C)
    # the other checkpoints (Table B.1): the head's share, same scale as the top bar
    sb = 0.13
    for i, (ck, nm) in enumerate((("flux1_krea_dev", "FLUX.1-Krea-dev"), ("flux2_dev", "FLUX.2-dev"))):
        q = d["pct"][ck]
        y0 = 0.30 - i * 0.30
        ax.add_patch(Rectangle((0, y0), 100 * q["head_f"], sb, fc=HEAD_C, ec="none"))
        ax.add_patch(Rectangle((100 * q["head_f"], y0), 100 * q["step_f"], sb, fc=STEP_C, ec="none"))
        ax.text(0, y0 + sb + 0.03, nm, ha="left", va="bottom", fontsize=FS_S, color=sf.MID)
        ax.text(97, y0 + sb / 2, f"{q['head']}%", ha="right", va="center", fontsize=FS_S, color="white",
                fontweight="bold")


def panel_models(fig, d, x, y, w, h):
    """(c) B: head and noise-level shares of the near share's variance in three checkpoints."""
    ax = inch_axes(fig, x, y, w, h)
    names = [("flux1_dev", "FLUX.1-dev"), ("flux1_krea_dev", "FLUX.1-Krea-dev"), ("flux2_dev", "FLUX.2-dev")]
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(names) - 0.2)
    ax.set_axis_off()
    for i, (ck, nm) in enumerate(names):
        p = d["pct"][ck]
        yy = len(names) - 1 - i
        ax.barh([yy], [100 * p["head_f"]], color=HEAD_C, height=0.5, lw=0)
        ax.barh([yy], [100 * p["step_f"]], left=[100 * p["head_f"]], color=STEP_C, height=0.5, lw=0)
        ax.text(4, yy, f"head {p['head']}%", ha="left", va="center", fontsize=FS_S, color="white",
                fontweight="bold")
        ax.text(0, yy + 0.3, nm, ha="left", va="bottom", fontsize=FS_S, color=sf.INK)


def render(d, cand: str):
    sf.use_style()
    sf.script_floor(FS_S)
    fig = plt.figure(figsize=(W, H))
    T, gap, x0, y_top = 0.655, 0.03, 0.22, 1.60
    title(fig, 0.0, 1.81, "a", "Two heads, one query")
    panel_tiles(fig, d, x0, y_top, T, gap)
    xb = x0 + 3 * T + 2 * gap + 0.36
    title(fig, xb - 0.30, 1.81, "b", "All 1,368 heads")
    if cand == "A":
        panel_lines(fig, d, xb, 0.40, 1.20, 1.20)
        xc = xb + 1.20 + 0.66
        title(fig, xc, 1.81, "c", "Variance split")
        panel_zoom(fig, d, xc, 0.03, W - xc - 0.02, 1.66)
    else:
        panel_matrix(fig, d, xb, 0.40, 1.20, 1.20)
        xc = xb + 1.20 + 0.66
        title(fig, xc, 1.81, "c", "Variance split")
        panel_models(fig, d, xc, 0.40, W - xc - 0.02, 1.25)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keep", choices=("A", "B"), default="A")
    a = ap.parse_args()
    d = load()
    BUILD.mkdir(parents=True, exist_ok=True)
    V6.mkdir(parents=True, exist_ok=True)
    out = {}
    for cand in ("A", "B"):
        fig = render(d, cand)
        p = BUILD / f"observation_{cand}.pdf"
        fig.savefig(p, dpi=600)
        plt.close(fig)
        out[cand] = p
    fig = render(d, a.keep)
    fig.savefig(V6 / "observation.pdf", dpi=600)
    plt.close(fig)
    p = d["pct"]
    prov = {
        "figure": "Figure 2 (fig:observation), v6", "script": rel(Path(__file__)), "kept": a.keep,
        "size_in": [W, H],
        "inputs": {rel(q): sha256(q) for q in (CALIB, VARSPLIT, HEADLINE, ATTN, IMG)},
        "rules": {
            "near_share": "mass_mean[radius 16] / (1 - text mass), as scripts/variance_split.py; tiles: share of "
                          "the full-attention map within Chebyshev distance 16 of the query",
            "bins": f"populated, sigma centre < {GATE} (11 bins), sigma falling left to right",
            "heads_a": "block 12, heads 4 and 7 = argmax and argmin of the 32-prompt share at the bin centred at "
                       "0.62 (asserted)",
            "variance": "variance_split.csv rows post_gate, image_share, radius 16; recomputed from the archive "
                        "(FLUX.1-dev) and checked against headline.tex",
        },
        "values": {"variance_percent": {k: {kk: v[kk] for kk in ("head", "step", "avg")} for k, v in p.items()},
                   "flux1_parts": d["parts"], "map_sigma": [float(s) for s in d["map_sigma"]],
                   "map_share": {f"head{h + 1}": [round(float(v), 4) for v in d["map_share"][:, h]]
                                 for h in HEADS},
                   "head_rows": {f"head{h + 1}": [round(float(v), 4) for v in d["head_rows"][h]] for h in HEADS},
                   "curve": [round(float(v), 4) for v in d["curve"]],
                   "sigma_cols": [round(float(s), 4) for s in d["sig_cols"]]},
        "candidates": {k: str(v) for k, v in out.items()},
        "output_sha256": sha256(V6 / "observation.pdf"),
    }
    (V6 / "observation_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print("kept", a.keep, "->", V6 / "observation.pdf")


if __name__ == "__main__":
    main()
