"""Observation (fig:observation): at the training resolution, how far a head looks is set by the head, not the step.

FLUX.1-dev at 1024^2, where full attention is still valid. Reach is the share of a query's image attention that falls
within 16 tokens of the query: mass_mean[radius 16] / clip(1 - text mass, 1e-6) of the reach statistics
(calib/k1_1024.npz, 32 prompts), exactly as scripts/variance_split.py computes it, over the populated noise bins
after the gate (sigma < 0.89, 11 bins).

(a) Full attention of two heads of block 12 (one-based) at one query (the leftmost sheep's face) of a real 1024^2
    image, read from results/attn_focus/ and drawn with the functions of fig_attn_focus.py (its Figure 3 column
    "Full attention"): amber inside the emulated r = 4 window (16 x 16 tokens), blue outside it, the query as a ring,
    and the near share m_i of the image attention. Head 4 is near-reaching, head 7 far-reaching.
(b) Head-by-step matrix: one row per head (57 blocks x 24 heads = 1,368), sorted by the head's mean share over the
    11 post-gate bins (most near-reaching at the top); columns are those bins with sigma falling left to right; the
    colour is the share within 16 tokens on the paper's near/far map (scafig.PI_CMAP: amber near, blue far, 0 to 1).
    A thin line in each head's colour runs from its tile in (a) to a mark at its row. Variance split of the share
    (results/heads_view/variance_split.csv, flux1_dev, 1024x1024, post_gate, image_share, radius 16): head
    1 - noise_within_head = 97.5%, noise level noise_within_head = 2.5%, head-averaged curve head_avg_curve = 0.7%;
    the script recomputes all three from the archive and checks them against the CSV and headline.tex
    (\\hlVarHead, \\hlVarStep, \\hlVarAvg, half-up to one decimal).

Variants (both are always drawn into paper/figures/candidates/observation_{a,b}.pdf):
  a  the matrix with a vertical colourbar and, in a right column, a slim variance bar (head | noise level);
  b  (kept) the matrix with the head-averaged curve as a one-row strip under it on the same colour scale (what a
     single shared schedule can follow, 0.7% of the variance), and a right column with a horizontal colour scale
     (far 0 to near 1) above the variance bar (head | noise level).
--variant (default KEPT) picks the one written as paper/figures/observation.{pdf,svg}, the web exports
web/assets/img/figs/observation_{light,dark}.{svg,png,webp} and paper/figures/observation_provenance.json.

Run from anywhere, on a compute node (CPU only, a few seconds):
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python paper/figures/src/fig_observation.py [--variant a|b]
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
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_attn_focus as af  # noqa: E402  (loading and rendering of the two heads' maps; nothing is re-extracted)
import scafig as sf  # noqa: E402

ROOT = HERE.parents[2]
FIGS = ROOT / "paper" / "figures"
CAND = FIGS / "candidates"
WEB = ROOT / "web" / "assets" / "img" / "figs"
CALIB = ROOT / "calib" / "k1_1024.npz"
VARSPLIT = ROOT / "results" / "heads_view" / "variance_split.csv"
HEADLINE = FIGS / "snippets" / "main_tables" / "headline.tex"
AF_NPZ = af.RES / "attn_focus.npz"
AF_IMG = af.ASSETS / "image_1024.png"
GATE = 0.89               # post-gate bins: sigma < GATE (scripts/variance_split.py)
RADIUS = 16               # tokens
KEPT = "b"                # written as observation.pdf: the strip shows the 0.7% a shared schedule can follow
W, H = sf.TEXT, 1.80      # inches, built at the printed size
REP = 24                  # each matrix column is repeated REP times so that no viewer smooths across columns
FS_LAB, FS_NUM = 7.0, 6.5  # labels; tick labels and numbers (house floor 6 pt)

THEMES = {
    "light": dict(af.THEMES["light"], bar_head="#3B4249", bar_step="#C3C9CF", bar_txt="white"),
    "dark": dict(af.THEMES["dark"], bar_head="#C3C9CF", bar_step="#59636C", bar_txt="#0D100F"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def half_up(x: Decimal) -> str:
    return str(x.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def split(X: np.ndarray):
    """Same arithmetic as scripts/variance_split.py: shares of the total variance of X[layer, head, bin]."""
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


# ------------------------------------------------------------------------------------------------------- data
def load() -> dict:
    z = np.load(CALIB, allow_pickle=False)
    sig = z["sigma_centers"].astype(float)
    pop = z["populated"].astype(bool)
    radii = [int(r) for r in z["radii"]]
    text = z["mass_mean"][0]
    share = z["mass_mean"][radii.index(RADIUS) + 1] / np.clip(1 - text, 1e-6, None)      # [57, 24, 14]
    post = np.where(pop & (sig < GATE))[0]
    cols = post[np.argsort(-sig[post], kind="stable")]                                   # sigma falls left to right
    X = share[:, :, cols]
    n_layer, n_head, n_bin = X.shape
    flat = X.reshape(n_layer * n_head, n_bin)
    order = np.argsort(-flat.mean(axis=1), kind="stable")                                # most near-reaching on top
    _, _, noise, curve = split(share[:, :, post])

    with open(VARSPLIT, newline="") as fh:
        row = next(r for r in csv.DictReader(fh) if r["checkpoint"] == "flux1_dev" and r["res"] == "1024x1024"
                   and r["sigma_range"] == "post_gate" and r["quantity"] == "image_share" and r["radius"] == "16")
    for key, mine in (("noise_within_head", noise), ("head_avg_curve", curve)):
        if abs(float(row[key]) - mine) > 1e-6:
            raise SystemExit(f"{key}: archive gives {mine:.6f}, variance_split.csv {row[key]}")
    if int(row["n_heads"]) != n_layer * n_head or int(row["n_bins"]) != n_bin:
        raise SystemExit(f"shape {n_layer * n_head} x {n_bin} vs csv {row['n_heads']} x {row['n_bins']}")
    noise_d, curve_d = Decimal(row["noise_within_head"]), Decimal(row["head_avg_curve"])
    pct = {"head": half_up(100 * (1 - noise_d)), "step": half_up(100 * noise_d), "avg": half_up(100 * curve_d)}
    tex = HEADLINE.read_text()
    for key, macro in (("head", "hlVarHead"), ("step", "hlVarStep"), ("avg", "hlVarAvg")):
        found = re.search(r"\\newcommand\{\\%s\}\{([^}]*)\}" % macro, tex)
        if found is None or found.group(1) != pct[key]:
            raise SystemExit(f"\\{macro}: headline.tex has {found and found.group(1)}, the CSV rounds to {pct[key]}")

    curve_vals = X.mean(axis=(0, 1))
    if abs(curve_vals.min() - float(row["curve_min"])) > 1e-6 or abs(curve_vals.max() - float(row["curve_max"])) > 1e-6:
        raise SystemExit("head-averaged curve range differs from variance_split.csv")

    a = np.load(AF_NPZ)
    d = {k: a[k] for k in a.files}
    block = int(d["block"])
    heads = [int(h) for h in d["heads"]]
    ranks = {tag: int(np.where(order == block * n_head + h)[0][0]) for tag, h in zip(("near", "far"), heads)}
    d.update(M=flat[order], sigma_cols=sig[cols], curve=curve_vals, ranks=ranks, n_heads=n_layer * n_head,
             pct=pct, noise=float(noise_d), curve_share=float(curve_d), head_rows={
                 tag: flat[block * n_head + h] for tag, h in zip(("near", "far"), heads)},
             row_range=flat.max(axis=1) - flat.min(axis=1), row_mean=flat.mean(axis=1))
    return d


# ------------------------------------------------------------------------------------------------------- draw
def fx(x):
    return x / W


def fy(y):
    return y / H


def tri(fig, x, y, color, direction=">", ms=4.6):
    fig.lines.append(mpl.lines.Line2D([fx(x)], [fy(y)], transform=fig.transFigure, marker=direction, ms=ms,
                                      mfc=color, mec="none", ls="none", zorder=20))


def draw_tiles(fig, d, th, x0, y_top, y_bot, gap=0.05):
    """Panel a: the two heads' full attention (amber inside the r = 4 window, blue outside), stacked."""
    px = 512
    T = (y_top - y_bot - gap) / 2
    img = Image.open(AF_IMG)
    base = af.grey_base(img, th, px)
    win = d["window"]
    r0, c0, w, _ = (int(v) for v in win)
    mask = af.window_mask(win)
    near_c, far_c = af.hex_rgb(th["near"]), af.hex_rgb(th["far"])
    s = px / af.N_SIDE
    qr, qc = int(d["query"][0]), int(d["query"][1])
    qy, qx = (qr + 0.5) * s, (qc + 0.5) * s
    rows = {}
    for ri, tag_ in enumerate(("near", "far")):
        y0 = y_top - (ri + 1) * T - ri * gap
        ax = fig.add_axes([fx(x0), fy(y0), fx(T), fy(T)])
        full = d[f"{tag_}_full"]
        dens = full / full.sum() * af.N_SIDE * af.N_SIDE
        af.show(ax, af.heat(af.heat(base, dens, near_c, mask), dens, far_c, ~mask), px)
        ax.add_patch(Rectangle((c0 * s, r0 * s), w * s, w * s, fill=False, ec=th["near"], lw=0.8, zorder=8))
        af.query_mark(ax, qx, qy, ms=3.4)
        af.tag(ax, rf"$m_i={float(d[f'{tag_}_m']):.2f}$", th, FS_NUM)
        h1 = int(d["heads"][ri]) + 1
        yc = y0 + T / 2
        fig.text(fx(x0 - 0.075), fy(yc), f"Head {h1}", rotation=90, ha="center", va="center", fontsize=FS_LAB,
                 fontweight="bold", color=th["ink"])
        rows[tag_] = (y0, T)
    return T, rows


def leaders(fig, d, th, tile_x1, rows, mx0, my0, mh):
    """Tie each tile of panel a to its head's row in the matrix: a thin line in the head's colour from the tile's
    right edge to a mark at the row (both rows fall inside their tile's height at this layout; checked)."""
    n = d["M"].shape[0]
    for tag_ in ("near", "far"):
        y = my0 + mh * (1 - (d["ranks"][tag_] + 0.5) / n)
        y0, T = rows[tag_]
        if not y0 <= y <= y0 + T:
            raise SystemExit(f"row of the {tag_} head ({y:.3f} in) lies outside its tile ({y0:.3f}..{y0 + T:.3f})")
        fig.lines.append(mpl.lines.Line2D([fx(tile_x1 + 0.015), fx(mx0 - 0.07)], [fy(y), fy(y)],
                                          transform=fig.transFigure, color=th[tag_], lw=0.7, zorder=19))
        tri(fig, mx0 - 0.045, y, th[tag_], direction=">", ms=4.4)


def draw_matrix(fig, d, th, mx0, my0, mw, mh):
    """Panel b: one row per head, sorted; sigma falls left to right."""
    M = d["M"]
    n, nb = M.shape
    ax = fig.add_axes([fx(mx0), fy(my0), fx(mw), fy(mh)])
    rgb = np.repeat(sf.PI_CMAP(np.clip(M, 0, 1))[..., :3], REP, axis=1)
    ax.imshow(rgb, extent=(0, nb, n, 0), interpolation="none", aspect="auto", zorder=1)
    ax.set_xlim(0, nb)
    ax.set_ylim(n, 0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_yticks([])
    ax.set_ylabel(f"{n:,} heads, sorted", fontsize=FS_LAB, labelpad=7.5, color=th["ink"])
    return ax


def draw_colorbar_h(fig, th, x0, y0, width, ch=0.075):
    """Variant b: the colour scale as a short horizontal bar, far (0) at the left and near (1) at the right."""
    cax = fig.add_axes([fx(x0), fy(y0), fx(width), fy(ch)])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=sf.PI_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.ax.set_xticklabels(["0", "0.5", "1"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2.0, width=0.5, pad=1.5, labelsize=FS_NUM, colors=th["ink"])
    fig.text(fx(x0 - 0.04), fy(y0 + ch / 2), "far", ha="right", va="center", fontsize=FS_NUM, color=th["far"])
    fig.text(fx(x0 + width + 0.04), fy(y0 + ch / 2), "near", ha="left", va="center", fontsize=FS_NUM,
             color=th["near"])
    fig.text(fx(x0), fy(y0 + ch + 0.05), "share within 16 tokens", ha="left", va="bottom", fontsize=FS_LAB,
             color=th["ink"])
    return cax


def draw_colorbar(fig, th, cx0, my0, mh, cw=0.065):
    cax = fig.add_axes([fx(cx0), fy(my0), fx(cw), fy(mh)])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=sf.PI_CMAP)
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0", "0.5", "1"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2.0, width=0.5, pad=1.5, labelsize=FS_NUM, colors=th["ink"])
    cb.set_label("share within 16 tokens", fontsize=FS_LAB, labelpad=2.0, color=th["ink"])
    cax.text(0.5, 1 + 0.035 / mh, "near", transform=cax.transAxes, ha="center", va="bottom", fontsize=FS_NUM,
             color=th["near"])
    cax.text(0.5, -0.035 / mh, "far", transform=cax.transAxes, ha="center", va="top", fontsize=FS_NUM,
             color=th["far"])
    return cax


def draw_var_horizontal(fig, d, th, x0, yc, width, heading, bh=0.12):
    """One stacked bar of the variance of the share: head | noise level, as in Figure 3c."""
    head, step = 100 - 100 * d["noise"], 100 * d["noise"]
    ax = fig.add_axes([fx(x0), fy(yc - bh / 2), fx(width), fy(bh)])
    ax.barh([0], [head], color=th["bar_head"], height=1.0, lw=0)
    ax.barh([0], [step], left=[head], color=th["bar_step"], height=1.0, lw=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_axis_off()
    ax.text(3.0, 0, f"head {d['pct']['head']}%", ha="left", va="center", fontsize=FS_NUM, color=th["bar_txt"],
            fontweight="bold")
    ax.annotate(f"noise level {d['pct']['step']}%", (100, -0.5), xytext=(0, -2.5), textcoords="offset points",
                ha="right", va="top", fontsize=FS_NUM, color=th["mid"])
    fig.text(fx(x0), fy(yc + bh / 2 + 0.045), heading, ha="left", va="bottom", fontsize=FS_LAB,
             color=th["ink"], linespacing=1.1)


def draw_strip(fig, d, th, mx0, sy0, mw, sh, share):
    """Variant b: the head-averaged curve on the matrix's colour scale, column for column, with its variance share.
    It shares the matrix's x axis and carries the sigma ticks for both."""
    nb = len(d["curve"])
    ax = fig.add_axes([fx(mx0), fy(sy0), fx(mw), fy(sh)], sharex=share)
    rgb = np.repeat(sf.PI_CMAP(np.clip(d["curve"][None, :], 0, 1))[..., :3], REP, axis=1)
    ax.imshow(rgb, extent=(0, nb, 1, 0), interpolation="none", aspect="auto")
    ax.set_xlim(0, nb)
    ax.set_ylim(1, 0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_yticks([])
    ax.set_ylabel("mean", rotation=0, ha="right", va="center", fontsize=FS_NUM, color=th["ink"])
    ax.yaxis.set_label_coords(-0.04 / mw, 0.5)
    fig.text(fx(mx0 + mw + 0.04), fy(sy0 + sh / 2), f"{d['pct']['avg']}% of variance", ha="left", va="center",
             fontsize=FS_NUM, color=th["mid"])
    return ax


def sigma_ticks(ax, d, th):
    nb = len(d["sigma_cols"])
    ticks = list(range(0, nb, 2))
    if ticks[-1] != nb - 1:
        ticks.append(nb - 1)
    ax.set_xticks([t + 0.5 for t in ticks], [f"{d['sigma_cols'][t]:.2f}" for t in ticks])
    ax.tick_params(axis="x", length=2.0, width=0.5, pad=1.5, labelsize=FS_NUM, colors=th["ink"])
    ax.set_xlabel(r"noise level $\sigma$", fontsize=FS_LAB, labelpad=1.5, color=th["ink"])


def letters(fig, th, items):
    for x, y, letter in items:
        fig.text(fx(x), fy(y), letter, ha="left", va="baseline", fontsize=sf.FS_PANEL, fontweight="bold",
                 color=th["ink"])


def draw(fig, d, th, variant: str):
    y_top = H - 0.13
    tile_x0 = 0.19
    T, rows = draw_tiles(fig, d, th, tile_x0, y_top, 0.03)
    mx0 = 1.43
    if variant == "a":
        mw, my0 = 2.30, 0.30
        mh = y_top - my0
        ax = draw_matrix(fig, d, th, mx0, my0, mw, mh)
        sigma_ticks(ax, d, th)
        cx0 = mx0 + mw + 0.08
        draw_colorbar(fig, th, cx0, my0, mh)
        vx0 = cx0 + 0.53
        draw_var_horizontal(fig, d, th, vx0, my0 + mh * 0.45, W - vx0 - 0.02, "variance of\nthe share")
    else:
        mw = 2.45
        sy0, sh = 0.30, 0.08
        my0 = sy0 + sh + 0.045
        mh = y_top - my0
        ax = draw_matrix(fig, d, th, mx0, my0, mw, mh)
        sigma_ticks(draw_strip(fig, d, th, mx0, sy0, mw, sh, share=ax), d, th)
        ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
        rx0, rw = mx0 + mw + 0.52, 0.86
        draw_colorbar_h(fig, th, rx0, my0 + mh - 0.30, rw)
        draw_var_horizontal(fig, d, th, rx0, my0 + mh * 0.30, rw, "variance of the share")
    leaders(fig, d, th, tile_x0 + T, rows, mx0, my0, mh)
    letters(fig, th, ((0.0, y_top + 0.035, "a"), (mx0 - 0.27, y_top + 0.035, "b")))


def render(d, variant: str, theme: str = "light"):
    th = THEMES[theme]
    sf.use_style()
    sf.script_floor(FS_NUM)
    mpl.rcParams.update({"text.color": th["ink"]})
    fig = plt.figure(figsize=(W, H))
    draw(fig, d, th, variant)
    return fig


def words(fig) -> tuple[list[str], int]:
    """Every visible text string in the figure, and the number of words in it (numbers and math tags excluded)."""
    fig.canvas.draw()
    out = []
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text().strip()
        if s and t.get_visible() and t.get_window_extent().width > 0:
            out.append(s.replace("\n", " "))
    n = sum(1 for s in out for w in s.split() if re.search(r"[A-Za-z]{2,}", w) and not w.startswith("$"))
    return out, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", choices=("a", "b"), default=KEPT,
                    help="the candidate written as observation.pdf/.svg, the web files and the provenance")
    a = ap.parse_args()
    d = load()
    CAND.mkdir(parents=True, exist_ok=True)
    cand = {}
    for v in ("a", "b"):
        fig = render(d, v)
        p = CAND / f"observation_{v}.pdf"
        fig.savefig(p, dpi=400)
        cand[v] = p
        plt.close(fig)
    fig = render(d, a.variant)
    stem = FIGS / "observation"
    fig.savefig(f"{stem}.pdf", dpi=400)
    fig.savefig(f"{stem}.svg", dpi=400)
    texts, n_words = words(fig)
    import orx_figstyle as base
    audit = base._audit(fig) + base._text_collisions(fig)
    small = min(t.get_fontsize() for t in fig.findobj(mpl.text.Text) if t.get_text().strip())
    if small < sf.MIN_PT:
        audit.append(f"text at {small} pt, below the {sf.MIN_PT} pt floor")
    print("figure audit:", "; ".join(audit) if audit else "clean", f"(smallest text {small} pt)")
    plt.close(fig)
    web = []
    WEB.mkdir(parents=True, exist_ok=True)
    for name in THEMES:
        fig = render(d, a.variant, name)
        for ext in ("svg", "png"):
            p = WEB / f"observation_{name}.{ext}"
            fig.savefig(p, dpi=300, transparent=True)
            web.append(p)
        Image.open(WEB / f"observation_{name}.png").save(WEB / f"observation_{name}.webp", "WEBP", quality=90,
                                                         method=6)
        web.append(WEB / f"observation_{name}.webp")
        plt.close(fig)
    M = d["M"]
    prov = {
        "figure": "fig:observation", "script": rel(Path(__file__)), "variant": a.variant,
        "size_in": [W, H],
        "inputs": {rel(p): sha256(p) for p in (CALIB, VARSPLIT, HEADLINE, AF_NPZ, AF_IMG)},
        "rules": {
            "share": "mass_mean[radii.index(16) + 1] / clip(1 - mass_mean[0], 1e-6), as scripts/variance_split.py",
            "bins": f"populated and sigma_centers < {GATE} (post-gate), columns ordered by sigma descending",
            "rows": "57 blocks x 24 heads, sorted by the mean share over the post-gate bins, descending (stable)",
            "variance": "variance_split.csv row flux1_dev, 1024x1024, post_gate, image_share, radius 16; head = "
                        "1 - noise_within_head, noise level = noise_within_head, shared curve = head_avg_curve; "
                        "recomputed from the archive (|diff| <= 1e-6) and checked half-up against headline.tex",
            "panel_a": "fig_attn_focus.py full-attention rendering of results/attn_focus/attn_focus.npz on "
                       "paper/figures/attn_focus/image_1024.png; window = emulated r = 4 window",
            "colour": "scafig.PI_CMAP, 0 (far, blue) to 1 (near, amber)",
        },
        "values": {
            "n_heads": int(d["n_heads"]), "n_bins": int(M.shape[1]),
            "sigma_columns": [round(float(s), 6) for s in d["sigma_cols"]],
            "variance_percent_printed": d["pct"],
            "noise_within_head": d["noise"], "head_avg_curve": d["curve_share"],
            "head_avg_curve_values": [round(float(v), 6) for v in d["curve"]],
            "row_mean_share": {"min": float(d["row_mean"].min()), "median": float(np.median(d["row_mean"])),
                               "max": float(d["row_mean"].max())},
            "row_range_over_bins": {"median": float(np.median(d["row_range"])),
                                    "p90": float(np.quantile(d["row_range"], 0.9)),
                                    "max": float(d["row_range"].max())},
            "matrix_min": float(M.min()), "matrix_max": float(M.max()),
            "panel_a": {"block_one_based": int(d["block"]) + 1,
                        "heads_one_based": [int(h) + 1 for h in d["heads"]],
                        "m_i": {t: round(float(d[f"{t}_m"]), 6) for t in ("near", "far")},
                        "sigma_of_map": float(d["sigma"]),
                        "rank_one_based_of_1368": {t: d["ranks"][t] + 1 for t in ("near", "far")},
                        "row_share": {t: [round(float(v), 6) for v in d["head_rows"][t]] for t in ("near", "far")}},
        },
        "text_in_figure": texts,
        "words_in_figure": n_words,
        "audit": audit or "clean",
        "outputs": {rel(p): sha256(p) for p in
                    [cand["a"], cand["b"], Path(f"{stem}.pdf"), Path(f"{stem}.svg")] + web},
    }
    (FIGS / "observation_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print("drew", rel(Path(f"{stem}.pdf")), "variant", a.variant, f"| {n_words} words:", " / ".join(texts))


if __name__ == "__main__":
    main()
