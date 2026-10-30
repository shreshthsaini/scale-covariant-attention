"""Heads view: the two key sets SCA reads at 4096^2, and the reach each head showed in calibration.

(a) A curated SCA output at 4096^2: the final operator's image (sca_v2, out/dupfix/sca_v2) of the
    Figure 1 strip prompt, minus8343371271909676658, seed 0; its run read this very pi table, and
    imgsel.final_twin checks it against the published image of the same prompt and seed. One query
    tile of 8x8 tokens (128 px, dark outline), its local key window of 64x64 tokens (1024 px, amber;
    the window from the fifth step after the gate on; the four steps before use 128, 112, 96 and 80
    tokens; centred on the tile and clamped to the image as in sca/attn.py window_index) and the global branch's pooled keys (blue): centre pooling
    keeps token 4j+2 of every 4x4-token block, pixel centre 64j + 40. Every KEY_STEP-th pooled key
    per axis is drawn (j = KEY_OFFSET, KEY_OFFSET + KEY_STEP, ...).
(b) Measured reach of three heads at sigma bin BIN: the same image stands in for the 1024^2 native
    view (64x64 tokens). Nested squares at Chebyshev radii 2, 4, 8, 16 tokens around a centred
    query; each ring's amber opacity is linear in its mass density (share of image attention mass
    in the ring over its share of the 4096 image tokens, so 1 = uniform), saturating at
    DENSITY_CAP. Each thumbnail is labelled with its share within 16 tokens.
(c) pi_h(sigma) of those heads in the scale-2 table, with the median and interquartile range of
    all 1368 heads at every populated bin; sigma > sigma_g = 0.89 runs the global branch only.
(d) pi_h against the share within 16 tokens for all 1368 heads at bin BIN, with Pearson r.

Heads: among heads that put at least IMG_FLOOR of their mass on image keys at bin BIN, rank pi
and share (mid-ranks scaled to (0, 1)) and take the head nearest (q, q) in rank space, with
q = 0.90 (local-leaning), 0.20 (mid, pi near 0.5) and 0.05 (global-leaning).

Data (CPU only): results/evidence/calib/k1_1024.npz, FLUX.1-dev calibration at 1024^2 (57 blocks x
24 heads x 14 noise bins, 32 prompts). mass_mean [5, 57, 24, 14] holds the full-attention softmax
mass on text keys (index 0) and within Chebyshev radius 2, 4, 8, 16 tokens of the query (1..4),
averaged over sampled queries, prompts and steps (sca/calib.py CalibStats.add_mass). The share
within r tokens is mass_mean[r] / (1 - mass_mean[0]). The script writes results/heads_view/*.csv
from the archive, then draws every number and curve from those CSVs alone. With --no-results-csv it
writes them to a temporary directory instead, requires them to be byte-identical to results/heads_view,
and draws from results/heads_view (nothing is written into results/).

Regenerate: <sca env python> paper/figures/src/fig_heads_view.py [--no-results-csv]
"""
from __future__ import annotations

import csv
import hashlib
import argparse
import filecmp
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical PDF on rerun
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath
from PIL import Image, ImageEnhance
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imgsel  # noqa: E402
import scafig as sf  # noqa: E402

ROOT = HERE.parents[2]
NPZ = ROOT / "results/evidence/calib/k1_1024.npz"
RES = ROOT / "results/heads_view"
OUT = ROOT / "paper/figures/heads_view"
CACHE = Path("/scratch/09032/saini_2/sca/figcache/heads_view")  # regenerable thumbnails
PID = "minus8343371271909676658"
METHOD, RESOLUTION = "sca_v2", "4096x4096"  # the final operator (imgsel.final_twin of main/sca_nb)

# calibration slice
SCALE = 2                 # the pi table drawn in (c) and (d)
BIN = 8                   # sigma bin of (b) and (d); centre 0.618
BINS = (4, 8, 11)         # bins written to the CSVs
RADII = (2, 4, 8, 16)     # Chebyshev radii in sca/calib.py
SIGMA_G = 0.89            # gate: sigma > SIGMA_G runs the global branch only
IMG_FLOOR = 0.5           # a selectable head puts at least half of its mass on image keys
TARGETS = (("local", 0.90), ("mid", 0.20), ("global", 0.05))
NATIVE = 64               # 1024^2 native grid side in tokens
DENSITY_CAP = 12.0        # ring opacity saturates at 12x the uniform density
ALPHA_MAX = 0.9

# 4096^2 geometry of the SCA run (sidecar: grid 256x256, tile 8, window 64, centre pool 4x4)
RES_PX, GRID = 4096, 256
PX = RES_PX // GRID       # 16 px per token
TILE, WINDOW, POOL = 8, 64, 4
TILE_RC = (12, 14)        # query tile (row, column) in tile units, on the church tower
KEY_OFFSET, KEY_STEP = 1, 4

W, H = sf.TEXT, 1.72      # 5.5 in by 1.72 in
COL = {"local": sf.LOCAL, "mid": sf.INK, "global": sf.GLOBAL}
DOT_GREY = "#A9B1B9"
BAND_GREY = "#ECEEF0"
SAT_A, VEIL_A = 0.6, 0.12     # photo in (a): partly desaturated, light white veil
SAT_B, VEIL_B = 0.0, 0.35     # thumbnails in (b): grey, lighter, so the amber reads


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f6(x) -> str:
    return f"{float(x):.6f}"


def write_csv(name: str, header: list[str], rows: list[list]) -> Path:
    path = RES / name
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def read_csv(name: str) -> list[dict]:
    with open(RES / name, newline="") as fh:
        return list(csv.DictReader(fh))


def window_start(t: int) -> int:
    """First token of the local key window of query tile t (sca/attn.py window_index)."""
    return int(np.clip(t * TILE + TILE // 2 - WINDOW // 2, 0, GRID - WINDOW))


# ----------------------------------------------------------------------------- archive -> CSVs
def compute() -> dict:
    z = np.load(NPZ)
    assert list(z["radii"]) == list(RADII) and list(z["scales"]) == [2, 4]
    m, pop, sig = z["mass_mean"], z["populated"], z["sigma_centers"]
    mn = z["mass_n"]
    assert np.allclose(m, z["mass"] / np.where(mn > 0, mn, 1.0)[None, :, None, :], rtol=1e-12, atol=0)
    pi = {2: z["s2_pi_head"], 4: z["s4_pi_head"]}
    nb, nh, _ = pi[2].shape
    text = m[0]
    img = 1.0 - text
    share = m[1:] / img[None]                           # [4 radii, blocks, heads, bins]
    assert np.all(np.diff(share, axis=0) >= -1e-12) and share.max() <= 1 + 1e-9
    RES.mkdir(parents=True, exist_ok=True)

    rows = []
    for b in BINS:
        for blk in range(nb):
            for h in range(nh):
                rows.append([blk, h, b, f6(sig[b]), f6(pi[2][blk, h, b]), f6(pi[4][blk, h, b])]
                            + [f6(share[k, blk, h, b]) for k in range(len(RADII))] + [f6(text[blk, h, b])])
    write_csv("pi_vs_mass.csv", ["block", "head", "bin", "sigma", "pi_s2", "pi_s4"]
              + [f"share_r{r}" for r in RADII] + ["text_mass"], rows)

    # three heads in rank space among heads with real image mass
    p, s16, im = pi[SCALE][:, :, BIN], share[3][:, :, BIN], img[:, :, BIN]
    elig = im >= IMG_FLOOR
    n = int(elig.sum())
    rp, rs = np.full(p.shape, np.nan), np.full(p.shape, np.nan)
    rp[elig] = (rankdata(p[elig]) - 0.5) / n
    rs[elig] = (rankdata(s16[elig]) - 0.5) / n
    heads = {}
    for role, q in TARGETS:
        d = np.where(elig, np.hypot(rp - q, rs - q), np.inf)
        b, h = np.unravel_index(int(np.argmin(d)), d.shape)
        heads[role] = (int(b), int(h))

    # (c) curves at the populated bins
    idx = np.flatnonzero(pop)
    flat = pi[SCALE].reshape(nb * nh, -1)
    q25, med, q75 = np.quantile(flat, [0.25, 0.5, 0.75], axis=0)
    rows = [[int(i), f6(sig[i])] + [f6(pi[SCALE][heads[r][0], heads[r][1], i]) for r, _ in TARGETS]
            + [f6(med[i]), f6(q25[i]), f6(q75[i])] for i in idx]
    write_csv("pi_sigma.csv", ["bin", "sigma"] + [f"pi_{r}" for r, _ in TARGETS]
              + ["median_all", "q25_all", "q75_all"], rows)

    # (b) ring densities of the three heads
    tokens = np.diff(np.r_[0, [(2 * r + 1) ** 2 for r in RADII]])   # 25, 56, 208, 800
    rows = []
    for role, _ in TARGETS:
        b, h = heads[role]
        ring = np.diff(np.r_[0.0, share[:, b, h, BIN]])
        dens = ring / (tokens / NATIVE ** 2)
        alpha = np.minimum(1.0, dens / DENSITY_CAP) * ALPHA_MAX
        r_in = [0] + [r + 1 for r in RADII[:-1]]
        for k, r in enumerate(RADII):
            rows.append([role, b, h, BIN, k + 1, r_in[k], r, int(tokens[k]), f6(ring[k]), f6(dens[k]), f6(alpha[k])])
    write_csv("rings.csv", ["role", "block", "head", "bin", "ring", "cheb_from", "cheb_to", "tokens",
                            "share", "density_vs_uniform", "opacity"], rows)

    # summary: Pearson r per scale, radius and bin (all heads, and heads with image mass >= floor)
    rows = []
    for s in (2, 4):
        for k, r in enumerate(RADII):
            for b in BINS:
                x, y = share[k][:, :, b].ravel(), pi[s][:, :, b].ravel()
                el = (img[:, :, b] >= IMG_FLOOR).ravel()
                rows.append(["pearson", "", "pearson_r", s, r, b, f6(sig[b]), "", "", x.size,
                             f6(np.corrcoef(x, y)[0, 1])])
                rows.append(["pearson_img_ge_0.5", "", "pearson_r", s, r, b, f6(sig[b]), "", "",
                             int(el.sum()), f6(np.corrcoef(x[el], y[el])[0, 1])])
    for role, q in TARGETS:
        b, h = heads[role]
        vals = {"pi_s2": pi[2][b, h, BIN], "pi_s4": pi[4][b, h, BIN]}
        vals.update({f"share_r{r}": share[k, b, h, BIN] for k, r in enumerate(RADII)})
        vals.update(text_mass=text[b, h, BIN], rank_pi_s2=rp[b, h], rank_share_r16=rs[b, h], rank_target=q)
        for key, v in vals.items():
            rows.append(["head", role, key, SCALE if key.startswith("rank") else "", "", BIN, f6(sig[BIN]),
                         b, h, n if key.startswith("rank") else "", f6(v)])
    write_csv("summary.csv", ["kind", "role", "metric", "scale", "radius", "bin", "sigma", "block", "head",
                              "n", "value"], rows)

    names = ["pi_sigma.csv", "pi_vs_mass.csv", "rings.csv", "summary.csv"]
    (RES / "SHA256SUMS").write_text("".join(f"{sha(RES / k)}  {k}\n" for k in names))
    return dict(heads=heads, eligible=n)


# ----------------------------------------------------------------------------- CSVs -> plot data
def load() -> dict:
    pvm = [r for r in read_csv("pi_vs_mass.csv") if int(r["bin"]) == BIN]
    x = np.array([float(r["share_r16"]) for r in pvm])
    y = np.array([float(r[f"pi_s{SCALE}"]) for r in pvm])
    summ = read_csv("summary.csv")
    r16 = [r for r in summ if r["kind"] == "pearson" and int(r["scale"]) == SCALE and int(r["radius"]) == 16
           and int(r["bin"]) == BIN]
    assert len(r16) == 1 and len(x) == int(r16[0]["n"]) == 1368
    r = float(r16[0]["value"])
    assert abs(np.corrcoef(x, y)[0, 1] - r) < 1e-4, "r from the CSV rows differs from summary.csv"
    heads = {}
    for row in summ:
        if row["kind"] == "head":
            hd = heads.setdefault(row["role"], dict(block=int(row["block"]), head=int(row["head"])))
            hd[row["metric"]] = float(row["value"])
    curves = read_csv("pi_sigma.csv")
    cv = {k: np.array([float(c[k]) for c in curves]) for k in curves[0]}
    rings = {}
    for row in read_csv("rings.csv"):
        rings.setdefault(row["role"], []).append(float(row["opacity"]))
    return dict(x=x, y=y, r=r, heads=heads, curves=cv, rings=rings)


# ----------------------------------------------------------------------------- source image
def source() -> tuple[Path, dict]:
    png = Path(imgsel.final_png(RESOLUTION, PID)[0])  # checked: twin of the published image, sca_v2 flags
    meta = json.loads(png.with_suffix(".json").read_text())
    assert meta["config"]["method"] == METHOD, meta["config"]["method"]
    if meta.get("complete") is not True or meta.get("png_sha256") != sha(png):
        raise ValueError(f"incomplete or hash-mismatched source {png}")
    c = meta["config"]["sca"]
    assert meta["config"]["grid"] == [GRID, GRID] and c["tile"] == TILE and c["window"] == WINDOW
    assert c["pool"] == "center" and c["pool_factors"] == [POOL, POOL] and c["gate_sigma"] == SIGMA_G
    assert c["pi_table_sha256"] == sha(NPZ), "the image was not generated with this calibration table"
    return png, meta


def thumb(img: Image.Image, px: int, sat: float, veil: float, name: str) -> tuple[np.ndarray, Path]:
    t = ImageEnhance.Color(img.resize((px, px), Image.LANCZOS)).enhance(sat)
    a = np.asarray(t, dtype=np.float64) * (1 - veil) + 255.0 * veil
    out = Image.fromarray(np.round(a).astype(np.uint8))
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}_{px}px.png"
    out.save(path)
    return np.asarray(out), path


# ----------------------------------------------------------------------------- figure
def annulus(q: float, r_out: int, r_in: int | None) -> MPath:
    """Square ring of token cells with Chebyshev distance in (r_in, r_out] around cell (q, q)."""
    def square(r, reverse):
        a, b = q - r, q + r + 1
        v = [(a, a), (b, a), (b, b), (a, b), (a, a)]
        return v[::-1] if reverse else v
    verts, codes = square(r_out, False), [MPath.MOVETO] + [MPath.LINETO] * 3 + [MPath.CLOSEPOLY]
    if r_in is not None:
        verts += square(r_in, True)
        codes += [MPath.MOVETO] + [MPath.LINETO] * 3 + [MPath.CLOSEPOLY]
    return MPath(verts, codes)


def letter(fig, x_in, y_in, ch, title):
    fig.text(x_in / W, y_in / H, ch, fontsize=sf.FS_PANEL, fontweight="bold", va="baseline", color=sf.INK)
    fig.text((x_in + 0.11) / W, y_in / H, title, fontsize=sf.FS_TICK, va="baseline", color=sf.INK)


def plot(d: dict, img_a: np.ndarray, img_b: np.ndarray) -> tuple:
    sf.use_style()
    sf.script_floor(sf.FS)  # the h of pi_h in the 8 pt axis label prints at 6 pt, not 5.6
    top = H - 0.17                       # top edge of every panel
    ya = 0.02
    sa = top - ya                        # (a) square side
    xa = 0.0
    gap_t = 0.025                        # 1.8 pt white gutters between thumbnails
    st = (sa - 2 * gap_t) / 3            # thumbnail side
    xb = xa + sa + 0.09
    xc, wc = 2.84, 1.27                  # (c) axes
    xd = xc + wc + 0.12                  # (d) axes, y shared with (c)
    wd = W - 0.06 - xd
    yc = 0.31
    hc = top - yc
    fig = plt.figure(figsize=(W, H))
    y_letter = top + 0.055

    # (a) the two key sets on the 4096^2 output
    ax = fig.add_axes([xa / W, ya / H, sa / W, sa / H])
    ax.imshow(img_a, extent=(0, RES_PX, RES_PX, 0), interpolation="none", zorder=0)
    ax.set_xlim(0, RES_PX)
    ax.set_ylim(RES_PX, 0)
    ax.axis("off")
    centres = (POOL * np.arange(KEY_OFFSET, RES_PX // (POOL * PX), KEY_STEP) + POOL // 2) * PX + PX / 2
    gx, gy = np.meshgrid(centres, centres)
    ax.scatter(gx.ravel(), gy.ravel(), s=2.6, color=sf.GLOBAL, edgecolors="white", linewidths=0.3, zorder=3)
    tr, tc = TILE_RC
    wr, wc_ = window_start(tr), window_start(tc)
    ax.add_patch(Rectangle((wc_ * PX, wr * PX), WINDOW * PX, WINDOW * PX, fill=False, ec=sf.LOCAL, lw=1.0,
                           zorder=4))
    halo = [patheffects.withStroke(linewidth=1.3, foreground="white")]
    ax.add_patch(Rectangle((tc * TILE * PX, tr * TILE * PX), TILE * PX, TILE * PX, fill=False, ec=sf.INK,
                           lw=0.65, zorder=5, path_effects=halo))
    letter(fig, xa, y_letter, "a", "keys at 4096²")

    # (b) three thumbnails, local-leaning on top
    roles = [r for r, _ in TARGETS]
    q = NATIVE // 2
    for k, role in enumerate(roles):
        y0 = ya + (2 - k) * (st + gap_t)
        tx = fig.add_axes([xb / W, y0 / H, st / W, st / H])
        tx.imshow(img_b, extent=(0, NATIVE, NATIVE, 0), interpolation="none", zorder=0)
        tx.set_xlim(0, NATIVE)
        tx.set_ylim(NATIVE, 0)
        tx.axis("off")
        for j, r in enumerate(RADII):
            tx.add_patch(PathPatch(annulus(q, r, RADII[j - 1] if j else None), fc=sf.LOCAL,
                                   alpha=d["rings"][role][j], lw=0, zorder=2))
            tx.add_patch(Rectangle((q - r, q - r), 2 * r + 1, 2 * r + 1, fill=False, ec=sf.MID, lw=0.3,
                                   alpha=0.8, zorder=3))
        tx.plot([q + 0.5], [q + 0.5], "o", ms=1.6, mfc=sf.INK, mec="none", zorder=4)
        ym = (y0 + st / 2) / H
        fig.add_artist(Line2D([(xb + st + 0.06) / W], [ym], marker="o", ms=4.0, mfc=COL[role], mec="white",
                              mew=0.5, ls="none", transform=fig.transFigure))
        fig.text((xb + st + 0.11) / W, ym, f"{d['heads'][role]['share_r16']:.2f}", fontsize=sf.FS_TICK,
                 va="center", ha="left", color=sf.INK)
    letter(fig, xb, y_letter, "b", "measured reach")

    # (c) pi(sigma)
    cx = fig.add_axes([xc / W, yc / H, wc / W, hc / H])
    cv = d["curves"]
    s = cv["sigma"]
    cx.axvspan(SIGMA_G, 1.0, color=BAND_GREY, lw=0, zorder=0)
    cx.fill_between(s, cv["q25_all"], cv["q75_all"], color=sf.SCA_LIGHT, lw=0, zorder=1)
    cx.plot(s, cv["median_all"], color=sf.SCA, lw=1.4, zorder=3, solid_capstyle="round")
    i8 = list(cv["bin"]).index(BIN)
    for role in roles:
        cx.plot(s, cv[f"pi_{role}"], color=COL[role], lw=1.1, zorder=4, solid_capstyle="round")
        cx.plot([s[i8]], [cv[f"pi_{role}"][i8]], "o", ms=3.4, mfc=COL[role], mec="white", mew=0.5, zorder=5)
    cx.set_xlim(0, 1)
    cx.set_ylim(-0.03, 1.09)
    cx.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    cx.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    cx.set_xlabel(r"noise level $\sigma$")
    cx.set_ylabel(r"local weight $\pi_h$")
    for side in ("top", "right"):
        cx.spines[side].set_visible(False)
    lab = dict(fontsize=sf.FS_TICK, ha="left")
    cx.text(0.12, 0.975, "local", color=sf.LOCAL, va="bottom", **lab)
    cx.text(0.46, 0.715, "median", color=sf.SCA, va="top", **lab)
    cx.text(0.12, 0.47, "mid", color=sf.INK, va="top", **lab)
    cx.text(0.20, 0.13, "global", color=sf.GLOBAL, va="bottom", **lab)
    letter(fig, xc, y_letter, "c", "calibrated π")

    # (d) pi against the share within 16 tokens
    dx = fig.add_axes([xd / W, yc / H, wd / W, hc / H], sharey=cx)
    dx.scatter(d["x"], d["y"], s=2.2, color=DOT_GREY, lw=0, alpha=0.75, zorder=2)
    for role in roles:
        hd = d["heads"][role]
        dx.plot([hd["share_r16"]], [hd[f"pi_s{SCALE}"]], "o", ms=4.0, mfc=COL[role], mec="white", mew=0.6,
                zorder=4)
    dx.set_xlim(0, 1)
    dx.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    dx.tick_params(axis="y", labelleft=False)
    dx.set_xlabel("share within 16 tokens")
    for side in ("top", "right"):
        dx.spines[side].set_visible(False)
    dx.text(0.97, 0.03, f"r = {d['r']:.2f}", transform=dx.transAxes, ha="right", va="bottom",
            fontsize=sf.FS_TICK, color=sf.INK)
    letter(fig, xd, y_letter, "d", "π vs. reach")

    geometry = dict(
        query_tile_rc=list(TILE_RC),
        query_tile_px_xywh=[tc * TILE * PX, tr * TILE * PX, TILE * PX, TILE * PX],
        local_window_px_xywh=[wc_ * PX, wr * PX, WINDOW * PX, WINDOW * PX],
        pooled_key_index_per_axis=f"j = {KEY_OFFSET}, {KEY_OFFSET + KEY_STEP}, ..., "
                                  f"{int(np.arange(KEY_OFFSET, RES_PX // (POOL * PX), KEY_STEP)[-1])} "
                                  f"of 0..{RES_PX // (POOL * PX) - 1} (every {KEY_STEP}th)",
        pooled_key_px_centres=[float(c) for c in centres],
        keys_drawn=int(centres.size ** 2), keys_total=int((RES_PX // (POOL * PX)) ** 2))
    return fig, geometry


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-results-csv", action="store_true",
                    help="write the CSVs to a temporary directory, require them to equal results/heads_view, "
                         "and draw from results/heads_view")
    args = ap.parse_args()
    if args.no_results_csv:
        kept = RES
        with tempfile.TemporaryDirectory() as tmp:
            RES = Path(tmp)
            sel = compute()
            names = ["pi_sigma.csv", "pi_vs_mass.csv", "rings.csv", "summary.csv", "SHA256SUMS"]
            same = [filecmp.cmp(RES / k, kept / k, shallow=False) for k in names]
            RES = kept
        assert all(same), f"results/heads_view differs from a fresh computation: {dict(zip(names, same))}"
    else:
        sel = compute()
    d = load()
    png, meta = source()
    img = Image.open(png).convert("RGB")
    assert img.size == (RES_PX, RES_PX)
    img_a, path_a = thumb(img, 768, SAT_A, VEIL_A, "a_church")
    img_b, path_b = thumb(img, 256, SAT_B, VEIL_B, "b_church")
    fig, geometry = plot(d, img_a, img_b)
    sf.save(fig, str(OUT), formats=("pdf",))
    cfg = meta["config"]
    provenance = dict(
        figure="paper/figures/heads_view.pdf", script="paper/figures/src/fig_heads_view.py",
        printed_size_in=[W, H], printed_size_texpt=[round(W * 72.27, 1), round(H * 72.27, 1)],
        calibration="FLUX.1-dev at 1024^2 (64x64 tokens), 57 blocks x 24 heads x 14 sigma bins, 32 prompts; "
                    "results/evidence/calib/k1_1024.npz",
        share_definition="share within r tokens = mass_mean[r] / (1 - mass_mean[0]): full-attention softmax mass "
                         "within Chebyshev radius r of the query over the mass on image keys",
        slice=dict(bin=BIN, sigma_centre=float(d["curves"]["sigma"][list(d["curves"]["bin"]).index(BIN)]),
                   pi_scale=SCALE),
        panel_a=dict(image=str(png), png_sha256=meta["png_sha256"], prompt_id=cfg["prompt_id"],
                     prompt=cfg["prompt"], method=cfg["method"], res=cfg["res"], seed=cfg["seed"],
                     steps=cfg["steps"], guidance=cfg["guidance"], sca=cfg["sca"],
                     used_in="Figure 1 bottom strip, SCA (ours) column (fig_hero.py STRIP_PID)",
                     operator=("final operator (sca_v2): the drawn 64-token window applies from the fifth step "
                               "after the gate; the four steps before use windows of 128, 112, 96 and 80 tokens"),
                     win_sched=cfg["sca"].get("win_sched"),
                     shown_as=f"downscaled to 768 px (LANCZOS), colour saturation x{SAT_A}, white veil {VEIL_A}",
                     **geometry),
        panel_b=dict(thumbnail="same image, 256 px, greyscale, white veil "
                               f"{VEIL_B}; stands in for the 1024^2 native view of 64x64 tokens",
                     query_token=[NATIVE // 2, NATIVE // 2], radii_tokens=list(RADII),
                     opacity=f"alpha = {ALPHA_MAX} * min(1, density / {DENSITY_CAP}), density = ring share / "
                             "(ring tokens / 4096), nominal (2r+1)^2 squares; label = share within 16 tokens",
                     note="measured masses average over sampled queries, including queries whose squares the "
                          "image border clips; the drawn rings use nominal areas"),
        panel_c="pi_h(sigma) of the three heads (scale-2 table) at the 13 populated bin centres, joined linearly; "
                "green = median of all 1368 heads, band = interquartile range; grey = sigma > 0.89 (global "
                "branch only at inference); dots mark the bin of panel (d)",
        panel_d=f"all 1368 heads at bin {BIN}: x = share within 16 tokens, y = pi_h (scale {SCALE}); Pearson r",
        selection=dict(rule="heads with image mass >= 0.5 at the bin; mid-ranks of pi and share scaled to (0,1); "
                            "nearest head to (q, q) in rank space",
                       targets=dict(TARGETS), eligible_heads=sel["eligible"], heads=d["heads"]),
        pearson_r_drawn=d["r"],
        csv_sha256={k: sha(RES / k) for k in ("pi_sigma.csv", "pi_vs_mass.csv", "rings.csv", "summary.csv")},
        sources={"results/evidence/calib/k1_1024.npz": sha(NPZ), str(png): sha(png),
                 str(png.with_suffix(".json")): sha(png.with_suffix(".json"))},
        cache={str(path_a): sha(path_a), str(path_b): sha(path_b)})
    OUT.with_name(OUT.name + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for role, hd in d["heads"].items():
        print(f"{role:6s} block {hd['block']:2d} head {hd['head']:2d}  pi_s2 {hd['pi_s2']:.3f}  "
              f"share16 {hd['share_r16']:.3f}  text {hd['text_mass']:.3f}  opacity {d['rings'][role]}")
    print(f"r = {d['r']:.4f}")


if __name__ == "__main__":
    main()
