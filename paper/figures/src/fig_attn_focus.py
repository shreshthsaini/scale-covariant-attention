"""Attention focus (fig:focus): what two heads of FLUX.1-dev look at, and the two views SCA gives them.

(a) One query token on a real 1024^2 FLUX.1-dev image (the native stage of progressive SCA for the Aesthetic-4K prompt
    a4k_eval_4096_minus8343371271909676658, seed 0, re-run on CPU by scripts/attn_focus.py) at step 20 of HRDiT's
    30-step schedule, sigma = 0.612, inside the reach table's noise bin centred at 0.618 (the bin of Figure 2c).
    Rows: two heads of block 12 (one-based), the block of Figure 2c: the heads with the highest and the lowest share of
    image attention within 16 tokens in the reach statistics (results/evidence/calib/k1_1024.npz, 32 prompts, that
    bin), heads 4 and 7. Columns: full attention over the image keys (Eq. 1), its near part inside the window of the
    emulated local operator at r = 4 (16 x 16 tokens around the query's tile) and its far part outside it, drawn on
    one scale so that near + far = full, with the near share m_i of the image attention (Lemma 1); then SCA's two
    views at r = 4: the local operator's attention inside the window (amber) and the global operator's attention over
    the 16 x 16 block-centre keys (blue dots, area proportional to weight), each drawn with an opacity that follows the
    head's fitted weight pi_h (s4_pi_head of the archive at that bin). Text keys are not drawn.
(b) Reach emulation on the same image: the 4096^2 target (progressive SCA's output for the same prompt and seed) with
    its native 64-token window and 64 x 64 block-centre grid, and the 1024^2 image with both views shrunk by r = 4.
(c) The variance split of the share within 16 tokens after the first 8 steps (results/heads_view/variance_split.csv,
    FLUX.1-dev, post_gate, radius 16): head 97.5%, noise level 2.5% (headline.tex \\hlVarHead, \\hlVarStep).

Colour: attention density relative to uniform over the 4,096 image tokens, log scale from 1x to DMAX x; amber inside
the window (near), blue outside (far), as everywhere in the paper (amber = local, blue = global).

Stages:
  --extract  read the capture (/scratch/09032/saini_2/sca/out/attn_focus/sheep_church_seed0) and write the small
             archive results/attn_focus/attn_focus.npz + summary.json and the two images in paper/figures/attn_focus/
  (default)  draw from results/attn_focus/ only: paper/figures/attn_focus.pdf/.svg, provenance, and the web files
             web/assets/img/figs/attn_focus_{light,dark}.{svg,png,webp}
Run with the sca env python on a compute node.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig as sf  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ROOT = HERE.parents[2]
FIGS = ROOT / "paper" / "figures"
ASSETS = FIGS / "attn_focus"
RES = ROOT / "results" / "attn_focus"
WEB = ROOT / "web" / "assets" / "img" / "figs"
ARCHIVE = ROOT / "results" / "evidence" / "calib" / "k1_1024.npz"
VARSPLIT = ROOT / "results" / "heads_view" / "variance_split.csv"
HEADLINE = FIGS / "snippets" / "main_tables" / "headline.tex"
CAPTURE = Path("/scratch/09032/saini_2/sca/out/attn_focus/sheep_church_seed0")
TARGET = Path("/scratch/09032/saini_2/sca/out/dupfix_prog/sca_v2_prog/4096x4096/"
              "a4k_eval_4096_minus8343371271909676658_0325f0a692_seed0.png")
PROMPT_ID = "a4k_eval_4096_minus8343371271909676658"
STEP, STEPS_EXTRA = 20, (8, 26)
QUERY = (43, 11)          # token (row, col): the face of the leftmost sheep
BLOCK = 11                # zero-based; block 12 one-based, the block of Figure 2c
BIN = 8                   # reach-table noise bin centred at sigma = 0.618
RATIO = 4
DMAX = 100.0              # colour saturates at 100 x the uniform density
N_SIDE = 64


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------- extract
def extract() -> None:
    import torch
    from attn_focus_data import Capture
    torch.set_num_threads(24)
    z = np.load(ARCHIVE)
    mm = z["mass_mean"][:, BLOCK, :, BIN]                     # [5, 24]: text mass, then radii 2, 4, 8, 16
    share16 = mm[4] / (1 - mm[0])
    near_head, far_head = int(np.argmax(share16)), int(np.argmin(share16))
    heads = (near_head, far_head)
    pi4 = z["s4_pi_head"][BLOCK, :, BIN]
    centre = float(z["sigma_centers"][BIN])
    out = {"heads": np.array(heads), "share16": share16[list(heads)], "pi_r4": pi4[list(heads)],
           "sigma_centre": np.array(centre), "query": np.array(QUERY), "ratio": np.array(RATIO),
           "block": np.array(BLOCK)}
    cap = Capture(CAPTURE, STEP)
    out["sigma"] = np.array(cap.sigma)
    r0, c0, w, t = cap.window(*QUERY, RATIO)
    out["window"] = np.array([r0, c0, w, t])
    for tag, h in zip(("near", "far"), heads):
        full, txt = cap.full(BLOCK, h, *QUERY)
        loc, ltxt = cap.local(BLOCK, h, *QUERY, RATIO)
        glo, gtxt = cap.global_(BLOCK, h, *QUERY, RATIO)
        _, _, m = cap.split(full, *QUERY, RATIO)
        out[f"{tag}_full"], out[f"{tag}_text"], out[f"{tag}_m"] = full, np.array(txt), np.array(m)
        out[f"{tag}_local"], out[f"{tag}_local_text"] = loc, np.array(ltxt)
        out[f"{tag}_global"], out[f"{tag}_global_text"] = glo, np.array(gtxt)
    del cap
    extra = {}
    for j in STEPS_EXTRA:
        c2 = Capture(CAPTURE, j)
        for tag, h in zip(("near", "far"), heads):
            full, _ = c2.full(BLOCK, h, *QUERY)
            extra[(tag, j)] = (c2.sigma, c2.split(full, *QUERY, RATIO)[2])
        del c2
    out["extra_steps"] = np.array(STEPS_EXTRA)
    out["extra_sigma"] = np.array([extra[("near", j)][0] for j in STEPS_EXTRA])
    out["near_m_extra"] = np.array([extra[("near", j)][1] for j in STEPS_EXTRA])
    out["far_m_extra"] = np.array([extra[("far", j)][1] for j in STEPS_EXTRA])
    RES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RES / "attn_focus.npz", **out)
    ASSETS.mkdir(parents=True, exist_ok=True)
    img = Image.open(CAPTURE / "image_1024.png").convert("RGB")
    img.save(ASSETS / "image_1024.png")
    tgt = Image.open(TARGET).convert("RGB")
    tgt.resize((1024, 1024), Image.LANCZOS).save(ASSETS / "target_4096_at1024.jpg", quality=93, subsampling=0)
    meta = json.loads((CAPTURE / "meta.json").read_text())
    summary = {
        "figure": "fig:focus", "script": "paper/figures/src/fig_attn_focus.py --extract",
        "capture": {"dir": str(CAPTURE), "meta": {k: meta[k] for k in ("prompt_id", "prompt", "seed", "derived_seed",
                                                                       "steps", "guidance", "mu", "capture_steps",
                                                                       "device", "code_revision")},
                    "image_1024_sha256": sha256(CAPTURE / "image_1024.png"),
                    "qk_files": {f"qk_step{j:02d}.pt": meta["files"][f"qk_step{j:02d}.pt"] for j in
                                 (STEP,) + STEPS_EXTRA}},
        "target_4096": {"path": str(TARGET), "sha256": sha256(TARGET)},
        "archive": {"path": str(ARCHIVE.relative_to(ROOT)), "sha256": sha256(ARCHIVE)},
        "rules": {"block": "block 12 one-based (zero-based 11), the block of Figure 2c",
                  "heads": "argmax and argmin over the block's 24 heads of the share of image attention within "
                           "16 tokens, mass_mean[4] / (1 - mass_mean[0]) at noise bin 8 (centre 0.618), 32 prompts",
                  "query": "token (row 43, col 11), pixel centre (184, 696): the face of the leftmost sheep",
                  "window": "emulated local window at r = 4: side 16 tokens around the query's 2 x 2 tile "
                            "(attn_focus_data.Capture.window, as sca/attn.py window_index)",
                  "split": "near = image keys inside the window, far = the other image keys; m = near share of the "
                           "image attention; text keys (both operators keep them) are not drawn"},
        "values": {"heads_one_based": [h + 1 for h in heads], "share16": [float(x) for x in out["share16"]],
                   "pi_r4": [float(x) for x in out["pi_r4"]], "sigma": float(out["sigma"]),
                   "sigma_bin_centre": centre, "m": {t: float(out[f"{t}_m"]) for t in ("near", "far")},
                   "text_mass": {t: float(out[f"{t}_text"]) for t in ("near", "far")},
                   "m_other_steps": {t: {f"{float(s):.4f}": float(v) for s, v in
                                         zip(out["extra_sigma"], out[f"{t}_m_extra"])} for t in ("near", "far")}},
        "outputs": {"results/attn_focus/attn_focus.npz": sha256(RES / "attn_focus.npz"),
                    "paper/figures/attn_focus/image_1024.png": sha256(ASSETS / "image_1024.png"),
                    "paper/figures/attn_focus/target_4096_at1024.jpg": sha256(ASSETS / "target_4096_at1024.jpg")},
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["values"], indent=1))


# ------------------------------------------------------------------------------------------------------- draw
THEMES = {
    "light": dict(ink=sf.INK, mid=sf.MID, near=sf.LOCAL, far=sf.GLOBAL, head="#3B4249", step="#C3C9CF",
                  bg_lo=0.30, bg_hi=0.80, tag_bg=(0.08, 0.09, 0.10, 0.62), tag_fg="white", rule=sf.RULE, face="white"),
    "dark": dict(ink="#E7EAE8", mid="#A3ACB3", near="#E39A47", far="#6E9FD6", head="#C3C9CF", step="#4A535B",
                 bg_lo=0.18, bg_hi=0.62, tag_bg=(0.02, 0.03, 0.03, 0.70), tag_fg="#E7EAE8", rule="#323A37",
                 face="#0D100F"),
}


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def grey_base(img: Image.Image, th, px: int) -> np.ndarray:
    L = np.asarray(img.convert("L").resize((px, px), Image.LANCZOS), dtype=np.float64) / 255
    return np.repeat((th["bg_lo"] + (th["bg_hi"] - th["bg_lo"]) * L)[..., None], 3, -1)


def heat(base: np.ndarray, dens: np.ndarray, color: np.ndarray, mask=None, fade: float = 1.0) -> np.ndarray:
    """Composite a token density map (x uniform) over base [px, px, 3]: alpha follows log density up to DMAX."""
    px = base.shape[0]
    v = np.clip(np.log10(np.maximum(dens, 1.0)) / np.log10(DMAX), 0, 1)
    if mask is not None:
        v = np.where(mask, v, 0.0)
    a = 0.95 * fade * v ** 0.75
    a = np.kron(a, np.ones((px // dens.shape[0], px // dens.shape[1])))[..., None]
    return base * (1 - a) + color[None, None, :] * a


def window_mask(win) -> np.ndarray:
    r0, c0, w, _ = (int(x) for x in win)
    m = np.zeros((N_SIDE, N_SIDE), dtype=bool)
    m[r0:r0 + w, c0:c0 + w] = True
    return m


def tag(ax, text, th, fs, loc="tl"):
    x, ha = (0.045, "left") if loc.endswith("l") else (0.955, "right")
    y, va = (0.95, "top") if loc.startswith("t") else (0.05, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fs, color=th["tag_fg"], zorder=12,
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", fc=th["tag_bg"], ec="none"))


def query_mark(ax, x, y, ms=5.0):
    """Hollow ring, so that the attention under the query stays visible."""
    ax.plot([x], [y], marker="o", ms=ms, mfc="none", mec="black", mew=1.5, zorder=11)
    ax.plot([x], [y], marker="o", ms=ms, mfc="none", mec="white", mew=0.7, zorder=12)


def show(ax, im, px):
    ax.imshow(np.clip(im, 0, 1), extent=(0, px, px, 0), interpolation="nearest")
    ax.set_xlim(0, px)
    ax.set_ylim(px, 0)
    ax.set_axis_off()


def draw(fig, d, th, px=512):
    """Lay the whole figure out on fig (5.5 in wide)."""
    W, H = fig.get_figwidth(), fig.get_figheight()
    img = Image.open(ASSETS / "image_1024.png")
    tgt = Image.open(ASSETS / "target_4096_at1024.jpg")
    base = grey_base(img, th, px)
    win = d["window"]
    r0, c0, w, _ = (int(x) for x in win)
    mask = window_mask(win)
    near_c, far_c = hex_rgb(th["near"]), hex_rgb(th["far"])
    s = px / N_SIDE
    qr, qc = int(d["query"][0]), int(d["query"][1])
    qy, qx = (qr + 0.5) * s, (qc + 0.5) * s
    fs, fss = sf.FS, sf.FS_SMALL
    # Zoomed window: the image crop and the token grid of the window at px resolution.
    zoom_base = grey_base(img.crop((c0 * 16, r0 * 16, (c0 + w) * 16, (r0 + w) * 16)), th, px)
    zs = px / w
    zqy, zqx = (qr - r0 + 0.5) * zs, (qc - c0 + 0.5) * zs
    # Geometry (inches).
    left, tile, gap, top_title = 0.31, 0.835, 0.04, 0.17
    y_top = H - top_title
    rows = ("near", "far")
    for ri, tag_ in enumerate(rows):
        y0 = y_top - (ri + 1) * tile - ri * gap
        axs = []
        for ci in range(4):
            x0 = left + ci * (tile + gap)
            axs.append(fig.add_axes([x0 / W, y0 / H, tile / W, tile / H]))
        full = d[f"{tag_}_full"]
        dens = full / full.sum() * N_SIDE * N_SIDE
        pi = float(d[f"{tag_}_pi"])
        m = float(d[f"{tag_}_m"])
        # (1) full attention: amber inside the window, blue outside.
        show(axs[0], heat(heat(base, dens, near_c, mask), dens, far_c, ~mask), px)
        # (2) near part, zoomed to the window.
        show(axs[1], heat(zoom_base, dens[r0:r0 + w, c0:c0 + w], near_c), px)
        # (3) far part.
        show(axs[2], heat(base, dens, far_c, ~mask), px)
        # (4) SCA: the local operator inside the window and the global operator on the block-centre grid, each with
        # an opacity that follows the head's weight.
        loc = d[f"{tag_}_local"]
        loc_dens = loc / max(loc.sum(), 1e-30) * N_SIDE * N_SIDE
        show(axs[3], heat(base, loc_dens, near_c, mask, fade=0.2 + 0.8 * pi), px)
        g = d[f"{tag_}_global"]
        gw = g / max(g.sum(), 1e-30)
        cen = (np.arange(N_SIDE // RATIO) * RATIO + RATIO // 2 + 0.5) * s
        X, Y = np.meshgrid(cen, cen)
        size = 0.35 + 7.5 * (gw / gw.max())
        axs[3].scatter(X.ravel(), Y.ravel(), s=size.ravel(), c=th["far"], alpha=0.2 + 0.8 * (1 - pi), linewidths=0,
                       zorder=9)
        for ci in (0, 2, 3):
            axs[ci].add_patch(Rectangle((c0 * s, r0 * s), w * s, w * s, fill=False, ec=th["near"], lw=0.8,
                                        zorder=8, alpha=1.0 if ci != 3 else 0.35 + 0.65 * pi))
            query_mark(axs[ci], qx, qy, ms=4.2)
        query_mark(axs[1], zqx, zqy, ms=5.2)
        for sp_ in axs[1].spines.values():
            sp_.set_visible(False)
        axs[1].add_patch(Rectangle((0, 0), px, px, fill=False, ec=th["near"], lw=1.1, zorder=13, clip_on=False))
        tag(axs[1], rf"$m_i={m:.2f}$", th, fss)
        tag(axs[2], rf"$1-m_i={1 - m:.2f}$", th, fss)
        tag(axs[3], rf"$\pi_h={pi:.2f}$", th, fss)
        # Row label.
        h1 = int(d["heads"][ri]) + 1
        word = "near-reaching" if tag_ == "near" else "far-reaching"
        yc = y0 + tile / 2
        fig.text((left - 0.19) / W, yc / H, f"Head {h1}", rotation=90, ha="center", va="center", fontsize=fs,
                 fontweight="bold", color=th["ink"])
        fig.text((left - 0.075) / W, yc / H, word, rotation=90, ha="center", va="center", fontsize=fss,
                 color=th["near"] if tag_ == "near" else th["far"])
    for ci, title in enumerate(("Full attention", "Near part (zoom)", "Far part", "SCA: window + grid")):
        x = left + ci * (tile + gap) + tile / 2
        fig.text(x / W, (y_top + 0.035) / H, title, ha="center", va="bottom", fontsize=fs, color=th["ink"])
    # (b) Reach emulation on the same image, right column.
    rx0 = left + 4 * tile + 3 * gap + 0.30
    rw = W - rx0 - 0.02
    em = 0.57                                     # image side, inches
    ey0 = y_top - em - 0.02
    ax_t = fig.add_axes([rx0 / W, ey0 / H, em / W, em / H])
    ax_e = fig.add_axes([(rx0 + rw - em) / W, ey0 / H, em / W, em / H])
    for ax, im in ((ax_t, tgt), (ax_e, img)):
        arr = np.asarray(im.convert("RGB").resize((px, px), Image.LANCZOS), dtype=np.float64) / 255
        ax.imshow(arr * 0.85 + 0.15 * arr.mean(-1, keepdims=True), extent=(0, px, px, 0), interpolation="lanczos")
        ax.set_xlim(0, px)
        ax.set_ylim(px, 0)
        ax.set_axis_off()
    for ax, n_grid, dot in ((ax_t, 64, 0.25), (ax_e, 16, 0.95)):
        ax.add_patch(Rectangle((c0 * s, r0 * s), w * s, w * s, fill=False, ec=th["near"], lw=0.9, zorder=8))
        step = px / n_grid
        cc = (np.arange(n_grid) + 0.5) * step
        X, Y = np.meshgrid(cc, cc)
        ax.scatter(X.ravel(), Y.ravel(), s=dot ** 2, c=th["far"], linewidths=0, zorder=7)
        query_mark(ax, qx, qy, ms=3.2)
    fig.text((rx0 + em / 2) / W, (ey0 - 0.03) / H, r"$4096^2$", ha="center", va="top", fontsize=fss, color=th["ink"])
    fig.text((rx0 + rw - em / 2) / W, (ey0 - 0.03) / H, r"$1024^2$", ha="center", va="top", fontsize=fss,
             color=th["ink"])
    arr_y = ey0 + em / 2
    fig.patches.append(FancyArrowPatch(((rx0 + em + 0.025) / W, arr_y / H), ((rx0 + rw - em - 0.025) / W, arr_y / H),
                                       transform=fig.transFigure, arrowstyle="-|>,head_length=3,head_width=1.8",
                                       color=th["ink"], lw=0.8))
    fig.text((rx0 + rw / 2) / W, (arr_y + 0.04) / H, r"$\div r$", ha="center", va="bottom", fontsize=fs,
             color=th["ink"])
    fig.text(rx0 / W, (y_top + 0.035) / H, "Reach emulation", ha="left", va="bottom", fontsize=fs, color=th["ink"])
    fig.text((rx0 + rw / 2) / W, (ey0 - 0.19) / H, "window 64 \u2192 16 tokens\ngrid $64^2$ \u2192 $16^2$ keys",
             ha="center", va="top", fontsize=fss, color=th["mid"], linespacing=1.15)
    # (c) Variance split, a single stacked bar.
    head, step_ = float(d["var_head"]), float(d["var_step"])
    by0 = y_top - 2 * tile - gap + 0.16
    bh = 0.13
    ax_b = fig.add_axes([rx0 / W, by0 / H, rw / W, bh / H])
    ax_b.barh([0], [head], color=th["head"], height=1.0, lw=0)
    ax_b.barh([0], [step_], left=[head], color=th["step"], height=1.0, lw=0)
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(-0.5, 0.5)
    ax_b.set_axis_off()
    ax_b.text(2.0, 0, f"head {head:.1f}%", ha="left", va="center", fontsize=fss, color=th["face"], fontweight="bold")
    ax_b.annotate(f"noise level {step_:.1f}%", (100, -0.5), xytext=(0, -3), textcoords="offset points",
                  ha="right", va="top", fontsize=fss, color=th["mid"])
    fig.text(rx0 / W, (by0 + bh + 0.045) / H, "Variance of the share\nwithin 16 tokens", ha="left", va="bottom",
             fontsize=fss, color=th["ink"], linespacing=1.1)
    # Panel letters.
    for x, y, letter in ((0.0, y_top + 0.035, "a"), (rx0 - 0.14, y_top + 0.035, "b"),
                         (rx0 - 0.14, by0 + bh + 0.045 + 0.13, "c")):
        fig.text(x / W, y / H, letter, ha="left", va="bottom", fontsize=sf.FS_PANEL, fontweight="bold",
                 color=th["ink"])


def load_drawn() -> dict:
    z = np.load(RES / "attn_focus.npz")
    d = {k: z[k] for k in z.files}
    d["near_pi"], d["far_pi"] = d["pi_r4"][0], d["pi_r4"][1]
    with open(VARSPLIT, newline="") as fh:
        row = next(r for r in csv.DictReader(fh) if r["checkpoint"] == "flux1_dev" and r["res"] == "1024x1024"
                   and r["sigma_range"] == "post_gate" and r["quantity"] == "image_share" and r["radius"] == "16")
    noise = float(row["noise_within_head"])
    d["var_step"], d["var_head"] = 100 * noise, 100 * (1 - noise)
    head_tex = re.search(r"\\newcommand\{\\hlVarHead\}\{([^}]*)\}", HEADLINE.read_text()).group(1)
    step_tex = re.search(r"\\newcommand\{\\hlVarStep\}\{([^}]*)\}", HEADLINE.read_text()).group(1)
    if f"{d['var_head']:.1f}" != head_tex or f"{d['var_step']:.1f}" != step_tex:
        raise SystemExit(f"variance split {d['var_head']:.2f}/{d['var_step']:.2f} vs headline {head_tex}/{step_tex}")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--extract", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract()
        return
    d = load_drawn()
    H = 1.93
    sf.use_style()
    sf.script_floor(sf.FS_SMALL)
    fig = plt.figure(figsize=(sf.TEXT, H))
    draw(fig, d, THEMES["light"])
    stem = FIGS / "attn_focus"
    fig.savefig(f"{stem}.pdf", dpi=400)
    fig.savefig(f"{stem}.svg", dpi=400)
    plt.close(fig)
    web = []
    WEB.mkdir(parents=True, exist_ok=True)
    for name, th in THEMES.items():
        sf.use_style()
        sf.script_floor(sf.FS_SMALL)
        mpl.rcParams.update({"text.color": th["ink"]})
        fig = plt.figure(figsize=(sf.TEXT, H))
        draw(fig, d, th)
        for ext in ("svg", "png"):
            p = WEB / f"attn_focus_{name}.{ext}"
            fig.savefig(p, dpi=300, transparent=True)
            web.append(p)
        Image.open(WEB / f"attn_focus_{name}.png").save(WEB / f"attn_focus_{name}.webp", "WEBP", quality=90,
                                                        method=6)
        web.append(WEB / f"attn_focus_{name}.webp")
        plt.close(fig)
    prov = json.loads((RES / "summary.json").read_text())
    prov["draw"] = {"script": "paper/figures/src/fig_attn_focus.py", "variance_split": str(VARSPLIT.relative_to(ROOT)),
                    "variance_split_sha256": sha256(VARSPLIT), "var_head": d["var_head"], "var_step": d["var_step"],
                    "outputs": {str(p.relative_to(ROOT)): sha256(p) for p in
                                [Path(f"{stem}.pdf"), Path(f"{stem}.svg")] + web}}
    (FIGS / "attn_focus_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print("drew", stem)


if __name__ == "__main__":
    main()
