"""Data assets of the v6 method figure (Figure 3) and method zoom: real images, real attention, real weights.

One image flows through the figure: the Aesthetic-4K prompt a4k_eval_4096_minus8343371271909676658, seed 0.
  img4096.jpg     progressive SCA's 4096^2 output (fig_attn_focus.TARGET, stored at 1024 px by fig_attn_focus.py)
  (the native 1024^2 stage of the same run, paper/figures/attn_focus/image_1024.png, is the base of the b_* tiles)
Attention (paper/figures/src/data/v6_attn_block12.npz, written by v6_attn_extract.py from the CPU capture of that
stage): block 12, head HEAD, query token (43, 11) (the face of the leftmost sheep), step 20 (sigma 0.612), r = 4:
  b_near.png      the emulated near view (window of 16 tokens around the query's tile): its attention, amber
  b_full.png      full attention over all 64 x 64 image keys: amber inside the window, blue outside (Lemma 1 split)
  b_far.png       the base image under the far view; the 16 x 16 block-centre keys are vector dots in data.tex,
                  dot area proportional to the far view's attention on that key
Weights (results/evidence/calib/k1_1024.npz, 32 emulation prompts, the r = 4 table):
  J(pi) = E - 2 pi B + pi^2 A of block 12 at the noise bin centred at sigma = 0.618 for heads 7 (far), HEAD and
  4 (near), rescaled to [0, 1] on pi in [0, 1]; minimisers clip(B / A, 0, 1)        -> data.tex \\JcFar ... \\JpNear
  table_r2.png, table_r4.png: block 12's 24 heads (rows, sorted by the mean r = 4 weight, near-reaching on top)
  by the 13 populated noise bins (columns, sigma falling left to right), scafig.PI_CMAP   -> data.tex \\TabRow ...
Geometry (data.tex): the 4096^2 native window (64 tokens) and the 1024^2 window at r = 4 (16 tokens) of the query's
tile, as fractions of the image side; they cover the same pixels by construction (Section 3.2).

Run on a compute node (CPU, seconds):
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I paper/figures/src/v6_method_assets.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig as sf  # noqa: E402

ROOT = HERE.parents[2]
FIGS = ROOT / "paper" / "figures"
OUT = FIGS / "v6" / "method_assets"
IMG1024 = FIGS / "attn_focus" / "image_1024.png"
IMG4096 = FIGS / "attn_focus" / "target_4096_at1024.jpg"
ATTN = HERE / "data" / "v6_attn_block12.npz"
CALIB = ROOT / "results" / "evidence" / "calib" / "k1_1024.npz"

BLOCK = 11                 # zero-based; block 12
HEAD = 9                   # zero-based head 10: rule below (asserted in main)
FAR_HEAD, NEAR_HEAD = 6, 3  # zero-based heads 7 and 4: lowest and highest share within 16 tokens in block 12
BIN = 8                    # noise bin centred at sigma = 0.618 (the capture's step 20 has sigma = 0.612)
STEP_INDEX = 1             # step 20 in v6_attn_block12.npz (steps 8, 20, 26)
RATIO = 4
N1024, N4096 = 64, 256     # token grid sides
TILE = 8
PX = 448                   # raster side of the attention tiles (about 600 dpi at 0.75 in)
DMAX = 100.0               # density (x uniform) at which the heat saturates, as fig_attn_focus.DMAX

NEAR = np.array([int(sf.LOCAL[i:i + 2], 16) / 255 for i in (1, 3, 5)])
FAR = np.array([int(sf.GLOBAL[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def light_base(img: Image.Image, px: int, lo: float = 0.58, hi: float = 0.97) -> np.ndarray:
    """Greyscale image mapped to a light range, so that amber and blue heat read on top of it."""
    L = np.asarray(img.convert("L").resize((px, px), Image.LANCZOS), dtype=np.float64) / 255
    return np.repeat((lo + (hi - lo) * L)[..., None], 3, -1)


def heat(base: np.ndarray, dens: np.ndarray, color: np.ndarray, mask=None) -> np.ndarray:
    """Alpha-composite a token density (x uniform over the image keys) onto base; alpha follows log density."""
    v = np.clip(np.log10(np.maximum(dens, 1.0)) / np.log10(DMAX), 0, 1)
    if mask is not None:
        v = np.where(mask, v, 0.0)
    a = np.minimum(1.0, 1.0 * v ** 0.6)
    k = base.shape[0] // dens.shape[0]
    a = np.kron(a, np.ones((k, k)))[..., None]
    return base * (1 - a) + color[None, None, :] * a


def veil(im: np.ndarray, mask_px: np.ndarray | None, amount: float) -> np.ndarray:
    """Lighten im towards white outside mask_px (everywhere if mask_px is None)."""
    out = im * (1 - amount) + amount
    if mask_px is None:
        return out
    return np.where(mask_px[..., None], im, out)


def save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8)).save(path, optimize=True)


def window(row: int, col: int, n: int, r: int) -> tuple[int, int, int]:
    """Top-left corner and side of the window at ratio r on an n-token grid (attn_focus_data.Capture.window)."""
    w = n // r
    t = max(1, round(TILE * w / 64))
    def start(p):
        return int(np.clip(t * (p // t) + t // 2 - w // 2, 0, n - w))
    return start(row), start(col), w


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ATTN)
    q = tuple(int(v) for v in z["query"])
    assert int(z["block"]) == BLOCK and q == (43, 11)
    # Rule for the illustrated head: of block 12's 24 heads, the one whose full attention at this query puts the
    # share closest to one half inside the r = 4 window (it needs both views).
    m4 = z["m4"][STEP_INDEX]
    assert int(np.argmin(np.abs(m4 - 0.5))) == HEAD, "selection rule picks another head"
    r0, c0, w = window(*q, N1024, RATIO)
    assert [r0, c0, w] == [int(v) for v in z["window_r4"][:3]], "window differs from the extraction"
    # the native 64-token window of the query's 8 x 8 tile at 4096^2: tokens (4 q, 4 q) lie in tile 4 q // 8
    R0, C0, W = window(4 * q[0], 4 * q[1], N4096, 4)
    assert W == 64 and (R0, C0) == (4 * r0, 4 * c0), "the r = 4 window at 1024^2 must cover the 4096^2 window"

    # images
    Image.open(IMG4096).convert("RGB").resize((640, 640), Image.LANCZOS).save(OUT / "img4096.jpg", quality=92,
                                                                              subsampling=0)
    img = Image.open(IMG1024)
    base = light_base(img, PX)
    s = PX // N1024
    win_tok = np.zeros((N1024, N1024), bool)
    win_tok[r0:r0 + w, c0:c0 + w] = True
    win_px = np.kron(win_tok, np.ones((s, s), bool))

    # (b) near view: the emulated local operator's attention inside the window (image keys only)
    loc = np.zeros((N1024, N1024))
    loc[r0:r0 + w, c0:c0 + w] = z["local_r4"][HEAD]
    d_loc = loc / loc.sum() * N1024 ** 2
    save_png(veil(heat(base, d_loc, NEAR, win_tok), win_px, 0.55), OUT / "b_near.png")
    # full attention: amber inside the window, blue outside
    full = z["full"][STEP_INDEX, HEAD].astype(np.float64)
    d_full = full / full.sum() * N1024 ** 2
    save_png(heat(heat(base, d_full, NEAR, win_tok), d_full, FAR, ~win_tok), OUT / "b_full.png")
    # far view: the base, veiled; dots are vector (data.tex)
    save_png(veil(base, None, 0.45), OUT / "b_far.png")
    glob = z["global_r4"][HEAD].astype(np.float64)       # [16, 16], image part of the far view's softmax
    gshare = glob / glob.sum()

    # Figure-2-style tiles of heads 4 and 7 are drawn by v6_observation.py; here only the method's head.
    m_full = float(full[win_tok].sum() / full.sum())

    # J(pi) and the table
    a = np.load(CALIB)
    i4 = list(a["scales"]).index(RATIO)
    S = a["stats"][i4][:, BLOCK, :, BIN]                  # A, B, E_fg, E_fl, F2, count
    A, B, E = S[0], S[1], S[2]
    pi_tab = a["s4_pi_head"][BLOCK, :, BIN]
    curves = {}
    grid = np.linspace(0, 1, 41)
    for tag, h in (("Far", FAR_HEAD), ("Mid", HEAD), ("Near", NEAR_HEAD)):
        J = E[h] - 2 * grid * B[h] + grid ** 2 * A[h]
        Jn = (J - J.min()) / (J.max() - J.min())
        star = float(np.clip(B[h] / A[h], 0, 1)) if A[h] > 0 else 1.0
        assert abs(star - pi_tab[h]) < 1e-9, (tag, star, pi_tab[h])
        curves[tag] = dict(head_one_based=h + 1, A=float(A[h]), B=float(B[h]), E=float(E[h]),
                           B_over_A=float(B[h] / A[h]), pi_star=star,
                           coords=" ".join(f"({x:.3f},{y:.4f})" for x, y in zip(grid, Jn)))
    pop = a["populated"].astype(bool)
    sig = a["sigma_centers"]
    cols = np.where(pop)[0][::-1]                         # sigma falls left to right
    tables = {}
    for r in (2, 4):
        T = a[f"s{r}_pi_head"][BLOCK][:, cols]            # [24, 13]
        tables[r] = T
    order = np.argsort(-tables[4].mean(1), kind="stable")  # near-reaching heads on top, the same order in both
    for r in (2, 4):
        rgb = sf.PI_CMAP(np.clip(tables[r][order], 0, 1))[..., :3]
        up = np.kron(rgb, np.ones((12, 12, 1)))
        save_png(up, OUT / f"table_r{r}.png")
    head_row = int(np.where(order == HEAD)[0][0])         # 0 = top row
    bin_col = int(np.where(cols == BIN)[0][0])            # 0 = left column

    # data.tex
    def frac(v, n):
        return f"{v / n:.5f}"
    lines = ["% Written by v6_method_assets.py; do not edit by hand.",
             "% Fractions of the image side measured from the top-left corner (x right, y down).",
             r"\def\WinX{%s}\def\WinY{%s}\def\WinS{%s}" % (frac(c0, N1024), frac(r0, N1024), frac(w, N1024)),
             r"\def\NatX{%s}\def\NatY{%s}\def\NatS{%s}" % (frac(C0, N4096), frac(R0, N4096), frac(W, N4096)),
             r"\def\QX{%s}\def\QY{%s}\def\QS{%s}" % (frac(q[1], N1024), frac(q[0], N1024), frac(1, N1024)),
             r"\def\TileX{%s}\def\TileY{%s}\def\TileS{%s}" % (frac(TILE * (4 * q[1] // TILE), N4096),
                                                            frac(TILE * (4 * q[0] // TILE), N4096),
                                                            frac(TILE, N4096))]
    for tag, c in curves.items():
        lines.append(r"\def\Jc%s{%s}" % (tag, c["coords"]))
        lines.append(r"\def\Jp%s{%.4f}" % (tag, c["pi_star"]))
    lines.append(r"\def\JpMidTwo{%.2f}" % curves["Mid"]["pi_star"])
    # far-view dots: centre fraction and radius fraction (area proportional to the attention share)
    dots = []
    rmax = 0.5 / 16 * 0.98                               # the largest dot nearly fills its block
    smax = gshare.max()
    for i in range(16):
        for j in range(16):
            rr = rmax * np.sqrt(gshare[i, j] / smax)
            if rr >= 0.0006:
                dots.append(f"{(4 * j + 2.5) / 64:.4f}/{(4 * i + 2.5) / 64:.4f}/{rr:.5f}")
    lines.append(r"\def\FarDots{%s}" % ",".join(dots))
    lines.append(r"\def\TabRow{%d}\def\TabCol{%d}\def\TabRows{24}\def\TabCols{%d}" % (head_row, bin_col, len(cols)))
    lines.append(r"\def\SigmaBin{%.2f}" % sig[BIN])
    (OUT / "data.tex").write_text("\n".join(lines) + "\n")
    grad = sf.PI_CMAP(np.linspace(1, 0, 256))[:, None, :3] * np.ones((1, 8, 1))
    save_png(grad, OUT / "picmap.png")

    prov = {
        "script": rel(Path(__file__)),
        "inputs": {rel(p): sha256(p) for p in (IMG1024, IMG4096, ATTN, CALIB)},
        "block_one_based": BLOCK + 1, "head_one_based": HEAD + 1, "query": list(q), "ratio": RATIO,
        "sigma_capture": float(z["sigma"][STEP_INDEX]), "sigma_bin_centre": float(sig[BIN]),
        "window_1024_r4": [r0, c0, w], "window_4096_native": [R0, C0, W],
        "near_share_in_window_full": m_full, "far_view_text_mass": float(z["global_r4_text"][HEAD]),
        "near_view_text_mass": float(z["local_r4_text"][HEAD]),
        "j_curves": {k: {kk: vv for kk, vv in v.items() if kk != "coords"} for k, v in curves.items()},
        "table": {"rows": "block 12, 24 heads sorted by mean r = 4 weight over populated bins, descending",
                  "cols": [round(float(sig[c]), 4) for c in cols], "head_row_of_head": head_row,
                  "bin_col": bin_col, "pi_r4_cell": float(tables[4][HEAD, bin_col]),
                  "pi_r2_cell": float(tables[2][HEAD, bin_col])},
        "selection_rule": "head 10: of block 12's 24 heads, the one whose full attention at this query and step "
                          "puts the share closest to one half inside the r = 4 window (m = %.3f); heads 4 and 7: "
                          "highest and lowest share within 16 tokens in block 12 at this bin (32 prompts)"
                          % float(m4[HEAD]),
        "outputs": {},
    }
    for p in sorted(OUT.glob("*")):
        if p.name != "provenance.json":
            prov["outputs"][rel(p)] = sha256(p)
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(json.dumps({k: prov[k] for k in ("window_1024_r4", "window_4096_native", "near_share_in_window_full",
                                           "j_curves", "table")}, indent=1))


if __name__ == "__main__":
    main()
