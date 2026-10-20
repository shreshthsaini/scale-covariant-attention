"""Assets and data for the SCA method figure (fig:method, figures/src/method.tex).

The figure is TikZ (figures/src/method.tex, compiled to figures/method.pdf).
This script writes what that file draws from, all into figures/method_flow/:

  input.jpg    the 4096^2 input latent x_sigma, drawn as the real SCA output
               x0 forward-noised in pixel space, x_sigma = (1 - sigma) x0 +
               sigma eps with sigma = 0.2748 (step 26 of 28) and one Gaussian
               sample per latent token (256 x 256 at 4096^2), shared by the
               three colour channels (luminance grain; per-channel noise hid
               the window content).
               This is an illustration of the noise level, not a recorded
               sampler state.
  global.png   what the global branch keys on: the token kappa(b) = s b +
               floor(s/2) of every s x s block, one cell per block, on the
               n0 x n0 native grid (cell colour = mean of that token's pixels
               in input.jpg's noisy image). Nearest-neighbour upsampled.
  local.jpg    what the local branch keys on: the n0 x n0 window of the query
               tile, cut from the same noisy image at full resolution.
  output.jpg   the clean SCA output at 4096^2 (the sample after 28 steps).
  data.tex     TikZ macros: the pi table of one block (24 heads x 14 noise
               bins, sigma decreasing left to right), the per-head weights
               pi_h(sigma) read at the drawn step's noise bin, and the per-step
               local share of the 28-step strip.

Geometry is schematic, as in Figure D.1: s = 4, n0 = 8, t = 2 on a 32 x 32
token grid laid over the image (FLUX.1 itself has n0 = 64 and t = 8, and the
4096^2 latent is 256 x 256 tokens). Window start a(tau) = clip(t tau +
floor(t/2) - floor(n0/2), 0, H - n0) (Section 2.2).

pi data: src/data/k1_1024_pi.npz, key s4_pi_head (57 blocks x 24 heads x 14
bins; the archived scale-4 table that inference reads). Bin 0 is unpopulated
and holds the nearest populated bin, as stored. The strip's local share is the
mean of that table over all blocks and heads, interpolated linearly between
the populated bin centres at each step's sigma (FLUX.1 schedule, mu = 1.15).

Run from paper/figures/src on a compute node (no GPU needed):
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I fig_method_flow.py
then compile:  ~/.local/bin/tectonic --outdir .. method.tex
then record:   python -I fig_method_flow.py --prov    (provenance and PDF check;
               needs pymupdf < 1.26, for example through
               uv run --no-project --with numpy --with matplotlib --with 'pymupdf<1.26')
or only check: python -I fig_method_flow.py --check
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig  # noqa: E402  (palette and the paper-wide pi colormap)

FIG = HERE.parent
OUT = FIG / "method_flow"
NPZ = HERE / "data" / "k1_1024_pi.npz"
TEX = HERE / "method.tex"
PDF = FIG / "method.pdf"
PROV = FIG / "method_provenance.json"

SRC = Path("/scratch/09032/saini_2/sca/out/main/sca_nb/4096x4096/"
           "a4k_eval_4096_minus1066207185399086750_d6e8faefc5_seed0.png")
PROMPT_ID = "a4k_eval_4096_minus1066207185399086750"

BLOCK = 7            # zero-based; "block 8"
BIN = 3              # noise bin read at the drawn step (centre 0.2748)
STEP = 26            # 1-based step of the 28-step schedule with sigma = 0.2748
SIGMA_G = 0.89
MU = 1.15
STEPS = 28
S, N0, TQ, HTOK = 4, 8, 2, 32   # schematic: scale, native side, tile side, target side
TILE = (8, 5)        # query tile index (x, y): tokens 16..17 x 10..11 (the eye)
LAT = 256            # latent tokens per side at 4096^2 (noise grain of input.jpg)
NOISE_SEED = 0
PX_IN = 768          # stored size of input.jpg and output.jpg
PX_LOC = 512         # stored size of local.jpg
PX_GLOB = 256        # stored size of global.png (8 x 8 cells, nearest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def schedule(steps: int = STEPS) -> list[float]:
    """FLUX.1 shifted schedule as in fig_method.py (mu = 1.15)."""
    t = np.linspace(1.0, 1.0 / steps, steps)
    return [float(math.exp(MU) / (math.exp(MU) + (1.0 / s - 1.0))) for s in t]


def window_start(tau: int) -> int:
    return int(np.clip(TQ * tau + TQ // 2 - N0 // 2, 0, HTOK - N0))


def rgb255(hexcolor: str) -> tuple[int, int, int]:
    h = hexcolor.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def build_images(sigma: float) -> dict:
    img = np.asarray(Image.open(SRC).convert("RGB"), dtype=np.float32) / 255.0
    side = img.shape[0]
    assert img.shape[:2] == (4096, 4096), img.shape
    rng = np.random.default_rng(NOISE_SEED)
    eps = rng.standard_normal((LAT, LAT, 1)).astype(np.float32)
    up = side // LAT                                   # 16 px per latent token
    eps_full = np.repeat(np.repeat(eps, up, 0), up, 1)
    x0 = 2.0 * img - 1.0
    noisy = np.clip(((1.0 - sigma) * x0 + sigma * eps_full + 1.0) / 2.0, 0.0, 1.0)
    del eps_full

    def save(arr: np.ndarray, size: int, name: str, resample=Image.LANCZOS, **kw):
        im = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))
        if im.size[0] != size:
            im = im.resize((size, size), resample)
        path = OUT / name
        im.save(path, **kw)
        return path

    paths = {}
    # input: noisy image, area-averaged to PX_IN (grain = 3 px per latent token)
    paths["input"] = save(noisy, PX_IN, "input.jpg", Image.BOX, quality=92)
    paths["output"] = save(img, PX_IN, "output.jpg", Image.LANCZOS, quality=92)
    # local window of the query tile, full resolution
    tok = side // HTOK                                 # 128 px per schematic token
    ax, ay = window_start(TILE[0]), window_start(TILE[1])
    crop = noisy[ay * tok:(ay + N0) * tok, ax * tok:(ax + N0) * tok]
    paths["local"] = save(crop, PX_LOC, "local.jpg", Image.BOX, quality=92)
    # global: block-centre token of every s x s block, on the n0 x n0 grid
    cells = np.zeros((N0, N0, 3), np.float32)
    for by in range(N0):
        for bx in range(N0):
            ky, kx = S * by + S // 2, S * bx + S // 2   # kappa(b)
            cells[by, bx] = noisy[ky * tok:(ky + 1) * tok, kx * tok:(kx + 1) * tok].mean((0, 1))
    glob = np.repeat(np.repeat(cells, PX_GLOB // N0, 0), PX_GLOB // N0, 1)
    paths["global"] = save(glob, PX_GLOB, "global.png", Image.NEAREST)
    return {"paths": paths, "window_start_xy": [ax, ay]}


def build_data() -> dict:
    d = np.load(NPZ)
    table = d["s4_pi_head"]                    # (57, 24, 14), bins in increasing sigma
    centres = d["sigma_centers"]
    pop = d["populated"].astype(bool)
    blk = table[BLOCK]                         # (24, 14)
    sig = schedule()
    mean_bin = table.mean(axis=(0, 1))         # (14,)
    share = [float(np.interp(s, centres[pop], mean_bin[pop])) for s in sig]
    assert abs(sig[STEP - 1] - centres[BIN]) < 5e-4, (sig[STEP - 1], centres[BIN])

    lines = ["% Written by fig_method_flow.py from src/data/k1_1024_pi.npz (s4_pi_head)."]
    # table: column c = 0..13 is bin 13 - c (sigma decreasing left to right), row r = head r
    cells = []
    for c in range(14):
        b = 13 - c
        for r in range(24):
            R, G, B = rgb255(scafig.pi_color(blk[r, b]))
            cells.append("%d/%d/%d/%d/%d" % (c, r, R, G, B))
    lines.append(r"\def\pitable{%s}" % ",".join(cells))
    chips = []
    for r in range(24):
        R, G, B = rgb255(scafig.pi_color(blk[r, BIN]))
        chips.append("%d/%d/%d/%d" % (r, R, G, B))
    lines.append(r"\def\pichips{%s}" % ",".join(chips))
    lines.append(r"\def\stepshare{%s}" % ",".join("%d/%.3f" % (k + 1, v) for k, v in enumerate(share)))
    bar = []
    for k in range(24):
        R, G, B = rgb255(scafig.pi_color((k + 0.5) / 24))
        bar.append("%d/%d/%d/%d" % (k, R, G, B))
    lines.append(r"\def\pibar{%s}" % ",".join(bar))
    lines.append(r"\def\readcol{%d}" % (13 - BIN))
    lines.append(r"\def\gatex{%.4f}" % ((1.0 - SIGMA_G) * 14))   # sigma_g in table columns
    n_global = sum(1 for s in sig if s > SIGMA_G)
    lines.append(r"\def\nglobal{%d}" % n_global)
    lines.append(r"\def\curstep{%d}" % STEP)
    (OUT / "data.tex").write_text("\n".join(lines) + "\n")
    return {
        "table": blk, "chips": blk[:, BIN], "share": share, "sigma": sig,
        "centres": centres, "populated": pop, "n_global": n_global,
        "n_prompts": int(d["n_prompts"]),
    }


def check_pdf() -> dict:
    import fitz  # pymupdf < 1.26
    doc = fitz.open(PDF)
    page = doc[0]
    w, h = page.rect.width, page.rect.height
    sizes, words, outside = [], [], []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if not span["text"].strip():
                    continue
                sizes.append(round(span["size"], 2))
                x0, y0, x1, y1 = span["bbox"]
                if x0 < -0.5 or y0 < -0.5 or x1 > w + 0.5 or y1 > h + 0.5:
                    outside.append(span["text"])
    words = [wd[4] for wd in page.get_text("words")]
    return {"width_bp": round(w, 2), "height_bp": round(h, 2), "ratio": round(h / w, 4),
            "min_text_pt": min(sizes) if sizes else None, "max_text_pt": max(sizes) if sizes else None,
            "words": words, "n_words": len(words), "outside": outside}


def main():
    OUT.mkdir(exist_ok=True)
    data = build_data()
    sigma = data["sigma"][STEP - 1]
    if "--prov" in sys.argv:   # provenance only: keep the assets the PDF was built from
        names = {"input": "input.jpg", "output": "output.jpg", "local": "local.jpg", "global": "global.png"}
        imgs = {"paths": {k: OUT / v for k, v in names.items()},
                "window_start_xy": [window_start(TILE[0]), window_start(TILE[1])]}
    else:
        imgs = build_images(sigma)
    prov = {
        "figure": "fig:method (main.tex Figure 3)",
        "tikz": "paper/figures/src/method.tex",
        "script": "paper/figures/src/fig_method_flow.py",
        "assets_dir": "paper/figures/method_flow",
        "fonts": "Helvetica-metric sans (helvet) for labels, Computer Modern math",
        "palette": {"method_green": scafig.SCA, "local_amber": scafig.LOCAL,
                    "global_blue": scafig.GLOBAL, "neutral_ink": scafig.INK,
                    "neutral_mid": scafig.MID, "pi_colormap": "scafig.PI_CMAP "
                    "(#1F4F8A, #2F6DB3, #F4F1EA, #E08A2B, #9C5512 at 0, .22, .5, .78, 1)"},
        "image": {
            "source_png": str(SRC), "source_sha256": sha256(SRC), "prompt_id": PROMPT_ID,
            "method": "sca_nb (SCA, default settings), FLUX.1-dev, 4096x4096, seed 0, 28 steps",
            "also_in": "figures/showcase/square1.jpg (showcase/provenance.json records[1])",
            "input_jpg": "forward-noised illustration of x_sigma: (1 - sigma) x0 + sigma eps in "
                         "pixel space, sigma = %.4f (step %d), eps ~ N(0, 1) per latent token "
                         "(%d x %d) shared by the RGB channels, numpy default_rng(%d); not a recorded sampler "
                         "state" % (sigma, STEP, LAT, LAT, NOISE_SEED),
            "global_png": "mean colour of the block-centre token kappa(b) = s b + floor(s/2) of "
                          "every s x s block of the noisy image, n0 x n0 cells",
            "local_jpg": "the n0 x n0 token window of the query tile, cut from the noisy image",
            "output_jpg": "the clean SCA output, downsampled",
        },
        "geometry": {
            "schematic": {"s": S, "n0": N0, "t": TQ, "target_side_tokens": HTOK},
            "flux": {"n0": 64, "t": 8, "s_at_4096": 4, "latent_side_at_4096": 256},
            "query_tile_xy": list(TILE),
            "query_tile_tokens_xy": [TQ * TILE[0], TQ * TILE[1]],
            "window_start_xy": imgs["window_start_xy"],
            "block_centre_tokens": [S * b + S // 2 for b in range(N0)],
            "query_coarse_cell_xy": [TQ * TILE[0] // S, TQ * TILE[1] // S],
            "rules": "kappa(b) = s b + floor(s/2); c(p) = floor(p/s); "
                     "a(tau) = clip(t tau + floor(t/2) - floor(n0/2), 0, H - n0) (Section 2.2)",
        },
        "pi": {
            "data": "paper/figures/src/data/k1_1024_pi.npz", "data_sha256": sha256(NPZ),
            "key": "s4_pi_head", "n_prompts": data["n_prompts"], "resolution": "1024x1024",
            "block_zero_based": BLOCK, "block_label": BLOCK + 1,
            "why_this_block": "shows both axes of pi_h(sigma): heads range from global to local "
                              "and several heads change with sigma (std over heads at the read bin = %.2f; "
                              "block 29 of the previous figure: 0.08 at bin 6)" % float(np.std(data["chips"])),
            "table_layout": "24 heads (rows, head 1 at top) x 14 noise bins of width 1/14 "
                            "(columns, sigma decreasing left to right); bin 0 unpopulated, holds "
                            "the nearest populated bin as stored; sigma > sigma_g veiled",
            "sigma_centers": [round(float(c), 6) for c in data["centres"]],
            "pi_head_by_bin": [[round(float(v), 4) for v in row] for row in data["table"]],
            "read_bin": BIN, "read_sigma": round(float(data["centres"][BIN]), 6),
            "chips_pi_h": [round(float(v), 4) for v in data["chips"]],
        },
        "schedule": {
            "mu": MU, "steps": STEPS, "sigma": [round(s, 4) for s in data["sigma"]],
            "sigma_g": SIGMA_G, "global_only_steps": list(range(1, data["n_global"] + 1)),
            "drawn_step": STEP,
            "strip_local_share": [round(v, 3) for v in data["share"]],
            "strip_rule": "steps with sigma > sigma_g global only (blue); later steps split into "
                          "the local share (amber, top) and 1 - share (blue), share = mean of "
                          "s4_pi_head over all 57 x 24 heads interpolated at the step's sigma",
        },
        "assets_sha256": {p.name: sha256(p) for p in imgs["paths"].values()},
    }
    prov["assets_sha256"]["data.tex"] = sha256(OUT / "data.tex")
    if PDF.exists() and PDF.stat().st_mtime >= TEX.stat().st_mtime:
        try:
            prov["pdf"] = check_pdf()
            prov["pdf"]["sha256"] = sha256(PDF)
        except ImportError:
            pass
    PROV.write_text(json.dumps(prov, indent=1) + "\n")
    print("wrote", OUT, PROV)


if __name__ == "__main__":
    if "--check" in sys.argv:
        print(json.dumps({k: v for k, v in check_pdf().items()}, indent=1))
    else:
        main()
