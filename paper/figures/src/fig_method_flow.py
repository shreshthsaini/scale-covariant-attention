"""Assets and data for the SCA method figure (fig:method, figures/src/method.tex), operator sca_v2.

The figure is TikZ (figures/src/method.tex, compiled to figures/method.pdf). This script writes what it
draws from into figures/method_flow/:

  thumb.jpg   the clean sca_v2 sample at 4096^2 (FLUX.1-dev, seed 0), downsampled; panel (a) draws the
              shrinking and native phases on it.
  global.png  the opening-phase view: the block-centre token kappa(b) = s b + floor(s/2) of every s x s
              block (s = 4) of the same sample, 64 x 64 cells, nearest-neighbour upsampled.
  data.tex    TikZ macros: the per-step local share of the 28-step strip, the window frames (to scale on
              the 256-token side), the query tile, the schematic bridge dots, and the two calibrated tables
              (r = 2 and r = 4) of one block for panel (b).

Rules, all from sca/attn.py with the sca_v2 flags (--bridge-cells 8 --bridge-k 16 --win-sched
0.88:128,0.86:112,0.84:96,0.82:80 --win-pi-match --win-pi-interp, gate 0.89):
  window   w = 128, 112, 96, 80 while sigma > 0.88, 0.86, 0.84, 0.82 (SCAContext.window_now), else n0 = 64;
           start a = clip(t tau + floor(t/2) - floor(w/2), 0, H - w) per axis (flex_layout), t = 8.
  weight   pi at ratio r = max(H, W) / w, log-linear between the scale-2 and scale-4 tables, t = log2(r/2)
           (SCAContext.table_interp); at w = n0 the scale-4 table (pi_table_now returns the published pi).
  lookup   linear in sigma between all 14 bin centres, clamped (PiTable.__call__).
  strip    share = mean of that weight over all 57 x 24 heads at the step's sigma; steps with sigma > 0.89
           are global only (share 0).
  bridge   1,024 tokens, 16 per cell of an 8 x 8 grid, chosen per block and step from the keys
           (bridge_index); the drawn dots are SCHEMATIC (one per selected cell, positions seeded), since the
           selection needs the model's keys, which were not recorded.
  dots     the opening view's blue dots are SCHEMATIC too: one per 16 x 16 tokens (the operator keeps one
           key per 4 x 4 block, 64 x 64 keys, which global.png shows as cells).

Run from paper/figures/src on a compute node (no GPU needed):
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I fig_method_flow.py
then compile:  ~/.local/bin/tectonic --outdir .. method.tex
then record:   python -I fig_method_flow.py --prov    (provenance with the PDF check; needs pymupdf < 1.26,
               e.g. uv run --no-project --with numpy --with pillow --with matplotlib --with 'pymupdf<1.26')
or only check: python -I fig_method_flow.py --check
"""
from __future__ import annotations

import hashlib
import json
import math
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

SRC = Path("/scratch/09032/saini_2/sca/out/dupfix/sca_v2/4096x4096/"
           "a4k_eval_4096_minus3145908358244441177_04bcf2b825_seed0.png")
PROMPT_ID = "a4k_eval_4096_minus3145908358244441177"

H = 256                      # latent tokens per side at 4096^2
N0, T_TILE, S = 64, 8, 4     # native side, query tile, block size
SIGMA_G, MU, STEPS = 0.89, 1.15, 28
WIN_SCHED = ((0.88, 128), (0.86, 112), (0.84, 96), (0.82, 80))
TILE = (13, 11)              # query tile (x, y): tokens 104..111 x 88..95, between the eyes
BLOCK = 7                    # zero-based; "block 8" (as in the previous version of the figure)
CELLS, PER = 8, 16           # bridge grid and tokens per cell (drawn: one dot per selected cell)
BRIDGE_SEED = 3
PX_THUMB, PX_GLOB = 720, 512


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def schedule() -> list[float]:
    t = np.linspace(1.0, 1.0 / STEPS, STEPS)
    return [float(math.exp(MU) / (math.exp(MU) + (1.0 / s - 1.0))) for s in t]


def window(sigma: float) -> int:
    for s, w in WIN_SCHED:
        if sigma > s:
            return w
    return N0


def start(tau: int, w: int) -> int:
    return int(np.clip(T_TILE * tau + T_TILE // 2 - w // 2, 0, H - w))


def lookup(table: np.ndarray, centres: np.ndarray, sigma: float) -> np.ndarray:
    """PiTable.__call__: linear between all bin centres, clamped at the ends."""
    j = int(np.searchsorted(centres, sigma))
    if j <= 0:
        return table[..., 0]
    if j >= len(centres):
        return table[..., -1]
    a = (sigma - centres[j - 1]) / (centres[j] - centres[j - 1])
    return (1 - a) * table[..., j - 1] + a * table[..., j]


def weight(d, sigma: float) -> np.ndarray:
    """pi of every head at this sigma under sca_v2 (pi_table_now with win_pi_match and win_pi_interp)."""
    w = window(sigma)
    s4 = lookup(d["s4_pi_head"], d["sigma_centers"], sigma)
    if w == N0:
        return s4
    r = H / w
    t = math.log(r / 2) / math.log(4 / 2)
    return (1 - t) * lookup(d["s2_pi_head"], d["sigma_centers"], sigma) + t * s4


def rgb255(hexcolor: str) -> tuple[int, int, int]:
    h = hexcolor.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def bridge_dots(phase: int, frames: list[tuple[int, int, int]], q: tuple[int, int]) -> list[tuple[int, int]]:
    """Schematic bridge dots: one token in each selected cell of the 8 x 8 grid, kept clear of the drawn
    frames and the query tile so that the marks stay legible."""
    rng = np.random.default_rng(BRIDGE_SEED + phase)
    side = H // CELLS
    dots = []
    for cy in range(CELLS):
        for cx in range(CELLS):
            if (cx + 3 * cy + 2 * phase) % 5:
                continue
            for _ in range(200):
                x = cx * side + int(rng.integers(4, side - 4))
                y = cy * side + int(rng.integers(4, side - 4))
                near = any(min(abs(x - a), abs(x - a - w)) < 6 and a - 6 <= y <= a + w + 6 or
                           min(abs(y - b), abs(y - b - w)) < 6 and b - 6 <= x <= b + w + 6
                           for w, a, b in frames)
                if not near and max(abs(x - q[0] - 4), abs(y - q[1] - 4)) > 14:
                    dots.append((x, y))
                    break
    return dots


def build_images() -> dict:
    img = Image.open(SRC).convert("RGB")
    assert img.size == (4096, 4096), img.size
    paths = {}
    th = img.resize((PX_THUMB, PX_THUMB), Image.LANCZOS)
    paths["thumb"] = OUT / "thumb.jpg"
    th.save(paths["thumb"], quality=92)
    arr = np.asarray(img, dtype=np.float32)
    tok = 4096 // H                                     # 16 px per latent token
    n = H // S                                          # 64 block centres per side
    cells = np.zeros((n, n, 3), np.float32)
    for by in range(n):
        for bx in range(n):
            ky, kx = S * by + S // 2, S * bx + S // 2   # kappa(b)
            cells[by, bx] = arr[ky * tok:(ky + 1) * tok, kx * tok:(kx + 1) * tok].mean((0, 1))
    glob = np.repeat(np.repeat(cells, PX_GLOB // n, 0), PX_GLOB // n, 1)
    paths["global"] = OUT / "global.png"
    Image.fromarray(np.clip(glob + 0.5, 0, 255).astype(np.uint8)).save(paths["global"])
    return paths


def build_data() -> dict:
    d = np.load(NPZ)
    sig = schedule()
    share = [0.0 if s > SIGMA_G else float(weight(d, s).mean()) for s in sig]
    n_global = sum(s > SIGMA_G for s in sig)
    shrink = [k for k, s in enumerate(sig) if s <= SIGMA_G and window(s) != N0]
    q = (T_TILE * TILE[0], T_TILE * TILE[1])
    frames = [(w, start(TILE[0], w), start(TILE[1], w)) for w in (128, 112, 96, 80, 64)]
    dots_a = bridge_dots(0, frames[:4], q)
    dots_b = bridge_dots(1, frames[4:], q)
    blk2, blk4 = d["s2_pi_head"][BLOCK], d["s4_pi_head"][BLOCK]
    lines = ["% Written by fig_method_flow.py (sca_v2) from src/data/k1_1024_pi.npz and the rules in its docstring."]
    lines.append(r"\def\stepshare{%s}" % ",".join("%d/%.3f" % (k + 1, v) for k, v in enumerate(share)))
    lines.append(r"\def\nglobal{%d}" % n_global)
    lines.append(r"\def\nshrink{%d}" % len(shrink))
    lines.append(r"\def\winframes{%s}" % ",".join("%d/%d/%d" % f for f in frames[:4]))
    lines.append(r"\def\nativeframe{%d/%d/%d}" % frames[4])
    lines.append(r"\def\qx{%d}\def\qy{%d}" % q)
    lines.append(r"\def\bridgeA{%s}" % ",".join("%d/%d" % p for p in dots_a))
    lines.append(r"\def\bridgeB{%s}" % ",".join("%d/%d" % p for p in dots_b))
    for name, blk in (("pitableA", blk2), ("pitableB", blk4)):
        cells = []
        for c in range(14):                       # column c = bin 13 - c (sigma decreasing left to right)
            for r in range(24):
                R, G, B = rgb255(scafig.pi_color(blk[r, 13 - c]))
                cells.append("%d/%d/%d/%d/%d" % (c, r, R, G, B))
        lines.append(r"\def\%s{%s}" % (name, ",".join(cells)))
    bar = []
    for k in range(24):
        R, G, B = rgb255(scafig.pi_color((k + 0.5) / 24))
        bar.append("%d/%d/%d/%d" % (k, R, G, B))
    lines.append(r"\def\pibar{%s}" % ",".join(bar))
    lines.append(r"\def\gatex{%.4f}" % ((1.0 - SIGMA_G) * 14))                 # sigma_g in table columns
    lines.append(r"\def\shrinkxa{%.4f}\def\shrinkxb{%.4f}" % ((1 - sig[shrink[0]]) * 14, (1 - sig[shrink[-1]]) * 14))
    (OUT / "data.tex").write_text("\n".join(lines) + "\n")
    return {"sigma": sig, "share": share, "n_global": n_global, "shrink": shrink, "frames": frames,
            "query": q, "dots_a": dots_a, "dots_b": dots_b, "blk2": blk2, "blk4": blk4,
            "n_prompts": int(d["n_prompts"]),
            "weights": {k + 1: {"sigma": round(sig[k], 4), "w": window(sig[k]), "r": round(H / window(sig[k]), 4),
                                "mean_pi": round(share[k], 4)} for k in range(STEPS) if sig[k] <= SIGMA_G}}


def check_pdf() -> dict:
    import fitz  # pymupdf < 1.26
    page = fitz.open(PDF)[0]
    w, h = page.rect.width, page.rect.height
    sizes, outside = [], []
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
    return {"width_bp": round(w, 2), "height_bp": round(h, 2), "min_text_pt": min(sizes), "max_text_pt": max(sizes),
            "words": words, "n_words": len(words), "outside": outside}


def main():
    OUT.mkdir(exist_ok=True)
    data = build_data()
    paths = {"thumb": OUT / "thumb.jpg", "global": OUT / "global.png"} if "--prov" in sys.argv else build_images()
    prov = {
        "figure": "fig:method (main.tex Figure 2), operator sca_v2",
        "tikz": "paper/figures/src/method.tex", "script": "paper/figures/src/fig_method_flow.py",
        "assets_dir": "paper/figures/method_flow",
        "palette": {"method_green": scafig.SCA, "local_amber": scafig.LOCAL, "global_blue": scafig.GLOBAL,
                    "neutral_ink": scafig.INK, "neutral_mid": scafig.MID,
                    "pi_colormap": "scafig.PI_CMAP (blue 0 = global, amber 1 = local)"},
        "image": {"source_png": str(SRC), "source_sha256": sha256(SRC), "prompt_id": PROMPT_ID,
                  "method": "sca_v2, FLUX.1-dev, 4096x4096, seed 0, 28 steps (out/dupfix/sca_v2)",
                  "thumb": "clean sample, Lanczos to %d px" % PX_THUMB,
                  "global": "block-centre token of every 4 x 4 block (64 x 64 cells), mean colour of its 16 x 16 px"},
        "geometry": {"latent_side": H, "n0": N0, "t": T_TILE, "s": S, "query_tile_xy": list(TILE),
                     "query_tokens_xy": list(data["query"]),
                     "frames_w_ax_ay": [list(f) for f in data["frames"]],
                     "drawn_to_scale": "frames and query tile in tokens of the 256-token side",
                     "opening_dots": "schematic: one blue dot per 16 x 16 tokens",
                     "bridge_dots": "schematic: one dot per selected cell of the 8 x 8 grid (real: 16 per cell, "
                                    "1,024 tokens, chosen from the keys per block and step), seed %d; "
                                    "shrink view %s, native view %s" % (BRIDGE_SEED, data["dots_a"], data["dots_b"])},
        "schedule": {"mu": MU, "steps": STEPS, "sigma": [round(s, 4) for s in data["sigma"]], "sigma_g": SIGMA_G,
                     "global_only_steps": list(range(1, data["n_global"] + 1)),
                     "shrink_steps": [k + 1 for k in data["shrink"]], "win_sched": [list(x) for x in WIN_SCHED],
                     "strip_local_share": [round(v, 3) for v in data["share"]],
                     "post_gate_weights": data["weights"]},
        "pi": {"data": "paper/figures/src/data/k1_1024_pi.npz", "data_sha256": sha256(NPZ),
               "keys": ["s2_pi_head", "s4_pi_head"], "n_prompts": data["n_prompts"],
               "block_zero_based": BLOCK, "block_label": BLOCK + 1,
               "table_layout": "24 heads (rows) x 14 bins (columns, sigma 1 -> 0); r = 2 top, r = 4 bottom",
               "s2_block": [[round(float(v), 4) for v in row] for row in data["blk2"]],
               "s4_block": [[round(float(v), 4) for v in row] for row in data["blk4"]]},
        "assets_sha256": {p.name: sha256(p) for p in paths.values()},
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
        print(json.dumps(check_pdf(), indent=1))
    else:
        main()
