"""Data assets for the method figure v3 (fig:method): thumbnails, reach-table cards, and three real quadratics J(pi).

The figure itself is TikZ (method_v3a.tex and method_v3b.tex are the two candidates, method_v3.tex the chosen one);
this script writes only the pieces drawn from data into paper/figures/method_v3/:

  canvas_1024.jpg   the 1,024^2 canvas of panels (a) and (b): emulation prompt 1 ("A candid street portrait of an
                    elderly fisherman ..."), seed 0. Stand-in, as in the earlier method figures: the emulation pass
                    stores sufficient statistics, not images, so this is the same prompt rendered by single-pass SCA
                    at 2048^2 and resized (paper/figures/method_provenance.json, "calibration_image").
  target_4096.jpg   the 4096^2 target of panel (b) and the output of panel (d): progressive SCA, Aesthetic-4K prompt
                    a4k_eval_4096_minus8343371271909676658 (sheep and church), seed 0.
  noise.png         the noise icon of panel (d) (an icon, not the sampler's latent).
  table_r2.png,     the reach table of one block (block 12 one-based): its 24 heads as rows, sorted by their mean
  table_r4.png      ratio-4 weight (local on top), by the 13 occupied noise bins as columns (sigma falling left to
                    right), coloured by scafig.PI_CMAP (amber = 1, local window; blue = 0, global grid).
  data.tex          the three quadratics of panel (c) as TikZ coordinates, with their minimizers.
  provenance.json   inputs with sha256, rules, outputs with sha256.

Panel (c) draws, for three heads of the same block at the noise bin centred at sigma = 0.618 and ratio r = 4,
J(pi) = E - 2 pi B + pi^2 A from the stored sums (results/evidence/calib/k1_1024.npz, quantities A, B, E_fg in the
order of sca/ QUANT), each rescaled to [0, 1] over pi in [0, 1] so that the three shapes share one box. The heads
are the block's lowest and highest B/A and the head closest to the block's median B/A, so the three minimizers are
clip(B/A, 0, 1) = 0, an interior value, and 1. No number is invented: every printed value is computed here.
Deterministic. Run with the sca env python:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_method_v3.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig as sf  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = HERE.parents[2]
NPZ = ROOT / "results" / "evidence" / "calib" / "k1_1024.npz"
FLOW = ROOT / "paper" / "figures" / "method_flow"
OUT = ROOT / "paper" / "figures" / "method_v3"

BLOCK = 11          # zero-based; block 12 one-based (the block of the table cards)
BIN = 8             # noise bin centred at sigma = 0.618, the bin of Figure 3
RATIO = 4
QUANT = ("A", "B", "E_fg", "E_fl", "F2", "count")   # order of the stored sums (sca/ statistics module)
NPTS = 41


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_cards(z) -> dict[str, Path]:
    pop = z["populated"]
    out = {}
    s4 = z["s4_pi_head"][BLOCK][:, pop]
    order = np.argsort(-s4.mean(axis=1), kind="stable")  # local heads on top
    for r in (2, 4):
        T = z[f"s{r}_pi_head"][BLOCK][:, pop][order][:, ::-1]  # sigma falling left to right
        rgba = (sf.PI_CMAP(np.clip(T, 0, 1)) * 255).round().astype(np.uint8)
        img = Image.fromarray(rgba[..., :3], "RGB")
        k = 16  # integer upscale, nearest: cells stay sharp in every PDF viewer
        img = img.resize((img.width * k, img.height * k), Image.NEAREST)
        path = OUT / f"table_r{r}.png"
        img.save(path, optimize=True)
        out[f"r{r}"] = path
    return out


def quadratics(z) -> dict:
    scales = [int(s) for s in z["scales"]]
    S = z["stats"][scales.index(RATIO)]
    A, B, E = (S[QUANT.index(q)][BLOCK, :, BIN] for q in ("A", "B", "E_fg"))
    ratio = B / A
    stored = z[f"s{RATIO}_pi_head"][BLOCK, :, BIN]
    pistar = np.clip(ratio, 0.0, 1.0)
    assert np.allclose(pistar, stored, atol=1e-9), "stored pi differs from clip(B/A, 0, 1)"
    lo, hi = int(np.argmin(ratio)), int(np.argmax(ratio))
    mid = int(np.argmin(np.abs(ratio - np.median(ratio))))
    assert ratio[lo] < 0 < ratio[mid] < 1 < ratio[hi], "need one head clipped at each end and one interior"
    p = np.linspace(0.0, 1.0, NPTS)
    curves = {}
    for role, h in (("glob", lo), ("mid", mid), ("loc", hi)):
        J = E[h] - 2 * p * B[h] + p * p * A[h]
        jmin = E[h] - 2 * pistar[h] * B[h] + pistar[h] ** 2 * A[h]
        jmax = float(J.max())
        y = (J - jmin) / (jmax - jmin)
        curves[role] = dict(head_one_based=h + 1, A=float(A[h]), B=float(B[h]), E=float(E[h]),
                            B_over_A=float(ratio[h]), pi_star=float(pistar[h]),
                            coords=" ".join(f"({x:.3f},{v:.4f})" for x, v in zip(p, y)))
    return dict(curves=curves, sigma_centre=float(z["sigma_centers"][BIN]), block_one_based=BLOCK + 1,
                ratio=RATIO, median_B_over_A=float(np.median(ratio)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(NPZ, allow_pickle=False)
    cards = table_cards(z)
    q = quadratics(z)
    copies = {"canvas_1024.jpg": FLOW / "calib.jpg", "target_4096.jpg": FLOW / "output_prog.jpg",
              "noise.png": FLOW / "noise.png"}
    for name, src in copies.items():
        shutil.copyfile(src, OUT / name)

    with open(OUT / "data.tex", "w") as fh:
        fh.write("% Written by fig_method_v3.py from results/evidence/calib/k1_1024.npz; do not edit by hand.\n")
        fh.write("%% Block %d, sigma bin centred at %.3f, ratio %d; J rescaled to [0, 1] over pi in [0, 1] per head.\n"
                 % (q["block_one_based"], q["sigma_centre"], q["ratio"]))
        for role, c in q["curves"].items():
            tag = {"glob": "G", "mid": "M", "loc": "L"}[role]
            fh.write("%% head %d: B/A = %.3f, pi* = %.3f\n" % (c["head_one_based"], c["B_over_A"], c["pi_star"]))
            fh.write("\\def\\Jc%s{%s}\n" % (tag, c["coords"]))
            fh.write("\\def\\Jp%s{%.3f}\n" % (tag, c["pi_star"]))

    prov = {
        "figure": "fig:method v3; TikZ sources paper/figures/src/method_v3a.tex, method_v3b.tex, method_v3.tex",
        "script": "paper/figures/src/fig_method_v3.py",
        "inputs": {
            "reach_statistics": {"path": str(NPZ.relative_to(ROOT)), "sha256": sha256(NPZ),
                                 "keys": ["stats", "s2_pi_head", "s4_pi_head", "populated", "sigma_centers"],
                                 "stats_order": list(QUANT)},
            "images": {name: {"source": str(src.relative_to(ROOT)), "sha256": sha256(src)}
                       for name, src in copies.items()},
            "image_notes": {
                "canvas_1024.jpg": "stand-in for the 1024^2 canvas: emulation prompt 1, seed 0, rendered by SCA at "
                                   "2048^2 and resized (method_provenance.json, calibration_image)",
                "target_4096.jpg": "progressive SCA output, 4096^2, prompt a4k_eval_4096_minus8343371271909676658, "
                                   "seed 0 (method_provenance.json, output_image_progressive)",
                "noise.png": "icon, numpy default_rng(20261102) (method_provenance.json, noise)",
            },
        },
        "panel_c": {k: v for k, v in q.items() if k != "curves"} | {
            "heads": {role: {k: v for k, v in c.items() if k != "coords"} for role, c in q["curves"].items()},
            "rule": "J(pi) = E - 2 pi B + pi^2 A per head (E = E_fg); rescaled per head to [0, 1] over pi in [0, 1]; "
                    "heads: lowest B/A, highest B/A, and closest to the block median B/A"},
        "rules": {
            "table_cards": "block 12, 24 heads sorted by mean ratio-4 weight (local on top) x 13 occupied noise bins, "
                           "sigma falling left to right, scafig.PI_CMAP, 16x nearest upscale",
            "key_maps": "vector, drawn to scale: 1024^2 canvas = 64 x 64 tokens with a 16-token window and a 16 x 16 "
                        "block-centre grid at r = 4; 4096^2 target = 256 x 256 tokens with a 64-token window and a "
                        "64 x 64 block-centre grid",
            "stage_tiles": "schematic (no intermediate stage images were saved)",
        },
        "outputs": {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.name != "provenance.json"},
    }
    with open(OUT / "provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)
        fh.write("\n")
    print(json.dumps(prov["panel_c"], indent=1))


if __name__ == "__main__":
    main()
