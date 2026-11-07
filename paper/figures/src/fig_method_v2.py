"""Data assets for the merged method figure (fig:method, v2): concept curves and calibration-table cards.

The figure itself is TikZ (method_v2a.tex, method_v2b.tex, final method_v2.tex in this directory); this script writes
only the pieces drawn from data, at their exact printed size, into paper/figures/method_v2/:

  cloud_<tag>.png   the 1,368 per-head weight curves pi_h(sigma) of results/heads_view/concept_curves.csv (ratio 4,
                    80 sigma values from 0.89 to 0.10), no axes or text; the PNG covers exactly the plot box whose
                    data range data.tex records, so TikZ draws axes, AReA's step and every label in the paper font.
                    Line colour: the head's mean weight (column mean_after_gate) on a blue-grey-amber map with the
                    end colours of scafig.PI_CMAP (amber = local window, blue = global grid).
  table_r2.png,     the calibrated table of one block (block 12 one-based, the block of Figure 3's bars), its 24 heads
  table_r4.png      as rows (sorted by their mean ratio-4 weight, local on top) by the 13 occupied noise bins as columns
                    (sigma falling left to right, as on the curve axis), coloured by scafig.PI_CMAP; from
                    calib/k1_1024.npz (s2_pi_head, s4_pi_head), the file concept_curves.csv is computed from.
  data.tex          plot ranges and the counts the TikZ prints (number of heads, gate).
  provenance.json   inputs with sha256, rules, outputs with sha256.

No number is invented: every value the figure prints is either in data.tex (computed here from the CSV) or a fixed
fact of the method (sizes, ratios, step counts). Deterministic. Run from anywhere with the sca env python:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_method_v2.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig as sf  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = HERE.parents[2]
CSV = ROOT / "results" / "heads_view" / "concept_curves.csv"
NPZ = ROOT / "calib" / "k1_1024.npz"
OUT = ROOT / "paper" / "figures" / "method_v2"

GATE = 0.89                  # AReA's switch (first 8 of 28 steps at maximum shift); the CSV grid starts here
X_LEFT, X_RIGHT = 1.0, 0.10  # sigma at the left and right edge of every curve plot (denoising runs left to right)
Y_LOW, Y_HIGH = -0.06, 1.06  # pi range of the plot box; 0 and 1 sit just inside it
BLOCK = 11                   # zero-based; block 12 one-based
DPI = 600

# Plot boxes (width, height) in pt at print size, one per candidate layout; the TikZ uses the same numbers.
CLOUDS = {
    "a": (84.0, 56.0),    # method_v2a: three columns, the SCA panel under AReA's, same size
    "b": (170.0, 38.0),   # method_v2b: three rows, the SCA panel beside AReA's
}

# Line colours: PI_CMAP's ends, but a mid grey instead of its near-white midpoint so that mid heads stay visible
LINE_CMAP = LinearSegmentedColormap.from_list(
    "sca_pi_lines", [(0.0, "#1F4F8A"), (0.22, sf.GLOBAL), (0.5, "#A39E96"), (0.78, sf.LOCAL), (1.0, "#9C5512")])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_curves():
    heads: dict[tuple[int, int], list[tuple[float, float]]] = {}
    mean: dict[tuple[int, int], float] = {}
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            key = (int(r["layer"]), int(r["head"]))
            heads.setdefault(key, []).append((float(r["sigma"]), float(r["pi"])))
            mean[key] = float(r["mean_after_gate"])
    keys = sorted(heads)
    sig = np.array([s for s, _ in heads[keys[0]]])
    Y = np.array([[p for _, p in heads[k]] for k in keys])
    m = np.array([mean[k] for k in keys])
    return keys, sig, Y, m


def cloud(tag: str, sig: np.ndarray, Y: np.ndarray, m: np.ndarray, lw: float = 0.3, alpha: float = 0.12) -> Path:
    w, h = CLOUDS[tag]
    fig = plt.figure(figsize=(w / 72.0, h / 72.0), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(X_LEFT, X_RIGHT)
    ax.set_ylim(Y_LOW, Y_HIGH)
    order = np.argsort(np.abs(m - 0.5))  # mid heads first, the clearly local and global heads on top
    segs = [np.column_stack([sig, Y[k]]) for k in order]
    cols = [LINE_CMAP(float(np.clip(m[k], 0, 1))) for k in order]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, alpha=alpha, capstyle="butt"))
    path = OUT / f"cloud_{tag}.png"
    fig.savefig(path, dpi=DPI, transparent=True, metadata={"Software": None})
    plt.close(fig)
    return path


def table_cards() -> dict[str, Path]:
    z = np.load(NPZ, allow_pickle=False)
    pop = z["populated"]
    out = {}
    s4 = z["s4_pi_head"][BLOCK][:, pop]  # [head, occupied bin], bins in increasing sigma
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys, sig, Y, m = read_curves()
    assert len(sig) == 80 and abs(sig[0] - GATE) < 1e-9 and abs(sig[-1] - 0.10) < 1e-9
    n_heads = len(keys)
    assert n_heads == 57 * 24 == 1368
    clouds = {tag: cloud(tag, sig, Y, m) for tag in CLOUDS}
    cards = table_cards()

    with open(OUT / "data.tex", "w") as fh:
        fh.write("% Written by fig_method_v2.py from results/heads_view/concept_curves.csv; do not edit by hand.\n")
        fh.write("\\def\\CvXL{%.2f}\\def\\CvXR{%.2f}\\def\\CvYL{%.2f}\\def\\CvYH{%.2f}\n" % (X_LEFT, X_RIGHT, Y_LOW, Y_HIGH))
        fh.write("\\def\\CvGate{%.2f}\n" % GATE)
        fh.write("\\def\\NHeads{%s}\n" % f"{n_heads:,}".replace(",", "{,}"))
        for tag, (w, h) in CLOUDS.items():
            fh.write("\\expandafter\\def\\csname CvW%s\\endcsname{%.1f}\\expandafter\\def\\csname CvH%s\\endcsname{%.1f}\n"
                     % (tag, w, tag, h))

    prov = {
        "figure": "fig:method v2 (merged concept and method figure); TikZ sources paper/figures/src/method_v2*.tex",
        "script": "paper/figures/src/fig_method_v2.py",
        "inputs": {
            "concept_curves": {"path": str(CSV.relative_to(ROOT)), "sha256": sha256(CSV),
                               "rows": int(n_heads * len(sig)), "heads": n_heads,
                               "sigma_grid": [float(sig[0]), float(sig[-1]), len(sig)],
                               "note": "ratio-4 weights pi_h(sigma) as the sampler reads the table (fig_concept.py)"},
            "calibration_table": {"path": str(NPZ.relative_to(ROOT)), "sha256": sha256(NPZ),
                                  "keys": ["s2_pi_head", "s4_pi_head", "populated"],
                                  "block_one_based": BLOCK + 1},
            "images_reused_read_only": {
                "calibration_image": "paper/figures/method_flow/calib.jpg (stand-in, see method_provenance.json)",
                "noise": "paper/figures/method_flow/noise.png (icon, numpy default_rng(20261102))",
                "progressive_output": "paper/figures/method_flow/output_prog.jpg (progressive SCA, 4096^2, "
                                      "prompt a4k_eval_4096_minus8343371271909676658, seed 0)",
            },
        },
        "rules": {
            "curves": "one line per head over sigma in [0.89, 0.10]; colour = head mean on PI_CMAP ends with a grey "
                      "midpoint; lw 0.3 pt, alpha 0.12; mid heads drawn first",
            "axes": {"sigma": [X_LEFT, X_RIGHT], "pi": [Y_LOW, Y_HIGH]},
            "area_step": "AReA in SCA's notation: weight 0 on its windows for sigma > 0.89, then 1, for every head "
                         "(drawn in TikZ, not data)",
            "table_cards": "block 12, 24 heads sorted by mean ratio-4 weight (local on top) x 13 occupied noise bins, "
                           "sigma falling left to right, scafig.PI_CMAP, 16x nearest upscale",
            "stage_tiles": "schematic (no intermediate stage images were saved); side grows with stage size",
        },
        "figures": {
            "paper/figures/src/method_v2a.tex": "candidate A: three columns (a) | (b) | (c), mix as a zoomed inset",
            "paper/figures/src/method_v2b.tex": "candidate B: three full-width rows, mix inside each SCA stage block",
            "paper/figures/src/method_v2.tex": "chosen: candidate B, compiled to paper/figures/method_v2.pdf",
        },
        "outputs": {p.name: sha256(p) for p in [*clouds.values(), *cards.values(), OUT / "data.tex"]},
    }
    with open(OUT / "provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)
        fh.write("\n")
    print(f"wrote {OUT}: {', '.join(sorted(prov['outputs']))}")


if __name__ == "__main__":
    main()
