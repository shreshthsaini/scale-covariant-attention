"""Appendix figure rel_err (v6, paper/figures/v6/rel_err.pdf): relative output error of every mixing rule.

FLUX.1-dev, reach emulation at 1024^2, 32 prompts, emulated ratios 2 (a) and 4 (b). Seven rules per ratio: the
per-head table (SCA), the per-head table of the other ratio, one weight per layer, one weight shared over heads, the
best hard switch from far to near, near view only and far view only. Every error is recomputed from the per-cell
statistics by k1data.errors and asserted against data/k1_1024.json by k1data.check, as fig_rel_err.py does; nothing is
typed. Same numbers as the v5.1 figure (fig_rel_err.py); only the presentation follows the v6 house style
(v6_emulation.py): "(a) Name" titles, near amber and far blue, the other greys neutral, values at the bar ends with the
lowest error of each ratio in bold.

  cd paper/figures/src && /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I v6_app_rel_err.py
Writes paper/figures/v6/rel_err.pdf and paper/figures/v6/rel_err_provenance.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig as sf  # noqa: E402
from k1data import check, errors, load  # noqa: E402

ROOT = HERE.parents[2]
V6 = ROOT / "paper" / "figures" / "v6"
STEM = "rel_err"
W, H = sf.TEXT, 1.62
FS_T, FS_L, FS_S, FS_V = 8.5, 7.5, 7.0, 6.5
SCA_LIGHT = "#9AD3BA"
G_DARK = "#7D8893"
XROOM = 1.17  # each axis reaches 1.17 times its longest bar: room for that bar's value
ROWS = [  # (key in k1data.errors, label, colour); plain words, a subscript would print small
    ("per_head", "per head (SCA)", sf.SCA),
    ("transfer_from", "per head, other ratio", SCA_LIGHT),
    ("per_layer", "per layer", G_DARK),
    ("uniform", "shared", G_DARK),
    ("switch", "hard switch", G_DARK),
    ("loc", "near only", sf.LOCAL),
    ("glob", "far only", sf.GLOBAL),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


# ------------------------------------------------------------------------------------------------------------ data
def data() -> dict:
    """fig_rel_err.py main(): load, check against the report, errors per ratio."""
    z, report = load()
    check(z, report)
    return {s: errors(z, s) for s in (2, 4)}


# ------------------------------------------------------------------------------------------------------------ draw
def axes_in(fig, x, y, w, h, **kw):
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def title(fig, x, y, letter, name):
    fig.text(x / W, y / H, f"({letter}) {name}", fontsize=FS_T, fontweight="bold", ha="left", va="baseline")


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=FS_S, length=2.2, width=0.5, pad=1.6)


def render(e: dict):
    sf.use_style()
    sf.script_floor(FS_S)
    fig = plt.figure(figsize=(W, H))
    left, gap, right, bottom, top = 1.02, 0.34, 0.04, 0.37, 1.36
    pw = (W - left - gap - right) / 2
    ty = 1.50
    n = len(ROWS)
    yc = np.arange(n)[::-1].astype(float)
    for j, s in enumerate((2, 4)):
        x0 = left + j * (pw + gap)
        ax = axes_in(fig, x0, bottom, pw, top - bottom)
        vals = [e[s][k] for k, _, _ in ROWS]
        ax.barh(yc, vals, height=0.64, color=[c for _, _, c in ROWS], lw=0, zorder=2)
        best = min(vals)
        for yy, v in zip(yc, vals):
            ax.annotate(f"{v:.3f}", (v, yy), xytext=(2, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=FS_V, color=sf.INK, fontweight="bold" if v == best else "normal")
        ax.set_xlim(0, max(vals) * XROOM)
        ax.set_ylim(-0.6, n - 0.4)
        ticks = [0, 0.1, 0.2, 0.3] if s == 2 else [0, 0.2, 0.4, 0.6]
        ax.set_xticks(ticks, ["0"] + [f"{t:g}" for t in ticks[1:]])
        ax.set_yticks(yc, [lab for _, lab, _ in ROWS] if j == 0 else [])
        sf.value_grid(ax, "x")
        clean(ax)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.spines["left"].set_visible(False)
        title(fig, 0.0 if j == 0 else x0, ty, "ab"[j], f"Ratio {s}")
    fig.text((left + (W - left - right) / 2) / W, 0.04 / H, r"relative output error $\downarrow$", ha="center",
             va="bottom", fontsize=FS_L)
    return fig


def main() -> None:
    e = data()
    V6.mkdir(parents=True, exist_ok=True)
    fig = render(e)
    out = V6 / f"{STEM}.pdf"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    ins = [HERE / "data" / "k1_1024_pi.npz", HERE / "data" / "k1_1024.json"]
    vals = {str(s): {k: e[s][k] for k, _, _ in ROWS} | {"sigma_star": e[s]["sigma_star"]} for s in (2, 4)}
    prov = {"figure": "Appendix figure rel_err (fig:relerr), v6", "script": rel(Path(__file__)),
            "data_code": "k1data.load, k1data.check, k1data.errors (as paper/figures/src/fig_rel_err.py)",
            "size_in": [W, H], "inputs": {rel(p): sha256(p) for p in ins}, "values": vals,
            "output_sha256": sha256(out)}
    (V6 / f"{STEM}_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    for s in (2, 4):
        print(s, {k: round(v, 4) for k, v in e[s].items()})


if __name__ == "__main__":
    main()
