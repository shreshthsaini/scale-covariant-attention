"""House style for every SCA paper figure: plots, diagrams and image grids.

One visual language across the paper, layered on the vendored orx_figstyle:

* Text is Helvetica-metric sans (Nimbus Sans on Vista; the paper's \\sffamily
  is Helvetica through the `times` package). Math uses Computer Modern, the
  same glyphs as the paper's math. Helvetica at 8.5 pt has about the x-height
  of the body's 10 pt Times, so labels print at body-text visual size; tick
  labels are one step smaller.
* Colour has fixed meanings. GLOBAL (blue) is the image-anchored branch and
  LOCAL (amber) the pixel-anchored branch. The local weight pi runs from blue
  (pi = 0, global) through a light neutral (0.5) to amber (pi = 1, local).
  SCA is the single accent (green). Baselines are neutral greys, darkest for
  the strongest baseline. Crop windows and their zoom tiles share one ZOOM
  colour.
* Panel letters are bold lowercase without parentheses, top left.

Diagrams and image grids written in TikZ load the same colours from
sca_colors.tex, which this module writes (python scafig.py).
"""
from __future__ import annotations

import os

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import orx_figstyle as base

TEXT = base.TEXT  # 5.5 in, the ICLR text width

INK = "#1F2328"      # primary text and outlines
MID = "#5B6570"      # secondary text
RULE = "#C3C9CF"     # thin structural lines
GRID = "#E6E9EC"     # gridlines
PAPER = "#FFFFFF"

GLOBAL = "#2F6DB3"   # global, image-anchored branch
GLOBAL_LIGHT = "#DCE7F3"
LOCAL = "#E08A2B"    # local, pixel-anchored branch
LOCAL_LIGHT = "#F9E6D0"
SCA = "#13865F"      # ours: the only accent
SCA_LIGHT = "#D7EEE4"
NEUTRAL = "#F4F1EA"  # midpoint of the pi map
ZOOM = "#FFC21A"     # crop windows and the frames of their zoom tiles
LOSS = "#B85C00"     # significant loss for SCA (matches the paper's \dn colour)

# Baselines: neutral and ordered by strength (AReA darkest).
METHOD_COLOR = {
    "sca": SCA,
    "area": "#3E4A57",
    "full": "#9AA3AC",
    "dype": "#6F7C89",
    "sega": "#828E9A",
    "sega_official": "#828E9A",
    "ultraimage": "#56626F",
}
METHOD_NAME = {
    "sca": "SCA", "area": "AReA", "full": "Full attention", "dype": "DyPE",
    "sega": "SEGA", "sega_official": "SEGA", "ultraimage": "UltraImage",
}

# pi colormap: global blue -> warm neutral -> local amber, perceptually even
# in lightness on both sides of 0.5.
PI_CMAP = LinearSegmentedColormap.from_list(
    "sca_pi", [(0.0, "#1F4F8A"), (0.22, GLOBAL), (0.5, NEUTRAL),
               (0.78, LOCAL), (1.0, "#9C5512")])

FS = 8.5        # labels, legends, annotations
FS_TICK = 7.5   # tick labels
FS_PANEL = 9.5  # panel letters
FS_SMALL = 7.0  # the smallest allowed text (in-plot value labels)


def use_style() -> None:
    """orx_figstyle rcParams plus the SCA sizes, colours and math font."""
    base.use_style("sans")
    mpl.rcParams.update({
        "mathtext.fontset": "cm",
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS,
        "legend.fontsize": FS - 0.5,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.linewidth": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "legend.handlelength": 1.6,
        "axes.labelpad": 2.5,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "svg.hashsalt": "sca-paper",
        "pdf.compression": 9,
    })
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")


def panel_label(fig, ax, letter: str, dx: float = -0.0, dy: float = 0.0, x_fig: float | None = None):
    """Bold lowercase letter at the top-left of `ax`, outside the axes.

    Pass `x_fig` (figure fraction) to align letters across rows.
    """
    bbox = ax.get_position()
    x = bbox.x0 + dx if x_fig is None else x_fig
    fig.text(x, bbox.y1 + dy, letter, fontsize=FS_PANEL, fontweight="bold",
             ha="left", va="bottom", color=INK)


def save(fig, stem: str, formats=("pdf", "svg")):
    return base.save(fig, stem, formats=formats)


def write_tikz_colors(path: str) -> str:
    """Write the same palette as xcolor definitions for TikZ figures."""
    names = {
        "scaInk": INK, "scaMid": MID, "scaRule": RULE, "scaGrid": GRID,
        "scaGlobal": GLOBAL, "scaGlobalLight": GLOBAL_LIGHT,
        "scaLocal": LOCAL, "scaLocalLight": LOCAL_LIGHT,
        "scaAccent": SCA, "scaAccentLight": SCA_LIGHT,
        "scaNeutral": NEUTRAL, "scaZoom": ZOOM, "scaLoss": LOSS,
        "scaArea": METHOD_COLOR["area"], "scaFull": METHOD_COLOR["full"],
    }
    lines = ["% Written by scafig.py; the same palette as the matplotlib figures."]
    lines += [r"\definecolor{%s}{HTML}{%s}" % (k, v.lstrip("#").upper()) for k, v in names.items()]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def pi_color(value: float) -> str:
    r, g, b, _ = PI_CMAP(float(np.clip(value, 0, 1)))
    return "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print(write_tikz_colors(os.path.join(here, "sca_colors.tex")))
