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
  SCA is the accent (green) and AReA, the strongest baseline, the one
  contrasting hue (purple); every other baseline is a neutral grey. Amber and
  blue mean only local and global. Crop windows and their zoom tiles share
  one ZOOM colour.
* Reference style (figures/REFERENCE_STYLE.md): text 6 to 8 pt, top and
  right spines off, direct labels instead of legend boxes. The binding plot
  standard of 2026-10-19 overrides that note's "no grid": plots and bar charts
  carry thin light-grey gridlines along the value axis, behind the data
  (value_grid below); heatmaps and image grids carry none. Bar charts print
  each value at the bar end, the best one bold (FS_VALUE).
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
AREA = "#7A4EA3"     # AReA: the one contrasting hue (not amber, blue or green)
AREA_LIGHT = "#E9E0F1"

# AReA purple; the other baselines neutral greys, darker for stronger ones.
METHOD_COLOR = {
    "sca": SCA,
    "area": AREA,
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

FS = 8.0        # labels, direct labels, the bold headline (house rule: 6 to 8 pt)
FS_TICK = 7.0   # tick labels
FS_PANEL = 8.0  # panel letters
FS_SMALL = 6.5  # the smallest text (small grey per-size ratios)
FS_VALUE = 6.5  # values printed at bar ends
MIN_PT = 6.0    # nothing prints smaller, sub- and superscripts included (gold-standard rubric)
GRIDLINE = "#E3E6EA"  # value-axis gridlines (plot standard of 2026-10-19)
GRIDLINE_LW = 0.45


def value_grid(ax, axis: str = "y") -> None:
    """Thin light-grey gridlines at the major ticks of the value axis, behind the data.

    `axis` is "y" for line plots and vertical bars, "x" for horizontal bars. The rcParams keep
    axes.axisbelow True, so the grid draws at zorder 0.5, under every artist at zorder >= 1.
    """
    ax.set_axisbelow(True)
    ax.grid(visible=True, axis=axis, which="major", color=GRIDLINE, linewidth=GRIDLINE_LW)


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
        "lines.markersize": 3.5,
        "axes.grid": False,
        "legend.handlelength": 1.6,
        "axes.labelpad": 2.5,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "svg.hashsalt": "sca-paper",
        "pdf.compression": 9,
    })
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")


def script_floor(base_pt: float, floor_pt: float = MIN_PT) -> float:
    """Print mathtext sub- and superscripts of `base_pt` text at `floor_pt` or more.

    matplotlib sets every script at a fixed 0.7 of its parent (matplotlib._mathtext.SHRINK_FACTOR),
    so the subscript of a 7 pt tick label prints at 4.9 pt and that of an 8 pt label at 5.6 pt.
    LaTeX raises the ratio at small sizes (8 pt text takes 6 pt scripts); this does the same for a
    whole figure. Pass the smallest size that carries a script, after use_style() and before the
    first draw; the parse cache is cleared because mathtext caches its layouts. Returns the factor.
    """
    from matplotlib import _mathtext, mathtext

    factor = max(0.7, round(floor_pt / base_pt + 0.006, 3))
    _mathtext.SHRINK_FACTOR = factor
    mathtext.MathTextParser._parse_cached.cache_clear()
    return factor


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
        "scaArea": METHOD_COLOR["area"], "scaAreaLight": AREA_LIGHT, "scaFull": METHOD_COLOR["full"],
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
