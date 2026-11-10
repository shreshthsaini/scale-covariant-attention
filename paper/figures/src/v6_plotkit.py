"""Shared helpers of the v6 result plots: Figures 6 (gains) and 7 (scaling) and the appendix result plots.

One plot style for every result figure of the paper, on top of scafig (palette, sizes, math font) and the vendored
orx_figstyle (rcParams, audit):
* Canvas widths are printed widths. TEXT (5.5 in) for a full-width figure, WRAP (0.48 TEXT = 2.64 in) for a wrap
  figure, HALF (0.5 TEXT = 2.75 in) for a figure included at half the text width. The paper includes each with
  width=\\linewidth inside a float of that width, so it prints at scale 1.0 and its 7 pt labels print at 7 pt.
* Labels 7 to 8 pt (scafig FS_TICK and FS), nothing under scafig.MIN_PT (6 pt), sub- and superscripts included
  (scafig.script_floor). No titles, no top or right spines, light gridlines along the value axis only, legends
  without frames, the direction of "better" stated on the value axis, no broken axes.
* Panel tags read "(a) name": a bold letter in parentheses, then a short name, top left above the axes.
* save() writes PDF and SVG and runs the orx_figstyle audit unchanged. The audit's width rule knows only 3.25, 5.5
  and 6.75 in, so for a figure built at a declared narrower printed width (WRAP or HALF) that one finding is
  dropped and the audit line names the width; every other check (fonts, titles, axis labels, text size, overlaps,
  clipping) applies as is.
* macro() reads the value of a \\newcommand or \\providecommand from a generated snippet, so a script can assert that
  what it plots rounds to what the paper prints; fmt() formats as the generators do (Python fixed point, the
  minus sign as in the macros).

Nothing here reads results/; each figure script reads its own sources.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical PDF and SVG on rerun
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import orx_figstyle as base  # noqa: E402
import scafig as sf  # noqa: E402

ROOT = HERE.parents[2]
FIGS = ROOT / "paper" / "figures"
V6 = FIGS / "v6"
CANDIDATES = V6 / "candidates"
SNIPPETS = FIGS / "snippets"
HEADLINE = SNIPPETS / "main_tables" / "headline.tex"

TEXT = sf.TEXT            # 5.5 in
WRAP = round(0.48 * TEXT, 4)   # 2.64 in, a 0.48 \textwidth wrap figure
HALF = round(0.50 * TEXT, 4)   # 2.75 in, included at 0.5 \linewidth
PRINTED_WIDTHS = (TEXT, WRAP, HALF)

# Interval colours shared by the paired-difference plots: direction of a significant result in colour, a result
# whose interval includes zero in a neutral grey. scafig has no named grey for it yet; NS falls back to the
# neutral baseline grey until the style owner adds one.
GAIN = sf.SCA
LOSS = sf.LOSS
NS = getattr(sf, "NS", sf.METHOD_COLOR["sega"])
ARROW = "\u2192"   # right arrow, set in the text face
DOWN = "\u2193"
UP = "\u2191"
TIMES = "\u00d7"
MINUS = "\u2212"


def use_style(script_pt: float | None = None) -> None:
    """scafig.use_style(); with `script_pt`, mathtext sub- and superscripts of text at that size print at 6 pt."""
    sf.use_style()
    if script_pt is not None:
        sf.script_floor(script_pt)


def panel_tag(fig, ax, letter: str, name: str = "", dy_pt: float = 3.0, dx_pt: float = 0.0,
              color: str | None = None) -> None:
    """"(a) name" above the top-left corner of `ax`: the letter bold, the name regular, both FS (8 pt)."""
    bb = ax.get_position()
    W, H = fig.get_size_inches()
    x, y = bb.x0 + dx_pt / 72 / W, bb.y1 + dy_pt / 72 / H
    t = fig.text(x, y, f"({letter})", fontsize=sf.FS_PANEL, fontweight="bold", ha="left", va="baseline",
                 color=sf.INK)
    if name:
        fig.canvas.draw()
        w = t.get_window_extent().width / fig.dpi / W
        fig.text(x + w + 2.5 / 72 / W, y, name, fontsize=sf.FS, ha="left", va="baseline",
                 color=color or sf.INK)


def clean_axes(ax, grid: str | None = "y", left: bool = True) -> None:
    """No top or right spine; light gridlines along the value axis (`grid` "x" or "y", None for none)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.grid(False)
    if grid:
        sf.value_grid(ax, grid)


def _audit_filtered(fig, printed_width: float) -> list[str]:
    problems = base._audit(fig)
    width = fig.get_size_inches()[0]
    if abs(width - printed_width) < 0.005 and any(abs(width - w) < 0.005 for w in PRINTED_WIDTHS):
        problems = [p for p in problems if not p.startswith(f"width {width:.2f}in is not a known column width")]
    return problems


def save(fig, stem, printed_width: float = TEXT, formats=("pdf", "svg")) -> list[str]:
    """Write STEM.pdf and STEM.svg (no tight bounding box: the canvas is the printed size) and print the audit.

    Returns the written paths. The figure must be built at `printed_width`, one of TEXT, WRAP and HALF."""
    stem = str(stem)
    width = fig.get_size_inches()[0]
    if abs(width - printed_width) > 0.005:
        raise ValueError(f"{stem}: canvas {width:.3f} in, declared printed width {printed_width:.3f} in")
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    paths = []
    for ext in formats:
        path = f"{stem}.{ext}"
        fig.savefig(path, format=ext)
        paths.append(path)
    try:
        problems = _audit_filtered(fig, printed_width)
    finally:
        plt.close(fig)
    rel = os.path.relpath(stem, ROOT) if stem.startswith(str(ROOT)) else stem
    if problems:
        print(f"FIGURE AUDIT {rel}: {len(problems)} problem(s), fix before using", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    else:
        note = "" if abs(width - TEXT) < 0.005 else f" (printed width {width:.2f} in = {width / TEXT:.2f} text width)"
        print(f"figure audit {rel}: clean{note}", file=sys.stderr)
    return paths


def min_text_pt(fig) -> float:
    """Smallest font size of any visible text artist of `fig` (mathtext scripts are checked by script_floor)."""
    sizes = [t.get_fontsize() for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
    return min(sizes) if sizes else float("inf")


_MACRO = r"\\(?:re|provide|new)command\{\\%s\}\{((?:[^{}]|\{[^{}]*\})*)\}"


def macro(name: str, path: Path = HEADLINE) -> str:
    """The literal value of \\name in a generated snippet (first definition)."""
    m = re.search(_MACRO % re.escape(name), Path(path).read_text())
    if m is None:
        raise SystemExit(f"{path}: no \\{name}")
    return m.group(1)


def has_macro(name: str, path: Path = HEADLINE) -> bool:
    return re.search(_MACRO % re.escape(name), Path(path).read_text()) is not None


def g6(x: float) -> float:
    """Six significant digits, as the archived paired CSVs store them."""
    return float(f"{x:.6g}")


def fmt(x: float, d: int, signed: bool = False) -> str:
    """Fixed point as gen_main_tables.triple prints it (Python formatting, ASCII minus)."""
    return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"


def check(cond: bool, what: str, bad: list[str]) -> None:
    if not cond:
        bad.append(what)


def fail_if(bad: list[str], script: str) -> None:
    if bad:
        raise SystemExit(f"{script}: plotted values differ from the paper or a caption claim fails:\n  "
                         + "\n  ".join(bad))
