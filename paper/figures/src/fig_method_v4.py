"""Build the method figure v4 (fig:method) and its web exports.

The figure is TikZ (method_v4a.tex and method_v4b.tex are the two candidates, method_v4.tex the chosen one, macros in
method_v4_common.tex). It adds no data assets: it draws v3's (paper/figures/method_v3/, written by fig_method_v3.py,
provenance.json there). This script only compiles and exports:

  paper/figures/method_v4.pdf                     the paper figure (light), 5.5 in wide
  paper/figures/candidates/method_v4{a,b}.pdf     the two candidates (with --candidates)
  web/assets/img/figs/method_v4_{light,dark}.{svg,png,webp}
                                                  transparent background; the dark variant is the same source
                                                  compiled with \\SCADARK defined (method_v4_common.tex recolours
                                                  ink, greys, tints, and the photo fades for the page's --bg #0D100F)

Compilation runs in a scratch build directory (symlinks to the sources and to ../method_v3), so no aux file lands in
the repository. CPU only; run on a compute node:
  ~/.local/bin/uv run --no-project --with 'pymupdf<1.26' --with pillow python -I fig_method_v4.py [--candidates]
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import fitz  # pymupdf
from PIL import Image

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent
ROOT = HERE.parents[2]
WEB = ROOT / "web" / "assets" / "img" / "figs"
TECTONIC = Path.home() / ".local" / "bin" / "tectonic"
BUILD = Path(os.environ.get("SCRATCH", "/scratch/09032/saini_2")) / "tmp" / "method_v4_build"
SOURCES = ("sca_colors.tex", "method_v4_common.tex", "method_v4a.tex", "method_v4b.tex", "method_v4.tex")
WEB_DPI = 300


def setup() -> Path:
    src = BUILD / "src"
    (BUILD / "out").mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True)
    link = BUILD / "method_v3"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(FIGS / "method_v3")
    for name in SOURCES:
        dst = src / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(HERE / name)
    (src / "method_v4_dark.tex").write_text("\\def\\SCADARK{1}\\input{method_v4.tex}\n")
    return src


def compile_tex(src: Path, name: str) -> Path:
    subprocess.run([str(TECTONIC), "--outdir", str(BUILD / "out"), name], cwd=src, check=True,
                   stdout=subprocess.DEVNULL)
    return BUILD / "out" / name.replace(".tex", ".pdf")


def export_web(pdf: Path, stem: str) -> list[Path]:
    WEB.mkdir(parents=True, exist_ok=True)
    page = fitz.open(pdf)[0]
    svg = WEB / f"{stem}.svg"
    svg.write_text(page.get_svg_image(text_as_path=True))
    png = WEB / f"{stem}.png"
    page.get_pixmap(dpi=WEB_DPI, alpha=True).save(png)
    webp = WEB / f"{stem}.webp"
    Image.open(png).save(webp, "WEBP", quality=90, method=6)
    return [svg, png, webp]


def main() -> None:
    src = setup()
    light = compile_tex(src, "method_v4.tex")
    shutil.copyfile(light, FIGS / "method_v4.pdf")
    dark = compile_tex(src, "method_v4_dark.tex")
    if "--candidates" in sys.argv:
        (FIGS / "candidates").mkdir(exist_ok=True)
        for c in ("a", "b"):
            shutil.copyfile(compile_tex(src, f"method_v4{c}.tex"), FIGS / "candidates" / f"method_v4{c}.pdf")
    out = [FIGS / "method_v4.pdf"]
    out += export_web(light, "method_v4_light")
    out += export_web(dark, "method_v4_dark")
    r = fitz.open(FIGS / "method_v4.pdf")[0].rect
    print(f"method_v4.pdf: {r.width / 72:.3f} x {r.height / 72:.3f} in")
    for p in out:
        print(p.relative_to(ROOT), p.stat().st_size)


if __name__ == "__main__":
    main()
