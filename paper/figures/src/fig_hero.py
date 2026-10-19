"""Page-1 hero (fig:teaser): curated SCA outputs plus a six-method strip.

Top row: two curated SCA outputs from the 191 evaluation prompts (seed 0,
28 steps, guidance 4.5), a 2048x6144 panorama and a 4096^2 interior, with
native-pixel zoom tiles. Bottom: one curated evaluation prompt at 4096^2 for
six methods, full image above the same 768x768 window. Seconds per image are
the clean batch-1 medians of results/latency_curve_summary.csv (one GH200,
three prompts after warm-up); methods without a clean timing show none.

Everything is generated: JPEG assets (deterministic PIL encoding), a TikZ
source with vector boxes, leader lines and labels, its PDF (tectonic), the
snippet, and provenance with source hashes. No metric chose any image.

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_hero.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

from PIL import Image

import imgsel
import scafig

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = FIG.parent.parent
ASSETS = FIG / "teaser"
TEX = HERE / "teaser.tex"
OUT_PDF = FIG / "teaser.pdf"
SNIPPET = FIG / "snippets" / "teaser_six.tex"
TECTONIC = os.path.expanduser("~/.local/bin/tectonic")

W = 5.5          # text width, inches
G = 0.05         # gutter
M = 0.02         # margin of insets and badges from image edges (0.035 let the AReA and SCA
                 # strip badges overlap the head covering and the pipe on it)
DPI = 600        # pixel budget for downsampled full images

# One leader per zoom tile, along the outer edge of the pair: the houses window joins
# its tile at the top (the line runs over the hillside) and the rowboat window at the
# bottom (over the lake), so no leader crosses the village or the crowded shore.
PANO = dict(method="sca_nb", res="2048x6144", pid="minus2742856837225126754",
            zooms=[(4064, 832, 384), (3488, 1536, 320)],   # houses, rowboat
            leaders=("top", "bottom"))
# The square's inset sits in its top-right corner, on the dark vault: chosen by eye as
# the least informative corner (the lower corners hold the pews and the receding
# chandeliers, the top-left holds the badge). Leaders join the window's right corners
# to the inset's left corners and stay on the dark stonework.
SQUARE = dict(method="sca_nb", res="4096x4096", pid="minus8737279395568207187",
              zooms=[(1792, 1536, 448)])                      # rose window
STRIP_PID = "3657338173664010496"
STRIP_BOX = (1600, 1280, 768)                                 # x, y, size (same as before)
STRIP = [("full", "Full attention"), ("dype", "DyPE"), ("sega_official", "SEGA"),
         ("area", "AReA"), ("ultraimage", r"UltraImage$^{*}$"), ("sca_nb", "SCA (ours)")]
TIMING_KEY = {"full": "full", "area": "area", "sca_nb": "sca_nb"}  # clean batch-1 only


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(method: str, res: str, pid: str) -> tuple[Path, dict]:
    root = Path(imgsel.OUT) / "main" / method / res
    hits = sorted(root.glob(f"a4k_eval_4096_{pid}_*_seed0.png"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{method} {res} {pid}: {hits}")
    png = hits[0]
    meta = json.loads(png.with_suffix(".json").read_text())
    if meta.get("complete") is not True or meta.get("png_sha256") != sha(png):
        raise ValueError(f"incomplete or hash-mismatched source {png}")
    return png, meta


def save_jpeg(img: Image.Image, path: Path, quality: int, subsampling: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=quality, optimize=True, subsampling=subsampling)
    return path


def preview(img: Image.Image, width_in: float) -> Image.Image:
    w, h = img.size
    tw = min(w, round(DPI * width_in))
    if tw == w:
        return img
    return img.resize((tw, round(h * tw / w)), Image.Resampling.LANCZOS)


def timings() -> dict[str, float]:
    with open(REPO / "results" / "latency_curve_summary.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["res"] == "4096x4096"]
    return {r["method"]: float(r["median_seconds"]) for r in rows}


def main() -> None:
    scafig.write_tikz_colors(str(HERE / "sca_colors.tex"))
    records, tex = [], []
    # Geometry, inches; origin at the top-left corner, y grows downward (negated in TikZ).
    h = (W - 1.5 * G) / 4.5            # mosaic row height
    zt = (h - G) / 2                   # zoom tile side
    pano_w = 3 * h
    zx = pano_w + G                    # zoom column x
    sq_x = zx + zt + G                 # square x
    col = (W - 5 * G) / 6              # strip column width
    head = 0.205                       # gap plus one header line
    gv = 0.025                         # gap between a full image and its crop
    y_strip = h + head                 # top of strip images

    def node(x, y, w, hh, name):
        tex.append(r"\node[anchor=north west,inner sep=0pt] at (%.4f,%.4f) "
                   r"{\includegraphics[width=%.4fin,height=%.4fin]{%s}};" % (x, -y, w, hh, name))

    def box(x0, y0, x1, y1, style="zoombox"):
        tex.append(r"\draw[halo] (%.4f,%.4f) rectangle (%.4f,%.4f);" % (x0, -y0, x1, -y1))
        tex.append(r"\draw[%s] (%.4f,%.4f) rectangle (%.4f,%.4f);" % (style, x0, -y0, x1, -y1))

    def frame(x0, y0, x1, y1, style):
        tex.append(r"\draw[%s] (%.4f,%.4f) rectangle (%.4f,%.4f);" % (style, x0, -y0, x1, -y1))

    def leader(a, b):
        tex.append(r"\draw[leaderhalo] (%.4f,%.4f) -- (%.4f,%.4f);" % (a[0], -a[1], b[0], -b[1]))
        tex.append(r"\draw[leader] (%.4f,%.4f) -- (%.4f,%.4f);" % (a[0], -a[1], b[0], -b[1]))

    def badge(x, y, text, anchor="north west"):
        tex.append(r"\node[badge,anchor=%s] at (%.4f,%.4f) {%s};" % (anchor, x, -y, text))

    # --- top row: panorama + two zoom tiles -------------------------------------------
    png, meta = source(PANO["method"], PANO["res"], PANO["pid"])
    img = Image.open(png).convert("RGB")
    assert img.size == (6144, 2048)
    full = save_jpeg(preview(img, pano_w), ASSETS / "pano.jpg", 88, 2)
    node(0, 0, pano_w, h, "pano.jpg")
    zooms = []
    for k, (x, y, s) in enumerate(PANO["zooms"]):
        p = save_jpeg(img.crop((x, y, x + s, y + s)), ASSETS / f"pano_zoom{k + 1}.jpg", 90, 0)
        zooms.append(p.name)
        sx = pano_w / img.width
        bx0, by0, bx1, by1 = x * sx, y * sx, (x + s) * sx, (y + s) * sx
        ty = 0 if k == 0 else zt + G
        node(zx, ty, zt, zt, p.name)
        if PANO["leaders"][k] == "top":
            leader((bx1, by0), (zx, ty))
        else:
            leader((bx1, by1), (zx, ty + zt))
        box(bx0, by0, bx1, by1)
        frame(zx, ty, zx + zt, ty + zt, "zoomframe")
    badge(M, M, r"2048\,$\times$\,6144")       # height x width, as in the text
    records.append(dict(role="mosaic panorama", source=str(png), png_sha256=sha(png),
                        prompt=meta["config"]["prompt"], config=meta["config"],
                        zoom_boxes_xys=PANO["zooms"], leaders=list(PANO["leaders"]),
                        preview=full.name, zooms=zooms))

    # --- top row: square with one inset ---------------------------------------------------
    png, meta = source(SQUARE["method"], SQUARE["res"], SQUARE["pid"])
    img = Image.open(png).convert("RGB")
    assert img.size == (4096, 4096)
    full = save_jpeg(preview(img, h), ASSETS / "square.jpg", 88, 2)
    node(sq_x, 0, h, h, "square.jpg")
    (x, y, s), = SQUARE["zooms"]
    p = save_jpeg(img.crop((x, y, x + s, y + s)), ASSETS / "square_zoom1.jpg", 90, 0)
    sx = h / img.width
    bx0, by0, bx1, by1 = sq_x + x * sx, y * sx, sq_x + (x + s) * sx, (y + s) * sx
    inset = 0.47
    ix, iy = sq_x + h - M - inset, M
    assert ix > bx1 and iy + inset < h, "inset must clear the window and stay inside the image"
    node(ix, iy, inset, inset, p.name)
    leader((bx1, by0), (ix, iy))
    leader((bx1, by1), (ix, iy + inset))
    box(bx0, by0, bx1, by1)
    frame(ix, iy, ix + inset, iy + inset, "zoomframe")
    badge(sq_x + M, M, r"4096\,$\times$\,4096")
    records.append(dict(role="mosaic square", source=str(png), png_sha256=sha(png),
                        prompt=meta["config"]["prompt"], config=meta["config"],
                        zoom_boxes_xys=SQUARE["zooms"], inset_at="top-right", inset_in=inset,
                        inset_xy_in=[round(ix - sq_x, 4), round(iy, 4)], preview=full.name,
                        zooms=[p.name]))

    # --- bottom: six-method strip ---------------------------------------------------------
    secs = timings()
    bx, by, bs = STRIP_BOX
    for k, (method, label) in enumerate(STRIP):
        png, meta = source(method, "4096x4096", STRIP_PID)
        img = Image.open(png).convert("RGB")
        assert img.size == (4096, 4096)
        x0 = k * (col + G)
        fullp = save_jpeg(preview(img, col), ASSETS / f"strip_{method}_full.jpg", 88, 2)
        cropp = save_jpeg(img.crop((bx, by, bx + bs, by + bs)), ASSETS / f"strip_{method}_crop.jpg", 90, 0)
        node(x0, y_strip, col, col, fullp.name)
        node(x0, y_strip + col + gv, col, col, cropp.name)
        sx = col / 4096
        box(x0 + bx * sx, y_strip + by * sx, x0 + (bx + bs) * sx, y_strip + (by + bs) * sx)
        frame(x0, y_strip + col + gv, x0 + col, y_strip + 2 * col + gv, "cropframe")
        ours = method == "sca_nb"
        name = r"\textbf{%s}" % label if ours else label
        tex.append(r"\node[head%s,anchor=base] at (%.4f,%.4f) {%s};"
                   % ("ours" if ours else "", x0 + col / 2, -(y_strip - 0.042), name))
        key = TIMING_KEY.get(method)
        t = secs.get(key) if key else None
        if t is not None:
            tex.append(r"\node[%s,anchor=north west] at (%.4f,%.4f) {%.1f\,s};"
                       % ("badgeours" if ours else "badge", x0 + M, -(y_strip + M), t))
        records.append(dict(role="strip " + method, source=str(png), png_sha256=sha(png),
                            prompt=meta["config"]["prompt"], config=meta["config"],
                            crop_xys=STRIP_BOX, seconds_batch1_median=t,
                            full=fullp.name, crop=cropp.name))

    total_h = y_strip + 2 * col + gv
    doc = r"""%% Generated by fig_hero.py; edit the script, not this file.
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{tikz}
\graphicspath{{../teaser/}}
\input{sca_colors.tex}
\begin{document}
\begin{tikzpicture}[x=1in,y=1in,
  halo/.style={draw=black,draw opacity=0.35,line width=1.15pt},
  leaderhalo/.style={draw=black,draw opacity=0.22,line width=0.85pt},
  zoombox/.style={draw=scaZoom,line width=0.55pt},
  zoomframe/.style={draw=scaZoom,line width=0.7pt},
  cropframe/.style={draw=scaZoom,line width=0.5pt},
  leader/.style={draw=scaZoom,line width=0.4pt},
  badge/.style={fill=black,fill opacity=0.58,text opacity=1,text=white,rounded corners=1.2pt,
                inner xsep=1.8pt,inner ysep=1.15pt,font=\sffamily\fontsize{6.5}{7}\selectfont},
  badgeours/.style={fill=scaAccent,fill opacity=0.92,text opacity=1,text=white,rounded corners=1.2pt,
                inner xsep=1.8pt,inner ysep=1.15pt,font=\sffamily\bfseries\fontsize{6.5}{7}\selectfont},
  head/.style={font=\sffamily\fontsize{8.5}{10}\selectfont,text=scaInk},
  headours/.style={font=\sffamily\fontsize{8.5}{10}\selectfont,text=scaAccent},
  ]
\useasboundingbox (0,0) rectangle (%.4f,%.4f);
%s
\end{tikzpicture}
\end{document}
""" % (W, -total_h, "\n".join(tex))
    TEX.write_text(doc)
    subprocess.run([TECTONIC, "-X", "compile", TEX.name, "--outdir", str(FIG)], cwd=HERE,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    produced = FIG / "teaser.pdf"
    assert produced == OUT_PDF and produced.exists()

    caption = (
        r"\textbf{\sca{} keeps layout and fine detail at $4096^2$ and $2048\times6144$ without "
        r"training, and at $4096^2$ runs $%.2f\times$ faster than AReA.} "
        r"Top: curated \sca{} outputs with native-pixel zooms. Bottom: one curated prompt and the "
        r"same $768\times768$ window for six methods, seed 0 ($^{*}$UltraImage uses its two-stage "
        r"defaults). Badges: batch-1 seconds per image on one GH200, where measured."
        % (secs["area"] / secs["sca_nb"]))
    SNIPPET.write_text("\n".join([
        "% Generated by src/fig_hero.py (TikZ source src/teaser.tex, assets figures/teaser/).",
        "% Intended placement: top of page 2 (or page 1), full text width; see DESIGN_CHANGES.md.",
        r"\begin{figure*}[t]",
        r"  \centering",
        r"  \includegraphics[width=\linewidth]{figures/teaser.pdf}",
        r"  \caption{" + caption + "}",
        r"  \label{fig:teaser}",
        r"\end{figure*}", ""]))
    (ASSETS / "provenance.json").write_text(json.dumps(dict(
        selection="Curated for the hero by visual inspection of SCA outputs; no metric ranking. "
                  "The six-method prompt and window are those of the earlier hero and teaser.",
        timings_source="results/latency_curve_summary.csv (4096x4096 medians, batch 1, one GH200)",
        width_in=W, height_in=round(total_h, 4), pdf=str(OUT_PDF.relative_to(REPO)),
        records=records, asset_sha256={p.name: sha(p) for p in sorted(ASSETS.glob("*.jpg"))}),
        indent=2) + "\n")
    print(f"teaser.pdf {OUT_PDF.stat().st_size/1e6:.2f} MB, height {total_h:.3f} in, "
          f"assets {sum(p.stat().st_size for p in ASSETS.glob('*.jpg'))/1e6:.2f} MB")


if __name__ == "__main__":
    main()
