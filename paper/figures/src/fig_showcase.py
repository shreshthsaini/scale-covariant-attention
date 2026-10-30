"""Appendix showcase (fig:hero): curated SCA outputs at three aspect ratios.

Layout at the 5.5 in text width: a full-width 2048x6144 panorama with two
native-pixel insets, a row of four 4096^2 images with one inset each, and a row
of two 2048x4096 images (sizes are height x width, as in the paper). Every image
is a final-operator SCA output (sca_v2, out/dupfix/sca_v2) for an evaluation prompt
at seed 0 (FLUX.1-dev, 28 steps, guidance 4.5; asserted from the sidecars), the twin
(same prompt, prompt id and derived seed; imgsel.final_twin) of the published
operator's (sca_nb) image on which the prompts and windows were chosen. The final
images were viewed whole and at native pixels on 2026-10-30 (FINAL_VIEW). Images were chosen by visual
inspection for variety and clean results; no metric ranked them, except that the
still-life replacement of 2026-10-20 came from a shortlist with ImageReward and MUSIQ_c in
the top half of the 195 prompts, then a full 1:1 audit by eye. Prompts shown
elsewhere in the paper (page-1 hero, main qualitative figure, ablation and
failure figures) and the development prompts are excluded.

Everything is generated: JPEG assets (deterministic PIL encoding), a TikZ
source with vector boxes, leader lines and badges, its PDF (tectonic), the
snippet, and provenance with source hashes. Styles follow fig_hero.py.

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_showcase.py
"""
from __future__ import annotations

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
ASSETS = FIG / "showcase"
TEX = HERE / "showcase.tex"
OUT_PDF = FIG / "showcase.pdf"
SNIPPET = FIG / "snippets" / "hero.tex"
TECTONIC = os.path.expanduser("~/.local/bin/tectonic")

W = 5.5           # text width, inches
G = 0.05          # gutter
DPI = 600         # pixel budget for downsampled full images
M = 0.035         # margin of insets and badges from image edges
LW = 0.6 / 72     # width of the one neutral frame line (windows and insets), inches
FRAME = "A6ADB4"  # the one neutral frame colour; leader lines use it too
PANO_INSET = 0.80
SQ_INSET = 0.44
METHOD = "sca_v2"  # the final operator; the choices below were made on the published operator (sca_nb)
FINAL_VIEW = {  # the final operator's images, viewed whole and at native pixels (2026-10-30)
    "Great Wall": ("the lit wall segment and watchtower on the left ridge of the published image are gone (bare "
                   "ridge); the first window moved to the lit watchtowers on the main wall (PANO); otherwise clean"),
    "portrait": "clean; one pair of earrings, a different earring design",
    "illustration": ("clean of repeats: one waterfall and one lit spire (the published image has a second "
                     "waterfall and a second lit tower); the spire's tower carries a boxy block; one small lit "
                     "tower on a hill at left"),
    "still life": "clean",
    "ice climber": "clean; the tiny figures at the top of the published image are gone",
    "interior": "clean of repeats; a dress form on a stand with a small shelf at the back",
    "river town": "clean; nearly identical to the published image"}

# Zoom windows are (x, y, size) in source pixels. `at` places the inset inside its
# image: a corner, or "top" (centred above the window). `margin` is the (horizontal,
# vertical) gap to the image edges, M by default; 0 docks that side against the edge.
# `side` says where the inset lies relative to the window and picks the corner pairs
# joined by leader lines; None draws none. Positions were chosen by eye: each inset
# covers the least informative region of its image that is large enough, and leader
# lines and badges stay off the main subject.
# 2026-10-20: the village street 7543519915112968451 was replaced. Its restaurant signs were gibberish at
# 1:1 and its right third (including the shutter inset) was smeared.
# 2026-10-30: the final operator's image has no wall on the left ridge, where the first window (1530, 1000, 384),
# inset "top", side "above", framed a lit watchtower; the window moved to the lit watchtowers on the main wall and
# its inset to the top-right corner, on the sky and the sea of clouds.
PANO = dict(res="2048x6144", pid="4230887929316158402", what="Great Wall",
            zooms=[dict(box=(3440, 560, 384), at="top-right", side="right"),      # lit watchtowers: inset on the sky
                   dict(box=(3330, 1600, 320), at="bottom-right", side="right",   # snowy parapet: inset on the
                        margin=(0.0, 0.0))])                                      # dark slope, docked
SQUARES = [
    # Earring: inset on the sweater and the wall. A leader from the ear to that corner
    # would cross the face and the necklace, so this inset has none; the shared yellow
    # pairs it with its window.
    dict(pid="minus1066207185399086750", what="portrait",
         zoom=dict(box=(1160, 2080, 384), at="bottom-right", side=None)),
    # Lit spire: inset on the empty sky. The badge sits on the dark rocks because the
    # top-left corner holds the second peak and its lit tower.
    dict(pid="minus6845347395255545991", what="illustration", badge="bottom-right",
         zoom=dict(box=(1850, 420, 384), at="top-right", side="right")),
    # Vase rim and daisies: inset docked in the corner on the empty tablecloth, clear of the vase. The badge moves to the bottom
    # left because the top-left corner holds the tip of the left branch. (Replaced the pear still life
    # minus947425340308551929 on 2026-10-20: SCA drew a see-through third bottle and a faded second bowl.)
    dict(pid="minus8919475713516982506", what="still life", badge="bottom-left",
         zoom=dict(box=(1880, 1900, 384), at="bottom-right", side="right", margin=(0.0, 0.0))),
    # Face: inset on the flat ice wall right of the crevice.
    dict(pid="997293625843621481", what="ice climber",
         zoom=dict(box=(1900, 1250, 384), at="top-right", side="right")),
]
WIDES = [dict(res="2048x4096", pid="7414267858891695058", what="interior"),
         dict(res="2048x4096", pid="minus5370774387421928008", what="river town")]

# Prompts shown elsewhere in the paper; the showcase must not repeat them.
SHOWN_ELSEWHERE = {
    "minus2742856837225126754", "minus8737279395568207187", "3657338173664010496",   # Figure 1 (teaser)
    "1269964769307095670", "5849277588805396896", "minus8343371271909676658",          # Figure 1 (revised)
    "minus5023570400257826529", "minus2679953061215100202",                            # progressive
    "minus1057810871814672649", "8142852677968042257", "2100482621711410240",        # qual_main
    "minus2729538226706404616", "minus6394968997939197542", "1052899493017962912",
    "minus2857555394095239941", "1516768667427572324", "4525259556400020626",        # ablation grid
    "7009273464006889449",
    "7396058529551312081", "1316997283360365550", "minus8648158440845565467",         # failures
    "minus702025498990232048",
    "7109958488611222679", "2232973817337527860", "minus548913239957610051",         # Figure 6 (zoom4k)
}


def norm(pid: str) -> str:
    return "minus" + pid[1:] if pid.startswith("m") and not pid.startswith("minus") else pid


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(res: str, pid: str) -> tuple[Path, dict]:
    png = Path(imgsel.final_png(res, pid)[0])  # checked: twin of the published image, sca_v2 flags
    meta = json.loads(png.with_suffix(".json").read_text())
    if meta.get("complete") is not True or meta.get("png_sha256") != sha(png):
        raise ValueError(f"incomplete or hash-mismatched source {png}")
    cfg = meta["config"]
    expect = dict(method=METHOD, res=res, seed=0, steps=28, guidance=4.5,
                  model="black-forest-labs/FLUX.1-dev")
    bad = {k: cfg.get(k) for k, v in expect.items() if cfg.get(k) != v}
    if bad:
        raise ValueError(f"{png.name}: config differs from the caption: {bad}")
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


def main() -> None:
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    scafig.write_tikz_colors(str(HERE / "sca_colors.tex"))
    dev = {norm(p) for p in imgsel.dev4_ids()}
    chosen = [PANO["pid"]] + [s["pid"] for s in SQUARES] + [w["pid"] for w in WIDES]
    clash = set(chosen) & ({norm(p) for p in SHOWN_ELSEWHERE} | dev)
    assert not clash and len(set(chosen)) == len(chosen), clash

    records, tex, written = [], [], []
    # Geometry, inches; origin at the top-left corner, y grows downward (negated in TikZ).
    pano_h = W / 3
    sq = (W - 3 * G) / 4
    y_sq = pano_h + G
    wide_w = (W - G) / 2
    y_wide = y_sq + sq + G
    total_h = y_wide + wide_w / 2

    def node(x, y, w, hh, name):
        tex.append(r"\node[anchor=north west,inner sep=0pt] at (%.4f,%.4f) "
                   r"{\includegraphics[width=%.4fin,height=%.4fin]{%s}};" % (x, -y, w, hh, name))

    def rect(x0, y0, x1, y1, style):
        tex.append(r"\draw[%s] (%.4f,%.4f) rectangle (%.4f,%.4f);" % (style, x0, -y0, x1, -y1))

    def leader(a, b):
        tex.append(r"\draw[leader] (%.4f,%.4f) -- (%.4f,%.4f);" % (a[0], -a[1], b[0], -b[1]))

    def badge(x, y, w, h, img, at="top-left"):
        """Size badge, height x width, top left; bottom right or left where the top left holds content."""
        if at == "top-left":
            anchor, bx, by = "north west", x + M, y + M
        elif at == "bottom-left":
            anchor, bx, by = "south west", x + M, y + h - M
        else:
            assert at == "bottom-right", at
            anchor, bx, by = "south east", x + w - M, y + h - M
        tex.append(r"\node[badge,anchor=%s] at (%.4f,%.4f) {%d\,$\times$\,%d};"   # height x width
                   % (anchor, bx, -by, img.height, img.width))

    def save_full(img, width_in, name):
        p = save_jpeg(preview(img, width_in), ASSETS / name, 90, 2)
        written.append(p)
        return p

    def inset(img, x0, y0, w, h, z, q, name):
        """Native crop of window z["box"] drawn as a framed q-inch inset inside the image."""
        bx, by, bs = z["box"]
        crop = save_jpeg(img.crop((bx, by, bx + bs, by + bs)), ASSETS / name, 90, 0)
        written.append(crop)
        s = w / img.width
        b0, b1 = (x0 + bx * s, y0 + by * s), (x0 + (bx + bs) * s, y0 + (by + bs) * s)
        at, (mx, my) = z["at"], z.get("margin", (M, M))
        if at == "top":
            ix = min(max((b0[0] + b1[0]) / 2 - q / 2, x0 + mx), x0 + w - mx - q)
            iy = y0 + my
        else:
            ix = x0 + mx if at.endswith("left") else x0 + w - mx - q
            iy = y0 + my if at.startswith("top") else y0 + h - my - q
        i0, i1 = (ix, iy), (ix + q, iy + q)
        assert not (i0[0] < b1[0] and b0[0] < i1[0] and i0[1] < b1[1] and b0[1] < i1[1]), name
        node(ix, iy, q, q, crop.name)
        if z["side"]:
            pairs = {"below": [((b0[0], b1[1]), i0), (b1, (i1[0], i0[1]))],
                     "above": [(b0, (i0[0], i1[1])), ((b1[0], b0[1]), i1)],
                     "right": [((b1[0], b0[1]), i0), (b1, (i0[0], i1[1]))],
                     "left": [(b0, (i1[0], i0[1])), ((b0[0], b1[1]), i1)]}[z["side"]]
            for a, b in pairs:
                leader(a, b)
        rect(*b0, *b1, "zoombox")
        # On a docked side the frame moves in by half its line width, so it stays inside the image.
        dx, dy = (LW / 2 if mx == 0 else 0.0), (LW / 2 if my == 0 else 0.0)
        rect(i0[0] + dx, i0[1] + dy, i1[0] - dx, i1[1] - dy, "zoomframe")
        return dict(box_xys=list(z["box"]), crop=crop.name, crop_px=bs,
                    inset_in=q, magnification=round((q / bs) / (w / img.width), 2),
                    inset_at=at, inset_margin_in=[mx, my],
                    inset_xy_in=[round(ix - x0, 4), round(iy - y0, 4)], leaders=z["side"] is not None)

    def record(role, png, meta, full, zooms):
        records.append(dict(role=role, source=str(png), png_sha256=sha(png),
                            prompt=meta["config"]["prompt"], config=meta["config"],
                            preview=full.name, zooms=zooms))

    # --- panorama ---------------------------------------------------------------------------
    png, meta = source(PANO["res"], PANO["pid"])
    img = Image.open(png).convert("RGB")
    assert img.size == (6144, 2048)
    full = save_full(img, W, "pano.jpg")
    node(0, 0, W, pano_h, full.name)
    zooms = [inset(img, 0, 0, W, pano_h, z, PANO_INSET, f"pano_zoom{k + 1}.jpg")
             for k, z in enumerate(PANO["zooms"])]
    badge(0, 0, W, pano_h, img)
    record("panorama: " + PANO["what"], png, meta, full, zooms)

    # --- four squares -----------------------------------------------------------------------
    for k, spec in enumerate(SQUARES):
        png, meta = source("4096x4096", spec["pid"])
        img = Image.open(png).convert("RGB")
        assert img.size == (4096, 4096)
        x0 = k * (sq + G)
        full = save_full(img, sq, f"square{k + 1}.jpg")
        node(x0, y_sq, sq, sq, full.name)
        z = inset(img, x0, y_sq, sq, sq, spec["zoom"], SQ_INSET, f"square{k + 1}_zoom.jpg")
        badge(x0, y_sq, sq, sq, img, spec.get("badge", "top-left"))
        record("square: " + spec["what"], png, meta, full, [z])

    # --- two 2048x4096 ----------------------------------------------------------------------
    for k, spec in enumerate(WIDES):
        png, meta = source(spec["res"], spec["pid"])
        img = Image.open(png).convert("RGB")
        assert img.size == (4096, 2048)
        x0 = k * (wide_w + G)
        full = save_full(img, wide_w, f"wide{k + 1}.jpg")
        node(x0, y_wide, wide_w, wide_w / 2, full.name)
        badge(x0, y_wide, wide_w, wide_w / 2, img)
        record("wide: " + spec["what"], png, meta, full, [])

    for stale in sorted(set(ASSETS.glob("*.jpg")) - set(written)):
        stale.unlink()

    doc = r"""%% Generated by fig_showcase.py; edit the script, not this file.
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{tikz}
\graphicspath{{../showcase/}}
\input{sca_colors.tex}
\definecolor{scaFrame}{HTML}{%s}
\begin{document}
\begin{tikzpicture}[x=1in,y=1in,
  zoombox/.style={draw=scaFrame,line width=0.6pt},
  zoomframe/.style={draw=scaFrame,line width=0.6pt},
  leader/.style={draw=scaFrame,line width=0.4pt},
  badge/.style={fill=black,fill opacity=0.58,text opacity=1,text=white,rounded corners=1.2pt,
                inner xsep=2.2pt,inner ysep=1.3pt,font=\sffamily\fontsize{7}{7.5}\selectfont},
  ]
\useasboundingbox (0,0) rectangle (%.4f,%.4f);
%s
\end{tikzpicture}
\end{document}
""" % (FRAME, W, -total_h, "\n".join(tex))
    TEX.write_text(doc)
    subprocess.run([TECTONIC, "-X", "compile", TEX.name, "--outdir", str(FIG)], cwd=HERE,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert OUT_PDF.exists()

    caption = (
        r"\textbf{\sca{} keeps layout and fine detail across three aspect ratios without training.} "
        r"Curated \sca{} outputs at $4096^2$, $2048\times4096$ and $2048\times6144$, picked by eye for "
        r"variety; insets are 1:1 crops of the outlined windows.")
    comments = ["%% %s %s (%s): %s" % (r["role"], r["config"]["prompt_id"].split("_")[-1],
                                       r["config"]["res"], " ".join(r["prompt"].split())[:140])
                for r in records]
    SNIPPET.write_text("\n".join([
        "% Generated by src/fig_showcase.py (TikZ source src/showcase.tex, assets figures/showcase/);",
        "% edit the script, not this file. Curated SCA outputs, seed 0; sources (role, prompt id, HxW):",
        *comments,
        r"\begin{figure*}[!ht]",
        r"  \centering",
        r"  \includegraphics[width=\linewidth]{figures/showcase.pdf}",
        r"  \caption{" + caption + "}",
        r"  \label{fig:hero}",
        r"\end{figure*}", ""]))
    (ASSETS / "provenance.json").write_text(json.dumps(dict(
        selection="Curated by visual inspection of SCA (sca_nb) seed-0 outputs for variety and clean "
                  "results; no metric ranking, except the 2026-10-20 still-life replacement, shortlisted "
                  "with ImageReward and MUSIQ_c in the top half of 195 and confirmed by a 1:1 audit. "
                  "Prompts shown in other paper figures and the four development prompts are excluded. "
                  "Since 2026-10-30 every image is the final operator's (sca_v2) image of the same prompt "
                  "and seed; the windows are unchanged except the panorama's first (see final_operator).",
        final_operator=dict(method=METHOD, images="out/dupfix/sca_v2 (imgsel.final_twin of each published image)",
                            viewed=FINAL_VIEW,
                            moved_window=dict(image="Great Wall 4230887929316158402", old_box_xys=[1530, 1000, 384],
                                              old_inset="top, leaders above", new_box_xys=[3440, 560, 384],
                                              new_inset="top-right, leaders right",
                                              why="the lit watchtower it framed does not exist in the final "
                                                  "image")),
        width_in=W, height_in=round(total_h, 4), dpi_full_images=DPI, pdf=str(OUT_PDF.relative_to(REPO)),
        records=records, asset_sha256={p.name: sha(p) for p in sorted(ASSETS.glob("*.jpg"))}),
        indent=2) + "\n")
    print(f"showcase.pdf {OUT_PDF.stat().st_size/1e6:.2f} MB, height {total_h:.3f} in, "
          f"assets {sum(p.stat().st_size for p in ASSETS.glob('*.jpg'))/1e6:.2f} MB")


if __name__ == "__main__":
    main()
