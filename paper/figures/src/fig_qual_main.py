"""Main qualitative comparison (fig:qual_main) as a TikZ-composed PDF.

Four prompts at 4096^2, seed 0, five tiles of one height per row: the full-attention image (it
collapses, so it needs no crop), then AReA and SCA, each a full image plus the same native-pixel
crop, which is wider than tall (CROP_W x CROP_H). House style: method names once, SCA (ours) last
in the accent colour, 1 to 2 pt white gutters, no borders, one neutral 0.6 pt frame colour for the
crop windows and crop tiles, no text inside the images, 8 pt headers.

Selection (recorded in figures/qual_main/cmp_provenance.json):
* Prompts are curated. The six prompts of fig_qual.py MAIN were picked by hand for variety from
  the 191 evaluation prompts; the 2026-10-18 redesign kept four of them. No metric ranked or
  selected any prompt.
* Crop windows are automatic, never placed by hand: auto_box takes the CROP_W x CROP_H window
  with the most edge energy (PIL FIND_EDGES response of the 8x box-downsampled greyscale image)
  averaged over full attention, AReA and SCA, at least 256 px from every border. With a square
  512 window it is exactly fig_qual.auto_box; --check-square verifies that against the windows
  recorded for the previous version.

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_qual_main.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import imgsel
import scafig

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = FIG.parent.parent
ASSETS = FIG / "qual_main"
TEX = HERE / "qual_main.tex"
OUT_PDF = FIG / "qual_main.pdf"
SNIPPET = FIG / "snippets" / "qual_main.tex"
TECTONIC = os.path.expanduser("~/.local/bin/tectonic")

W = 5.5
PROMPTS = ["2100482621711410240", "m6394968997939197542", "m2729538226706404616", "m1057810871814672649"]
METHODS = [("full", "Full attention"), ("area", "AReA"), ("sca_nb", "SCA (ours)")]
CROP_W, CROP_H = 736, 512   # native crop window, width x height
G_IN, G_OUT = 1 / 72, 2 / 72  # white gutters (in): 1 pt inside a group, 2 pt between groups
HEAD, GV = 0.14, 2 / 72     # header band and row gap (in)
DPI = 600                   # downscaled full images: at most 600 dpi of printed width
FRAME = "A6ADB4"            # the one neutral frame colour (crop windows and crop tiles)
OLD_BOXES = {"2100482621711410240": (928, 3288), "m6394968997939197542": (1408, 1880),
             "m2729538226706404616": (1520, 256), "m1057810871814672649": (1896, 3328)}  # 512^2, v1


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(method: str, pid: str) -> tuple[Path, dict]:
    root = Path(imgsel.OUT) / "main" / method / "4096x4096"
    stem = "minus" + pid[1:] if pid.startswith("m") else pid   # short id -> file id
    hits = sorted(root.glob(f"a4k_eval_4096_{stem}_*_seed0.png"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{method} {pid}: {hits}")
    meta = json.loads(hits[0].with_suffix(".json").read_text())
    if meta.get("complete") is not True or meta.get("png_sha256") != sha(hits[0]):
        raise ValueError(f"incomplete or hash-mismatched source {hits[0]}")
    return hits[0], meta


def detail_map(img, f=8):
    g = img.convert("L").resize((img.width // f, img.height // f), Image.BOX)
    a = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 0
    return a


def auto_box(imgs, w, h, f=8):
    """fig_qual.auto_box for a w x h window: max mean edge energy, at least 1/16 side from the border."""
    a = np.mean([detail_map(i, f) for i in imgs], axis=0)
    kh, kw = h // f, w // f
    ii = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = ii[kh:, kw:] - ii[:-kh, kw:] - ii[kh:, :-kw] + ii[:-kh, :-kw]
    m = max(1, a.shape[0] // 16)
    s[:m, :] = s[-m:, :] = -1
    s[:, :m] = s[:, -m:] = -1
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return int(x * f), int(y * f), w, h


def layout():
    """Column x positions and the tile height; every tile in a row shares the height."""
    aspect = CROP_W / CROP_H
    tile = (W - 2 * G_IN - 2 * G_OUT) / (3 + 2 * aspect)
    crop_w = aspect * tile
    x_area = tile + G_OUT
    xs = dict(full=0.0, area=x_area, area_crop=x_area + tile + G_IN)
    xs["sca_nb"] = xs["area_crop"] + crop_w + G_OUT
    xs["sca_nb_crop"] = xs["sca_nb"] + tile + G_IN
    assert abs(xs["sca_nb_crop"] + crop_w - W) < 1e-9
    return xs, tile, crop_w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="trial run: write the PDF, snippet and assets here instead of the paper tree")
    ap.add_argument("--check-square", action="store_true",
                    help="also verify that auto_box with a 512x512 window reproduces the v1 windows")
    a = ap.parse_args()
    assets, out_pdf, snippet, tex = ASSETS, OUT_PDF, SNIPPET, TEX
    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        assets = a.out_dir / "qual_main"
        assets.mkdir(exist_ok=True)
        out_pdf, snippet, tex = a.out_dir / "qual_main.pdf", a.out_dir / "qual_main_snippet.tex", \
            a.out_dir / "qual_main.tex"
    scafig.write_tikz_colors(str(HERE / "sca_colors.tex"))
    xs, tile, crop_w = layout()
    sx = tile / 4096
    tex_lines, records, boxes = [], [], {}

    def node(x, y, w, h, name):
        tex_lines.append(r"\node[anchor=north west,inner sep=0pt] at (%.4f,%.4f) "
                         r"{\includegraphics[width=%.4fin,height=%.4fin]{%s}};" % (x, -y, w, h, name))

    def rect(x0, y0, x1, y1, style):
        tex_lines.append(r"\draw[%s] (%.4f,%.4f) rectangle (%.4f,%.4f);" % (style, x0, -y0, x1, -y1))

    for r, pid in enumerate(PROMPTS):
        y = HEAD + r * (tile + GV)
        imgs, metas, pngs = {}, {}, {}
        for m, _ in METHODS:
            pngs[m], metas[m] = source(m, pid)
            imgs[m] = Image.open(pngs[m]).convert("RGB")
            assert imgs[m].size == (4096, 4096)
        trio = [imgs[m] for m, _ in METHODS]
        if a.check_square:
            old = auto_box(trio, 512, 512)[:2]
            assert old == OLD_BOXES[pid], (pid, old, OLD_BOXES[pid])
            print(f"{pid}: square 512 window reproduces v1 {old}")
        bx, by, bw, bh = auto_box(trio, CROP_W, CROP_H)
        boxes[pid] = [bx, by, bw, bh]
        for m, _ in METHODS:
            small = imgs[m].resize((round(DPI * tile),) * 2, Image.Resampling.LANCZOS)
            fp = assets / f"cmp_{pid}_{m}.jpg"
            small.save(fp, "JPEG", quality=88, optimize=True, subsampling=2)
            node(xs[m], y, tile, tile, fp.name)
            rec = dict(method=m, source=str(pngs[m]), png_sha256=sha(pngs[m]), config=metas[m]["config"],
                       full=fp.name, full_px=small.width)
            if m != "full":  # full attention has no crop tile, so no window either
                rect(xs[m] + bx * sx, y + by * sx, xs[m] + (bx + bw) * sx, y + (by + bh) * sx, "zoombox")
                cp = assets / f"cmp_{pid}_{m}_crop.jpg"
                imgs[m].crop((bx, by, bx + bw, by + bh)).save(cp, "JPEG", quality=90, optimize=True,
                                                              subsampling=0)
                node(xs[m + "_crop"], y, crop_w, tile, cp.name)
                i = 0.3 / 72  # half the line width: the frame stays inside the tile
                rect(xs[m + "_crop"] + i, y + i, xs[m + "_crop"] + crop_w - i, y + tile - i, "cropframe")
                rec["crop"] = cp.name
            records.append(dict(prompt_id=pid, crop_xywh=[bx, by, bw, bh], **rec))
    # headers: each method named once, centred over its group; SCA (ours) last, bold, in the accent colour
    centers = {"full": tile / 2, "area": (xs["area"] + xs["area_crop"] + crop_w) / 2,
               "sca_nb": (xs["sca_nb"] + xs["sca_nb_crop"] + crop_w) / 2}
    labels = {"full": "Full attention", "area": "AReA", "sca_nb": r"\textbf{SCA (ours)}"}
    for m, _ in METHODS:
        style = "headours" if m == "sca_nb" else "head"
        tex_lines.append(r"\node[%s,anchor=base] at (%.4f,%.4f) {%s};" % (style, centers[m], -(HEAD - 0.04),
                                                                          labels[m]))
    total = HEAD + 4 * tile + 3 * GV
    doc = r"""%% Generated by fig_qual_main.py; edit the script, not this file.
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{graphicx}
\usepackage{tikz}
\graphicspath{{%s/}}
\input{sca_colors.tex}
\definecolor{scaFrame}{HTML}{%s}
\begin{document}
\begin{tikzpicture}[x=1in,y=1in,
  zoombox/.style={draw=scaFrame,line width=0.6pt},
  cropframe/.style={draw=scaFrame,line width=0.6pt},
  head/.style={font=\sffamily\fontsize{8}{9}\selectfont,text=scaInk},
  headours/.style={font=\sffamily\fontsize{8}{9}\selectfont,text=scaAccent}]
\useasboundingbox (0,0) rectangle (%.4f,%.4f);
%s
\end{tikzpicture}
\end{document}
""" % (os.path.relpath(assets, HERE) if not a.out_dir else str(assets), FRAME, W, -total, "\n".join(tex_lines))
    tex.write_text(doc)
    if a.out_dir:
        shutil.copy(HERE / "sca_colors.tex", a.out_dir / "sca_colors.tex")
    subprocess.run([TECTONIC, "-X", "compile", tex.name, "--outdir", str(out_pdf.parent)], cwd=tex.parent,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    caption = (r"\textbf{At $4096^2$ full attention collapses, AReA blurs or tiles fine structure, and "
               r"\sca{} keeps both layout and detail.} Four curated prompts, seed~0, picked for variety from "
               r"the 191 evaluation prompts. Beside each AReA and \sca{} image is a 1:1 crop of the same "
               r"${%d\times%d}$ window (placement: Appendix~\ref{app:galleries}). Uncurated samples: "
               r"Figure~\ref{fig:gallery_4096}." % (CROP_W, CROP_H))
    snippet.write_text("\n".join([
        "% Generated by src/fig_qual_main.py (TikZ source src/qual_main.tex, assets figures/qual_main/cmp_*).",
        r"\begin{figure}[tp]", r"  \centering",
        r"  \includegraphics[width=\linewidth]{figures/qual_main.pdf}",
        r"  \caption{" + caption + "}", r"  \label{fig:qual_main}", r"\end{figure}", ""]))
    (assets / "cmp_provenance.json").write_text(json.dumps(dict(
        selection=("Prompts are curated: the six fig_qual.py MAIN prompts were picked by hand for variety from "
                   "the 191 evaluation prompts, and the 2026-10-18 redesign kept four of them (dropped "
                   "8142852677968042257 and 1052899493017962912). No metric ranked or selected any prompt."),
        crop_rule=(f"Automatic, not hand-placed: fig_qual_main.auto_box over full, area and sca_nb, i.e. the "
                   f"{CROP_W}x{CROP_H} (width x height) window with the largest sum of the mean PIL FIND_EDGES "
                   f"response of the 8x box-downsampled greyscale images, at least 256 px from every border. "
                   f"With a 512x512 window this is exactly fig_qual.auto_box (checked by --check-square)."),
        crop_windows_xywh=boxes, previous_square_windows_xy_512=OLD_BOXES,
        layout_in=dict(width=W, height=round(total, 4), tile=round(tile, 4), crop_width=round(crop_w, 4),
                       gutters=dict(inside_group=round(G_IN, 4), between_groups=round(G_OUT, 4),
                                    rows=round(GV, 4)), header=HEAD, frame_hex=FRAME, frame_pt=0.6),
        jpeg=dict(full="Lanczos to 600 dpi of printed width, quality 88, 4:2:0",
                  crop="native pixels, quality 90, 4:4:4"),
        records=records, height_in=round(total, 4),
        asset_sha256={p.name: sha(p) for p in sorted(assets.glob("cmp_*.jpg"))}), indent=2) + "\n")
    size = sum(p.stat().st_size for p in assets.glob("cmp_*.jpg"))
    print(f"{out_pdf} {out_pdf.stat().st_size / 1e6:.2f} MB, height {total:.3f} in, tile {tile:.3f} in, "
          f"crop {crop_w:.3f} in, assets {size / 1e6:.2f} MB, windows {boxes}")


if __name__ == "__main__":
    main()
