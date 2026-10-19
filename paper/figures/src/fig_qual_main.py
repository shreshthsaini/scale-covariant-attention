"""Main qualitative comparison (fig:qual_main) as a TikZ-composed PDF.

Four of the six prompts of the earlier qual_main figure, at 4096^2, seed 0:
full attention (full image only, since it collapses), AReA and SCA (full image plus
the same native 512x512 crop). Crop windows use the earlier automatic rule:
the window with the most edge energy averaged over full attention, AReA and
SCA (fig_qual.py auto_box, reproduced here). No metric selected any prompt.

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_qual_main.py
"""
from __future__ import annotations

import hashlib
import json
import os
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
CROP = 512
DPI = 600


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


def auto_box(imgs, size, f=8):
    """Same rule as fig_qual.auto_box: max mean edge energy, away from the border."""
    a = np.mean([detail_map(i, f) for i in imgs], axis=0)
    k = size // f
    ii = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    m = max(1, a.shape[0] // 16)
    s[:m, :] = s[-m:, :] = -1
    s[:, :m] = s[:, -m:] = -1
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return int(x * f), int(y * f), size


def main() -> None:
    scafig.write_tikz_colors(str(HERE / "sca_colors.tex"))
    g_in, g_out = 0.025, 0.085          # gutter inside a method group, between groups
    tile = (W - 2 * g_in - 2 * g_out) / 5
    head = 0.17
    gv = 0.035
    xs = [0.0]
    xs.append(xs[-1] + tile + g_out)    # AReA full
    xs.append(xs[-1] + tile + g_in)     # AReA crop
    xs.append(xs[-1] + tile + g_out)    # SCA full
    xs.append(xs[-1] + tile + g_in)     # SCA crop
    tex, records = [], []

    def node(x, y, w, h, name):
        tex.append(r"\node[anchor=north west,inner sep=0pt] at (%.4f,%.4f) "
                   r"{\includegraphics[width=%.4fin,height=%.4fin]{%s}};" % (x, -y, w, h, name))

    def rect(x0, y0, x1, y1, style):
        tex.append(r"\draw[%s] (%.4f,%.4f) rectangle (%.4f,%.4f);" % (style, x0, -y0, x1, -y1))

    for r, pid in enumerate(PROMPTS):
        y = head + r * (tile + gv)
        imgs, metas, pngs = {}, {}, {}
        for m, _ in METHODS:
            pngs[m], metas[m] = source(m, pid)
            imgs[m] = Image.open(pngs[m]).convert("RGB")
            assert imgs[m].size == (4096, 4096)
        bx, by, bs = auto_box([imgs[m] for m, _ in METHODS], CROP)
        sx = tile / 4096
        for m, _ in METHODS:
            col = {"full": 0, "area": 1, "sca_nb": 3}[m]
            small = imgs[m].resize((round(DPI * tile),) * 2, Image.Resampling.LANCZOS)
            fp = ASSETS / f"cmp_{pid}_{m}.jpg"
            small.save(fp, "JPEG", quality=88, optimize=True, subsampling=2)
            node(xs[col], y, tile, tile, fp.name)
            if m != "full":  # full attention has no crop tile, so no window either
                rect(xs[col] + bx * sx, y + by * sx, xs[col] + (bx + bs) * sx, y + (by + bs) * sx, "halo")
                rect(xs[col] + bx * sx, y + by * sx, xs[col] + (bx + bs) * sx, y + (by + bs) * sx, "zoombox")
            rec = dict(method=m, source=str(pngs[m]), png_sha256=sha(pngs[m]), config=metas[m]["config"],
                       full=fp.name)
            if m != "full":
                cp = ASSETS / f"cmp_{pid}_{m}_crop.jpg"
                imgs[m].crop((bx, by, bx + bs, by + bs)).save(cp, "JPEG", quality=90, optimize=True,
                                                              subsampling=0)
                node(xs[col + 1], y, tile, tile, cp.name)
                rect(xs[col + 1], y, xs[col + 1] + tile, y + tile, "cropframe")
                rec["crop"] = cp.name
            records.append(dict(prompt_id=pid, crop_xys=[bx, by, bs], **rec))
    # headers
    centers = [xs[0] + tile / 2, (xs[1] + xs[2] + tile) / 2, (xs[3] + xs[4] + tile) / 2]
    for (m, label), cx in zip(METHODS, centers):
        style = "headours" if m == "sca_nb" else "head"
        txt = r"\textbf{%s}" % label if m == "sca_nb" else label
        tex.append(r"\node[%s,anchor=base] at (%.4f,%.4f) {%s};" % (style, cx, -(head - 0.055), txt))
    total = head + 4 * tile + 3 * gv
    doc = r"""%% Generated by fig_qual_main.py; edit the script, not this file.
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{graphicx}
\usepackage{tikz}
\graphicspath{{../qual_main/}}
\input{sca_colors.tex}
\begin{document}
\begin{tikzpicture}[x=1in,y=1in,
  halo/.style={draw=black,draw opacity=0.35,line width=1.1pt},
  zoombox/.style={draw=scaZoom,line width=0.55pt},
  cropframe/.style={draw=scaZoom,line width=0.5pt},
  head/.style={font=\sffamily\fontsize{8.5}{10}\selectfont,text=scaInk},
  headours/.style={font=\sffamily\fontsize{8.5}{10}\selectfont,text=scaAccent},
  sub/.style={font=\sffamily\fontsize{7}{8}\selectfont,text=scaMid}]
\useasboundingbox (0,0) rectangle (%.4f,%.4f);
%s
\end{tikzpicture}
\end{document}
""" % (W, -total, "\n".join(tex))
    TEX.write_text(doc)
    subprocess.run([TECTONIC, "-X", "compile", TEX.name, "--outdir", str(FIG)], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    caption = (r"\textbf{At $4096^2$ full attention collapses, AReA blurs or tiles fine structure, and "
               r"\sca{} keeps both layout and detail.} Four evaluation prompts, seed 0. Beside each AReA "
               r"and \sca{} image is the same native ${512\times512}$ window, placed automatically where "
               r"the three methods have the most edge energy. Random uncurated samples are in "
               r"Figure~\ref{fig:gallery_4096}.")
    SNIPPET.write_text("\n".join([
        "% Generated by src/fig_qual_main.py (TikZ source src/qual_main.tex, assets figures/qual_main/cmp_*).",
        r"\begin{figure}[tp]", r"  \centering",
        r"  \includegraphics[width=\linewidth]{figures/qual_main.pdf}",
        r"  \caption{" + caption + "}", r"  \label{fig:qual_main}", r"\end{figure}", ""]))
    (ASSETS / "cmp_provenance.json").write_text(json.dumps(dict(
        selection="Four of the six prompts of the earlier qual_main figure (fig_qual.py MAIN); "
                  "no metric ranking. Crop rule: fig_qual.auto_box over full, area, sca.",
        records=records, height_in=round(total, 4),
        asset_sha256={p.name: sha(p) for p in sorted(ASSETS.glob("cmp_*.jpg"))}), indent=2) + "\n")
    size = sum(p.stat().st_size for p in ASSETS.glob("cmp_*.jpg"))
    print(f"qual_main.pdf {OUT_PDF.stat().st_size/1e6:.2f} MB, height {total:.3f} in, assets {size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
