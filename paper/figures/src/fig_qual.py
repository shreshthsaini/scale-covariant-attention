"""Qualitative figures for the SCA paper: image tiles (JPEG) plus a LaTeX snippet per figure.

Regenerate one figure:   <sca env python> paper/figures/src/fig_qual.py hero
All figures:             <sca env python> paper/figures/src/fig_qual.py all
Options: --methods area sca (gallery columns; any imgsel.ROOTS label, e.g. a new baseline once its
         ROOTS entry exists), --res 4096x4096, --n 32, --seed 0, --workers 16.

Every tile is either a downscaled full image (Lanczos) or a native 1:1 crop (never resampled).
Crop windows are given by hand in the specs below, or chosen automatically as the window with the
highest mean absolute Laplacian, averaged over all methods shown in that row so no method is
favoured. Output is deterministic: tiles go to paper/figures/<figure>/ and the snippet to
paper/figures/snippets/<figure>.tex. Sources live on scratch (purge-eligible):
/scratch/09032/saini_2/sca/out/...

The appendix grids (gallery, gallery_wide, ablation, failures) share snippets/_scagrid.tex: clean
JPEG tiles, vector crop windows and frames drawn by TikZ, and one tile height per figure. Their
prompts, crop windows and source hashes are frozen in paper/figures/<figure>/provenance.json on the
first run; later runs reuse that record and refuse changed sources. hero and qual_main keep the
older raster layout. failure_fallback re-encodes the tiles of the fallback fig:failure in main.tex.
"""

import argparse
import json
import os
import random
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import imgsel

SNIP = os.path.join(imgsel.FIGDIR, "snippets")
COLORS = ["#E69F00", "#56B4E9", "#009E73", "#D55E00"]  # Okabe-Ito, matches orx_figstyle
GAP = 0.004  # gap between tiles, fraction of \linewidth
LABW = 0.022  # width reserved for a rotated row label
NAMES = {"sca": r"\textbf{SCA (ours)}", "area": "AReA", "full": "Full attention",
         "abl_v0": "No gate", "abl_local": r"\scaTwo{$\pi{=}1$}{local only}", "abl_uniform": r"Uniform $\pi(\sigma)$",
         "abl_global": r"\scaTwo{$\pi{=}0$}{global only}", "abl_mean": "Mean pooling", "abl_layer": r"Per-layer $\pi$",
         "dev_w16": "No gate, window 16", "dev_w32": "No gate, window 32",
         "dypes": "DyPE", "sega": "SEGA", "ultraimage": "UltraImage"}


def wxh(res):
    h, w = res.split("x")
    return rf"{w}$\times${h}"


# ---------------------------------------------------------------- image ops
def detail_map(img, f=8):
    g = img.convert("L").resize((img.width // f, img.height // f), Image.BOX)
    a = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 0
    return a


def auto_box(imgs, size, f=8):
    """Native box (x, y, size) maximising mean edge energy, averaged over imgs (all same size)."""
    a = np.mean([detail_map(i, f) for i in imgs], axis=0)
    k = size // f
    ii = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    m = max(1, a.shape[0] // 16)  # keep away from the border
    s[:m, :] = s[-m:, :] = -1
    s[:, :m] = s[:, -m:] = -1
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return (int(x * f), int(y * f), size)


def outlined(img, max_side, boxes):
    """Downscaled full image with each crop window outlined in its colour."""
    t = imgsel.fit(img, max_side)
    s = t.width / img.width
    lw = max(2, round(t.width / 220))
    d = ImageDraw.Draw(t)
    for (x, y, sz), c in boxes:
        d.rectangle([x * s, y * s, (x + sz) * s - 1, (y + sz) * s - 1], outline=c, width=lw)
    return t


def framed(img, color):
    t = img.copy()
    lw = max(3, round(t.width / 128))
    ImageDraw.Draw(t).rectangle([0, 0, t.width - 1, t.height - 1], outline=color, width=lw)
    return t


def make_tiles(job):
    """job: dict(out, label, res, sid, side, boxes=[(box, color)], crops=[(box, color, name)])."""
    img = imgsel.load(imgsel.find(job["label"], job["res"], job["sid"]))
    out = {}
    if job.get("side"):
        p = os.path.join(job["out"], f"{job['name']}.jpg")
        imgsel.save_jpeg(outlined(img, job["side"], job.get("boxes", [])), p)
        out["full"] = p
    for box, color, cname in job.get("crops", []):
        p = os.path.join(job["out"], f"{job['name']}_{cname}.jpg")
        imgsel.save_jpeg(framed(imgsel.crop(img, box), color), p, max_side=box[2])
        out[cname] = p
    return job["name"], out


def find_boxes(job):
    """job: dict(key, labels, res, sid, size) -> key, auto box over all labels."""
    imgs = [imgsel.load(imgsel.find(l, job["res"], job["sid"])) for l in job["labels"]]
    return job["key"], auto_box(imgs, job["size"])


# ---------------------------------------------------------------- latex layout
def rel(p):
    return os.path.relpath(p, imgsel.PAPER)


def tile(path, h, overlay=None):
    s = rf"\includegraphics[height={h:.4f}\scaW]{{{rel(path)}}}"
    if overlay:
        s += (r"\llap{\raisebox{0.5ex}{\colorbox{black}{\color{white}\tiny " + overlay
              + r"}}\hspace{0.4ex}}")
    return s


def row(items, label=None, total=0.999):
    """items: list of (aspect, path, overlay) or ('stack', [(path, overlay), ...]) (squares)."""
    asp = sum(0.5 if it[0] == "stack" else it[0] for it in items)
    avail = total - (LABW if label is not None else 0) - GAP * (len(items) - 1)
    h = avail / asp
    parts = []
    for it in items:
        if it[0] == "stack":
            hs = (h - GAP) / 2
            a, b = it[1]
            parts.append(r"\begin{minipage}[b]{%.4f\scaW}" % hs + tile(a[0], hs, a[1])
                         + r"\\[%.4f\scaW]" % GAP + tile(b[0], hs, b[1]) + r"\end{minipage}")
        else:
            parts.append(tile(it[1], h, it[2] if len(it) > 2 else None))
    lab = ""
    if label is not None:
        lab = (r"\makebox[%.4f\scaW][l]{\rotatebox{90}{\makebox[%.4f\scaW][c]{%s}}}"
               % (LABW, h, label))
    body = lab + (r"\hspace{%.4f\scaW}" % GAP).join(parts)
    return r"\makebox[\textwidth][l]{" + body + "}", h


def header(cells, h, label=False):
    """cells: list of (aspect, text); same geometry as a row of height h."""
    parts = [r"\makebox[%.4f\scaW][c]{%s}" % (a * h, t) for a, t in cells]
    pre = r"\hspace*{%.4f\scaW}" % LABW if label else ""
    return r"\makebox[\textwidth][l]{" + pre + (r"\hspace{%.4f\scaW}" % GAP).join(parts) + "}"


def snippet(name, body_lines, caption, label, env="figure*", comments=()):
    os.makedirs(SNIP, exist_ok=True)
    lines = [f"% Generated by paper/figures/src/fig_qual.py {name}; do not edit by hand."]
    lines += [f"% {c}" for c in comments]
    lines += [rf"\begin{{{env}}}[t]", r"\centering\sffamily\scriptsize\ifdefined\scaW\else\newlength{\scaW}\fi\setlength{\scaW}{\textwidth}",
              r"\setlength{\fboxsep}{1pt}\setlength{\lineskip}{0pt}\setlength{\lineskiplimit}{0pt}"]
    lines += [b + r"\\[%.4f\textwidth]" % GAP if i < len(body_lines) - 1 else b
              for i, b in enumerate(body_lines)]
    lines += [r"\rmfamily\normalsize", rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\end{{{env}}}", ""]
    p = os.path.join(SNIP, f"{name}.tex")
    with open(p, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", p)


def prompt_comments(pairs):
    out = []
    for label, res, sid in pairs:
        try:
            out.append(f"{sid} ({res}): {imgsel.prompt(imgsel.find(label, res, sid))[:150].replace(chr(0x2014), ' - ')}")
        except FileNotFoundError:
            out.append(f"{sid} ({res}): missing for {label}")
    return out


def run(jobs, workers):
    with Pool(workers) as pool:
        return dict(pool.map(make_tiles, jobs, chunksize=1))


def boxes_for(specs, workers):
    with Pool(workers) as pool:
        return dict(pool.map(find_boxes, specs, chunksize=1))


# ---------------------------------------------------------------- appendix grids: one vector style
# The appendix galleries, the ablation grid and the failure grid share the macros in
# snippets/_scagrid.tex. Every tile of a figure has one height, \scaT, which TeX sets so that the
# widest row fills \linewidth. A full image is followed 1 pt later by its native crop; method groups
# are 4 pt apart, rows 2 pt, prompts set side by side 9 pt. The crop window is a vector rectangle in
# scaZoom over a thin dark casing, so it reads on light and dark images, and its crop tile carries
# the same frame. The JPEGs stay clean: full images at GRID_DPI of their printed size, crops native.
JPEG_Q = 85  # quality of every grid tile; imgsel.save_jpeg uses 4:2:0 and optimize (deterministic)
GRID_DPI = 500  # full-image previews, pixels per printed inch
TEXT_PT = 397.48  # \linewidth of the 5.5 in text block; sizes the JPEGs only, TeX sizes the tiles
G_PAIR, G_GROUP, G_ROW, G_BLOCK, G_SECTION = 1, 4, 2, 9, 5  # gutters in pt
LAB_PT = 11  # \scaLW + \scaLG: the rotated row-label column and its gap
ACCENT = {"sca", "flux2_sca", "krea_sca"}  # columns headed "SCA (ours)" in the accent colour

GRID_MACROS = r"""% Shared macros for the SCA image grids. Written by paper/figures/src/fig_qual.py; do not edit by hand.
% Each grid snippet inputs this file first. The guard makes repeated input harmless, and every
% definition is global, so input inside a group is safe too. Needs graphicx, xcolor and tikz.
\ifcsname scaGridLoaded\endcsname\else
\expandafter\gdef\csname scaGridLoaded\endcsname{}%
\xglobal\providecolor{scaInk}{HTML}{1F2328}%
\xglobal\providecolor{scaMid}{HTML}{5B6570}%
\xglobal\providecolor{scaAccent}{HTML}{13865F}%
\xglobal\providecolor{scaZoom}{HTML}{FFC21A}%
\newlength{\scaT}% tile height; each figure sets it so that its widest row fills \linewidth
\newlength{\scaLW}\global\scaLW=9.5pt% rotated row-label column
\newlength{\scaLG}\global\scaLG=1.5pt% gap between a row label and its tiles
\newlength{\scaZW}\global\scaZW=0.8pt% crop window and crop frame, in scaZoom
\newlength{\scaZC}\global\scaZC=1.4pt% dark casing under that line, 0.3 pt on each side
\newlength{\scaZI}\global\scaZI=0.7pt% half the casing: frames and border windows stay inside the tile
\newlength{\scaHT}\global\scaHT=1.5pt% trim at each end of a header rule
\newlength{\scaTW}\newlength{\scaTH}%
% \scaImg{width}{height}{file}: a plain tile.
\gdef\scaImg#1#2#3{\includegraphics[width=#1,height=#2]{#3}}%
% \scaImgTag{width}{height}{file}{tag}: a plain tile with a small label in its top left corner.
\gdef\scaImgTag#1#2#3#4{\scaTW=#1\relax\scaTH=#2\relax%
\begin{tikzpicture}[x=\scaTW,y=\scaTH]%
\node[anchor=south west,inner sep=0pt,outer sep=0pt] at (0,0) {\includegraphics[width=\scaTW,height=\scaTH]{#3}};%
\useasboundingbox (0,0) rectangle (1,1);%
\node[anchor=north west,inner sep=1.2pt,fill=black,fill opacity=.6,text opacity=1,text=white,font=\sffamily\scriptsize] at ([shift={(1.5pt,-1.5pt)}]0,1) {#4};%
\end{tikzpicture}}%
% \scaFull{width}{height}{file}{(x0,y0)}{(x1,y1)}: a full image with its crop window outlined.
% Corners are fractions of the tile, origin at the bottom left.
\gdef\scaFull#1#2#3#4#5{\scaTW=#1\relax\scaTH=#2\relax%
\begin{tikzpicture}[x=\scaTW,y=\scaTH]%
\node[anchor=south west,inner sep=0pt,outer sep=0pt] at (0,0) {\includegraphics[width=\scaTW,height=\scaTH]{#3}};%
\useasboundingbox (0,0) rectangle (1,1);%
\draw[black,draw opacity=.55,line width=\scaZC] #4 rectangle #5;%
\draw[scaZoom,line width=\scaZW] #4 rectangle #5;%
\end{tikzpicture}}%
% \scaCrop{width}{height}{file}: a native crop, framed like its window.
\gdef\scaCrop#1#2#3{\scaTW=#1\relax\scaTH=#2\relax%
\begin{tikzpicture}[x=\scaTW,y=\scaTH]%
\node[anchor=south west,inner sep=0pt,outer sep=0pt] at (0,0) {\includegraphics[width=\scaTW,height=\scaTH]{#3}};%
\useasboundingbox (0,0) rectangle (1,1);%
\draw[black,draw opacity=.55,line width=\scaZC] ([shift={(\scaZI,\scaZI)}]0,0) rectangle ([shift={(-\scaZI,-\scaZI)}]1,1);%
\draw[scaZoom,line width=\scaZW] ([shift={(\scaZI,\scaZI)}]0,0) rectangle ([shift={(-\scaZI,-\scaZI)}]1,1);%
\end{tikzpicture}}%
% \scaHead{width}{rule colour}{text}: a one-line header centred over a group of tiles, ruled like
% \cmidrule(lr); text a little wider than its group may overhang into the gutters.
\gdef\scaHead#1#2#3{\parbox[b]{#1}{\centering\makebox[\linewidth][c]{\color{scaInk}\strut#3}\par\nointerlineskip
\vskip1.2pt\hbox to\hsize{\hskip\scaHT{\color{#2}\leaders\hrule height .5pt\hfill}\hskip\scaHT}}}%
% \scaTwo{line}{line}: two-line header text, bottom aligned with one-line headers.
\gdef\scaTwo#1#2{\begin{tabular}[b]{@{}c@{}}#1\\#2\end{tabular}}%
% \scaRowLabel{height}{text}: a row label rotated 90 degrees and centred on a row of that height.
\gdef\scaRowLabel#1#2{\makebox[\scaLW][l]{\rotatebox{90}{\makebox[#1][c]{\color{scaInk}\strut#2}}}}%
% \scaGridStyle: typography and exact spacing for a grid; use inside a group.
\gdef\scaGridStyle{\centering\sffamily\footnotesize\offinterlineskip\parskip=0pt\relax}%
\fi
"""


def write_grid_macros():
    os.makedirs(SNIP, exist_ok=True)
    p = os.path.join(SNIP, "_scagrid.tex")
    with open(p, "w") as fh:
        fh.write(GRID_MACROS)
    return p


def grid_tiles(job):
    """Clean JPEG tiles for a grid; the TikZ overlay draws windows and frames.

    job: dict(out, name, png or label/res/sid, px=None, crops=[(box, cname)]). The full image is
    downscaled (Lanczos) to px on its long side; crops are native 1:1, never resampled.
    """
    img = imgsel.load(job.get("png") or imgsel.find(job["label"], job["res"], job["sid"]))
    out = {}
    if job.get("px"):
        out["full"] = imgsel.save_jpeg(img, os.path.join(job["out"], f"{job['name']}.jpg"), JPEG_Q, job["px"])
    for box, cname in job.get("crops", []):
        out[cname] = imgsel.save_jpeg(imgsel.crop(img, box), os.path.join(job["out"], f"{job['name']}_{cname}.jpg"),
                                      JPEG_Q, box[2])
    return job["name"], out


def run_grid(jobs, workers):
    with Pool(max(1, min(workers, len(jobs)))) as pool:
        return dict(pool.map(grid_tiles, jobs, chunksize=1))


class Geo:
    """Tile height of one grid: its widest row, `units` tile heights plus `gaps` pt, fills \\linewidth."""

    def __init__(self, units, gaps, label=False):
        self.units, self.gaps, self.label = units, gaps, label

    def tile_in(self):
        return (TEXT_PT - self.gaps - (LAB_PT if self.label else 0)) / self.units / 72.27

    def px(self, aspect=1.0):
        """Long side, in pixels, of a full image with this width/height at GRID_DPI of its print size."""
        return round(max(aspect, 1.0) * self.tile_in() * GRID_DPI)

    def tex(self):
        lab = r"-\scaLW-\scaLG" if self.label else ""
        return r"\setlength{\scaT}{\dimexpr(\linewidth-%gpt%s)*1000/%d\relax}%%" % (
            self.gaps, lab, round(self.units * 1000))


def tdim(units, gaps=0):
    """TeX width of `units` tile heights plus `gaps` pt."""
    u = r"\scaT" if units == 1 else r"%g\scaT" % units
    return r"\dimexpr%s+%gpt\relax" % (u, gaps) if gaps else u


def t_img(path, aspect=1):
    return r"\scaImg{%s}{\scaT}{%s}" % (tdim(aspect), rel(path))


def t_tag(path, tag, aspect=1):
    return r"\scaImgTag{%s}{\scaT}{%s}{%s}" % (tdim(aspect), rel(path), tag)


def t_crop(path):
    return r"\scaCrop{\scaT}{\scaT}{%s}" % rel(path)


def t_full(path, box, res):
    """Full image at `res` (HxW) with its crop window box = (x, y, size), native pixels, outlined."""
    w, h = imgsel.res_size(res)
    x, y, s = box

    def corner(fx, fy, sx, sy):  # a side on the image border moves inside by half the casing
        shift = [f"{a}shift={'' if d > 0 else '-'}\\scaZI" for a, d in (("x", sx), ("y", sy)) if d]
        return "(%s%.5f,%.5f)" % ("[%s]" % ",".join(shift) if shift else "", fx, fy)

    lo = corner(x / w, 1 - (y + s) / h, x <= 0, y + s >= h)
    hi = corner((x + s) / w, 1 - y / h, -(x + s >= w), -(y <= 0))
    return r"\scaFull{%s}{\scaT}{%s}{%s}{%s}" % (tdim(w / h), rel(path), lo, hi)


def g_row(cells, label=None, indent=False):
    """One row of a grid, exactly \\linewidth wide: TeX cells interleaved with gutters in pt."""
    body = "".join(r"\hspace{%gpt}" % c if isinstance(c, (int, float)) else c for c in cells)
    if label is not None:
        body = r"\scaRowLabel{\scaT}{%s}\hspace{\scaLG}" % label + body
    elif indent:
        body = r"\hspace*{\dimexpr\scaLW+\scaLG\relax}" + body
    return r"\makebox[\linewidth][c]{%s}" % body


def head_cell(text, units, gaps=0, accent=False):
    if accent:
        text = r"\textcolor{scaAccent}{\textbf{%s}}" % text
    return r"\scaHead{%s}{%s}{%s}" % (tdim(units, gaps), "scaAccent" if accent else "scaMid", text)


def method_head(m, units, gaps=0, names=None):
    if m in ACCENT:
        return head_cell("SCA (ours)", units, gaps, accent=True)
    return head_cell((names or NAMES).get(m, m), units, gaps)


def grid_snippet(name, geo, rows, caption, label, comments=(), source=None, env="figure*"):
    """Write snippets/<name>.tex. rows: [(tex, gap after in pt)]; the last gap is unused."""
    write_grid_macros()
    lines = [f"% Generated by paper/figures/src/{source or 'fig_qual.py ' + name}; do not edit by hand."]
    lines += [f"% {c}" for c in comments]
    lines += [r"\input{figures/snippets/_scagrid}%", rf"\begin{{{env}}}[t]", r"\begingroup\scaGridStyle", geo.tex()]
    lines += [tex + (r"\par\vskip%gpt" % gap if i < len(rows) - 1 else r"\par") for i, (tex, gap) in enumerate(rows)]
    lines += [r"\endgroup", rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\end{{{env}}}", ""]
    p = os.path.join(SNIP, f"{name}.tex")
    with open(p, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", p)
    return p


# ---------------------------------------------------------------- frozen selection (provenance.json)
def _manifest_path(name):
    return os.path.join(imgsel.FIGDIR, name, "provenance.json")


def _write_manifest(name, man):
    os.makedirs(os.path.dirname(_manifest_path(name)), exist_ok=True)
    with open(_manifest_path(name), "w") as fh:
        fh.write(json.dumps(man, indent=2) + "\n")


def superseded(name):
    """True if fig_gallery_large.py lays out this gallery (a "display" record in its provenance.json).

    The gallery modes then skip it, so a rerun cannot restore the superseded small-tile layout or
    overwrite its larger full-image tiles. Delete the display record to rebuild the old layout.
    """
    path = _manifest_path(name)
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        shown = "display" in json.load(fh)
    if shown:
        print(f"{name}: skipped; fig_gallery_large.py lays out this gallery (display record in provenance.json)")
    return shown


def frozen(name, compute, workers):
    """Prompts, crop windows and source hashes of one figure, frozen in <figure>/provenance.json.

    The first run calls compute() -> dict(selection=..., prompts=[dict(short_id, res, crop_xy_size,
    methods)], ...) and records every source PNG with its SHA256. Later runs reuse that record and
    fail if any source changed, so a growing output pool cannot change a published figure.
    """
    path = _manifest_path(name)
    if os.path.exists(path):
        with open(path) as fh:
            man = json.load(fh)
        srcs = [s for e in man["prompts"] for s in e["sources"].values()]
        with Pool(max(1, min(workers, len(srcs)))) as pool:
            digests = pool.map(imgsel.sha256, [s["png"] for s in srcs])
        for s, d in zip(srcs, digests):
            if d != s["sha256"]:
                raise ValueError(f"Changed source image: {s['png']}")
        return man
    man = dict(schema=1, figure=name, generation_seed=0, **compute())
    pngs = {}
    for e in man["prompts"]:
        e["sources"] = {m: {"png": imgsel.find(m, e["res"], e["short_id"])} for m in e.pop("methods")}
        pngs.update({s["png"]: None for s in e["sources"].values()})
    with Pool(max(1, min(workers, len(pngs)))) as pool:
        digests = dict(zip(pngs, pool.map(imgsel.sha256, list(pngs))))
    for e in man["prompts"]:
        e["prompt"] = imgsel.prompt(next(iter(e["sources"].values()))["png"])
        for s in e["sources"].values():
            s["sha256"] = digests[s["png"]]
    _write_manifest(name, man)
    return man


def record_outputs(name, man, tiles):
    """Store the SHA256 of every tile written for this figure (the only part that changes on a re-render)."""
    paths = sorted(p for t in tiles.values() for p in t.values())
    man["outputs"] = {rel(p): imgsel.sha256(p) for p in paths}
    _write_manifest(name, man)


# ---------------------------------------------------------------- figures
HERO = dict(
    pano=("sca", "2048x6144", "m2742856837225126754", (3456, 128, 768)),
    portrait=("sca", "4096x4096", "3657338173664010496", [(1600, 1280, 768), (1408, 2304, 640)]),
    cathedral=("sca", "4096x4096", "m8737279395568207187"),
    train=("sca", "2048x4096", "6978695891801939003"),
    still=("sca", "4096x4096", "m947425340308551929"),
    compare=("4096x4096", "m3651203496105491158", (1280, 1280, 768)),
)

MAIN = ["m1057810871814672649", "8142852677968042257", "2100482621711410240",
        "m2729538226706404616", "m6394968997939197542", "1052899493017962912"]
MAIN_BOX = {}  # sid -> (x, y, size) overrides of the automatic crop window

ABL = ["1139740195494332405", "1516768667427572324", "4525259556400020626", "7009273464006889449"]
ABL_METHODS = ["abl_v0", "abl_local", "abl_uniform", "abl_global", "abl_mean", "sca"]

FAIL_DUP = ["7354534197758042049", "3614061571823823193"]  # development prompts
FAIL_SCA = [  # (sid, crop box, short description used in the caption order)
    ("7396058529551312081", (2304, 1408, 1024)),  # thin diagonal sword: staircase blocks
    ("1316997283360365550", (2560, 1024, 768)),   # blocky patch beside the coat
    ("m8648158440845565467", (128, 1280, 1024)),  # dancer: a foot where the rear arm should be
    ("m702025498990232048", (0, 2560, 1024)),     # dense illustration: border dissolves into speckle
]


def used_ids():
    return {HERO[k][2] for k in ("pano", "portrait", "cathedral", "train", "still")} | {HERO["compare"][1]} \
        | set(MAIN) | {s for s, _ in FAIL_SCA}


def fig_hero(a):
    out = os.path.join(imgsel.FIGDIR, "hero")
    l, r, sid, pbox = HERO["pano"]
    ql, qr, qsid, (b1, b2) = HERO["portrait"]
    cres, csid, cbox = HERO["compare"]
    jobs = [
        dict(out=out, name="pano", label=l, res=r, sid=sid, side=1600, boxes=[(pbox, COLORS[0])],
             crops=[(pbox, COLORS[0], "crop")]),
        dict(out=out, name="portrait", label=ql, res=qr, sid=qsid, side=1000,
             boxes=[(b1, COLORS[1]), (b2, COLORS[2])], crops=[(b1, COLORS[1], "crop1"), (b2, COLORS[2], "crop2")]),
        dict(out=out, name="cathedral", label="sca", res=HERO["cathedral"][1], sid=HERO["cathedral"][2], side=1000),
        dict(out=out, name="train", label="sca", res=HERO["train"][1], sid=HERO["train"][2], side=1600),
        dict(out=out, name="still", label="sca", res=HERO["still"][1], sid=HERO["still"][2], side=1000),
        dict(out=out, name="cmp_sca", label="sca", res=cres, sid=csid, side=1000, boxes=[(cbox, COLORS[3])],
             crops=[(cbox, COLORS[3], "crop")]),
        dict(out=out, name="cmp_area", label="area", res=cres, sid=csid, crops=[(cbox, COLORS[3], "crop")]),
        dict(out=out, name="cmp_full", label="full", res=cres, sid=csid, crops=[(cbox, COLORS[3], "crop")]),
    ]
    t = run(jobs, a.workers)
    rA, _ = row([(3, t["pano"]["full"], wxh(r)), (1, t["pano"]["crop"], "1:1 crop")])
    rB, _ = row([(1, t["portrait"]["full"], wxh(qr)),
                 ("stack", [(t["portrait"]["crop1"], None), (t["portrait"]["crop2"], None)]),
                 (1, t["cathedral"]["full"], wxh(HERO["cathedral"][1])),
                 (2, t["train"]["full"], wxh(HERO["train"][1]))])
    rC, _ = row([(1, t["cmp_sca"]["full"], wxh(cres)), (1, t["cmp_full"]["crop"], "Full attention"),
                 (1, t["cmp_area"]["crop"], "AReA"), (1, t["cmp_sca"]["crop"], r"\textbf{SCA}"),
                 (1, t["still"]["full"], wxh(HERO["still"][1]))])
    cap = (r"\textbf{SCA generates coherent, detailed images up to 6144$\times$2048 from FLUX.1-dev without "
           r"training.} Unless labeled otherwise, images are SCA outputs at seed 0 (labels give width$\times$height); framed tiles are "
           r"1:1 crops at native resolution from the outlined windows. Bottom row: the same prompt and seed at "
           r"4096$\times$4096 with full attention, which collapses into a blurred field, with AReA, whose sky breaks "
           r"into tiled facets, and with SCA.")
    snippet("hero", [rA, rB, rC], cap, "fig:hero",
            comments=prompt_comments([(l, r, sid), (ql, qr, qsid), ("sca", HERO["cathedral"][1], HERO["cathedral"][2]),
                                      ("sca", HERO["train"][1], HERO["train"][2]),
                                      ("sca", cres, csid), ("sca", HERO["still"][1], HERO["still"][2])]))


def fig_main(a):
    out = os.path.join(imgsel.FIGDIR, "qual_main")
    methods = a.methods or ["full", "area", "sca"]
    res, size = a.res, 512
    auto = boxes_for([dict(key=s, labels=methods, res=res, sid=s, size=size) for s in MAIN if s not in MAIN_BOX],
                     a.workers)
    box = {s: MAIN_BOX.get(s, auto.get(s)) for s in MAIN}
    jobs = [dict(out=out, name=f"{s}_{m}", label=m, res=res, sid=s, side=700, boxes=[(box[s], COLORS[0])],
                 crops=[(box[s], COLORS[0], "crop")]) for s in MAIN for m in methods]
    t = run(jobs, a.workers)
    lines = []
    for blk in (MAIN[:3], MAIN[3:]):
        for m in methods:
            items = []
            for s in blk:
                items += [(1, t[f"{s}_{m}"]["full"]), (1, t[f"{s}_{m}"]["crop"])]
            lines.append(row(items, label=NAMES.get(m, m))[0])
    cap = (r"\textbf{At 4096$\times$4096, full attention collapses, AReA loses detail and adds texture "
           r"artifacts, and SCA keeps both layout and fine detail.} FLUX.1-dev, same prompt and seed (0) per "
           r"column pair; each full image is followed by a 1:1 crop of its outlined 512$\times$512 window. "
           r"Crop windows are chosen automatically as the region with the most edge energy averaged over the "
           r"three methods. Prompts were picked for variety from the 191 evaluation prompts; uncurated random "
           r"samples are in Figure~\ref{fig:gallery_4096}.")
    snippet("qual_main", lines, cap, "fig:qual_main", comments=prompt_comments([("sca", res, s) for s in MAIN]))


def gallery_pool(res, labels, exclude=()):
    common = None
    for l in labels:
        ids = {imgsel.short_id(k) for k in imgsel.index(l, res)}
        common = ids if common is None else common & ids
    return sorted(common - imgsel.dev4_ids() - used_ids() - set(exclude))


def gallery_ids(res, n, seed, labels, exclude=()):
    pool = gallery_pool(res, labels, exclude)
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


def count_word(n):
    return ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"][n] if n <= 10 else str(n)


CROP_RULE = "Window of size crop_xy_size[2] with the highest mean edge energy, averaged over every method shown"
FLUX1 = r"FLUX.1-dev, seed~0, 28 steps, guidance 4.5"


def random_selection(name, res, n, seed, methods, size, workers, exclude=()):
    """Frozen random gallery selection: seeded sample of the eligible pool, automatic crop windows."""
    def compute():
        pool = gallery_pool(res, methods, exclude)
        ids = gallery_ids(res, n, seed, methods, exclude)
        auto = boxes_for([dict(key=s, labels=methods, res=res, sid=s, size=size) for s in ids], workers)
        return dict(
            resolution=res, methods=methods,
            selection=f"random.Random({seed}).sample(sorted(eligible_ids), {n}); eligible = prompts with "
                      f"seed-0 outputs for every method, minus the development prompts, the prompts in the "
                      f"main-text figures (fig_qual.used_ids) and {len(set(exclude))} prompts of earlier galleries",
            crop_rule=CROP_RULE, eligible_count=len(pool), eligible_ids=pool,
            prompts=[dict(short_id=s, res=res, crop_xy_size=list(auto[s]), methods=methods) for s in ids])
    man = frozen(name, compute, workers)
    if man["methods"] != methods:
        raise ValueError(f"{name} is frozen for methods {man['methods']}; remove its provenance.json to reselect")
    return man


def pair_cells(t, s, methods, box, res):
    """Full image then its crop, for each method, groups 4 pt apart."""
    cells = []
    for k, m in enumerate(methods):
        cells += ([G_GROUP] if k else []) + [t_full(t[f"{s}_{m}"]["full"], box, res), G_PAIR,
                                             t_crop(t[f"{s}_{m}"]["crop"])]
    return cells


def pair_heads(methods, aspect, names=None):
    cells = []
    for k, m in enumerate(methods):
        cells += ([G_GROUP] if k else []) + [method_head(m, aspect + 1, G_PAIR, names)]
    return cells


def fig_gallery_square(a):
    """AReA vs SCA (or --methods) at a square resolution, random prompts, 16 per page."""
    methods = a.methods or ["area", "sca"]
    res = a.res
    tag = res.split("x")[0] if len(set(res.split("x"))) == 1 else res
    name = f"gallery_{tag}"
    if superseded(name):
        return
    out = os.path.join(imgsel.FIGDIR, name)
    size = 512 * int(res.split("x")[0]) // 4096 if int(res.split("x")[0]) > 4096 else 512
    man = random_selection(name, res, a.n, a.seed, methods, size, a.workers)
    ids = [e["short_id"] for e in man["prompts"]]
    box = {e["short_id"]: tuple(e["crop_xy_size"]) for e in man["prompts"]}
    nm = len(methods)
    per_row = 2 if nm <= 2 else 1
    per_page = 16 if per_row == 2 else 6
    geo = Geo(per_row * 2 * nm, per_row * (nm * G_PAIR + (nm - 1) * G_GROUP) + (per_row - 1) * G_BLOCK)
    jobs = [dict(out=out, name=f"{s}_{m}", label=m, res=res, sid=s, px=geo.px(), crops=[(box[s], "crop")])
            for s in ids for m in methods]
    t = run_grid(jobs, a.workers)
    head = []
    for j in range(per_row):
        head += ([G_BLOCK] if j else []) + pair_heads(methods, 1)
    vs = " and ".join(r"\textbf{SCA (ours)}" if m in ACCENT else NAMES.get(m, m) for m in methods)
    pages = [ids[i:i + per_page] for i in range(0, len(ids), per_page)]
    for pi, page in enumerate(pages):
        rows = [(g_row(head), G_ROW)]
        for i in range(0, len(page), per_row):
            cells = []
            for j, s in enumerate(page[i:i + per_row]):
                cells += ([G_BLOCK] if j else []) + pair_cells(t, s, methods, box[s], res)
            rows.append((g_row(cells), G_ROW))
        cap = rf"\textbf{{Uncurated samples at {wxh(res)}: {vs}{', continued' if pi else ''}.}} "
        if pi == 0:
            cap += (rf"{count_word(len(ids))} evaluation prompts drawn at random (selection seed~{a.seed}), excluding "
                    rf"development and main-text prompts. {FLUX1}. Each image is followed by a native-pixel crop of "
                    rf"its outlined {size}$\times${size} window of highest edge energy averaged over the methods.")
        else:
            cap += r"Same protocol as Figure~\ref{fig:" + name + "}."
        part = name + ("" if pi == 0 else f"_p{pi + 1}")
        grid_snippet(part, geo, rows, cap, f"fig:{part}",
                     comments=prompt_comments([("sca" if "sca" in methods else methods[0], res, s) for s in page]),
                     source=f"fig_qual.py gallery_{tag}")
    record_outputs(name, man, t)


def fig_gallery_wide(a):
    methods = a.methods or ["area", "sca"]
    sq = _manifest_path("gallery_4096")
    if os.path.exists(sq):  # the square gallery's frozen prompts
        with open(sq) as fh:
            taken = {e["short_id"] for e in json.load(fh)["prompts"]}
    else:
        taken = set(gallery_ids("4096x4096", a.n, a.seed, ["area", "sca"]))
    for res in (a.wide or ["2048x4096", "2048x6144"]):
        h, w = (int(v) for v in res.split("x"))
        name = f"gallery_{w}x{h}"
        if superseded(name):  # its frozen prompts still count as taken for later wide galleries
            with open(_manifest_path(name)) as fh:
                taken |= {e["short_id"] for e in json.load(fh)["prompts"]}
            continue
        out = os.path.join(imgsel.FIGDIR, name)
        # disjoint from the square gallery and from earlier wide galleries, so every prompt appears once
        man = random_selection(name, res, a.n_wide, a.seed + 1 + len(taken), methods, 512, a.workers,
                               exclude=sorted(taken))
        ids = [e["short_id"] for e in man["prompts"]]
        box = {e["short_id"]: tuple(e["crop_xy_size"]) for e in man["prompts"]}
        taken |= set(ids)
        asp = w / h
        nm = len(methods)
        geo = Geo(nm * (asp + 1), nm * G_PAIR + (nm - 1) * G_GROUP)
        jobs = [dict(out=out, name=f"{s}_{m}", label=m, res=res, sid=s, px=geo.px(asp), crops=[(box[s], "crop")])
                for s in ids for m in methods]
        t = run_grid(jobs, a.workers)
        rows = [(g_row(pair_heads(methods, asp)), G_ROW)]
        rows += [(g_row(pair_cells(t, s, methods, box[s], res)), G_ROW) for s in ids]
        vs = " and ".join(r"\textbf{SCA (ours)}" if m in ACCENT else NAMES.get(m, m) for m in methods)
        cap = (rf"\textbf{{Uncurated samples at {w}$\times${h} (width$\times$height): {vs}.}} "
               rf"{count_word(len(ids))} evaluation prompts drawn at random, disjoint from the other galleries, "
               rf"excluding development and main-text prompts. {FLUX1}. Each image is followed by a native-pixel "
               r"crop of its outlined 512$\times$512 window of highest edge energy averaged over the methods.")
        grid_snippet(name, geo, rows, cap, f"fig:{name}", comments=prompt_comments([("sca", res, s) for s in ids]),
                     source="fig_qual.py gallery_wide")
        record_outputs(name, man, t)


def fig_ablation(a):
    out = os.path.join(imgsel.FIGDIR, "ablation")
    res, size = "4096x4096", 512
    methods = a.methods or ABL_METHODS

    def compute():
        auto = boxes_for([dict(key=s, labels=methods, res=res, sid=s, size=size) for s in ABL], a.workers)
        return dict(resolution=res, methods=methods, selection="Hand-picked ablation prompts (fig_qual.ABL)",
                    crop_rule=CROP_RULE,
                    prompts=[dict(short_id=s, res=res, crop_xy_size=list(auto[s]), methods=methods) for s in ABL])

    man = frozen("ablation", compute, a.workers)
    if man["methods"] != methods:
        raise ValueError(f"ablation is frozen for methods {man['methods']}")
    nm = len(methods)
    geo = Geo(nm, (nm - 1) * G_GROUP, label=True)
    jobs = [dict(out=out, name=f"{e['short_id']}_{m}", label=m, res=res, sid=e["short_id"], px=geo.px(),
                 crops=[(tuple(e["crop_xy_size"]), "crop")]) for e in man["prompts"] for m in methods]
    t = run_grid(jobs, a.workers)
    head = []
    for k, m in enumerate(methods):
        head += ([G_GROUP] if k else []) + [method_head(m, 1)]
    rows = [(g_row(head, indent=True), G_ROW)]
    for e in man["prompts"]:
        s, b = e["short_id"], tuple(e["crop_xy_size"])
        full, crop = [], []
        for k, m in enumerate(methods):
            full += ([G_GROUP] if k else []) + [t_full(t[f"{s}_{m}"]["full"], b, res)]
            crop += ([G_GROUP] if k else []) + [t_crop(t[f"{s}_{m}"]["crop"])]
        rows += [(g_row(full, label="Full image"), G_PAIR), (g_row(crop, label="1:1 crop"), G_SECTION)]
    cap = (r"\textbf{Without the gate the subject repeats; $\pi=0$ leaves the pooled branch's coarse grid, and "
           r"mean pooling loses the layout.} Four ablation prompts at 4096$\times$4096, seed~0, 28 steps, guidance "
           r"4.5; variants as in Table~\ref{tab:ablation}; crops show native pixels of the outlined 512$\times$512 "
           r"windows. $\pi=1$ and uniform $\pi(\sigma)$ look close to SCA here; their deficits show in the metrics.")
    grid_snippet("ablation", geo, rows, cap, "fig:ablation_grid",
                 comments=prompt_comments([("sca", res, s) for s in ABL]))
    record_outputs("ablation", man, t)


def fig_failures(a):
    out = os.path.join(imgsel.FIGDIR, "failures")
    res = "4096x4096"
    dup_methods = ["abl_v0", "dev_w16", "sca"]

    def compute():
        return dict(resolution=res, methods=dup_methods + ["area"],
                    selection="Hand-picked: two development prompts (FAIL_DUP) and four SCA errors with "
                              "hand-placed windows (FAIL_SCA) in fig_qual.py",
                    crop_rule="Hand-placed windows; the AReA crop uses the same window",
                    prompts=[dict(short_id=s, res=res, crop_xy_size=None, methods=dup_methods) for s in FAIL_DUP]
                    + [dict(short_id=s, res=res, crop_xy_size=list(b), methods=["sca", "area"]) for s, b in FAIL_SCA])

    man = frozen("failures", compute, a.workers)
    box = {e["short_id"]: tuple(e["crop_xy_size"]) for e in man["prompts"] if e["crop_xy_size"]}
    geo = Geo(6, 2 * (G_PAIR + G_GROUP) + G_BLOCK)
    jobs = [dict(out=out, name=f"dup_{s}_{m}", label=m, res=res, sid=s, px=geo.px())
            for s in FAIL_DUP for m in dup_methods]
    for s, _ in FAIL_SCA:
        jobs.append(dict(out=out, name=f"err_{s}_sca", label="sca", res=res, sid=s, px=geo.px(),
                         crops=[(box[s], "crop")]))
        jobs.append(dict(out=out, name=f"err_{s}_area", label="area", res=res, sid=s, crops=[(box[s], "crop")]))
    t = run_grid(jobs, a.workers)
    head1, row1, head2 = [], [], []
    for j, s in enumerate(FAIL_DUP):  # duplication without the gate, two development prompts
        head1 += ([G_BLOCK] if j else []) + [head_cell("No gate", 2, 2), 3, method_head("sca", 1)]
        row1 += ([G_BLOCK] if j else []) + [t_tag(t[f"dup_{s}_abl_v0"]["full"], r"$w{=}64$"), 2,
                                            t_tag(t[f"dup_{s}_dev_w16"]["full"], r"$w{=}16$"), 3,
                                            t_img(t[f"dup_{s}_sca"]["full"])]
        head2 += ([G_BLOCK] if j else []) + [method_head("sca", 2, G_PAIR), G_GROUP, method_head("area", 1)]
    rows = [(g_row(head1), G_ROW), (g_row(row1), G_SECTION), (g_row(head2), G_ROW)]
    for pair in (FAIL_SCA[:2], FAIL_SCA[2:]):  # errors SCA still makes, each beside AReA's same window
        cells = []
        for j, (s, _) in enumerate(pair):
            cells += ([G_BLOCK] if j else []) + [t_full(t[f"err_{s}_sca"]["full"], box[s], res), G_PAIR,
                                                 t_crop(t[f"err_{s}_sca"]["crop"]), G_GROUP,
                                                 t_crop(t[f"err_{s}_area"]["crop"])]
        rows.append((g_row(cells), G_ROW))
    cap = (r"\textbf{Without the gate, windows duplicate the subject; SCA still makes local errors.} "
           r"Top (development prompts): no gate with 64- and 16-token windows ($w$), then SCA. Below: a staircase "
           r"sword, a grid beside a coat, a foot for an arm, border speckle, each next to AReA's same window; AReA "
           r"composes differently and avoids them. 4096$\times$4096, seed~0, 28 steps, guidance 4.5.")
    grid_snippet("failures", geo, rows, cap, "fig:failures",
                 comments=prompt_comments([("sca", res, s) for s in FAIL_DUP] + [("sca", res, s) for s, _ in FAIL_SCA]))
    record_outputs("failures", man, t)


# The fallback fig:failure in main.tex reads figures/failure/<prompt>_<variant>.jpg, written first by
# fig_qualitative.py at 600 px. Same sources and names here, sized to GRID_DPI of the printed tile.
FALLBACK = [("3614061571823823193", "moai"), ("m6873203881450547484", "boats"), ("7354534197758042049", "bear")]
FALLBACK_VARIANTS = [("v0w64", "abl_v0"), ("v0w16", "dev_w16"), ("local", "abl_local"), ("sca", "sca")]


def fig_failure_fallback(a):
    out = os.path.join(imgsel.FIGDIR, "failure")
    px = round(0.246 * 0.6 * TEXT_PT / 72.27 * GRID_DPI)  # width=0.246\linewidth of a 0.6\linewidth minipage
    run_grid([dict(out=out, name=f"{n}_{v}", label=lab, res="4096x4096", sid=s, px=px)
              for s, n in FALLBACK for v, lab in FALLBACK_VARIANTS], a.workers)


FIGS = {"hero": fig_hero, "qual_main": fig_main, "gallery": fig_gallery_square, "gallery_wide": fig_gallery_wide,
        "ablation": fig_ablation, "failures": fig_failures, "failure_fallback": fig_failure_fallback}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fig", choices=list(FIGS) + ["all"])
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--res", default="4096x4096")
    ap.add_argument("--wide", nargs="*", help="non-square resolutions HxW for gallery_wide")
    ap.add_argument("--n", type=int, default=32, help="square gallery prompts")
    ap.add_argument("--n-wide", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    for name in (FIGS if a.fig == "all" else [a.fig]):
        FIGS[name](a)


if __name__ == "__main__":
    main()
