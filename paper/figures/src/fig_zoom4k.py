"""Zoom comparison at 4096^2 (replaces fig:qual_main), in the style of MFlow Figure 22, as a TikZ PDF.

Three prompts at 4096^2, seed 0, one row each: SCA's full image with the crop window outlined in red,
then that window at native resolution (1:1) from full attention, AReA, HRDiT and SCA (ours, last).
Above each crop is the image's MUSIQ_c (MUSIQ averaged over four fixed random 1024^2 crops per image,
Appendix app:metrics), read from the per-image CSVs behind Table tab:main; the best value in each row
is bold. House style: method names once, SCA (ours) last in the bold accent, 1 to 2 pt white gutters,
no image borders except the red window boxes, sans labels of 6 to 8 pt at printed size.

Selection (recorded in figures/zoom4k_provenance.json):
* Prompts are curated: the first three, in order, of the four prompts of the previous fig:qual_main,
  which were picked by hand for variety from the 191 evaluation prompts. The fourth, the ivy archway
  m1057810871814672649, is left out for space. A first draft dropped the portrait instead because its
  crops differed least; its values, read afterwards, made it the one prompt of the four where SCA does
  not have the best MUSIQ_c, so that criterion favoured SCA and was replaced by the order rule, which
  does not look at the images or the values.
* Crop windows are automatic, never placed by hand: fig_qual_main.auto_box, the CROP_W x CROP_H window
  with the most edge energy averaged over the four methods shown, at least 256 px from every border.

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_zoom4k.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from fig_qual_main import auto_box  # the crop rule of the previous fig:qual_main, unchanged

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = FIG.parent.parent
TECTONIC = os.path.expanduser("~/.local/bin/tectonic")

EVAL = Path("/scratch/09032/saini_2/sca/eval")         # per-image CSVs behind Table tab:main
ARCHIVE = REPO / "results" / "evidence" / "eval"         # byte-identical copies archived in the repo
PROMPTS = ["2100482621711410240", "m6394968997939197542", "m2729538226706404616"]  # fig_qual_main order
DROPPED = "m1057810871814672649"
SEED, RES = "0", "4096x4096"
METRIC = "musiq_crop1024"                                # MUSIQ_c in the paper
# key, header, per-image CSV holding this method's value, method name in that CSV
METHODS = [("full", "Full attention", EVAL / "final4k_full_base" / "per_image.csv", "full"),
           ("area", "AReA", EVAL / "hrdit191_base" / "per_image.csv", "area"),
           ("hrdit", "HRDiT", EVAL / "hrdit191_base" / "per_image.csv", "hrdit_dev_author"),
           ("sca", r"\textbf{SCA (ours)}", EVAL / "hrdit191_base" / "per_image.csv", "sca_nb")]
CROSS = {  # other per-image CSVs that score the same PNG; the value must agree exactly
    "full": [ARCHIVE / "final4k_full_base" / "per_image.csv"],
    "area": [EVAL / "final4k_nb_base" / "per_image.csv", ARCHIVE / "final4k_nb_base" / "per_image.csv",
             ARCHIVE / "final4k_full_base" / "per_image.csv"],
    "sca": [EVAL / "final4k_nb_base" / "per_image.csv", ARCHIVE / "final4k_nb_base" / "per_image.csv"]}
TABLE = {"full": 33.2, "area": 50.4, "hrdit": 53.5, "sca": 61.2}  # Table tab:main, MUSIQ_c column

W = 397.5                    # text width (TeX pt); the layout unit is the TeX point
CROP_W, CROP_H = 320, 240    # native window (px), shown 1:1 at about 280 pixels per printed inch
G_OUT, G_IN = 2.0, 1.5       # white gutters (pt): full image to crops, between crops
HEAD, BAND = 10.0, 8.5       # method-name band and the value band above every row (pt)
THUMB_DPI = 800              # SCA's full image, Lanczos-downscaled to this many px per printed inch
RED, BOX_LW = "D62728", 0.7  # the one frame colour (window boxes only) and its width (pt)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_id(pid: str) -> str:
    return "a4k_eval_4096_" + ("minus" + pid[1:] if pid.startswith("m") else pid)


_rows: dict[Path, list[dict]] = {}


def rows(path: Path) -> list[dict]:
    if path not in _rows:
        with open(path, newline="") as f:
            _rows[path] = list(csv.DictReader(f))
    return _rows[path]


def lookup(path: Path, method: str, pid: str) -> dict:
    hits = [r for r in rows(path) if r["method"] == method and r["prompt_id"] == file_id(pid)
            and r["seed"] == SEED and r["res"] == RES]
    if len(hits) != 1:
        raise LookupError(f"{path} {method} {pid}: {len(hits)} rows")
    return hits[0]


def table_mean(path: Path, method: str) -> float:
    vals = [float(r[METRIC]) for r in rows(path) if r["method"] == method and r["seed"] == SEED and r["res"] == RES]
    assert len(vals) == 191, (path, method, len(vals))
    return sum(vals) / len(vals)


def source(path: Path, method: str, pid: str) -> tuple[dict, Path, dict]:
    """The metric row and the exact PNG it scored, checked against its sidecar."""
    row = lookup(path, method, pid)
    png = Path(row["path"])
    meta = json.loads(png.with_suffix(".json").read_text())
    if meta.get("complete") is not True or meta.get("png_sha256") != sha(png):
        raise ValueError(f"incomplete or hash-mismatched source {png}")
    if not png.name.endswith(f"_seed{SEED}.png"):
        raise ValueError(f"not seed {SEED}: {png}")
    return row, png, meta


def layout() -> tuple[float, float, list[float]]:
    """Row height h (the full image is h x h), crop width cw, and the crop x positions."""
    a = CROP_W / CROP_H
    h = (W - G_OUT - 3 * G_IN) / (1 + 4 * a)
    cw = a * h
    xs = [h + G_OUT + i * (cw + G_IN) for i in range(4)]
    assert abs(xs[-1] + cw - W) < 1e-9
    return h, cw, xs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="trial run: write the PDF, assets and provenance here instead of the paper tree")
    a = ap.parse_args()
    base = a.out_dir if a.out_dir else FIG
    assets, out_pdf, prov_path = base / "zoom4k", base / "zoom4k.pdf", base / "zoom4k_provenance.json"
    assets.mkdir(parents=True, exist_ok=True)
    for old in assets.glob("*.jpg"):  # this folder holds only this script's output
        old.unlink()
    h, cw, xs = layout()
    total = HEAD + len(PROMPTS) * (BAND + h)
    tex, records, examples = [], [], []

    def node(x, y, w, hh, name):
        tex.append(r"\node[anchor=north west,inner sep=0pt] at (%.4f,%.4f) "
                   r"{\includegraphics[width=%.4fpt,height=%.4fpt]{%s}};" % (x, -y, w, hh, name))

    def text(style, anchor, x, y, s):
        tex.append(r"\node[%s,anchor=%s,inner sep=0pt] at (%.4f,%.4f) {%s};" % (style, anchor, x, -y, s))

    # the per-image means must reproduce Table tab:main before any value is printed
    table_check = {}
    for key, _, path, name in METHODS:
        mean = table_mean(path, name)
        assert round(mean, 1) == TABLE[key], (key, mean, TABLE[key])
        table_check[key] = dict(file=str(path), method=name, n=191, mean=round(mean, 4), table=TABLE[key])

    for r, pid in enumerate(PROMPTS):
        top = HEAD + r * (BAND + h) + BAND
        got = {key: source(path, name, pid) for key, _, path, name in METHODS}
        prompts = {m["config"]["prompt"] for _, _, m in got.values()}
        seeds = {m["config"]["derived_seed"] for _, _, m in got.values()}
        assert len(prompts) == 1 and len(seeds) == 1, (pid, prompts, seeds)  # same prompt and noise seed
        imgs = {}
        for key, (_, png, _) in got.items():
            imgs[key] = Image.open(png).convert("RGB")
            assert imgs[key].size == (4096, 4096), (png, imgs[key].size)
        bx, by, bw, bh = auto_box([imgs[k] for k, *_ in METHODS], CROP_W, CROP_H)

        # SCA's full image with the window in red
        px = round(THUMB_DPI * h / 72.27)
        thumb = assets / f"{pid}_sca_full.jpg"
        imgs["sca"].resize((px, px), Image.Resampling.LANCZOS).save(thumb, "JPEG", quality=90, optimize=True,
                                                                    subsampling=0)
        node(0, top, h, h, thumb.name)
        s = h / 4096
        tex.append(r"\draw[win] (%.4f,%.4f) rectangle (%.4f,%.4f);" % (bx * s, -(top + by * s), (bx + bw) * s,
                                                                       -(top + (by + bh) * s)))
        # crops at native resolution, values above them, the best one bold
        vals = {key: float(got[key][0][METRIC]) for key, *_ in METHODS}
        best = f"{max(vals.values()):.1f}"  # every value that prints as the maximum is bold
        for i, (key, _, path, name) in enumerate(METHODS):
            row, png, meta = got[key]
            cp = assets / f"{pid}_{key}_crop.jpg"
            imgs[key].crop((bx, by, bx + bw, by + bh)).save(cp, "JPEG", quality=95, optimize=True, subsampling=0)
            node(xs[i], top, cw, h, cp.name)
            shown = f"{vals[key]:.1f}"
            text("val", "base", xs[i] + cw / 2, top - 2.2, r"\textbf{%s}" % shown if shown == best else shown)
            cross = []
            for other in CROSS.get(key, []):
                o = lookup(other, name, pid)
                assert o[METRIC] == row[METRIC] and Path(o["path"]).resolve() == png.resolve(), (other, pid)
                cross.append(str(other))
            records.append(dict(
                prompt_id=pid, csv_prompt_id=row["prompt_id"], seed=int(row["seed"]), method=key, csv_method=name,
                metric_file=str(path), column=METRIC, value=vals[key], printed=shown, best=shown == best,
                same_value_in=cross, image=str(png), image_sha256=sha(png),
                derived_seed=meta["config"]["derived_seed"], crop=cp.name, crop_xywh=[bx, by, bw, bh]))
        if r == 0:  # the metric is named once, left of the first row of values
            text("lab", "base east", h, top - 2.2, r"MUSIQ\raisebox{-0.3ex}{\fontsize{6}{6}\selectfont c}\,$\uparrow$")
        examples.append(dict(prompt_id=pid, prompt=prompts.pop(), derived_seed=seeds.pop(), crop_xywh=[bx, by, bw, bh],
                             full_image=thumb.name, full_image_px=px))

    for i, (key, head, _, _) in enumerate(METHODS):
        text("headours" if key == "sca" else "head", "base", xs[i] + cw / 2, HEAD - 2.6, head)

    doc = r"""%% Generated by paper/figures/src/fig_zoom4k.py; edit the script, not this file.
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{graphicx}
\usepackage{tikz}
\input{sca_colors.tex}
\definecolor{zoomRed}{HTML}{%s}
\begin{document}
\begin{tikzpicture}[x=1pt,y=1pt,
  win/.style={draw=zoomRed,line width=%.2fpt},
  head/.style={font=\sffamily\fontsize{8}{9}\selectfont,text=scaInk},
  headours/.style={font=\sffamily\fontsize{8}{9}\selectfont,text=scaAccent},
  val/.style={font=\sffamily\fontsize{7}{8}\selectfont,text=scaInk},
  lab/.style={font=\sffamily\fontsize{7}{8}\selectfont,text=scaMid}]
\useasboundingbox (0,0) rectangle (%.4f,%.4f);
%s
\end{tikzpicture}
\end{document}
""" % (RED, BOX_LW, W, -total, "\n".join(tex))
    (assets / "zoom4k.tex").write_text(doc)
    shutil.copy(HERE / "sca_colors.tex", assets / "sca_colors.tex")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([TECTONIC, "-X", "compile", "zoom4k.tex", "--outdir", tmp], cwd=assets, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.copy(Path(tmp) / "zoom4k.pdf", out_pdf)

    dropped = {key: float(lookup(path, name, DROPPED)[METRIC]) for key, _, path, name in METHODS}
    prov = dict(
        figure="zoom4k (replaces fig:qual_main)", script="paper/figures/src/fig_zoom4k.py",
        pdf=str(out_pdf), pdf_sha256=sha(out_pdf), pdf_bytes=out_pdf.stat().st_size,
        selection=("Curated. The first three, in order, of the four prompts of the previous fig:qual_main "
                   "(fig_qual_main.PROMPTS), which were picked by hand for variety from the 191 evaluation "
                   "prompts (fig_qual.py MAIN, kept by the 2026-10-18 redesign). The fourth, the ivy archway "
                   "%s, is left out for space. History: a first draft dropped the portrait "
                   "m2729538226706404616 instead, because its crops differed least between methods, a choice "
                   "made before any per-image value was read. Its values, read afterwards, made it the one "
                   "prompt of the four where SCA does not have the best MUSIQ_c (HRDiT does), so that visual "
                   "criterion favoured SCA. It was replaced by the order rule, which looks at neither the images "
                   "nor the values. No metric ranked the prompts. All six fig_qual.py MAIN prompts are excluded "
                   "from the uncurated galleries." % DROPPED),
        crop_rule=("Automatic, not hand-placed: fig_qual_main.auto_box over full, area, hrdit and sca, i.e. the "
                   "%dx%d (width x height) window with the largest mean PIL FIND_EDGES response of the 8x "
                   "box-downsampled greyscale images, at least 256 px from every border. The same window is "
                   "cropped from every method; layouts differ between methods, so the crops differ in content."
                   % (CROP_W, CROP_H)),
        metric=("MUSIQ_c (CSV column %s): pyiqa MUSIQ averaged over four fixed random 1024^2 crops per image, "
                "crop positions set by the prompt id and shared across methods (Appendix app:metrics). It is "
                "a per-image score, not a score of the displayed window. Printed to one decimal; the best "
                "printed value per row is bold." % METRIC),
        sources=dict(full_attention="out/main/full", area="out/main/area",
                     hrdit="out/baseline_expansion_20261017_{v1,rest145}/hrdit_dev_author via out/hrdit191eval",
                     sca="out/main/sca_nb"),
        metric_files={str(p): sha(p) for p in sorted({r["metric_file"] for r in records}
                                                     | {c for r in records for c in r["same_value_in"]})},
        table_check=table_check, examples=examples, values=records,
        dropped=dict(prompt_id=DROPPED, values_read_after_selection=dropped),
        layout_pt=dict(width=W, height=round(total, 3), row_height=round(h, 3), full_image=round(h, 3),
                       crop_width=round(cw, 3), gutters=dict(full_to_crops=G_OUT, between_crops=G_IN),
                       header_band=HEAD, value_band=BAND, box=dict(colour="#" + RED, width_pt=BOX_LW),
                       crop_ppi=round(CROP_W / (cw / 72.27), 1)),
        jpeg=dict(full=f"SCA full image, Lanczos to {THUMB_DPI} px per printed inch, quality 90, 4:4:4",
                  crop="native pixels (1:1), quality 95, 4:4:4"),
        asset_sha256={p.name: sha(p) for p in sorted(assets.glob("*.jpg"))})
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    size = sum(p.stat().st_size for p in assets.glob("*.jpg"))
    print(f"{out_pdf} {out_pdf.stat().st_size / 1e6:.2f} MB, {W} x {total:.1f} pt, row {h:.2f} pt, crop {cw:.2f} pt "
          f"({CROP_W / (cw / 72.27):.0f} ppi), assets {size / 1e6:.2f} MB")
    for e in examples:
        print(e["prompt_id"], e["crop_xywh"])
    for r in records:
        print(r["prompt_id"], r["method"], r["printed"], "best" if r["best"] else "")


if __name__ == "__main__":
    main()
