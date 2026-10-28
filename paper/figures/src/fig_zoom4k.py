"""Zoom comparison at 4096^2 (replaces fig:qual_main), in the style of MFlow Figure 22, as a TikZ PDF.

Three prompts at 4096^2, seed 0, one row each: SCA's full image with the crop window outlined in red,
then that window at native resolution (1:1) from full attention, AReA, HRDiT and SCA (ours, last).
Above each crop is the image's MUSIQ_c (MUSIQ averaged over four fixed random 1024^2 crops per image,
Appendix app:metrics), read from the 195-prompt per-image CSVs behind Table tab:main (eval/a4k195_main;
HRDiT from eval/a4k195_panel), and checked to be identical, for the same PNG, in every older per-image CSV
that scored it (hrdit191_base, final4k_*, and their archived copies); the best value in each row
is bold. House style: method names once, SCA (ours) last in the bold accent, 1 to 2 pt white gutters,
no image borders except the red window boxes, sans labels of 6 to 8 pt at printed size.

Selection (recorded in figures/zoom4k_provenance.json):
* Prompts are curated (audits of 2026-10-20). The previous three, the first three prompts of the older
  fig:qual_main, were viewed whole and at native pixels. Two SCA images had artifacts: the blue-skinned
  character (2100482621711410240) has a smaller copy of itself at a candle-lit desk in the background and
  floating hands, and the woman in clouds (m2729538226706404616) has a second ear with an earring at the
  back of her head. The night street (m6394968997939197542) is clean, but AReA sits far below its own
  per-image median there (HPSv2.1 p7, ImageReward p21, MUSIQ_c p2), so the row did not meet the rule
  below either. All three were replaced. The medieval square (m3268147817845827695) that took the middle
  row was replaced in turn: its prompt asks for a lone figure, and SCA adds about three small
  pedestrians, one of them inside the automatic window.
* Rule: among the 195 evaluation prompts that are not displayed in any other figure and not used in
  development, shortlist by SCA's ImageReward and MUSIQ_c percentiles (eval/a4k195_main), keeping
  prompts where AReA and HRDiT each have ImageReward and HPSv2.1 at or above their own 25th percentile
  (FAIR_LO, asserted for every row; full attention is not constrained); then view SCA whole and at native
  pixels and reject any image with extra or repeated subjects, objects the prompt does not ask for,
  broken anatomy, seams, lattice or speckle; prefer varied subjects. The middle row was meant to show
  architecture or a street, but every such candidate failed the viewing or the rule (REJECTED,
  METRIC_STAGE), so it shows an interior still life: pink lilies in a vase by a paneled door
  (2232973817337527860). Ranking on SCA's metrics favours SCA in the printed values, so the caption says
  the prompts are curated and Table tab:main carries the means.
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
PROMPTS = ["7109958488611222679", "2232973817337527860", "m548913239957610051"]  # portrait, lilies, waterfall
REPLACED = {  # previous prompts of this figure and why each was replaced (audits of 2026-10-20)
    "2100482621711410240": "SCA: a smaller copy of the character at a desk in the background; floating hands",
    "m6394968997939197542": "SCA clean; AReA far below its own median (HPSv2.1 p7, ImageReward p21)",
    "m2729538226706404616": "SCA: a second ear with an earring at the back of the head",
    "m3268147817845827695": ("SCA: the prompt asks for a lone figure; SCA adds about three small pedestrians, "
                             "one of them inside the automatic window")}
FAIR_LO = 25  # AReA and HRDiT: ImageReward and HPSv2.1 percentiles at or above this on every shown prompt
REJECTED = {  # candidates viewed whole and at native pixels and rejected, with the reason (SCA's image)
    # first audit (shortlist for all three rows)
    "m7393293802729135153": "a third hand under the veil",
    "m3108444262580777290": "two extra small men on the cliff",
    "m3118670343091498341": "five canines for three wolves",
    "4820545723762191815": "five moons and extra bears",
    "3943607367684487957": "repeated baseballs",
    "m7264311869050248176": "extra owls",
    # second audit (middle row; architecture or street preferred)
    "7543519915112968451": ("flower alley: two pedestrians beyond the couple the prompt asks for; garbled "
                            "restaurant signs, one inside the automatic window"),
    "175243114551627459": ("snowy church: a second, faded spire behind the right-hand tree; also repeats the "
                           "sheep-and-church subject of Figure 1"),
    "m9039328566106297566": ("neon street: SCA's MUSIQ_c only at its median, and the automatic window falls on "
                             "a neon sign of invented characters"),
    "m7824758602266642611": "character at a desk: two smaller bespectacled copies of the character behind him",
    "m5606822722792416000": "woman at a balcony door: an extra arm at the right door edge and one on the left door",
    "7265494513082458083": "woman at a stone window: two extra women, one behind the door and one behind her",
    "m5795767365495583614": "polar bear: four or five moons, extra bears and tiny people",
    "m5332089088153794307": ("pitcher: a grey hook like a second handle beside the flowers; two citrus slices "
                             "and several petals beyond the three lemons and one petal asked for"),
    "m836402013429605880": ("sheep and stone church: about five houses for the two asked for; the window falls "
                            "on bare branches; repeats the subject of Figure 1"),
    "8612805720316271716": "woman on a balcony: a smaller copy of her, in the same gown, on a ledge at top left",
    "m7459007800202269666": "couple with parasols: each figure has a third hand",
    "m4460699761507361961": "five children: eight or more children; garbled text on a shirt",
    "m5691444384207183173": "window with shutters: parallel echo outlines around the shutters, lintel and box"}
METRIC_STAGE = {  # architecture or street prompts dropped before viewing
    "m1057810871814672649": "ivy-covered stone archway: a baseline below FAIR_LO",
    "m210790534622161044": "dining room: a baseline below FAIR_LO",
    "6978695891801939003": "town with a temple: a baseline below FAIR_LO",
    "m3112502764021036396": "arched stone bridge: a baseline below FAIR_LO",
    "m2014927182109364975": "Paris panorama: a baseline below FAIR_LO",
    "6066373865342273390": "white church with a golden dome: SCA's MUSIQ_c and HPSv2.1 low",
    "5826245701465116957": "wooden cabin: SCA's MUSIQ_c low (AReA's printed value would be higher)"}
SEED, RES = "0", "4096x4096"
METRIC = "musiq_crop1024"                                # MUSIQ_c in the paper
# key, header, per-image CSV holding this method's value, method name in that CSV
MAIN195 = EVAL / "a4k195_main" / "base" / "per_image.csv"    # 195-prompt protocol of Table tab:main
PANEL195 = EVAL / "a4k195_panel" / "base" / "per_image.csv"  # same protocol, HRDiT and the panel
EXTRA = {MAIN195: EVAL / "a4k195_main" / "extra" / "extra_per_image.csv",    # ImageReward (ir), HPSv2.1
         PANEL195: EVAL / "a4k195_panel" / "extra" / "extra_per_image.csv"}
N_PROMPTS = 195
METHODS = [("full", "Full attention", MAIN195, "full"),
           ("area", "AReA", MAIN195, "area"),
           ("hrdit", "HRDiT", PANEL195, "hrdit_dev_author"),
           ("sca", r"\textbf{SCA (ours)}", MAIN195, "sca_nb")]
CROSS = {  # other per-image CSVs that score the same PNG; the value must agree exactly
    "full": [EVAL / "final4k_full_base" / "per_image.csv", ARCHIVE / "final4k_full_base" / "per_image.csv"],
    "area": [EVAL / "hrdit191_base" / "per_image.csv", EVAL / "final4k_nb_base" / "per_image.csv",
             ARCHIVE / "final4k_nb_base" / "per_image.csv", ARCHIVE / "final4k_full_base" / "per_image.csv"],
    "hrdit": [EVAL / "hrdit191_base" / "per_image.csv"],
    "sca": [PANEL195, EVAL / "hrdit191_base" / "per_image.csv", EVAL / "final4k_nb_base" / "per_image.csv",
            ARCHIVE / "final4k_nb_base" / "per_image.csv"]}
TABLE = {"full": 33.1, "area": 50.4, "hrdit": 53.5, "sca": 61.2}  # Table tab:main (195 prompts), MUSIQ_c column

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
    assert len(vals) == N_PROMPTS, (path, method, len(vals))
    return sum(vals) / len(vals)


def percentiles(pid: str) -> dict:
    """Per method: ImageReward, MUSIQ_c and HPSv2.1 of this prompt's image as a percentile of that method's own
    195 seed-0 images (floor of 100 times the share at or below it)."""
    out = {}
    for key, _, path, name in METHODS:
        base = lookup(path, name, pid)
        assert lookup(EXTRA[path], name, pid)["path"] == base["path"], (pid, key)  # same PNG in both CSVs
        out[key] = {}
        for col, src in (("ir", EXTRA[path]), (METRIC, path), ("hpsv21", EXTRA[path])):
            vals = [float(r[col]) for r in rows(src) if r["method"] == name and r["seed"] == SEED and r["res"] == RES]
            assert len(vals) == N_PROMPTS, (src, name, col, len(vals))
            x = float(lookup(src, name, pid)[col])
            out[key][col] = 100 * sum(v <= x for v in vals) // len(vals)
    return out


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
        table_check[key] = dict(file=str(path), method=name, n=N_PROMPTS, mean=round(mean, 4), table=TABLE[key])
    pct = {pid: percentiles(pid) for pid in PROMPTS}
    for pid, p in pct.items():  # the selection rule holds for every shown prompt
        for key in ("area", "hrdit"):
            assert p[key]["ir"] >= FAIR_LO and p[key]["hpsv21"] >= FAIR_LO, (pid, key, p[key])
    assert not set(PROMPTS) & (set(REJECTED) | set(METRIC_STAGE) | set(REPLACED))

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
            text("lab", "base east", h, top - 2.2, r"MUSIQ\raisebox{-0.3ex}{\fontsize{6.1}{6}\selectfont c}\,$\uparrow$")
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

    replaced = {pid: dict(why=why, values={key: float(lookup(path, name, pid)[METRIC]) for key, _, path, name in METHODS},
                          percentiles=percentiles(pid))
                for pid, why in REPLACED.items()}
    prov = dict(
        figure="zoom4k (replaces fig:qual_main)", script="paper/figures/src/fig_zoom4k.py",
        pdf=str(out_pdf), pdf_sha256=sha(out_pdf), pdf_bytes=out_pdf.stat().st_size,
        selection=("Curated (audits of 2026-10-20). The first three prompts of the older fig:qual_main were viewed "
                   "whole and at native pixels and all three were replaced; the medieval square that then took the "
                   "middle row was replaced in turn (see 'replaced'). Eligible: the 195 evaluation prompts minus the "
                   "four development prompts, the prompts of Figure 1 and every prompt displayed in another figure "
                   "(galleries as displayed, showcase, ablation, failures, progressive). Rule: shortlist by SCA's "
                   "ImageReward and MUSIQ_c percentiles within its own 195 images (eval/a4k195_main), keeping "
                   "prompts where AReA and HRDiT each have ImageReward and HPSv2.1 at or above their own %dth "
                   "percentile (asserted for every row; full attention is not constrained); view SCA whole and at "
                   "native pixels and reject extra or repeated subjects, objects the prompt does not ask for, broken "
                   "anatomy, seams, lattice or speckle; prefer varied subjects. The middle row was meant to show "
                   "architecture or a street; every such candidate failed the viewing or the rule (see "
                   "percentiles.rejected and percentiles.metric_stage), so it shows an interior still life, "
                   "2232973817337527860 (pink lilies in a vase by a paneled door). Its native pixels are clean; "
                   "minor background quirks: a flat beige door with a brass knob and a wall plate beside the "
                   "paneled door, and a squared back piece on the round table top; no subject is repeated. Clean "
                   "at overview but not used, being a second landscape: 3638740675706761149. The rule constrains "
                   "ImageReward and HPSv2.1 only; the baselines' MUSIQ_c percentiles are recorded in "
                   "percentiles.shown. Ranking on SCA's metrics favours SCA in the printed values. None of the "
                   "shown prompts is displayed in another figure." % FAIR_LO),
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
                     hrdit="out/baseline_expansion_20261017_{v1,rest145}/hrdit_dev_author (paths as listed in "
                           "eval/a4k195_panel)",
                     sca="out/main/sca_nb"),
        metric_files={str(p): sha(p) for p in sorted({r["metric_file"] for r in records}
                                                     | {c for r in records for c in r["same_value_in"]})},
        table_check=table_check, examples=examples, values=records,
        replaced=replaced,
        percentiles=dict(
            definition=("per method, the image's ImageReward (ir), MUSIQ_c (%s) and HPSv2.1 (hpsv21) as a percentile "
                        "of that method's own %d seed-0 images: floor of 100 times the share at or below it" % (METRIC, N_PROMPTS)),
            files={str(p): sha(p) for p in sorted(set(EXTRA.values()) | {MAIN195, PANEL195})},
            shown=pct,
            rejected={pid: dict(why=why, percentiles=percentiles(pid)) for pid, why in REJECTED.items()},
            metric_stage={pid: dict(why=why, percentiles=percentiles(pid)) for pid, why in METRIC_STAGE.items()}),
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
    for pid, p in pct.items():
        print(pid, " ".join("%s %d/%d/%d" % (k, v["ir"], v[METRIC], v["hpsv21"]) for k, v in p.items()))


if __name__ == "__main__":
    main()
