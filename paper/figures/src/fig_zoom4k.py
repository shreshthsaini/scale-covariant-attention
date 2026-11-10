"""Zoom comparison at 4096^2 (fig:qual_main) in the style of MFlow Figure 22, as a TikZ PDF.

Three curated prompts at 4096^2, seed 0, one row each: progressive SCA's full image with the crop window
outlined in red (1 pt, drawn just outside the window), then that CROP_W x CROP_H (352 x 176 px) window at
native resolution (1:1) from AReA, HRDiT, single-pass SCA and progressive SCA (the headline configuration,
last). Above each crop is its image's MUSIQ_c (MUSIQ averaged over four fixed 1024^2 crops per image,
Appendix app:metrics), read from the 195-prompt per-image CSVs behind Table tab:main: AReA from
eval/a4k195_main, HRDiT from eval/a4k195_panel, single-pass SCA from eval/a4k195_main_sca_v2 and progressive
SCA from eval/prog195_sca_v2. Each value is checked to be identical, for the same PNG, in every other per-image
CSV that scores it (CROSS), and each method's 195 per-image values must reproduce its mean in the tables
(TABLE). The best of the four printed values in each row is bold. Method names appear once, above the
columns; both SCA headers are in the accent colour and progressive SCA's is bold.

Progressive switch (2026-11-07). Progressive SCA (sca_v2_prog, out/dupfix_prog/sca_v2_prog) runs HRDiT's
released three-stage schedule: 1024^2 for 30 steps with stock attention, then the last 17 steps at 2048^2 and
the last 10 at 4096^2 with SCA attention (asserted from the sidecars). It became the paper's headline
configuration and single-pass SCA (sca_v2, 28 steps at 4096^2) the fast mode, so the full image of each row is
now progressive SCA's and a progressive-SCA crop column was added after single-pass SCA. Four crops beside the
full image fit the text width within MAX_HEIGHT (165 pt; the three-crop version of review round 1 was 219.5 pt)
only at about 2:1, so the window went from 432 x 240 to 352 x 176 px, which keeps the print scale of the crops
near that of review round 1 (293 against 283 px per inch), and the value and header bands went from 8.5 and
10 pt to 8 and 9.5 pt. The automatic window rule was re-run over the four methods drawn. The prompts were not
re-curated. Progressive SCA's images of the three prompts were viewed whole, and every new window at 1:1 and
at 2x for all four methods (FINAL_VIEW_PROG, WINDOW_VIEW_R2, recorded in the provenance).

Earlier history, kept in the provenance. Review round 1 (2026-11-04) dropped the full-attention column
(NOT_DRAWN): it spent three cells on the flat texture that Figure 1 (fig:teaser) already shows, and Table
tab:main carries its means. Full attention is still checked (its table mean, the same prompt and derived seed as
the drawn images, its value in every CSV of CROSS) and recorded as not drawn; nothing of it is displayed. On
2026-10-30 the SCA column switched from the published operator (sca_nb) to the final operator (sca_v2,
out/dupfix/sca_v2): the same prompts and seed; imgsel.final_twin checks that each single-pass image is the
published image's twin (same prompt, prompt id and derived seed) and carries the sca_v2 flags (FINAL_VIEW).

Selection (recorded in figures/zoom4k_provenance.json):
* Prompts are curated (audits of 2026-10-20). The previous three, the first three prompts of the older
  fig:qual_main, were viewed whole and at native pixels. Two SCA images had artifacts: the blue-skinned
  character (2100482621711410240) has a smaller copy of itself at a candle-lit desk in the background and
  floating hands, and the woman in clouds (m2729538226706404616) has a second ear with an earring at the
  back of her head. The night street (m6394968997939197542) is clean, but AReA sits far below its own
  per-image median there (HPSv2.1 p7, ImageReward p21, MUSIQ_c p2), so the row did not meet the rule
  below either. All three were replaced. The medieval square (m3268147817845827695) that took the middle
  row was replaced in turn: its prompt asks for a lone figure, and SCA adds about three small
  pedestrians, one of them inside the automatic window. On 2026-10-30 the final operator's lily image
  (2232973817337527860) showed ghost lines along the table top's back edge, a white glow over its rim and streaks
  across the drape, so the middle row was re-chosen by the same rule on the final operator's metrics
  (eval/a4k195_main_sca_v2) and a native audit of every tile: the still life m947425340308551929 failed (a
  see-through third bottle, pears floating inside both bottles) and the blue-skinned character at a desk
  (m7824758602266642611) passed. The published operator's image of that prompt had two small copies of the
  character and was rejected on 2026-10-20; the final operator's image has none (READMITTED_V2).
* Rule: among the 195 evaluation prompts that are not displayed in any other figure and not used in
  development, shortlist by SCA's ImageReward and MUSIQ_c percentiles (eval/a4k195_main), keeping
  prompts where AReA and HRDiT each have ImageReward and HPSv2.1 at or above their own 25th percentile
  (FAIR_LO, asserted for every row; full attention is not constrained); then view SCA whole and at native
  pixels and reject any image with extra or repeated subjects, objects the prompt does not ask for,
  broken anatomy, seams, lattice or speckle; prefer varied subjects. Ranking on single-pass SCA's metrics
  favours single-pass SCA in the printed values, so the caption says the prompts are curated and Table
  tab:main carries the means. Progressive SCA's values played no part in the selection.
* Crop windows are automatic, never placed by hand: fig_qual_main.auto_box, the CROP_W x CROP_H window
  with the most edge energy averaged over the four methods drawn, at least 256 px from every border.

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

import imgsel
from fig_qual_main import auto_box  # the crop rule of the previous fig:qual_main, unchanged

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = FIG.parent.parent
TECTONIC = os.path.expanduser("~/.local/bin/tectonic")

EVAL = Path("/scratch/09032/saini_2/sca/eval")         # per-image CSVs behind Table tab:main
ARCHIVE = REPO / "results" / "evidence" / "eval"         # byte-identical copies archived in the repo
PROMPTS = ["7109958488611222679", "m7824758602266642611", "m548913239957610051"]  # portrait, character, waterfall
REPLACED = {  # previous prompts of this figure and why each was replaced (audits of 2026-10-20)
    "2100482621711410240": "SCA: a smaller copy of the character at a desk in the background; floating hands",
    "m6394968997939197542": "SCA clean; AReA far below its own median (HPSv2.1 p7, ImageReward p21)",
    "m2729538226706404616": "SCA: a second ear with an earring at the back of the head",
    "m3268147817845827695": ("SCA: the prompt asks for a lone figure; SCA adds about three small pedestrians, "
                             "one of them inside the automatic window"),
    "2232973817337527860": ("final operator (2026-10-30): ghost lines repeat the table top's back edge (x 1100-1750, "
                            "y 3290-3430, 16 px period), a white glow replaces the left rim (x 1030-1330, y 3380-3850), "
                            "streaks cross the drape; curtain ribbing locked to the 16 px token grid")}
REPLACED_V2 = {"2232973817337527860"}  # replaced after the final operator's audit; values and percentiles: sca_v2
REJECTED_V2 = {  # final-operator candidates for the middle row (2026-10-30), in rank order; percentiles in provenance
    "m947425340308551929": ("native audit: a see-through third bottle neck inside the tall bottle (x 965-1020, y 1585-1860) "
                            "and pears floating inside both bottles, visible at print size"),
    "m9039328566106297566": "overview: neon signs of invented characters",
    "m795146951737249951": "overview: two or three double basses and extra men for one bass player and one seated man",
    "m5332089088153794307": "overview: two citrus slices beyond the three lemons and one petal asked for",
    "m5996982714209002611": "overview: lattice along the coat and sleeve edges",
    "175243114551627459": "not viewed: repeats the sheep-and-church subject of Figure 1",
    "5941097972571555184": "not viewed: repeats the sheep-and-church subject of Figure 1"}
READMITTED_V2 = {  # rejected on 2026-10-20 for the published operator's image; the final operator's image passed
    "m7824758602266642611": ("native audit 2026-10-30: no copy of the character (shelves, wall, poster checked); flaws "
                             "only at 1:1: six digits on the character's left hand (x 2100-2580, y 2630-3010, about "
                             "0.4 mm per finger in print), pseudo-lettering on the plaque, book spines, jar and map, a "
                             "melted chain tangle (x 350-730, y 3130-3470); the automatic window (the eye) is clean")}
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
MAIN195_V2 = EVAL / "a4k195_main_sca_v2" / "base" / "per_image.csv"  # single-pass SCA, final operator (sca_v2)
EXTRA[MAIN195_V2] = EVAL / "a4k195_main_sca_v2" / "extra" / "extra_per_image.csv"
PROG195 = EVAL / "prog195_sca_v2" / "base" / "per_image.csv"  # progressive SCA (sca_v2_prog), same protocol
EXTRA[PROG195] = EVAL / "prog195_sca_v2" / "extra" / "extra_per_image.csv"
N_PROMPTS = 195
METHODS = [("full", "Full attention", MAIN195, "full"),
           ("area", "AReA", MAIN195, "area"),
           ("hrdit", "HRDiT", PANEL195, "hrdit_dev_author"),
           ("sca", "Single-pass SCA", MAIN195_V2, "sca_v2"),
           ("prog", r"\textbf{Progressive SCA}", PROG195, "sca_v2_prog")]
BASE = "prog"           # every row's full image: progressive SCA, the headline configuration (since 2026-11-07)
OURS = {"sca", "prog"}  # headers in the accent colour (house style); progressive SCA's is also bold
PROG_STAGES = (["1024x1024", "2048x2048", "4096x4096"], 30, [17, 10], "stock@native+sca@highres")  # asserted
# The final-operator audits of 2026-10-30 (REPLACED_V2, REJECTED_V2, READMITTED_V2) ranked these four methods.
V2_METHODS = METHODS[:4]
# The audits of 2026-10-20 (REPLACED, REJECTED, METRIC_STAGE and the selection rule) viewed and ranked the published
# operator's images; their records keep its values.
AUDIT_METHODS = METHODS[:3] + [("sca", "SCA (published operator)", MAIN195, "sca_nb")]
FINAL_VIEW = {  # the final operator's images of the shown prompts, viewed whole and at native pixels (2026-10-30)
    "7109958488611222679": "clean; nearly identical to the published image (double shoulder straps instead of an O-ring)",
    "m7824758602266642611": ("one character, one ornate globe, map and compass; no copy of the character; flaws only "
                             "at 1:1 (READMITTED_V2); the automatic window falls on the eye behind the glasses"),
    "m548913239957610051": ("clean; the two misty background peaks of the published image are gone; a few faint "
                            "light spots in the sky")}
FINAL_VIEW_PROG = {  # progressive SCA's images of the shown prompts, viewed whole at 1024 px (2026-11-07)
    "7109958488611222679": ("one woman in an olive dress before a dark green wall, hands at the lower edge; no extra "
                            "subject; at 1:1 her hair is soft and partly defocused (WINDOW_VIEW_R2)"),
    "m7824758602266642611": ("one blue-skinned character with round glasses at a desk; candle, two framed pictures, "
                             "globe, map and compass; no copy of the character; flatter cartoon shading than "
                             "single-pass SCA's image"),
    "m548913239957610051": ("one conical mountain over a river; a main fall and a second, smaller fall to its right "
                            "on the same river (the prompt asks for one cascading waterfall); both spray plumes fan "
                            "out sideways; no repeated mountain")}
CROSS = {  # other per-image CSVs that score the same PNG; the value must agree exactly
    "full": [EVAL / "final4k_full_base" / "per_image.csv", ARCHIVE / "final4k_full_base" / "per_image.csv"],
    "area": [EVAL / "hrdit191_base" / "per_image.csv", EVAL / "final4k_nb_base" / "per_image.csv",
             ARCHIVE / "final4k_nb_base" / "per_image.csv", ARCHIVE / "final4k_full_base" / "per_image.csv",
             MAIN195_V2],
    "hrdit": [EVAL / "hrdit191_base" / "per_image.csv", PROG195],
    "sca": [],   # no other per-image CSV scores the final operator's PNGs
    "prog": []}  # no other per-image CSV scores the progressive SCA PNGs
TABLE = {"full": 33.1, "area": 50.4, "hrdit": 53.5, "sca": 60.1, "prog": 54.1}  # Table tab:main, MUSIQ_c column
NOT_DRAWN = {  # loaded, checked and recorded in the provenance, but given no column (review round 1, 2026-11-04)
    "full": ("review round 1: the column spent three cells on the flat texture that Figure 1 (fig:teaser) "
             "already shows; Table tab:main carries its means; HRDiT is already shown")}
DRAWN = [m for m in METHODS if m[0] not in NOT_DRAWN]  # the crop columns
assert [k for k, *_ in DRAWN] == ["area", "hrdit", "sca", "prog"]  # progressive SCA (the headline) last
WINDOW_VIEW_R1 = {  # review round 1 (2026-11-04): each row's new window (x, y), viewed at 300 dpi and at 2x native
    "7109958488611222679": ((1288, 1360), (
        "hair at the image-left edge of her face, the cheek in the top right corner; SCA clean (no seam, lattice "
        "or extra subject); AReA clean; HRDiT soft, its layout puts dark background behind the strands")),
    "m7824758602266642611": ((2120, 1312), (
        "the eye behind the image-right lens, the bridge at the left edge (the previous window widened 72 px to "
        "the left and 40 px to the right, 24 px lower); SCA clean; the dark panes and white arcs inside the lens "
        "are a reflection already in the previous window; AReA shows its echo outlines (concentric contour lines "
        "around the eye and lens, hatching on the frame), as before; HRDiT clean")),
    "m548913239957610051": ((704, 1440), (
        "the lip of the left fall with rocks in the river above it; SCA clean; AReA clean; HRDiT soft, its "
        "layout differs (a lower ledge and a rock pillar)"))}
WINDOW_VIEW_R2 = {  # progressive switch (2026-11-07): each row's window (x, y) over the four drawn methods, viewed
    # at 1:1 for all four and at 2x (nearest) for single-pass and progressive SCA
    "7109958488611222679": ((2416, 1288), (
        "hair beside the image-right side of her face; AReA fine strands, slightly smeared; HRDiT soft strands; "
        "single-pass SCA the sharpest, separate strands and flyaways; progressive SCA soft: the left half is "
        "defocused hair and the right half thicker strands over dark background with stair-stepped edges, as if "
        "upsampled; no seam, lattice or extra subject in any crop")),
    "m7824758602266642611": ((2088, 1352), (
        "the eye behind the image-right lens; AReA shows its echo outlines (concentric contour lines around the eye "
        "and lens), as in review round 1; HRDiT a clean, soft cartoon eye; single-pass SCA a clean eye with a "
        "detailed, striated iris and painted highlights; progressive SCA a clean, flat cartoon eye with the nose at "
        "the left edge, smooth shading and slightly soft outlines")),
    "m548913239957610051": ((760, 1480), (
        "rocks in the river and the lip of the left fall; AReA crisp rocks and water; HRDiT softer; single-pass SCA "
        "the sharpest, crisp rock edges and fine water streaks at the lip; progressive SCA rocks and smooth water "
        "above a lower ledge, softer, with mild stair-stepping on rock outlines; layouts differ between methods"))}
REVIEW_ROUND_1 = dict(  # the three-crop layout of 2026-11-04 (zoom4k_provenance.json before the progressive switch)
    date="2026-11-04",
    comment=("The full-attention column spends three cells on texture already shown in Figure 1 (flat field). "
             "Source window to 1 pt."),
    changes=["full-attention column dropped (not_drawn); HRDiT is already shown",
             "window widened from 320x240 to 432x240 px to use the freed width at about the same row height; "
             "crops stay 1:1",
             "window outline on the full image from 0.7 to 1 pt",
             "every new window viewed at 300 dpi and at 2x native pixels"],
    height_pt=dict(before=220.711, after=219.484, budget=226.711),
    row_height_pt=dict(before=61.737, after=61.328), crop_px=[432, 240], crop_ppi=282.8,
    windows={pid: dict(xy=list(xy), view=v) for pid, (xy, v) in WINDOW_VIEW_R1.items()},
    printed={"7109958488611222679": dict(area="55.7", hrdit="55.3", sca="57.4"),
             "m7824758602266642611": dict(area="51.2", hrdit="49.1", sca="70.6"),
             "m548913239957610051": dict(area="45.2", hrdit="48.6", sca="65.3")},
    pdf_sha256="a5915f2f6c2d9f235fbf658fb1234366f73fecd3d0fdf3de3325a0363377e626")

W = 397.5                    # text width (TeX pt); the layout unit is the TeX point
CROP_W, CROP_H = 352, 176    # native window (px), 1:1 at about 293 px per printed inch; 432 x 240 in review round 1
G_OUT, G_IN = 2.0, 1.5       # white gutters (pt): full image to crops, between crops
HEAD, BAND = 9.5, 8.0        # method-name band and the value band above every row (pt); 10 and 8.5 before 2026-11-07
THUMB_DPI = 800              # the full image, Lanczos-downscaled to this many px per printed inch
RED, BOX_LW = "D62728", 1.0  # the one frame colour (window boxes only) and its width (pt), drawn outside the window
MAX_HEIGHT = 165.0           # page budget of the progressive switch (2026-11-07), pt


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


def percentiles(pid: str, methods=METHODS) -> dict:
    """Per method: ImageReward, MUSIQ_c and HPSv2.1 of this prompt's image as a percentile of that method's own
    195 seed-0 images (floor of 100 times the share at or below it)."""
    out = {}
    for key, _, path, name in methods:
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


def layout(n: int | None = None, crop_w: int = CROP_W) -> tuple[float, float, list[float]]:
    """Row height h (the full image is h x h), crop width cw, and the x positions of n crops (default: one per
    drawn method) of crop_w x CROP_H px that fill the width W beside the full image."""
    n = len(DRAWN) if n is None else n
    a = crop_w / CROP_H
    h = (W - G_OUT - (n - 1) * G_IN) / (1 + n * a)
    cw = a * h
    xs = [h + G_OUT + i * (cw + G_IN) for i in range(n)]
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
    assert total <= MAX_HEIGHT, (total, MAX_HEIGHT)  # page budget of the progressive switch
    tex, records, examples, not_drawn = [], [], [], []

    def node(x, y, w, hh, name):
        tex.append(r"\node[anchor=north west,inner sep=0pt] at (%.4f,%.4f) "
                   r"{\includegraphics[width=%.4fpt,height=%.4fpt]{%s}};" % (x, -y, w, hh, name))

    def text(style, anchor, x, y, s):
        tex.append(r"\node[%s,anchor=%s,inner sep=0pt] at (%.4f,%.4f) {%s};" % (style, anchor, x, -y, s))

    # the per-image means must reproduce the tables before any value is printed
    table_check = {}
    for key, _, path, name in METHODS:
        mean = table_mean(path, name)
        assert round(mean, 1) == TABLE[key], (key, mean, TABLE[key])
        table_check[key] = dict(file=str(path), method=name, n=N_PROMPTS, mean=round(mean, 4), table=TABLE[key])
    pct = {pid: percentiles(pid) for pid in PROMPTS}
    for pid, p in pct.items():  # the selection rule holds for every shown prompt
        for key in ("area", "hrdit"):
            assert p[key]["ir"] >= FAIR_LO and p[key]["hpsv21"] >= FAIR_LO, (pid, key, p[key])
    assert not set(PROMPTS) & ((set(REJECTED) - set(READMITTED_V2)) | set(METRIC_STAGE) | set(REPLACED) | set(REJECTED_V2))

    drawn = [k for k, *_ in DRAWN]
    for r, pid in enumerate(PROMPTS):
        top = HEAD + r * (BAND + h) + BAND
        got = {key: source(path, name, pid) for key, _, path, name in METHODS}
        twin, twin_sha = imgsel.final_png(RES, pid)
        assert Path(got["sca"][1]) == Path(twin) and sha(got["sca"][1]) == twin_sha, (pid, got["sca"][1], twin)
        prompts = {m["config"]["prompt"] for _, _, m in got.values()}
        seeds = {m["config"]["derived_seed"] for _, _, m in got.values()}
        assert len(prompts) == 1 and len(seeds) == 1, (pid, prompts, seeds)  # same prompt and noise seed
        pc = got["prog"][2]["config"]  # progressive SCA: HRDiT's released stages, SCA attention above 1024^2
        assert (pc["method"], pc["steps"], pc["attention"]) == ("sca_v2_prog", 57, PROG_STAGES[3]), pid
        assert (pc["progressive"]["stages"], pc["progressive"]["base_steps"], pc["progressive"]["highres_steps"]) \
            == PROG_STAGES[:3], pid
        assert [s["res"] for s in pc["sca"]["stages"]] == PROG_STAGES[0][1:], pid
        imgs = {}
        for key in drawn:  # full attention is checked through source() and the CSVs; its pixels are not needed
            imgs[key] = Image.open(got[key][1]).convert("RGB")
            assert imgs[key].size == (4096, 4096), (got[key][1], imgs[key].size)
        bx, by, bw, bh = auto_box([imgs[k] for k in drawn], CROP_W, CROP_H)
        assert WINDOW_VIEW_R2.get(pid, (None,))[0] == (bx, by), (pid, (bx, by), "view the new window first")

        # progressive SCA's full image; the red outline lies just outside the window, which stays visible
        px = round(THUMB_DPI * h / 72.27)
        thumb = assets / f"{pid}_{BASE}_full.jpg"
        imgs[BASE].resize((px, px), Image.Resampling.LANCZOS).save(thumb, "JPEG", quality=90, optimize=True,
                                                                   subsampling=0)
        node(0, top, h, h, thumb.name)
        s, d = h / 4096, BOX_LW / 2
        tex.append(r"\draw[win] (%.4f,%.4f) rectangle (%.4f,%.4f);" % (bx * s - d, -(top + by * s - d),
                                                                       (bx + bw) * s + d, -(top + (by + bh) * s + d)))
        # crops at native resolution, values above them, the best drawn one bold; full attention checked, not drawn
        vals = {key: float(got[key][0][METRIC]) for key, *_ in METHODS}
        best = f"{max(vals[k] for k in drawn):.1f}"  # every drawn value that prints as the maximum is bold
        for key, _, path, name in METHODS:
            row, png, meta = got[key]
            cross = []
            for other in CROSS.get(key, []):
                o = lookup(other, name, pid)
                assert o[METRIC] == row[METRIC] and Path(o["path"]).resolve() == png.resolve(), (other, pid)
                cross.append(str(other))
            rec = dict(prompt_id=pid, csv_prompt_id=row["prompt_id"], seed=int(row["seed"]), method=key,
                       csv_method=name, metric_file=str(path), column=METRIC, value=vals[key])
            if key in NOT_DRAWN:  # recorded with its checks; no crop, no printed value
                not_drawn.append(dict(rec, drawn=False, same_value_in=cross, image=str(png), image_sha256=sha(png),
                                      derived_seed=meta["config"]["derived_seed"]))
                continue
            i = drawn.index(key)
            cp = assets / f"{pid}_{key}_crop.jpg"
            imgs[key].crop((bx, by, bx + bw, by + bh)).save(cp, "JPEG", quality=95, optimize=True, subsampling=0)
            node(xs[i], top, cw, h, cp.name)
            shown = f"{vals[key]:.1f}"
            text("val", "base", xs[i] + cw / 2, top - 2.0, r"\textbf{%s}" % shown if shown == best else shown)
            records.append(dict(rec, printed=shown, best=shown == best, same_value_in=cross, image=str(png),
                                image_sha256=sha(png), derived_seed=meta["config"]["derived_seed"], crop=cp.name,
                                crop_xywh=[bx, by, bw, bh]))
        if r == 0:  # the metric is named once, left of the first row of values
            text("lab", "base east", h, top - 2.0, r"MUSIQ\raisebox{-0.3ex}{\fontsize{6.1}{6}\selectfont c}\,$\uparrow$")
        examples.append(dict(prompt_id=pid, prompt=prompts.pop(), derived_seed=seeds.pop(), crop_xywh=[bx, by, bw, bh],
                             window_view=WINDOW_VIEW_R2[pid][1], base_image=BASE, base_image_view=FINAL_VIEW_PROG[pid],
                             full_image=thumb.name, full_image_px=px))

    for i, (key, head, _, _) in enumerate(DRAWN):
        text("headours" if key in OURS else "head", "base", xs[i] + cw / 2, HEAD - 2.4, head)

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
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env=dict(os.environ, SOURCE_DATE_EPOCH="0"))  # byte-identical PDF per run
        shutil.copy(Path(tmp) / "zoom4k.pdf", out_pdf)

    replaced = {pid: dict(why=why, operator="sca_v2" if pid in REPLACED_V2 else "sca_nb",
                          values={key: float(lookup(path, name, pid)[METRIC])
                                  for key, _, path, name in (V2_METHODS if pid in REPLACED_V2 else AUDIT_METHODS)},
                          percentiles=percentiles(pid, V2_METHODS if pid in REPLACED_V2 else AUDIT_METHODS))
                for pid, why in REPLACED.items()}
    published = {pid: dict(value=float(lookup(MAIN195, "sca_nb", pid)[METRIC]),
                           printed=f"{float(lookup(MAIN195, 'sca_nb', pid)[METRIC]):.1f}",
                           image=lookup(MAIN195, "sca_nb", pid)["path"],
                           percentiles=percentiles(pid, AUDIT_METHODS)["sca"])
                 for pid in PROMPTS}
    prov = dict(
        figure="zoom4k (fig:qual_main)", script="paper/figures/src/fig_zoom4k.py",
        pdf=str(out_pdf), pdf_sha256=sha(out_pdf), pdf_bytes=out_pdf.stat().st_size,
        base_image=dict(method=BASE, csv_method="sca_v2_prog",
                        what="every row's full image, with the window outlined in red just outside it"),
        methods_drawn=drawn,
        headers={key: dict(tex=head, style="headours" if key in OURS else "head") for key, head, _, _ in DRAWN},
        not_drawn={key: dict(why=why, values=[r for r in not_drawn if r["method"] == key],
                             checks=("its Table tab:main mean (table_check); the same prompt and derived seed as the "
                                     "drawn images; its value identical in every CSV of same_value_in"))
                   for key, why in NOT_DRAWN.items()},
        progressive_switch=dict(
            date="2026-11-07",
            why=("progressive SCA became the headline configuration and single-pass SCA the fast mode; the full image "
                 "of every row is now progressive SCA's and a progressive-SCA crop column follows single-pass SCA"),
            changes=["full image: single-pass SCA (sca_v2) -> progressive SCA (sca_v2_prog)",
                     "crop columns: AReA, HRDiT, SCA -> AReA, HRDiT, single-pass SCA, progressive SCA",
                     "window %dx%d -> %dx%d px (four crops at about 2:1 within MAX_HEIGHT); the rule re-run over "
                     "the four methods drawn; crops stay 1:1" % (432, 240, CROP_W, CROP_H),
                     "header and value bands 10 and 8.5 pt -> %g and %g pt" % (HEAD, BAND),
                     "the red outline drawn just outside the window, so the window itself stays visible",
                     "prompts unchanged (not re-curated)"],
            height_pt=dict(before=REVIEW_ROUND_1["height_pt"]["after"], after=round(total, 3), budget=MAX_HEIGHT),
            viewed=dict(base_images=FINAL_VIEW_PROG, windows={p: v for p, (_, v) in WINDOW_VIEW_R2.items()})),
        review_round_1=REVIEW_ROUND_1,
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
                   "percentiles.rejected and percentiles.metric_stage), so it showed an interior still life, "
                   "2232973817337527860 (pink lilies in a vase by a paneled door). Its native pixels were clean; "
                   "minor background quirks: a flat beige door with a brass knob and a wall plate beside the "
                   "paneled door, and a squared back piece on the round table top; no subject is repeated. Clean "
                   "at overview but not used, being a second landscape: 3638740675706761149. The rule constrains "
                   "ImageReward and HPSv2.1 only; the baselines' MUSIQ_c percentiles are recorded in "
                   "percentiles.shown. Ranking on SCA's metrics favours SCA in the printed values. None of the "
                   "shown prompts is displayed in another figure. All of this ranked and viewed the published "
                   "operator's images (sca_nb). On 2026-10-30 the SCA column switched to the final operator "
                   "(sca_v2), the same prompts and seed; its images were viewed whole and at native pixels "
                   "(final_operator.viewed) and its printed values come from eval/a4k195_main_sca_v2. The "
                   "final operator's lily image streaks the table top, so the middle row was re-chosen on "
                   "2026-10-30 by the same rule on the final operator's metrics (eval/a4k195_main_sca_v2; the "
                   "development prompts and every prompt shown in any figure excluded) with a native audit of "
                   "every tile: the still life m947425340308551929 failed (see-through third bottle, floating "
                   "pears) and the character at a desk m7824758602266642611 passed (its published-operator "
                   "rejection does not hold for the final operator's image; percentiles.readmitted_final_operator). "
                   "The other two rows were not re-ranked (their final-operator percentiles are in "
                   "percentiles.shown). On 2026-11-07 progressive SCA was added and became the full image "
                   "(progressive_switch); the prompts were kept, and progressive SCA's values played no part in "
                   "their selection." % FAIR_LO),
        final_operator=dict(
            method="sca_v2", role="single-pass SCA (the fast mode since 2026-11-07)",
            images="out/dupfix/sca_v2/4096x4096 (imgsel.final_twin of the published image)",
            metric_file=str(MAIN195_V2), viewed=FINAL_VIEW,
            published_operator=dict(metric_file=str(MAIN195), method="sca_nb", values=published,
                                    note="the values this figure printed for SCA before the switch")),
        progressive=dict(
            method="sca_v2_prog", role="progressive SCA (the headline configuration since 2026-11-07)",
            images="out/dupfix_prog/sca_v2_prog/4096x4096 (paths as listed in eval/prog195_sca_v2)",
            metric_file=str(PROG195),
            schedule=("HRDiT's released schedule: stages %s, %d steps at the first stage with stock attention, then "
                      "the last %d steps at 2048^2 and the last %d at 4096^2 with SCA attention (57 steps in all; "
                      "asserted from every sidecar)" % (", ".join(PROG_STAGES[0]), PROG_STAGES[1], *PROG_STAGES[2])),
            viewed=FINAL_VIEW_PROG),
        crop_rule=("Automatic, not hand-placed: fig_qual_main.auto_box over the four drawn methods (area, hrdit, sca "
                   "and prog), i.e. the %dx%d (width x height) window with the largest mean PIL FIND_EDGES response "
                   "of the 8x box-downsampled greyscale images, at least 256 px from every border. The same window "
                   "is cropped from every drawn method; layouts differ between methods, so the crops differ in "
                   "content. Review round 1 used a 432x240 window over area, hrdit and sca (review_round_1.windows)."
                   % (CROP_W, CROP_H)),
        metric=("MUSIQ_c (CSV column %s): pyiqa MUSIQ averaged over four fixed 1024^2 crops per image, "
                "crop positions set by the prompt id and shared across methods (Appendix app:metrics). It is "
                "a per-image score, not a score of the displayed window. Printed to one decimal; the best "
                "printed value per row is bold." % METRIC),
        sources=dict(full_attention="out/main/full (checked, not drawn)", area="out/main/area",
                     hrdit="out/baseline_expansion_20261017_{v1,rest145}/hrdit_dev_author (paths as listed in "
                           "eval/a4k195_panel)",
                     sca="out/dupfix/sca_v2 (single-pass, final operator; the selection audits used out/main/sca_nb)",
                     prog="out/dupfix_prog/sca_v2_prog (progressive SCA)"),
        metric_files={str(p): sha(p) for p in sorted({r["metric_file"] for r in records + not_drawn}
                                                     | {c for r in records + not_drawn for c in r["same_value_in"]})},
        table_check=table_check, examples=examples, values=records,
        replaced=replaced,
        percentiles=dict(
            definition=("per method, the image's ImageReward (ir), MUSIQ_c (%s) and HPSv2.1 (hpsv21) as a percentile "
                        "of that method's own %d seed-0 images: floor of 100 times the share at or below it" % (METRIC, N_PROMPTS)),
            files={str(p): sha(p) for p in sorted(set(EXTRA.values()) | {MAIN195, PANEL195, MAIN195_V2, PROG195})},
            shown=pct,
            note=("shown: sca is single-pass SCA (the final operator, sca_v2) and prog progressive SCA "
                  "(sca_v2_prog); rejected and metric_stage: SCA is the published operator (sca_nb), whose images "
                  "the audits viewed; the final-operator records rank area, hrdit and sca_v2"),
            rejected={pid: dict(why=why, percentiles=percentiles(pid, AUDIT_METHODS)) for pid, why in REJECTED.items()},
            metric_stage={pid: dict(why=why, percentiles=percentiles(pid, AUDIT_METHODS))
                          for pid, why in METRIC_STAGE.items()},
            rejected_final_operator={pid: dict(why=why, percentiles=percentiles(pid, V2_METHODS))
                                     for pid, why in REJECTED_V2.items()},
            readmitted_final_operator={pid: dict(why=why, percentiles=percentiles(pid, V2_METHODS))
                                       for pid, why in READMITTED_V2.items()}),
        layout_pt=dict(width=W, height=round(total, 3), row_height=round(h, 3), full_image=round(h, 3),
                       crops=len(DRAWN), crop_width=round(cw, 3), crop_px=[CROP_W, CROP_H],
                       gutters=dict(full_to_crops=G_OUT, between_crops=G_IN),
                       header_band=HEAD, value_band=BAND,
                       box=dict(colour="#" + RED, width_pt=BOX_LW, placement="just outside the window"),
                       crop_ppi=round(CROP_W / (cw / 72.27), 1), max_height=MAX_HEIGHT,
                       text_pt=dict(headers=8, values=7, metric_label=7, metric_subscript=6.1)),
        jpeg=dict(full=f"progressive SCA's full image, Lanczos to {THUMB_DPI} px per printed inch, quality 90, 4:4:4",
                  crop="native pixels (1:1), quality 95, 4:4:4"),
        asset_sha256={p.name: sha(p) for p in sorted(assets.glob("*.jpg"))})
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    size = sum(p.stat().st_size for p in assets.glob("*.jpg"))
    print(f"{out_pdf} {out_pdf.stat().st_size / 1e6:.2f} MB, {W} x {total:.1f} pt (review round 1 "
          f"{REVIEW_ROUND_1['height_pt']['after']}), row {h:.2f} pt, crop {cw:.2f} pt ({CROP_W / (cw / 72.27):.0f} ppi), "
          f"assets {size / 1e6:.2f} MB")
    for e in examples:
        print(e["prompt_id"], e["crop_xywh"])
    for r in records:
        print(r["prompt_id"], r["method"], r["printed"], "best" if r["best"] else "")
    for pid, p in pct.items():
        print(pid, " ".join("%s %d/%d/%d" % (k, v["ir"], v[METRIC], v["hpsv21"]) for k, v in p.items()))

if __name__ == "__main__":
    main()
