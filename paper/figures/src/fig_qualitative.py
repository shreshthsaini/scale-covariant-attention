"""JPEG tiles for the two image figures; LaTeX in main.tex assembles them.

Tiles stay JPEG so the PDF embeds them without re-encoding (a matplotlib composite
would store raw pixels and add tens of MB). Every tile is a downscale or a 1:1 crop
of one generated 4096x4096 PNG; nothing is retouched.

  fig:qual     4 prompts x {full attention, AReA, SCA}: downscaled image with the
               crop outlined, plus the crop at native resolution (1:1 pixels).
  fig:failure  development prompts x {v0 window 64, v0 window 16, local only, SCA}.

Regenerate: <sca env python> paper/figures/src/fig_qualitative.py
Sources live on scratch (purge-eligible): /scratch/09032/saini_2/sca/out/...
"""

import glob
import os

from PIL import Image, ImageDraw

OUT = "/scratch/09032/saini_2/sca/out"
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(HERE))
RES = "4096x4096"
EDGE = "#E69F00"  # PALETTE["orange"] in orx_figstyle

THUMB = 480  # px of a downscaled full image (shown about 0.9 in wide)
CROP = 512  # px of a native-resolution crop (1:1)
FAIL = 600  # px of a failure-figure tile (shown about 1.3 in wide)

# Main comparison: methods (column dir) and prompts, none in dev4.jsonl.
MAIN_METHODS = [("full", "main/full"), ("area", "main/area"), ("sca", "main/sca_nb")]
# id suffix, crop center (x, y) in 4096 px, short name
MAIN_PROMPTS = [
    ("minus5711416573811300201", (2200, 1640), "man"),        # person: face
    ("minus2447481760626661988", (2048, 900), "bear"),        # animal: head
    ("1046736963520652161", (1100, 3150), "lake"),            # landscape: water
    ("7009273464006889449", (2048, 3040), "sign"),            # text + castle
]

# Failure modes: development prompts (excluded from the main table).
FAIL_METHODS = [
    ("v0w64", "abl/abl_v0"),           # calibrated mix from the first step, 64 window, no gate
    ("v0w16", "dev/sca_w16"),          # same with a 16-token window
    ("local", "abl/abl_localonly_nb"),  # gate, then pi = 1 (local only)
    ("sca", "main/sca_nb"),            # SCA final
]
FAIL_PROMPTS = [
    ("3614061571823823193", "moai"),
    ("minus6873203881450547484", "boats"),
    ("7354534197758042049", "bear"),
]


def find(method_dir, pid):
    hits = glob.glob(f"{OUT}/{method_dir}/{RES}/a4k_eval_4096_{pid}_*_seed0.png")
    if len(hits) != 1:
        raise FileNotFoundError(f"{method_dir} {pid}: {hits}")
    return hits[0]


def save_jpg(im, path, quality):
    im.save(path, "JPEG", quality=quality, optimize=True, progressive=False)
    print(f"{os.path.relpath(path, FIGDIR)}  {im.size[0]}x{im.size[1]}  {os.path.getsize(path) // 1024} KB")


def main_figure():
    d = os.path.join(FIGDIR, "qual")
    os.makedirs(d, exist_ok=True)
    for pid, (cx, cy), name in MAIN_PROMPTS:
        for mname, mdir in MAIN_METHODS:
            im = Image.open(find(mdir, pid)).convert("RGB")
            W, H = im.size
            x0 = min(max(cx - CROP // 2, 0), W - CROP)
            y0 = min(max(cy - CROP // 2, 0), H - CROP)
            save_jpg(im.crop((x0, y0, x0 + CROP, y0 + CROP)), f"{d}/{name}_{mname}_crop.jpg", 88)
            th = im.resize((THUMB, THUMB), Image.LANCZOS)
            k = THUMB / W
            ImageDraw.Draw(th).rectangle(
                [x0 * k, y0 * k, (x0 + CROP) * k - 1, (y0 + CROP) * k - 1], outline=EDGE, width=3)
            save_jpg(th, f"{d}/{name}_{mname}_full.jpg", 85)


def failure_figure():
    d = os.path.join(FIGDIR, "failure")
    os.makedirs(d, exist_ok=True)
    for pid, name in FAIL_PROMPTS:
        for mname, mdir in FAIL_METHODS:
            im = Image.open(find(mdir, pid)).convert("RGB").resize((FAIL, FAIL), Image.LANCZOS)
            save_jpg(im, f"{d}/{name}_{mname}.jpg", 85)


if __name__ == "__main__":
    main_figure()
    failure_figure()
