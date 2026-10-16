"""Shared helpers for the qualitative figures: locate generated images, read prompts, crop, save.

Images are FLUX.1-dev outputs (seed 0) written by scripts/sample.py and scripts/run_area.py as
<prompt_id>_<hash>_seed<k>.png with a JSON sidecar holding the prompt. A method is addressed by a
label that maps to a directory pattern with a {res} slot (see ROOTS); new methods or resolutions
(DyPE, SEGA, UltraImage, 6144^2, 8192^2, HRGen prompts) only need a new ROOTS entry or a
--root override in the figure scripts.
"""

import glob
import json
import os
import re

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

OUT = os.environ.get("SCA_OUT", "/scratch/09032/saini_2/sca/out")
DEV4 = "/scratch/09032/saini_2/sca/dev4.jsonl"
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, ".."))
PAPER = os.path.normpath(os.path.join(FIGDIR, ".."))

ROOTS = {  # label -> directory pattern; {res} is HxW
    "sca": OUT + "/main/sca_nb/{res}",
    "area": OUT + "/main/area/{res}",
    "full": OUT + "/main/full/{res}",
    "abl_v0": OUT + "/abl/abl_v0/{res}",
    "abl_local": OUT + "/abl/abl_localonly_nb/{res}",
    "abl_uniform": OUT + "/abl/abl_uniform_nb/{res}",
    "abl_layer": OUT + "/abl/abl_layer_nb/{res}",
    "abl_global": OUT + "/abl/abl_globalonly/{res}",
    "abl_mean": OUT + "/abl/abl_mean_nb/{res}",
    "dev_w16": OUT + "/dev/sca_w16/{res}",
    "dev_w32": OUT + "/dev/sca_w32/{res}",
}

NAME = re.compile(r"^(?P<pid>.+)_(?P<hash>[0-9a-f]{10})_seed(?P<seed>\d+)\.png$")


def root(label, res):
    pat = ROOTS.get(label, label)  # an absolute pattern may be passed directly
    return pat.format(res=res)


def index(label, res, seed=0):
    """prompt_id -> png path for one method and resolution."""
    out = {}
    for p in sorted(glob.glob(os.path.join(root(label, res), "*.png"))):
        m = NAME.match(os.path.basename(p))
        if m and int(m["seed"]) == seed:
            out[m["pid"]] = p
    return out


def short_id(pid):
    """a4k_eval_4096_minus123 -> m123 ; a4k_eval_4096_123 -> 123 (stable, file-name safe)."""
    s = pid.rsplit("_", 1)[-1]
    return s.replace("minus", "m")


def find(label, res, sid, seed=0):
    for pid, p in index(label, res, seed).items():
        if short_id(pid) == sid or pid == sid:
            return p
    raise FileNotFoundError(f"{label} {res} {sid}")


def prompt(png):
    with open(png[:-4] + ".json") as fh:
        d = json.load(fh)
    return d.get("config", d).get("prompt", "")


def dev4_ids():
    with open(DEV4) as fh:
        return {short_id(json.loads(line)["id"]) for line in fh if line.strip()}


def load(png):
    return Image.open(png).convert("RGB")


def fit(img, max_side):
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return img


def crop(img, box):
    """box = (x0, y0, size) or (x0, y0, w, h) in native pixels; a 1:1 crop, never resampled."""
    if len(box) == 3:
        box = (box[0], box[1], box[2], box[2])
    x, y, w, h = box
    return img.crop((x, y, x + w, y + h))


def save_jpeg(img, path, quality=88, max_side=1600):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = fit(img, max_side)
    # optimize + fixed subsampling keeps the bytes deterministic for identical input
    img.save(path, "JPEG", quality=quality, optimize=True, subsampling=0 if quality >= 90 else 2)
    return path
