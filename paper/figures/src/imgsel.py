"""Shared helpers for the qualitative figures: locate generated images, read prompts, crop, save.

Images are FLUX.1-dev outputs (seed 0) written by scripts/sample.py and scripts/run_area.py as
<prompt_id>_<hash>_seed<k>.png with a JSON sidecar holding the prompt. A method is addressed by a
label that maps to a directory pattern with a {res} slot (see ROOTS); new methods or resolutions
(DyPE, SEGA, UltraImage, 6144^2, 8192^2, HRGen prompts) only need a new ROOTS entry or a
--root override in the figure scripts.
"""

import glob
import hashlib
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

# The final operator (sca_v2): published SCA (label "sca", directory main/sca_nb) plus the window schedule after the
# opening (windows of 128, 112, 96 and 80 tokens on the four steps after the gate, then 64) and 1,024 bridge tokens in
# the local softmax. Its FLUX.1-dev images exist at three sizes only, under the published file names. Figures that
# show SCA as ours at these sizes read them through final_twin(); the published images stay where they are.
PUBLISHED_ROOT = OUT + "/main/sca_nb/{res}"
FINAL_ROOT = OUT + "/dupfix/sca_v2/{res}"
FINAL_RES = ("4096x4096", "2048x4096", "2048x6144")
FINAL_FLAGS = {"gate_sigma": 0.89, "window": 64, "pi_variant": "head",
               "win_sched": [[0.88, 128], [0.86, 112], [0.84, 96], [0.82, 80]],
               "bridge_cells": 8, "bridge_k": 16, "win_pi_match": True, "win_pi_interp": True}
ROOTS["sca_v2"] = FINAL_ROOT


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


def final_twin(png):
    """The final operator's image of one published SCA image: main/sca_nb/<res>/<name> -> dupfix/sca_v2/<res>/<name>.

    Checked before it is returned: a complete sidecar whose png_sha256 matches the file, method sca_v2 with the
    FINAL_FLAGS, and the same prompt id, prompt, resolution, seed, derived seed, steps, guidance and model as the
    published image (same prompt, same initial noise). Returns (path, sha256 of the PNG)."""
    png = os.path.abspath(png)
    res = os.path.basename(os.path.dirname(png))
    if res not in FINAL_RES or os.path.dirname(png) != PUBLISHED_ROOT.format(res=res):
        raise ValueError(f"not a published SCA image at a size the final operator has: {png}")
    twin = os.path.join(FINAL_ROOT.format(res=res), os.path.basename(png))
    with open(png[:-4] + ".json") as fh:
        pub = json.load(fh)["config"]
    with open(twin[:-4] + ".json") as fh:
        meta = json.load(fh)
    digest = sha256(twin)
    if meta.get("complete") is not True or meta.get("png_sha256") != digest:
        raise ValueError(f"incomplete or hash-mismatched final-operator image: {twin}")
    cfg = meta["config"]
    flags = {k: cfg.get("sca", {}).get(k) for k in FINAL_FLAGS}
    if cfg.get("method") != "sca_v2" or flags != FINAL_FLAGS:
        raise ValueError(f"{twin}: not the final operator (method {cfg.get('method')}, flags {flags})")
    same = ("prompt_id", "prompt", "res", "seed", "derived_seed", "steps", "guidance", "model")
    diff = {k: (pub.get(k), cfg.get(k)) for k in same if pub.get(k) != cfg.get(k)}
    if diff:
        raise ValueError(f"{twin} differs from the published image in {diff}")
    return twin, digest


def final_png(res, sid, seed=0):
    """(path, sha256) of the final operator's image for a prompt (short id, file-name id or full id), via its
    published twin."""
    return final_twin(find("sca", res, short_id(sid), seed))


def sha256(path):
    """Hex SHA256 of a file, for provenance records."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def res_size(res):
    """'HxW' -> (W, H) in pixels, the PIL size of an image at that resolution."""
    h, w = (int(v) for v in res.split("x"))
    return w, h
