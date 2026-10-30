"""Pick the 4096^2 comparison prompts and their shared 1:1 crop windows.

Curated prompts come from .agent/notes/figure_curation_20261020.md (clean SCA, every baseline
near its own median). Random prompts are a seeded draw from the other 195 Aesthetic-4K prompts
minus the four development prompts. One crop window per prompt, shared by all methods, placed
automatically: the CROP x CROP window with the largest mean edge response (PIL FIND_EDGES),
averaged over the three methods' images and searched on a 64 px grid inside the central 60 percent
of each axis, where the subject usually sits.

Writes web/build/compare_selection.json. Run with the sca env python (PIL, numpy).

--add-final keeps the prompts and crop windows of the existing file and adds, per prompt, the final operator's
image of the same prompt and seed (paths["sca_v2"], out/dupfix/sca_v2, found and checked by
paper/figures/src/imgsel.final_png) with its SHA-256. The windows stay the ones chosen on the previous operator's
image, so they were fixed before the final operator's images existed.
"""
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "paper", "figures", "src"))
import imgsel  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
RES = "4096x4096"
CROP = 512           # native pixels, shown 1:1
STEP = 8             # search grid in thumbnail px (64 native px)
CENTRAL = (0.2, 0.8)  # search region per axis, as a fraction of the image
SEED = 20261027
N_RANDOM = 3
METHODS = ["sca", "area", "full"]
CURATED = [  # short id, role in the paper (figure_curation_20261020.md)
    ("m8343371271909676658", "Figure 1 strip: sheep before a stone church"),
    ("5849277588805396896", "Figure 1 portrait: man with hat and guitar"),
    ("7109958488611222679", "Figure 6 portrait"),
    ("2232973817337527860", "Figure 6 lilies by a paneled door"),
    ("m548913239957610051", "Figure 6 waterfall"),
    ("m8737279395568207187", "Figure 1 cathedral"),
    ("m8919475713516982506", "Figure R.1 blossoms with an apple"),
    ("997293625843621481", "Figure R.1 ice climber"),
]


def edge_map(path):
    im = Image.open(path).convert("L").reduce(8)            # 512 x 512
    return np.asarray(im.filter(ImageFilter.FIND_EDGES), dtype=np.float32)


def best_box(paths):
    e = sum(edge_map(p) for p in paths) / len(paths)
    k = CROP // 8                                               # window in thumbnail px
    ii = np.pad(e, ((1, 0), (1, 0))).cumsum(0).cumsum(1)       # integral image
    best, arg = -1.0, None
    h, w = e.shape
    lo_y, hi_y = int(CENTRAL[0] * h), int(CENTRAL[1] * h) - k
    lo_x, hi_x = int(CENTRAL[0] * w), int(CENTRAL[1] * w) - k
    for y in range(lo_y, hi_y + 1, STEP):
        for x in range(lo_x, hi_x + 1, STEP):
            s = ii[y + k, x + k] - ii[y, x + k] - ii[y + k, x] + ii[y, x]
            if s > best:
                best, arg = s, (x, y)
    x, y = arg
    return [int(x * 8), int(y * 8), CROP], float(best / (k * k))


def add_final():
    path = os.path.join(HERE, "compare_selection.json")
    sel = json.load(open(path))
    for it in sel["items"]:
        png, digest = imgsel.final_png(RES, it["sid"])
        it["paths"]["sca_v2"], it["sca_v2_sha256"] = png, digest
        print(it["sid"], png)
    sel["final_operator"] = ("paths.sca_v2: the final operator (sca_v2) at the same prompt and seed, out/dupfix/sca_v2; "
                             "crop windows unchanged (chosen on the previous operator's image, label sca)")
    with open(path, "w") as fh:
        json.dump(sel, fh, indent=1)


def main():
    if "--add-final" in sys.argv[1:]:
        return add_final()
    idx = {m: imgsel.index(m, RES) for m in METHODS}
    common = sorted(set.intersection(*(set(v) for v in idx.values())))
    dev = imgsel.dev4_ids()
    curated_ids = [c[0] for c in CURATED]
    pool = sorted(imgsel.short_id(p) for p in common
                  if imgsel.short_id(p) not in dev and imgsel.short_id(p) not in curated_ids)
    rnd = random.Random(SEED).sample(pool, N_RANDOM)
    items = []
    for sid, label in [(c[0], "curated") for c in CURATED] + [(r, "random") for r in rnd]:
        paths = {m: imgsel.find(m, RES, sid) for m in METHODS}
        box, energy = best_box(list(paths.values()))
        items.append(dict(sid=sid, label=label, role=dict(CURATED).get(sid, "random draw"),
                          prompt=imgsel.prompt(paths["sca"]), paths=paths, box_xys=box,
                          edge_energy=round(energy, 3)))
        print(sid, label, box, items[-1]["prompt"][:70])
    out = dict(res=RES, seed_label=0, crop_px=CROP, methods=METHODS,
               random_draw=dict(seed=SEED, n=N_RANDOM, pool_size=len(pool),
                                pool="195 Aesthetic-4K prompts with all three methods at 4096x4096, "
                                     "minus the 4 development prompts and the 8 curated prompts"),
               crop_rule="512x512 native window with the largest mean FIND_EDGES response averaged "
                         "over SCA, AReA and full attention, searched on a 64 px grid inside the "
                         "central 60 percent of each axis",
               items=items)
    with open(os.path.join(HERE, "compare_selection.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
