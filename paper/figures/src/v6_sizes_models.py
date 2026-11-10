"""Main-text figure (paper/figures/v6/sizes_models.pdf, label fig:sizes_models): SCA beyond the 4096^2 FLUX.1-dev setting,
5.5 in wide and at most 2.4 in tall. Each tile is one image at its own size or checkpoint with a native 1:1 inset of
its 256 x 256 window of highest edge energy and a small tag naming the setting.

Settings (fixed order): FLUX.1-dev single-pass SCA at 2048 x 4096 and 2048 x 6144 (evaluation prompts) and at
6144 x 4096 (showcase prompts, portrait orientation); FLUX.1-dev SCA without the schedule at 6144^2 and 8192^2
(the completed evaluation prompts); FLUX.1-Krea-dev single-pass SCA and FLUX.2-dev SCA without the schedule at 4096^2.
Selection rule (one clause in the caption): one prompt per setting, drawn with seed 0
(random.Random("0:<setting>").choice) among that setting's prompts that name no person (v6_common.PERSON; showcase
prompts must also lie outside the showcase portrait category), no prompt twice, each setting from a keyword category no earlier setting used while one remains. Every setting is drawn;
a setting whose draw is not presentable is left out afterwards (DROP_REASON; recorded in visuals.md and provenance).

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I v6_sizes_models.py [--layout A|B] [--out-dir DIR]
      [--drop SETTING ...]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image  # noqa: E402

import imgsel  # noqa: E402
import v6_common as vc  # noqa: E402
import v6_hero  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = "/scratch/09032/saini_2/sca/out"
WIN = 256
H_MAX = 2.4 * 72.27


def _eval(run, method, res, root_csv=None):
    path = Path(root_csv) if root_csv else vc.EVAL / run / "base" / "per_image.csv"
    with open(path, newline="") as fh:
        return {imgsel.short_id(r["prompt_id"]): r["path"] for r in csv.DictReader(fh)
                if r["method"] == method and r["seed"] == "0" and r["res"] == res}


def _dir(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        m = imgsel.NAME.match(Path(p).name)
        if m and int(m["seed"]) == 0:
            out[imgsel.short_id(m["pid"])] = p
    return out


def _showcase(res):
    out = {}
    for p in sorted(glob.glob(f"{OUT}/showcase/flux1_sca_v2/{res}/sc_*_seed0.png")):
        out[Path(p).name.split("_seed")[0].rsplit("_", 1)[0]] = p
    return out


# key, tag (TeX), configuration, pool loader
SETTINGS = [
    ("f1_2048x4096", r"2048\,\texttimes\,4096", "FLUX.1-dev, single-pass SCA",
     lambda: _eval("a4k195_2048x4096_sca_v2", "sca_v2", "2048x4096")),
    ("f1_6144", r"6144\,\texttimes\,6144", "FLUX.1-dev, SCA without the schedule",
     lambda: _dir(f"{OUT}/main/sca_nb/6144x6144/*.png")),
    ("f1_8192", r"8192\,\texttimes\,8192", "FLUX.1-dev, SCA without the schedule",
     lambda: _dir(f"{OUT}/main/sca_nb/8192x8192/*.png")),
    ("f1_6144x4096", r"6144\,\texttimes\,4096", "FLUX.1-dev, single-pass SCA (showcase prompt)",
     lambda: _showcase("6144x4096")),
    ("f1_4096x6144", r"4096\,\texttimes\,6144", "FLUX.1-dev, single-pass SCA (showcase prompt)",
     lambda: _showcase("4096x6144")),
    ("f1_2048x6144", r"2048\,\texttimes\,6144", "FLUX.1-dev, single-pass SCA",
     lambda: _eval("a4k195_2048x6144_sca_v2", "sca_v2", "2048x6144")),
    ("krea", "FLUX.1-Krea-dev", "FLUX.1-Krea-dev, single-pass SCA, 4096^2",
     lambda: _eval("krea195_sca_v2", "sca_v2_krea", "4096x4096")),
    ("flux2", "FLUX.2-dev", "FLUX.2-dev, SCA without the schedule, 4096^2",
     lambda: _eval(None, "flux2_sca", "4096x4096", vc.EVAL / "flux2_4k_base" / "per_image.csv")),
]
ROWS = {"A": [["f1_2048x6144", "krea"], ["f1_2048x4096", "f1_4096x6144", "flux2"]],
        "A0": [["f1_2048x4096", "f1_6144", "f1_8192", "f1_6144x4096"], ["f1_2048x6144", "krea", "flux2"]],
        "B": [["f1_2048x6144", "f1_2048x4096", "f1_4096x6144", "krea", "flux2"]]}
# The 6144^2 and 8192^2 draws (SCA without the schedule) show block and stripe artifacts in their windows, so the paper
# figure leaves both sizes out (visuals.md); layout A0 keeps them for the record.
DROP_DEFAULT = ("f1_6144", "f1_8192", "f1_6144x4096")
DROP_REASON = {
    "f1_6144x4096": "the 1:1 window lands on a lattice of blocks and stripes over the defocused petals (bottom right)",
    "f1_6144": "at native pixels the stone wall of the 1:1 window smears into horizontal streaks",
    "f1_8192": "horizontal stripes across the planet and its ring at native pixels (exploratory 8192^2)"}


def texts_for(key, ids):
    if key in ("f1_6144x4096", "f1_4096x6144"):
        sc = v6_hero.showcase()
        return {s: sc[s]["prompt"] for s in ids}
    return {s: json.loads(Path(p).with_suffix(".json").read_text())["config"]["prompt"] for s, p in ids.items()}


DRAW_ORDER = ["f1_2048x4096", "f1_2048x6144", "f1_6144x4096", "f1_4096x6144", "krea", "flux2", "f1_6144", "f1_8192"]


def draw(drop=()):
    """One prompt per setting in DRAW_ORDER: Random('0:<setting>').choice among the setting's prompts that name no
    person (showcase prompts also outside the portrait category), not drawn before, and in a keyword category
    (v6_common.category) that no earlier setting used, while such a category remains (variety rule)."""
    picks, drawn, used = {}, set(), []
    loaders = {k: (tag, conf, loader) for k, tag, conf, loader in SETTINGS}
    for key in DRAW_ORDER:              # every setting is drawn; dropped ones are removed after, so a drop never
        tag, conf, loader = loaders[key]  # changes another setting's draw
        pool = loader()
        texts = texts_for(key, pool)
        sc = v6_hero.showcase() if key in ("f1_6144x4096", "f1_4096x6144") else {}
        cand = sorted(s for s in pool if s not in drawn and not vc.names_person(texts[s])
                      and sc.get(s, {}).get("category") != "portrait")  # showcase prompts: also not the portrait category
        fresh = [s for s in cand if vc.category(texts[s]) not in used]
        cand = fresh or cand
        s = random.Random(f"0:{key}").choice(cand)
        drawn.add(s)
        used.append(vc.category(texts[s]))
        picks[key] = dict(id=s, path=pool[s], tag=tag, config=conf, prompt=texts[s], pool_size=len(pool),
                          candidates=len(cand), category=used[-1])
    dropped = {k: picks.pop(k) for k in list(picks) if k in drop}
    return picks, dropped


def _prep(p):
    im = Image.open(p).convert("RGB")
    return vc.detail_map(im), im.size


def build(layout, out_dir, drop=()):
    picks, dropped = draw(drop)
    rows = [[k for k in r if k in picks] for r in ROWS[layout]]
    keys = [k for r in rows for k in r]
    with Pool(len(keys)) as mp:
        prep = dict(zip(keys, mp.map(_prep, [picks[k]["path"] for k in keys])))
    for k in keys:
        vc.sidecar(picks[k]["path"])          # complete sidecar with matching hash
    W, G = vc.TEXT_PT, vc.G
    asp = {k: prep[k][1][0] / prep[k][1][1] for k in keys}
    heights = [(W - (len(r) - 1) * G) / sum(asp[k] for k in r) for r in rows]
    crop_h = 0.0
    if layout == "B":
        crop_h = min(heights[0] * 0.95, H_MAX - heights[0] - G)
    H = sum(heights) + (len(rows) - 1) * G + (crop_h + G if crop_h else 0)
    scale = 1.0
    if H > H_MAX:                      # shrink every tile to the height cap; the rows then end a hair short of W
        scale = (H_MAX - (len(rows) - 1) * G) / (H - (len(rows) - 1) * G)
        heights = [h * scale for h in heights]
        H = H_MAX
    assets = (out_dir or vc.ASSETS) / ("sizes_models" if not out_dir else f"sizes_models_{layout}")
    pdf = (out_dir or vc.V6) / ("sizes_models.pdf" if not out_dir else f"sizes_models_{layout}.pdf")
    cv = vc.Canvas(W, H, assets)
    rec = dict(tiles=[])
    used_w = [sum(h * asp[k] for k in r) + (len(r) - 1) * G for r, h in zip(rows, heights)]
    x0 = (W - max(used_w)) / 2           # centre a tiling scaled to the height cap
    y = 0.0
    INSET, M = 34.0, 2.5                 # inset side 0.47 in, as in Figure 5
    for r, h in zip(rows, heights):
        x = x0
        for k in r:
            w = h * asp[k]
            im = Image.open(picks[k]["path"]).convert("RGB")
            tn = f"{k}_full.jpg"
            vc.save_jpeg(vc.thumb(im, w, h, dpi=250), assets / tn, vc.THUMB_Q)
            cv.image(x, y, w, h, tn)
            win = vc.best_windows([prep[k][0]], WIN, WIN, k=1)[0]
            sc = w / im.width
            b0, b1 = (x + win[0] * sc, y + win[1] * sc), (x + (win[0] + WIN) * sc, y + (win[1] + WIN) * sc)
            cn = f"{k}_crop_{win[0]}_{win[1]}.jpg"
            vc.save_jpeg(im.crop((win[0], win[1], win[0] + WIN, win[1] + WIN)), assets / cn, 92)
            if layout in ("A", "A0"):            # inset in a corner that does not cover the box
                cv.box(*b0, *b1, inside=False)
                for c in ("br", "bl", "tr", "tl"):
                    ix = x + M if c[1] == "l" else x + w - M - INSET
                    iy = y + M if c[0] == "t" else y + h - M - INSET
                    if not (ix < b1[0] + 1 and b0[0] - 1 < ix + INSET and iy < b1[1] + 1 and b0[1] - 1 < iy + INSET):
                        break
                cv.image(ix, iy, INSET, INSET, cn)
                cv.box(ix, iy, ix + INSET, iy + INSET, inside=True)
            else:                                # crop strip beneath, as wide as the tile
                cw = w
                ch = crop_h
                wpx, hpx = round(WIN * cw / ch) if cw > ch else WIN, WIN if cw > ch else round(WIN * ch / cw)
                win2 = vc.best_windows([prep[k][0]], wpx // 8 * 8, hpx // 8 * 8, k=1)[0]
                cn2 = f"{k}_strip_{win2[0]}_{win2[1]}.jpg"
                vc.save_jpeg(im.crop((win2[0], win2[1], win2[0] + win2[2], win2[1] + win2[3])), assets / cn2, 92)
                cv.box(x + win2[0] * sc, y + win2[1] * sc, x + (win2[0] + win2[2]) * sc, y + (win2[1] + win2[3]) * sc,
                       inside=False)
                cv.image(x, y + h + G, cw, ch, cn2)
                cv.box(x, y + h + G, x + cw, y + h + G + ch, inside=True)
            cv.text(x + 2.5, y + 2.5, picks[k]["tag"], size=vc.FS_TAG, chip=True)
            rec["tiles"].append(dict(key=k, **{kk: picks[k][kk] for kk in ("id", "path", "config", "prompt",
                                                                              "pool_size", "candidates", "category")},
                                     size_px=list(prep[k][1]), window_px=list(win), tile_pt=[round(w, 2), round(h, 2)],
                                     png_sha256=vc.sidecar(picks[k]["path"])["png_sha256"]))
            x += w + G
        y += h + G
    cv.compile(pdf, "sizes_models")
    rec.update(figure="sizes_models (fig:sizes_models)", layout=layout, pdf=str(pdf), pdf_bytes=pdf.stat().st_size,
               size_in=[round(W / 72.27, 3), round(H / 72.27, 3)], used_width_pt=round(max(used_w), 2),
               dropped={k: dict(id=v["id"], path=v["path"], reason=DROP_REASON.get(k, "")) for k, v in dropped.items()},
               assets=str(assets),
               rule=("one prompt per setting in DRAW_ORDER, random.Random('0:<setting>').choice over that setting's "
                     "prompts that name no person, no prompt twice, each from a keyword category no earlier setting "
                     "used while one remains"),
               history=("rule 1 (no variety clause) drew m6845347395255545991 for FLUX.1-Krea-dev, whose image has a "
                        "second lit structure on the left peak for the one tower the prompt asks for, and three "
                        "mountain scenes; the variety clause was added before its draw was viewed"))
    prov = (out_dir / f"sizes_models_{layout}_provenance.json") if out_dir else vc.V6 / "sizes_models_provenance.json"
    prov.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{pdf}: {pdf.stat().st_size / 1e6:.2f} MB, {W / 72.27:.2f} x {H / 72.27:.2f} in")
    for t in rec["tiles"]:
        print(t["key"], t["id"], t["size_px"], t["candidates"], t.get("category"), t["prompt"][:70])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="A", choices=["A", "A0", "B"])
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--drop", nargs="*", default=None)
    a = ap.parse_args()
    drop = tuple(a.drop) if a.drop is not None else (() if a.layout == "A0" else DROP_DEFAULT)
    build(a.layout, a.out_dir, drop)


if __name__ == "__main__":
    main()
