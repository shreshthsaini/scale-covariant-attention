"""Page side of the showcase: web/assets/js/showcase.js (window.SCA_SHOWCASE) and the page's own WebP files.

Inputs (read only): web/build/showcase_manifest.json (build_showcase.py, the showcase campaign of HANDOVER Section 32:
one entry per image with prompt, model, method, size "HxW", seed, steps, PNG path and four crop windows shared by the
group's images) and web/build/showcase_page.json (which images the page shows: "hero", "cards", "backbones", each a
list of {group, prompt_id}). The PNGs are decoded here; nothing depends on which files build_showcase.py exported.

Outputs, per shown image, in web/assets/img/zoom/<group>_<prompt id>/:
  view.webp            display view (hero 2000 px, cards 1600 px on the long side)
  c<k>.webp            the k-th crop window at native pixels (768 px, the manifest's boxes)
  t/<c>_<r>.webp       hero only: the native image in 512 px tiles for the page's deep zoom
  <method>_view.webp, <method>_c<k>.webp, <method>_t/
                       the same view, windows and (hero) tiles of each comparison image of the group (AReA, or the
                       same model sampled directly at the target size with full attention), when the manifest lists
                       it. Native images upscaled with Lanczos are never shown (user directive 2026-11-08).
Each item carries the page's zoom grid (assets/js/zoomgrid.js): one panel per image, the four windows as the
auto-zoom stops (they come from build_showcase.py's edge-energy map, the same boxes for every image of the group)
and as native-pixel tiles, and the hero's whole-image tiles. Every panel of a grid gets the same tile coverage.
Sizes on the page are width x height (the manifest's "size" is "HxW"). Files are rewritten only when missing or
older than their PNG.

Run (repository root, CPU; about a minute):
  nice -n 19 /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/build_showcase_page.py [--procs 6]
Writes an empty SCA_SHOWCASE when the manifest does not exist.
"""
import argparse
import json
import os
from multiprocessing import Pool

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
MANIFEST = os.path.join(HERE, "showcase_manifest.json")
PAGE_CFG = os.path.join(HERE, "showcase_page.json")
ZOOM = os.path.join(WEB, "assets", "img", "zoom")
JS = os.path.join(WEB, "assets", "js", "showcase.js")
TILE = 512
SCA_METHODS = ("sca_v2_prog", "sca_v2", "flux2_sca_v2", "qwen_sca")
CONFIG = {"sca_v2_prog": "Progressive SCA", "sca_v2": "Single-pass SCA", "flux2_sca_v2": "SCA, single pass",
          "qwen_sca": "SCA, single pass", "area": "AReA", "qwen_area": "AReA", "flux2_full": "Full attention",
          "qwen_full": "Full attention"}
NOT_SHOWN = ("lanczos",)    # user directive 2026-11-08: compare against methods only
CATS = {"product": "Products", "landscape": "Landscapes", "architecture": "Architecture", "portrait": "Portraits",
        "animal": "Animals", "dense": "Dense scenes"}


def rel(p):
    return os.path.relpath(p, WEB).replace(os.sep, "/")


def wh(size):
    h, w = (int(v) for v in size.lower().split("x"))
    return [w, h]


def fresh(path, src):
    return os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(src)


def render(job):
    """One image: view, crops and (hero) tiles. job = (png, out dir, prefix, view_px, boxes, tiles, ref_size)."""
    png, out, prefix, view_px, boxes, want_tiles, ref = job
    os.makedirs(out, exist_ok=True)
    vp = os.path.join(out, f"{prefix}view.webp")
    crops = [os.path.join(out, f"{prefix}c{k}.webp") for k in range(len(boxes))]
    todo = not all(fresh(p, png) for p in [vp] + crops)
    tdir = os.path.join(out, prefix + "t")
    if want_tiles:
        with Image.open(png) as probe:
            W, H = probe.size
        cols, rows = -(-W // TILE), -(-H // TILE)
        todo = todo or not fresh(os.path.join(tdir, f"{cols - 1}_{rows - 1}.webp"), png)
    if todo:
        im = Image.open(png).convert("RGB")
        W, H = im.size
        RW, RH = ref or (W, H)              # a smaller native image is shown at the SCA image's size
        s = min(1.0, view_px / max(RW, RH))
        v = im.resize((round(RW * s), round(RH * s)), Image.LANCZOS)
        v.save(vp, "WEBP", quality=80, method=6)
        kx, ky = W / RW, H / RH
        for (x, y, b), p in zip(boxes, crops):
            c = im.crop((round(x * kx), round(y * ky), round((x + b) * kx), round((y + b) * ky)))
            if c.size != (b, b):
                c = c.resize((b, b), Image.LANCZOS)
            c.save(p, "WEBP", quality=82, method=6)
        if want_tiles:
            os.makedirs(tdir, exist_ok=True)
            for r in range(-(-H // TILE)):
                for c in range(-(-W // TILE)):
                    box = (c * TILE, r * TILE, min(W, (c + 1) * TILE), min(H, (r + 1) * TILE))
                    im.crop(box).save(os.path.join(tdir, f"{c}_{r}.webp"), "WEBP", quality=80, method=4)
    with Image.open(vp) as v:
        vs = list(v.size)
    rec = {"view": rel(vp), "view_size": vs, "crops": [rel(p) for p in crops]}
    if want_tiles:
        with Image.open(png) as probe:
            W, H = probe.size
        rec["tiles"] = {"path": rel(os.path.join(tdir, "{c}_{r}.webp")), "size": TILE, "cols": -(-W // TILE), "rows": -(-H // TILE)}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    if not os.path.exists(MANIFEST):
        with open(JS, "w") as f:
            f.write("// Generated by web/build/build_showcase_page.py; do not edit. Empty until web/build/showcase_manifest.json exists.\n")
            f.write("window.SCA_SHOWCASE = {};\n")
        print("no manifest: wrote an empty", JS)
        return
    man = json.load(open(MANIFEST))
    cfg = json.load(open(PAGE_CFG)) if os.path.exists(PAGE_CFG) else {"hero": [], "cards": [], "backbones": []}
    by = {}
    for e in man.get("images", []):
        by.setdefault((e["group"], e["prompt_id"]), []).append(e)

    jobs, plan = [], []
    for role in ("hero", "cards", "backbones"):
        for sel in cfg.get(role, []):
            key = (sel["group"], sel["prompt_id"])
            es = by.get(key, [])
            sca = [e for e in es if e["method"] in SCA_METHODS]
            if not sca or not os.path.exists(sca[0]["png"]):
                print(f"skip {role} {key}: no SCA image yet")
                continue
            s = sca[0]
            out = os.path.join(ZOOM, f"{key[0]}_{key[1]}")
            boxes = [list(b) for b in s["crops"]]
            ref = wh(s["size"])
            members = [(s, "")] + [(o, o["method"] + "_") for o in es if o is not s and os.path.exists(o["png"])
                                   and o["method"] not in NOT_SHOWN and o["method"] not in SCA_METHODS]
            for e, prefix in members:
                jobs.append((e["png"], out, prefix, 2000 if role == "hero" else 1600, boxes, role == "hero", ref))
            plan.append((role, s, members))
    with Pool(a.procs) as pool:
        results = pool.map(render, jobs)
    it = iter(results)
    data = {"hero": [], "groups": [], "backbones": []}
    cards = {}
    for role, s, members in plan:
        recs = [next(it) for _ in members]
        r0 = recs[0]
        item = {"id": f"{s['group']}_{s['prompt_id']}", "title": s["title"], "category": s["category"], "prompt": s["prompt"],
                "model": s["model"], "config": CONFIG.get(s["method"], s["method"]), "size": wh(s["size"]),
                "seed": s.get("seed", 0), "steps": s.get("steps"), "view": r0["view"], "view_size": r0["view_size"],
                "crops": [{"box": b, "file": f} for b, f in zip(s["crops"], r0["crops"])], "compare": []}
        if "tiles" in r0:
            item["tiles"] = r0["tiles"]
        item["regions"] = [[b[0] + b[2] // 2, b[1] + b[2] // 2, b[2]] for b in s["crops"]]
        item["panels"] = []
        for (o, _), r in zip(members, recs):
            pan = {"method": o["method"], "label": CONFIG.get(o["method"], o["method"]), "view": r["view"], "vs": r["view_size"]}
            if o is s:
                pan["ours"] = True
            if "tiles" in r:
                pan["tiles"] = r["tiles"]
            else:
                pan["rt"] = [[b[0], b[1], b[2], b[2], f] for b, f in zip(s["crops"], r["crops"])]
            item["panels"].append(pan)
            if o is not s:
                item["compare"].append({"method": o["method"], "label": pan["label"], "view": r["view"], "crops": r["crops"]})
        if role == "hero":
            data["hero"].append(item)
        elif role == "cards":
            cards.setdefault(s["category"], []).append(item)
        else:
            data["backbones"].append(item)
    for cat, title in CATS.items():
        if cards.get(cat):
            data["groups"].append({"id": cat, "title": title, "items": cards[cat]})
    with open(JS, "w") as f:
        f.write("// Generated by web/build/build_showcase_page.py from web/build/showcase_manifest.json and showcase_page.json; do not edit.\n")
        f.write("window.SCA_SHOWCASE = " + json.dumps(data, indent=1) + ";\n")
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(ZOOM) for f in fs) if os.path.isdir(ZOOM) else 0
    print(f"wrote {JS}: hero {len(data['hero'])}, cards {sum(len(g['items']) for g in data['groups'])}, "
          f"backbones {len(data['backbones'])}; assets/img/zoom {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
