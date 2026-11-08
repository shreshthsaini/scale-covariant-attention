"""Page side of the showcase: web/assets/js/showcase.js (window.SCA_SHOWCASE) from web/build/showcase_manifest.json.

The manifest and the WebP files under web/assets/img/showcase/<group>/ come from build_showcase.py (the showcase
campaign, HANDOVER Section 32): per image an overview (2400 px long side) and four 768 px native-pixel crops at the
group's shared windows. This script only reads them, plus the native PNGs of the hero images, for which it cuts
native-pixel lens tiles into web/assets/img/zoom/<group>_<prompt>/ (512 px WebP tiles, loaded under the lens only).

Page sections it fills
  hero       the images named in web/build/showcase_page.json "hero" (group and prompt id), else the first exported
             image of flux1_5120 and of flux1_aspect: overview, lens tiles, the four native crops;
  groups     FLUX.1-dev SCA images by prompt category (products, landscapes and architecture, portraits, animals,
             dense scenes): overview, four native crops, and the same views of AReA and of the native image upscaled
             with Lanczos when exported (compared on the page with synced sliders);
  backbones  FLUX.2-dev and Qwen-Image SCA beside full attention at the target (or the native image with Lanczos).
Only images that build_showcase.py exported ("web" in the manifest entry) are used. Sizes are width x height on the
page; the manifest's "size" is "HxW".

Run (repository root, CPU, a few seconds per hero image):
  nice -n 19 /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/build_showcase_page.py [--no-tiles]
Writes an empty SCA_SHOWCASE when the manifest does not exist.
"""
import argparse
import json
import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
MANIFEST = os.path.join(HERE, "showcase_manifest.json")
PAGE_CFG = os.path.join(HERE, "showcase_page.json")
ZOOM = os.path.join(WEB, "assets", "img", "zoom")   # lens tiles; build_showcase.py owns assets/img/showcase
JS = os.path.join(WEB, "assets", "js", "showcase.js")
TILE = 512
SCA_METHODS = ("sca_v2_prog", "sca_v2", "flux2_sca_v2", "qwen_sca")
CATS = [("product", "Products and materials"), ("landscape", "Landscapes"), ("architecture", "Architecture"),
        ("portrait", "Portraits"), ("animal", "Animals"), ("dense", "Dense scenes")]
CONFIG = {"sca_v2_prog": "Progressive SCA", "sca_v2": "Single-pass SCA", "flux2_sca_v2": "SCA, single pass",
          "qwen_sca": "SCA, single pass", "area": "AReA", "flux2_full": "Full attention", "qwen_full": "Full attention",
          "lanczos": "Native + Lanczos"}


def rel(p):
    return os.path.relpath(p, WEB).replace(os.sep, "/")


def wh(size):
    h, w = (int(v) for v in size.lower().split("x"))
    return [w, h]


def tiles(png, key):
    """Native-pixel lens tiles of one PNG (skipped when they already exist and are newer than the PNG)."""
    out = os.path.join(ZOOM, key)
    im = None
    with Image.open(png) as probe:
        W, H = probe.size
    cols, rows = -(-W // TILE), -(-H // TILE)
    last = os.path.join(out, f"{cols - 1}_{rows - 1}.webp")
    if not (os.path.exists(last) and os.path.getmtime(last) > os.path.getmtime(png)):
        os.makedirs(out, exist_ok=True)
        im = Image.open(png).convert("RGB")
        for r in range(rows):
            for c in range(cols):
                box = (c * TILE, r * TILE, min(W, (c + 1) * TILE), min(H, (r + 1) * TILE))
                im.crop(box).save(os.path.join(out, f"{c}_{r}.webp"), "WEBP", quality=80, method=4)
    return {"path": rel(os.path.join(out, "{c}_{r}.webp")), "size": TILE, "cols": cols, "rows": rows}


def view_size(path):
    with Image.open(os.path.join(WEB, path)) as im:
        return list(im.size)


def item(e, others, with_tiles):
    w = e["web"]
    rec = {"id": f"{e['group']}_{e['prompt_id']}", "group": e["group"], "prompt_id": e["prompt_id"], "title": e["title"],
           "category": e["category"], "prompt": e["prompt"], "model": e["model"],
           "config": CONFIG.get(e["method"], e.get("method_name", e["method"])), "size": wh(e["size"]),
           "seed": e.get("seed", 0), "steps": e.get("steps"), "view": w["overview"], "view_size": view_size(w["overview"]),
           "crops": [{"box": b, "file": f} for b, f in zip(e["crops"], w["crops"])], "compare": []}
    for o in others:
        if o.get("web"):
            rec["compare"].append({"method": o["method"], "label": CONFIG.get(o["method"], o.get("method_name", o["method"])),
                                   "view": o["web"]["overview"], "crops": o["web"]["crops"],
                                   "note": ("native " + "x".join(reversed(o["native_size"].split("x"))) + ", upscaled with Lanczos")
                                   if o["method"] == "lanczos" and o.get("native_size") else None})
    if with_tiles:
        rec["tiles"] = tiles(e["png"], rec["id"])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tiles", action="store_true", help="skip cutting lens tiles for the hero images")
    a = ap.parse_args()
    if not os.path.exists(MANIFEST):
        with open(JS, "w") as f:
            f.write("// Generated by web/build/build_showcase_page.py; do not edit. Empty until web/build/showcase_manifest.json exists.\n")
            f.write("window.SCA_SHOWCASE = {};\n")
        print("no manifest: wrote an empty", JS)
        return
    man = json.load(open(MANIFEST))
    cfg = json.load(open(PAGE_CFG)) if os.path.exists(PAGE_CFG) else {}
    by = {}
    for e in man.get("images", []):
        by.setdefault((e["group"], e["prompt_id"]), []).append(e)

    def sca_of(key):
        es = by.get(key, [])
        s = [e for e in es if e["method"] in SCA_METHODS and e.get("web")]
        return (s[0], [e for e in es if e is not s[0]]) if s else (None, es)

    hero_keys = [(h["group"], h["prompt_id"]) for h in cfg.get("hero", [])]
    if not hero_keys:
        for g in ("flux1_5120", "flux1_aspect"):
            ks = [k for k in by if k[0] == g and sca_of(k)[0]]
            hero_keys += ks[:1]
    hero = []
    for k in hero_keys:
        s, others = sca_of(k)
        if s:
            hero.append(item(s, others, not a.no_tiles))
    used = set(hero_keys)

    groups = []
    order = cfg.get("group_order", ["flux1_4096", "flux1_5120", "flux1_aspect"])
    for cat, title in CATS:
        items = []
        for g in order:
            for k in sorted(by, key=lambda k: k[1]):
                if k[0] != g or k in used:
                    continue
                s, others = sca_of(k)
                if s and s["category"] == cat:
                    items.append(item(s, others, False))
                    used.add(k)
        if items:
            groups.append({"id": cat, "title": title, "items": items})

    backbones = []
    for g in ("flux2_5120", "flux2_4096", "qwen_5312", "qwen_4096"):
        for k in sorted(by, key=lambda k: k[1]):
            if k[0] != g:
                continue
            s, others = sca_of(k)
            if not s:
                continue
            it = item(s, others, False)
            it["note"] = cfg.get("backbone_notes", {}).get(k[1])
            backbones.append(it)

    data = {"source": "web/build/showcase_manifest.json", "crop_rule": man.get("crop_rule"), "hero": hero,
            "groups": groups, "backbones": backbones}
    with open(JS, "w") as f:
        f.write("// Generated by web/build/build_showcase_page.py from web/build/showcase_manifest.json; do not edit.\n")
        f.write("window.SCA_SHOWCASE = " + json.dumps(data, indent=1) + ";\n")
    n = sum(len(g["items"]) for g in groups)
    print(f"wrote {JS}: hero {len(hero)}, {n} images in {len(groups)} groups, {len(backbones)} backbone pairs")


if __name__ == "__main__":
    main()
