"""Page side of the showcase: web/assets/js/showcase.js (window.SCA_SHOWCASE) and the page's own WebP files.

Inputs (read only): web/build/showcase_manifest.json (build_showcase.py, the showcase campaign of HANDOVER Section 32:
one entry per image with prompt, model, method, size "HxW", seed, steps, PNG path and four crop windows shared by the
group's images) and web/build/showcase_page.json (which images the page shows: "hero", "cards", "backbones", "qwen_area", each a
list of {group, prompt_id}). "qwen_area" is the Qwen-Image block (one synced grid and a prompt strip: SCA beside AReA's
released Qwen-Image code and, where it exists, full attention); its summary numbers come from
results/qwen_5312_vs_area (scripts/qwen_area_eval.py) when that exists. The PNGs are decoded here; nothing depends on which files build_showcase.py exported.

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
import csv
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
QA_RESULTS = os.path.join(os.path.dirname(WEB), "results", "qwen_5312_vs_area")
THUMB = 176
QA_BASE = {"qwen_sca": "SCA", "qwen_sca40": "SCA", "qwen_area": "AReA", "qwen_full": "Full attention"}
METRIC_NAMES = (("ir", "ImageReward"), ("hpsv21", "HPSv2.1"), ("hpsv2", "HPSv2"), ("pickscore", "PickScore"),
                ("aesthetic", "Aesthetic"), ("clip", "CLIP score"), ("clipiqa", "CLIP-IQA"), ("arniqa", "ARNIQA"),
                ("musiq_crop1024", "MUSIQ, crops"), ("musiq_4096", "MUSIQ"))
TILE = 512
SCA_METHODS = ("sca_v2_prog", "sca_v2", "flux2_sca_v2", "qwen_sca")
CONFIG = {"sca_v2_prog": "Progressive SCA", "sca_v2": "Single-pass SCA", "flux2_sca_v2": "SCA, single pass",
          "qwen_sca": "SCA, single pass", "qwen_sca40": "SCA, 40 steps", "area": "AReA", "qwen_area": "AReA", "flux2_full": "Full attention",
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


def thumb(view, out, px=THUMB):
    """Small WebP of a view for a prompt strip; rewritten when older than the view."""
    if not fresh(out, view):
        with Image.open(view) as v:
            s = px / max(v.size)
            v.convert("RGB").resize((max(1, round(v.size[0] * s)), max(1, round(v.size[1] * s))), Image.LANCZOS).save(
                out, "WEBP", quality=78, method=6)
    return rel(out)


def qa_results(path=QA_RESULTS):
    """Compact numbers of results/qwen_5312_vs_area for the page: per-method means over the prompts that SCA (50 and
    40 steps) and AReA share, SCA minus AReA at each step count with its 95% bootstrap interval, the clean median
    seconds per image, and the step-matched timing (SCA and AReA both at 40 steps). None until it exists."""
    rows = lambda name: list(csv.DictReader(open(os.path.join(path, name)))) if os.path.exists(os.path.join(path, name)) else []
    means = {r["method"]: r for r in rows("summary_5312_paired_prompts.csv")}
    pairs = {r["metric"]: r for r in rows("sca_minus_area_5312.csv")}
    pairs40 = {r["metric"]: r for r in rows("sca40_minus_area_5312.csv")}
    timing = {(r["method"], r["res"]): r for r in rows("timing_summary.csv")}
    matched = {r["prompt_id"]: r for r in rows("timing_step_matched.csv")}.get("median")
    if not means or not pairs:
        return None
    f = lambda x: None if x in (None, "") else float(x)
    d = lambda r: None if r is None else {"delta": f(r["delta"]), "lo": f(r["ci_low"]), "hi": f(r["ci_high"]),
                                          "n": int(r["n_prompts"])}
    metrics = []
    for key, name in METRIC_NAMES:
        if key in pairs and key in means.get("qwen_sca", {}) and key in means.get("qwen_area", {}):
            metrics.append({"key": key, "name": name, "sca": f(means["qwen_sca"][key]), "area": f(means["qwen_area"][key]),
                            "sca40": f(means.get("qwen_sca40", {}).get(key)), "d50": d(pairs[key]),
                            "d40": d(pairs40.get(key)), "n": int(pairs[key]["n_prompts"])})
    times = {}
    for m in ("qwen_sca", "qwen_sca40", "qwen_area", "qwen_full"):
        r = timing.get((m, "5312x5312"))
        if r:
            times[m] = {"steps": int(r["steps"]), "n": int(r["n"]), "n_clean": int(r["n_clean"]),
                        "median_clean": f(r["median_seconds_clean"]), "median_all": f(r["median_seconds_all"]),
                        "per_step_clean": f(r["median_seconds_per_step_clean"])}
    step = None if not matched else {"n": int(matched["both_clean"]), "sca40": f(matched["sca40_seconds"]),
                                     "area": f(matched["area_seconds"]), "ratio": f(matched["area_over_sca40"])}
    return {"n_prompts": max((m["n"] for m in metrics), default=0), "metrics": metrics, "timing": times,
            "step_matched": step, "source": "results/qwen_5312_vs_area"}


def qa_image_flags(path=QA_RESULTS):
    """{(method, prompt id): gpu_shared} per 5312^2 image from results/qwen_5312_vs_area/timing.csv (the evaluator
    times each image against the task's co-tenant polls; the manifest flags whole tasks). Empty until it exists."""
    f = os.path.join(path, "timing.csv")
    if not os.path.exists(f):
        return {}
    return {(r["method"], r["prompt_id"]): r["gpu_shared"] == "True" for r in csv.DictReader(open(f))
            if r["res"] == "5312x5312"}


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
    cfg = json.load(open(PAGE_CFG)) if os.path.exists(PAGE_CFG) else {"hero": [], "cards": [], "backbones": [],
                                                                        "qwen_area": []}
    by = {}
    for e in man.get("images", []):
        by.setdefault((e["group"], e["prompt_id"]), []).append(e)

    jobs, plan = [], []
    for role in ("hero", "cards", "backbones", "qwen_area"):
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
    data = {"hero": [], "groups": [], "backbones": [], "qwen_area": {"items": [], "results": qa_results()}}
    qa_flags = qa_image_flags()
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
            pan = {"method": o["method"], "m": o["method"], "label": CONFIG.get(o["method"], o["method"]), "view": r["view"],
                   "vs": r["view_size"]}
            if role == "qwen_area" and o["method"] in QA_BASE:      # the Qwen block names each panel's step count
                pan["label"] = QA_BASE[o["method"]] + (f", {o['steps']} steps" if o.get("steps") else "")
            if o is s or (role == "qwen_area" and o["method"] == "qwen_sca40"):
                pan["ours"] = True
            if "tiles" in r:
                pan["tiles"] = r["tiles"]
            else:
                pan["rt"] = [[b[0], b[1], b[2], b[2], f] for b, f in zip(s["crops"], r["crops"])]
            item["panels"].append(pan)
            if o is not s:
                item["compare"].append({"method": o["method"], "label": pan["label"], "view": r["view"], "crops": r["crops"]})
        if role == "qwen_area":
            item["thumb"] = thumb(os.path.join(WEB, r0["view"]), os.path.join(WEB, os.path.dirname(r0["view"]), "thumb.webp"))
            item["timing"] = {o["method"]: {"seconds": None if o.get("seconds") is None else round(o["seconds"], 1),
                                            "steps": o.get("steps"),
                                            "shared": qa_flags.get((o["method"], o["prompt_id"]), o.get("gpu_shared"))}
                              for o, _ in members}
            data["qwen_area"]["items"].append(item)
        elif role == "hero":
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
          f"backbones {len(data['backbones'])}, qwen_area {len(data['qwen_area']['items'])} "
          f"(results {'yes' if data['qwen_area']['results'] else 'not yet'}); assets/img/zoom {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
