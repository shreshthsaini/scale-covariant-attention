"""Deep-zoom grids of the page and of web/review/: tour regions, overviews and native-pixel tiles (zoomlib.py).

Sets (all FLUX.1-dev at 4096^2, seed label 0, the same prompt and seed in every panel of a grid):
  compare  the eight-method comparison of gallery.html: the 15 evaluation prompts of web/build/gallery_draw.json
           (web/build/v5_gallery_draw.py: random.Random(0).sample over the sorted evaluation prompt ids, minus the
           main page's figure prompts), eight methods each (progressive and single-pass SCA, AReA, HRDiT,
           UltraImage, DyPE, SEGA, full attention); three tour stops. (Until 2026-11-14 this set was the 4 prompts of
           the paper's v5 Figure 5 and the 11 of web/build/compare_selection.json.)
  gallery  the fixed-rule gallery (web/assets/js/gallery.js): progressive SCA and AReA; three stops.
  review   the 96-prompt page (web/review/items.json): five methods; three stops; overviews at 1024 px and 640 px
           stop tiles (its panels are about 250 px wide, so a stop at 1:1 needs at most 640 native pixels).
Every image gets one native-pixel tile per stop (1024 px unless noted, quality 80). Whole-image tiles (zoomlib.cut_grid) would
put web/ far past its 250 MB budget: 0.5 to 2 MB per image, against about 0.2 MB for the stops. Overviews already
on the page are reused (compare and qual views, gallery views); missing ones are written at 1600 px. Outputs: web/assets/img/deep/<set>/<sid>/<method>/ (tiles, new views), web/assets/js/zoomsets.js
(window.SCA_ZOOMSETS = {compare, gallery}) and web/review/items.js (window.ITEMS). Files are rewritten only when
missing or older than their PNG; the regions are cached per grid on scratch (web_assets/zoom_cache/), never in
web/, because the cache names the source PNGs.

Run (repository root, CPU, a few minutes):
  nice -n 19 /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/build_zoom.py [compare gallery review] [--procs 10]
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zoomlib  # noqa: E402

WEB = os.path.dirname(HERE)
REPO = os.path.dirname(WEB)
DEEP = os.path.join(WEB, "assets", "img", "deep")
O = "/scratch/09032/saini_2/sca/out"
DIRS = {"prog": f"{O}/dupfix_prog/sca_v2_prog/4096x4096", "sca": f"{O}/dupfix/sca_v2/4096x4096",
        "area": f"{O}/main/area/4096x4096", "ultra": f"{O}/main/ultraimage/4096x4096",
        "dype": f"{O}/main/dype/4096x4096", "sega": f"{O}/main/sega_official/4096x4096",
        "full": f"{O}/main/full/4096x4096"}
LABEL = {"prog": "Progressive SCA", "sca": "Single-pass SCA", "area": "AReA", "hrdit": "HRDiT", "ultra": "UltraImage",
         "dype": "DyPE", "sega": "SEGA", "full": "Full attention"}
COMPARE_METHODS = ("prog", "sca", "area", "hrdit", "ultra", "dype", "sega", "full")
REVIEW_HRDIT = f"{O}/hrdit191eval/hrdit_dev_author/4096x4096"
CACHE = "/scratch/09032/saini_2/sca/web_assets/zoom_cache"


def rel(p, base=WEB):
    return os.path.relpath(p, base).replace(os.sep, "/")


def js_data(name):
    p = os.path.join(WEB, "assets", "js", name)
    t = open(p).read()
    return json.loads(t[t.index("{"):t.rindex("}") + 1])


def stem_png(d, stem):
    p = os.path.join(d, stem + ".png")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def compare_jobs():
    """The seeded draw of web/build/gallery_draw.json; overviews are written fresh (1600 px) for these prompts."""
    draw = json.load(open(os.path.join(HERE, "gallery_draw.json")))
    jobs = []
    for g in draw["picks"]:
        members = []
        for m in COMPARE_METHODS:
            png = g["paths"][m]
            if not os.path.exists(png):
                raise FileNotFoundError(png)
            members.append(dict(m=m, png=png, view=None, mode="regions"))
        jobs.append(dict(set="compare", sid=g["sid"], prompt=g["prompt"], members=members, base=WEB, k=3))
    return jobs


def gallery_jobs():
    G = js_data("gallery.js")
    jobs = []
    for it in G["items"]:
        full = ("minus" + it["sid"][1:]) if it["sid"].startswith("m") else it["sid"]
        members = []
        for m in ("prog", "area"):
            hits = sorted(glob.glob(f"{DIRS[m]}/a4k_eval_4096_{full}_*_seed0.png"))
            if len(hits) != 1:
                raise FileNotFoundError(f"gallery {m} {it['sid']}: {hits}")
            members.append(dict(m=m, png=hits[0], view=os.path.join(WEB, it["img"][m]["view"]), mode="regions"))
        jobs.append(dict(set="gallery", sid=it["sid"], index=it["index"], prompt=it["prompt"], members=members, base=WEB))
    return jobs


def review_jobs():
    R = os.path.join(WEB, "review")
    items = json.load(open(os.path.join(R, "items.json")))
    dirs = dict(DIRS, hrdit=REVIEW_HRDIT)
    jobs = []
    for it in items:
        members = []
        for m in ("prog", "sca", "area", "hrdit", "ultra"):
            members.append(dict(m=m, png=stem_png(dirs[m], it["id"]), view=None, view_px=1024, mode="regions", tile=640))
        jobs.append(dict(set="review", sid=f"{it['i']:02d}", i=it["i"], id=it["id"], prompt=it["prompt"], members=members,
                         base=R, outroot=os.path.join(R, "img", "z")))
    return jobs


def run(job):
    root = job.get("outroot") or os.path.join(DEEP, job["set"])
    gdir = os.path.join(root, job["sid"])
    os.makedirs(gdir, exist_ok=True)
    cache = os.path.join(CACHE, job["set"], job["sid"] + ".json")      # scratch: it names the source PNGs
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    pngs = [mm["png"] for mm in job["members"]]
    regs = None
    if os.path.exists(cache) and all(zoomlib.fresh(cache, p) for p in pngs):
        c = json.load(open(cache))
        if c.get("pngs") == pngs:
            regs = c["regions"]
    ims = {}
    if regs is None:
        maps, size = [], None
        for mm in job["members"]:
            im = Image.open(mm["png"]).convert("RGB")
            ims[mm["m"]] = im
            size = im.size
            maps.append(zoomlib.edge_map(im))
        regs = zoomlib.regions(maps, size, k=job.get("k", 3))
        json.dump({"pngs": pngs, "size": list(size), "regions": regs}, open(cache, "w"))
    panels = []
    for mm in job["members"]:
        m, png = mm["m"], mm["png"]
        odir = os.path.join(gdir, m)
        im = ims.get(m)

        def need():
            nonlocal im
            if im is None:
                im = Image.open(png).convert("RGB")
            return im
        if mm["view"]:
            view = mm["view"]
            with Image.open(view) as v:
                vs = list(v.size)
        else:
            view = os.path.join(odir, "v.webp")
            px = mm.get("view_px", 1600)
            vs = zoomlib.save_view(need() if not zoomlib.fresh(view, png) else None, view, long_side=px, src=png)
        with Image.open(png) as probe:
            W, H = probe.size
        p = {"m": m, "label": LABEL[m], "view": rel(view, job["base"]), "vs": vs}
        if m in ("prog", "sca"):
            p["ours"] = True
        if mm["mode"] == "grid":
            cols, rows = -(-W // zoomlib.TILE), -(-H // zoomlib.TILE)
            last = os.path.join(odir, f"{cols - 1}_{rows - 1}.webp")
            if not zoomlib.fresh(last, png) or not zoomlib.fresh(os.path.join(odir, "0_0.webp"), png):
                zoomlib.cut_grid(need(), odir, src=png)
            p["tiles"] = {"path": rel(os.path.join(odir, "{c}_{r}.webp"), job["base"]), "size": zoomlib.TILE, "cols": cols, "rows": rows}
        else:
            names = [os.path.join(odir, f"r{k}.webp") for k in range(len(regs))]
            if not all(zoomlib.fresh(n, png) for n in names):
                rt = zoomlib.cut_regions(need(), regs, odir, src=png, tile=mm.get("tile", zoomlib.TILE))
            else:   # boxes only, the files are fresh
                rt = zoomlib.cut_regions(Image.new("RGB", (W, H)), regs, odir, src=png, tile=mm.get("tile", zoomlib.TILE))
            p["rt"] = [[x, y, w, h, rel(os.path.join(odir, n), job["base"])] for x, y, w, h, n in rt]
        panels.append(p)
        im = None
        ims.pop(m, None)
    rec = {"sid": job["sid"], "prompt": job["prompt"], "size": [W, H], "regions": regs, "panels": panels}
    if job["set"] == "compare":         # prompt thumbnail for the selector row, from the first panel's overview
        th = os.path.join(gdir, "thumb.webp")
        v0 = os.path.join(job["base"], panels[0]["view"])
        if not zoomlib.fresh(th, v0):
            with Image.open(v0) as v:
                v.convert("RGB").resize((240, round(240 * v.size[1] / v.size[0])), Image.LANCZOS).save(th, "WEBP", quality=78, method=6)
        rec["thumb"] = rel(th, job["base"])
    for k in ("index", "i", "id"):
        if k in job:
            rec[k] = job[k]
    print(job["set"], job["sid"], regs, flush=True)
    return job["set"], rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*", default=["compare", "gallery", "review"])
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="first N grids per set (size test)")
    a = ap.parse_args()
    jobs = []
    for s in a.sets:
        js = {"compare": compare_jobs, "gallery": gallery_jobs, "review": review_jobs}[s]()
        jobs += js[: a.limit] if a.limit else js
    with Pool(a.procs) as pool:
        out = pool.map(run, jobs, chunksize=1)
    by = {}
    for s, rec in out:
        by.setdefault(s, []).append(rec)
    zs = os.path.join(WEB, "assets", "js", "zoomsets.js")
    old = js_data("zoomsets.js") if os.path.exists(zs) else {}
    for s in ("compare", "gallery"):
        if s in by:
            old[s] = by[s]
    with open(zs, "w") as f:
        f.write("// Generated by web/build/build_zoom.py; do not edit.\n")
        f.write("window.SCA_ZOOMSETS = " + json.dumps(old, separators=(",", ":")) + ";\n")
    if "review" in by:
        with open(os.path.join(WEB, "review", "items.js"), "w") as f:
            f.write("// Generated by web/build/build_zoom.py; do not edit.\n")
            f.write("window.ITEMS = " + json.dumps(sorted(by["review"], key=lambda r: r["i"]), separators=(",", ":")) + ";\n")
    for s in a.sets:
        d = os.path.join(WEB, "review", "img", "z") if s == "review" else os.path.join(DEEP, s)
        print(f"{s}: {len(by.get(s, []))} grids, tiles {zoomlib.dir_bytes(d) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
