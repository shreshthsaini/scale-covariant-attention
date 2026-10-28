"""Build every WebP image the page uses from the original PNGs on scratch.

Usage (sca env python, from the repo root):
    nice -n 19 python web/build/build_images.py compare|teaser|scale|failures|method|all

Inputs: generated PNGs under /scratch/09032/saini_2/sca/out (located with paper/figures/src/imgsel.py),
web/build/compare_selection.json (select_compare.py), web/build/page_images.json (teaser, scale strip and
failure picks with their crop windows), paper/figures/method.svg.
Outputs: web/assets/img/**.webp and web/assets/js/images.js (window.SCA_IMAGES: what the page shows, with
prompt text, labels, sizes and crop windows). Crops are cut from the original PNG at native pixels (1:1).
"""
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(WEB, ".."))
IMG = os.path.join(WEB, "assets", "img")
sys.path.insert(0, os.path.join(REPO, "paper", "figures", "src"))
import imgsel  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
FULL_PX = 1600      # long side of the slider images
FULL_Q = 80
CROP_Q = 88
DATA = {}


def rel(path):
    return os.path.relpath(path, WEB).replace(os.sep, "/")


def save_webp(im, path, q, lossless=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, "WEBP", quality=q, method=6, lossless=lossless)
    return rel(path)


def resized(im, long_side):
    w, h = im.size
    s = long_side / max(w, h)
    if s >= 1:
        return im
    return im.resize((round(w * s), round(h * s)), Image.LANCZOS)


def compare():
    sel = json.load(open(os.path.join(HERE, "compare_selection.json")))
    items = []
    for it in sel["items"]:
        x, y, s = it["box_xys"]
        rec = dict(sid=it["sid"], label=it["label"], prompt=it["prompt"], box=[x, y, s], size=[4096, 4096], img={})
        for m in sel["methods"]:
            im = Image.open(it["paths"][m]).convert("RGB")
            full = save_webp(resized(im, FULL_PX), os.path.join(IMG, "compare", f"{it['sid']}_{m}.webp"), FULL_Q)
            crop = save_webp(im.crop((x, y, x + s, y + s)),
                             os.path.join(IMG, "compare", f"{it['sid']}_{m}_crop.webp"), CROP_Q)
            rec["img"][m] = dict(full=full, crop=crop)
            if m == "sca":
                rec["thumb"] = save_webp(resized(im, 240), os.path.join(IMG, "compare", f"{it['sid']}_thumb.webp"), 78)
        items.append(rec)
        print("compare", it["sid"])
    DATA["compare"] = dict(items=items, crop_rule=sel["crop_rule"], random_draw=sel["random_draw"])


def picks():
    return json.load(open(os.path.join(HERE, "page_images.json")))


AUTO_CROP = 384
_REL = {}   # entry id -> relative centre of its first crop


def locate(entry):
    if entry.get("path"):
        hits = sorted(glob.glob(entry["path"]))
        if not hits:
            raise FileNotFoundError(entry["path"])
        return hits[0]
    return imgsel.find(entry["src"], entry["res"], entry["sid"])


def auto_box(im, side=AUTO_CROP, central=(0.2, 0.8)):
    """side x side window with the largest mean FIND_EDGES response inside the central region."""
    f = max(1, max(im.size) // 1024)
    e = np.asarray(im.convert("L").reduce(f).filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    k = max(1, side // f)
    ii = np.pad(e, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h, w = e.shape
    best, arg = -1.0, (0, 0)
    for y in range(int(central[0] * h), int(central[1] * h) - k + 1, max(1, k // 6)):
        for x in range(int(central[0] * w), int(central[1] * w) - k + 1, max(1, k // 6)):
            v = ii[y + k, x + k] - ii[y, x + k] - ii[y + k, x] + ii[y, x]
            if v > best:
                best, arg = v, (x, y)
    return [arg[0] * f, arg[1] * f, side]


def one_image(entry, outdir, long_side):
    """entry: see page_images.json -> record with WebP paths, size, prompt and crop windows."""
    path = locate(entry)
    im = Image.open(path).convert("RGB")
    W, H = im.size
    crops = entry.get("crops", [])
    if crops == "auto":
        crops = [auto_box(im)]
    elif isinstance(crops, str) and crops.startswith("rel_from:"):
        cx, cy = _REL[crops.split(":", 1)[1]]
        s = AUTO_CROP
        crops = [[min(max(0, round(cx * W - s / 2)), W - s), min(max(0, round(cy * H - s / 2)), H - s), s]]
    if crops:
        x, y, s = crops[0]
        _REL[entry["id"]] = ((x + s / 2) / W, (y + s / 2) / H)
    rec = {k: v for k, v in entry.items() if k not in ("path", "crops")}
    rec["crops"] = crops
    rec["size"] = [W, H]
    rec["prompt"] = entry.get("prompt") or imgsel.prompt(path)
    rec["full"] = save_webp(resized(im, long_side), os.path.join(IMG, outdir, f"{entry['id']}.webp"), FULL_Q)
    rec["crop_files"] = [save_webp(im.crop((x, y, x + s, y + s)),
                                   os.path.join(IMG, outdir, f"{entry['id']}_crop{k}.webp"), CROP_Q)
                         for k, (x, y, s) in enumerate(crops)]
    if entry.get("native"):
        rec["native_file"] = save_webp(im, os.path.join(IMG, outdir, f"{entry['id']}_native.webp"), 78)
    print(outdir, entry["id"], rec["size"], crops)
    return rec


def section(name, long_side):
    entries = picks()[name]
    later = [e for e in entries if isinstance(e.get("crops"), str) and e["crops"].startswith("rel_from:")]
    done = {e["id"]: one_image(e, name, e.get("long_side", long_side)) for e in entries if e not in later}
    done.update({e["id"]: one_image(e, name, e.get("long_side", long_side)) for e in later})
    DATA[name] = [done[e["id"]] for e in entries]


PDF_PY = os.environ.get("SCA_PDF_PY", "/work/09032/saini_2/vista/library/uv-envs/stationary-paper-qc/bin/python")  # any env with PyMuPDF
SCRATCH = "/scratch/09032/saini_2/sca/web_assets/build"


def pdf_to_webp(pdf, out_id, width_px=2400):
    """Rasterise page 1 of a paper figure PDF with PyMuPDF, then save it as WebP on a white ground."""
    os.makedirs(SCRATCH, exist_ok=True)
    png = os.path.join(SCRATCH, out_id + ".png")
    code = ("import fitz,sys; d=fitz.open(sys.argv[1]); p=d[0]; z=float(sys.argv[3])/p.rect.width; "
            "p.get_pixmap(matrix=fitz.Matrix(z,z), alpha=False).save(sys.argv[2])")
    subprocess.run([PDF_PY, "-c", code, pdf, png, str(width_px)], check=True)
    im = Image.open(png).convert("RGB")
    return save_webp(im, os.path.join(IMG, "method", out_id + ".webp"), 90), list(im.size)


def method():
    figs = os.path.join(REPO, "paper", "figures")
    DATA["method"] = {}
    for name in ("method", "method_zoom", "pi_heads"):
        path, size = pdf_to_webp(os.path.join(figs, name + ".pdf"), name)
        DATA["method"][name] = dict(file=path, size=size, source="paper/figures/%s.pdf" % name)
        print("method", name, size)


def write_js():
    path = os.path.join(WEB, "assets", "js", "images.js")
    old = {}
    if os.path.exists(path):
        txt = open(path).read()
        old = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    old.update(DATA)
    with open(path, "w") as fh:
        fh.write("// Generated by web/build/build_images.py. Do not edit by hand.\n")
        fh.write("window.SCA_IMAGES = " + json.dumps(old, indent=1) + ";\n")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = dict(compare=compare, teaser=lambda: section("teaser", 1600), scale=lambda: section("scale", 1100),
                failures=lambda: section("failures", 1100), method=method)
    for name, fn in jobs.items():
        if what in (name, "all"):
            fn()
    write_js()
