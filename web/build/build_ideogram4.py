"""Ideogram 4 at 8192^2 for the page's synced-zoom grids (assets/js/zoomgrid.js), plus its native 4096^2 references.

Per prompt one grid at 8192^2: single-pass SCA first, then full attention, SEGA, NTK and YaRN where they exist (NTK and
YaRN ran on three prompts). Every panel gets a 2400 px overview WebP (zoomlib.save_view, quality 80) and its whole
image in 1024 px native-pixel tiles (zoomlib.cut_grid: 8 x 8 at 8192^2, quality 80, <col>_<row>.webp), loaded by the
page only past the overview's resolution. Tour stops: zoomlib.regions over the grid's mean-normalised edge maps (k 3,
span 1024 native px), shared by every panel. Ideogram 4 at 4096^2 is listed separately as the model's native
reference (2400 px overview, 4 x 4 tiles), never as an upscale comparison.
Inputs: /scratch/09032/saini_2/sca/out/ideogram4/ideogram4_<method>/<HxW>/ (PNG + sidecar). Outputs:
web/assets/img/ideogram4/<prompt>/<method>_<size>/{view.webp,t/<c>_<r>.webp}, web/build/ideogram4_manifest.json and
its mirror web/assets/js/ideogram4.js (window.SCA_IDEOGRAM4, one zoomgrid spec per prompt in "grids").
Re-running skips tiles newer than their PNG. Run with the sca env python from the repo root:
    python web/build/build_ideogram4.py [--budget-mb 600] [--dry]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from PIL import Image

HERE = Path(__file__).resolve().parent
WEB = HERE.parent
sys.path.insert(0, str(HERE))
import zoomlib  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = Path("/scratch/09032/saini_2/sca/out/ideogram4")
PROMPTS = HERE / "ideogram4_prompts.jsonl"
MANIFEST = HERE / "ideogram4_manifest.json"
JS = WEB / "assets" / "js" / "ideogram4.js"
IMG = WEB / "assets" / "img" / "ideogram4"
ORDER = ("sca", "full", "sega", "ntk", "yarn")
LABELS = {"sca": "SCA (ours)", "full": "Full attention", "sega": "SEGA", "ntk": "NTK", "yarn": "YaRN"}
OVERVIEW = 2400


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def find(out, m, res, pid):
    for meta in sorted((out / f"ideogram4_{m}" / res).glob(f"{pid}_*seed0.json")):
        r = json.loads(meta.read_text())
        if r.get("complete") and r["config"]["prompt_id"] == pid and r["config"]["method"] == m:
            return meta.with_suffix(".png"), meta, r
    return None


def gpu_shared(out, pid, m):
    tag = {"sca": "sca8k", "full": "full8k", "sega": "sega8k", "ntk": "ntk8k", "yarn": "yarn8k"}[m]
    for f in (out / "_gpu").glob(f"*{tag}_{pid}.apps"):
        if any(x and not x.startswith(("start ", "end ", "cmd ")) for x in f.read_text().splitlines()):
            return True
    return False


def panel(png, meta, rec, m, pid, size_tag, regs, dry, web, img_root):
    rel = f"{pid}/{m}_{size_tag}"
    d = img_root / rel
    view = d / "view.webp"
    with Image.open(png) as im:
        W, H = im.size
        if dry:
            vs, tiles = [OVERVIEW, round(OVERVIEW * H / W)], {"size": zoomlib.TILE, "cols": -(-W // zoomlib.TILE),
                                                                "rows": -(-H // zoomlib.TILE)}
        else:
            im = im.convert("RGB")
            vs = zoomlib.save_view(im, str(view), long_side=OVERVIEW, q=80, src=str(png))
            tiles = zoomlib.cut_grid(im, str(d / "t"), src=str(png))
    base = os.path.relpath(img_root, web).replace(os.sep, "/")
    tiles = dict(tiles, path=f"{base}/{rel}/t/{{c}}_{{r}}.webp")
    return {"method": m, "label": LABELS[m], "ours": m == "sca", "view": f"{base}/{rel}/view.webp", "vs": vs,
            "tiles": tiles, "png": str(png), "sidecar": str(meta), "sha256": sha256(png),
            "seconds": round(rec["seconds"], 1), "peak_memory_gb": round((rec.get("peak_memory_bytes") or 0) / 1e9, 1),
            "gpu": rec.get("gpu"), "steps": rec["config"]["steps"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", type=Path, default=OUT)
    p.add_argument("--prompts", type=Path, default=PROMPTS)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--js", type=Path, default=JS)
    p.add_argument("--img", type=Path, default=IMG)
    p.add_argument("--web", type=Path, default=WEB)
    p.add_argument("--budget-mb", type=float, default=600.0)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args(argv)
    prompts = [json.loads(line) for line in open(a.prompts) if line.strip()]
    grids, native, missing = [], [], []
    for r in prompts:
        pid = r["id"]
        found = {m: find(a.outputs, m, "8192x8192", pid) for m in ORDER}
        have = [m for m in ORDER if found[m]]
        missing += [f"{pid}/{m}" for m in ORDER if not found[m] and m in ("sca", "full", "sega")]
        if "sca" in have:
            maps = []
            for m in have:
                with Image.open(found[m][0]) as im:
                    maps.append(zoomlib.edge_map(im.convert("RGB"), f=8))
            with Image.open(found["sca"][0]) as im:
                size = im.size
            regs = zoomlib.regions(maps, size, k=3, span=1024, f=8)
            panels = []
            for m in have:
                q = panel(*found[m], m, pid, "8192", regs, a.dry, a.web, a.img)
                q["gpu_shared"] = gpu_shared(a.outputs, pid, m)
                panels.append(q)
            grids.append({"prompt_id": pid, "title": r["title"], "category": r["category"], "prompt": r["prompt"],
                          "seed": 0, "size": list(size), "regions": regs, "panels": panels})
        nat = find(a.outputs, "full", "4096x4096", pid)
        if nat:
            q = panel(*nat, "full", pid, "4096", None, a.dry, a.web, a.img)
            q.update(label="Ideogram 4 at 4096² (native)", ours=False)
            native.append({"prompt_id": pid, "title": r["title"], "prompt": r["prompt"], "seed": 0,
                           "size": [4096, 4096], **q})
    used = zoomlib.dir_bytes(str(a.img))
    man = {"schema_version": 1,
           "description": "Ideogram 4 (ideogram-ai/ideogram-4-nf4-diffusers) at 8192x8192, twice its native 4096x4096: "
                          "single-pass SCA against full attention, SEGA, NTK and YaRN at the same size; Ideogram 4 at "
                          "4096x4096 as the native reference. Seed 0, the 10 showcase prompts in the campaign's order "
                          "(web/build/ideogram4_prompts.jsonl); NTK and YaRN on the first three.",
           "model": "ideogram-ai/ideogram-4-nf4-diffusers", "steps": 48, "seed": 0,
           "overview_px": OVERVIEW, "tile_px": zoomlib.TILE, "webp_quality": zoomlib.QUALITY,
           "regions_rule": "zoomlib.regions over the grid's edge maps, k 3, span 1024 native px",
           "grids": grids, "native_4096": native, "missing": missing, "web_bytes": used}
    if not a.dry:
        a.manifest.write_text(json.dumps(man, indent=1) + "\n")
        a.js.parent.mkdir(parents=True, exist_ok=True)
        a.js.write_text("// Generated by web/build/build_ideogram4.py; do not edit by hand.\nwindow.SCA_IDEOGRAM4 = "
                        + json.dumps({k: man[k] for k in ("grids", "native_4096", "seed", "steps", "model")}) + ";\n")
    print(f"grids {len(grids)}, native {len(native)}, missing {len(missing)}, web bytes {used / 1e6:.1f} MB", flush=True)
    if used > a.budget_mb * 1e6:
        raise SystemExit(f"over the {a.budget_mb} MB budget: {used / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
