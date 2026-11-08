#!/usr/bin/env python3
"""Showcase manifest, Lanczos references and web WebP files (HANDOVER Section 32). CPU only; run with the sca env
python under nice on a compute node.

Reads the showcase outputs /scratch/09032/saini_2/sca/out/showcase/<model>_<method>/<HxW>/ (PNG + JSON sidecar per
prompt, written by scripts/sample.py, sample_progressive.py, sample_flux2.py, sample_qwen.py and run_area.py) and the
prompts web/build/showcase_prompts.jsonl, and writes:

  out/showcase/<model>_lanczos/<HxW>/<stem>.png   each model's native image upscaled to the showcase size with PIL
                                                  LANCZOS (the reference baseline), once per (model, size, prompt)
  web/build/showcase_manifest.json                one entry per image: prompt, model, method, size, seed, seconds (from
                                                  the sidecar), whether other processes shared the GPU during the task
                                                  (out/showcase/_gpu/<task>.apps), scratch paths, SHA-256, crop boxes
  web/assets/img/showcase/<group>/<prompt>_<method>[_cK].webp
                                                  for the selected groups: an overview at 2400 px on the long side and
                                                  four 1:1 native-pixel 768 px crops per image

Crop boxes: per (group, prompt), the four non-overlapping 768 px windows with the most gradient energy in the group's SCA
image (mean |dx| + |dy| of the luminance on a 2x box-downsampled copy, windows on a 64 px stride), ordered by energy; the
comparison images of the group get the same boxes, so crops line up.

Groups: flux1_4096 (progressive SCA vs AReA vs FLUX.1-dev 1024^2 + Lanczos), flux1_5120 and flux1_aspect (single-pass
SCA vs AReA vs Lanczos), flux2_4096 and flux2_5120 (SCA vs full attention vs 1024^2 + Lanczos), qwen_4096 and qwen_5312
(SCA vs full attention vs 1328^2 + Lanczos).

Usage: build_showcase.py [--web GROUP:PROMPTS ...] [--budget-mb 60] [--dry]
  --web flux1_4096:all --web qwen_5312:sc_p01_skeleton_watch,sc_l01_fjord_village   (which images go to web/)
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = Path("/scratch/09032/saini_2/sca/out/showcase")
PROMPTS = HERE / "showcase_prompts.jsonl"
MANIFEST = HERE / "showcase_manifest.json"
WEB = REPO / "web/assets/img/showcase"
OVERVIEW, CROP, NCROP, STRIDE = 2400, 768, 4, 64
MODELS = {"flux1": "FLUX.1-dev", "flux2": "FLUX.2-dev", "qwen": "Qwen-Image"}
NAMES = {"sca_v2_prog": "progressive SCA", "sca_v2": "SCA (single pass)", "flux2_sca_v2": "SCA (single pass)",
         "qwen_sca": "SCA (single pass)", "area": "AReA", "flux2_full": "full attention", "qwen_full": "full attention",
         "lanczos": "native + Lanczos"}


def aspect(row, land, port):
    return land if row["orient"] == "landscape" else port


# group: (model, size(row), SCA dir, [(method, dir)], native dir(row) for the Lanczos reference)
GROUPS = {
    "flux1_4096": ("flux1", lambda r: "4096x4096", "flux1_sca_v2_prog", [("area", "flux1_area")], lambda r: "flux1_full/1024x1024"),
    "flux1_5120": ("flux1", lambda r: "5120x5120", "flux1_sca_v2", [("area", "flux1_area")], lambda r: "flux1_full/1024x1024"),
    "flux1_aspect": ("flux1", lambda r: aspect(r, "4096x6144", "6144x4096"), "flux1_sca_v2", [("area", "flux1_area")],
                     lambda r: "flux1_full/" + aspect(r, "1024x1536", "1536x1024")),
    "flux2_4096": ("flux2", lambda r: "4096x4096", "flux2_sca_v2", [("flux2_full", "flux2_full")], lambda r: "flux2_full/1024x1024"),
    "flux2_5120": ("flux2", lambda r: "5120x5120", "flux2_sca_v2", [("flux2_full", "flux2_full")], lambda r: "flux2_full/1024x1024"),
    "qwen_4096": ("qwen", lambda r: "4096x4096", "qwen_sca", [("qwen_full", "qwen_full")], lambda r: "qwen_full/1328x1328"),
    "qwen_5312": ("qwen", lambda r: "5312x5312", "qwen_sca", [("qwen_full", "qwen_full")], lambda r: "qwen_full/1328x1328"),
}


def stem(prompt_id, seed=0):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", prompt_id)[:80].strip(".") or "prompt"
    return f"{safe}_{hashlib.sha256(prompt_id.encode()).hexdigest()[:10]}_seed{seed}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(png):
    """Sidecar of a complete image, else None."""
    meta = png.with_suffix(".json")
    if not (png.exists() and meta.exists()):
        return None
    try:
        rec = json.loads(meta.read_text())
    except ValueError:
        return None
    return rec if rec.get("complete") else None


def task_logs():
    """{(out dir, prompt id): shared} from out/showcase/_gpu/<task>.apps. Each attempt of a task appends a start snapshot
    of its GPU's compute processes, its command line (--prompts, --start, --end, --out) and an end snapshot; our sampler
    is not running at either snapshot, so any process listed there belongs to someone else. An image counts as shared
    when any attempt of its task saw another process."""
    shared = {}
    for f in sorted((OUT / "_gpu").glob("*.apps")):
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        cmds = [l[4:] for l in lines if l.startswith("cmd ")]
        apps = [l for l in lines if not l.startswith(("start ", "end ", "cmd "))]
        for c in cmds[-1:]:
            arg = lambda k, d=None: (re.search(rf"--{k} (\S+)", c).group(1) if re.search(rf"--{k} (\S+)", c) else d)
            prompts = [json.loads(l)["id"] for l in open(arg("prompts")) if l.strip()] if arg("prompts") else []
            a, b = int(arg("start", 0)), int(arg("end", len(prompts)))
            for pid in prompts[a:b]:
                key = (str(Path(arg("out")).resolve()), pid)
                shared[key] = shared.get(key, False) or bool(apps)
    return shared


def gradient_energy(img):
    """[h/2, w/2] float32 mean |dx| + |dy| of the luminance after a 2x box downsample."""
    a = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))
    g = np.zeros_like(a)
    g[:, :-1] += np.abs(np.diff(a, axis=1))
    g[:-1, :] += np.abs(np.diff(a, axis=0))
    return g


def crop_boxes(img, n=NCROP, side=CROP, stride=STRIDE):
    """n non-overlapping side x side windows (x, y, side) with the most gradient energy, in full-resolution pixels."""
    g = gradient_energy(img)
    s, st = side // 2, stride // 2
    ii = np.pad(g, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ys = np.arange(0, g.shape[0] - s + 1, st)
    xs = np.arange(0, g.shape[1] - s + 1, st)
    E = ii[ys[:, None] + s, xs[None] + s] - ii[ys[:, None], xs[None] + s] - ii[ys[:, None] + s, xs[None]] + ii[ys[:, None], xs[None]]
    boxes = []
    for k in np.argsort(E, axis=None)[::-1]:
        y, x = int(ys[k // len(xs)]) * 2, int(xs[k % len(xs)]) * 2
        if all(abs(x - bx) >= side or abs(y - by) >= side for bx, by, _ in boxes):
            boxes.append((x, y, side))
            if len(boxes) == n:
                break
    return boxes


def lanczos(native_png, size, out_png):
    """Native image upscaled to size (HxW) with PIL LANCZOS, saved once."""
    if out_png.exists():
        return out_png
    h, w = (int(v) for v in size.split("x"))
    with Image.open(native_png) as im:
        up = im.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_png.with_name(out_png.name + f".tmp.{os.getpid()}")
    up.save(tmp, format="PNG")
    os.replace(tmp, out_png)
    return out_png


def webp(img, path, quality):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    img.save(tmp, format="WEBP", quality=quality, method=6)
    os.replace(tmp, path)
    return path.stat().st_size


def export(entry, img, group, quality_overview, quality_crop):
    """Overview + crops of one image into web/assets/img/showcase/<group>/; returns (paths, bytes)."""
    pid, m = entry["prompt_id"], entry["method"]
    w, h = img.size
    f = OVERVIEW / max(w, h)
    ov = img.resize((round(w * f), round(h * f)), Image.Resampling.LANCZOS) if f < 1 else img
    base = WEB / group / f"{pid}_{m}"
    total = webp(ov, base.with_name(base.name + ".webp"), quality_overview)
    crops = []
    for k, (x, y, s) in enumerate(entry["crops"]):
        p = base.with_name(f"{base.name}_c{k}.webp")
        total += webp(img.crop((x, y, x + s, y + s)), p, quality_crop)
        crops.append(str(p.relative_to(REPO / "web")))
    return {"overview": str(base.with_name(base.name + ".webp").relative_to(REPO / "web")), "crops": crops}, total


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--web", action="append", default=[], metavar="GROUP:PROMPTS",
                   help="export WebP for these prompt ids (comma list or 'all') of GROUP")
    p.add_argument("--budget-mb", type=float, default=60.0)
    p.add_argument("--quality-overview", type=int, default=80)
    p.add_argument("--quality-crop", type=int, default=88)
    p.add_argument("--no-lanczos", action="store_true", help="skip writing Lanczos references (manifest only lists them)")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args(argv)
    rows = [json.loads(l) for l in PROMPTS.read_text().splitlines() if l.strip()]
    shared = task_logs()
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    old_boxes = {(e["group"], e["prompt_id"]): e["crops"] for e in old.get("images", []) if e.get("crops")}
    want = {}
    for spec in a.web:
        g, ids = spec.split(":", 1)
        if g not in GROUPS:
            raise SystemExit(f"unknown group {g}")
        want[g] = {r["id"] for r in rows} if ids == "all" else set(ids.split(","))
    images, web_bytes, missing = [], 0, []
    for g, (model, size_of, sca_dir, comps, native_of) in GROUPS.items():
        for row in rows:
            size = size_of(row)
            st = stem(row["id"])
            sca_png = OUT / sca_dir / size / f"{st}.png"
            rec = record(sca_png)
            if rec is None:
                missing.append(f"{g}/{row['id']}/sca")
                continue
            native_png = OUT / native_of(row) / f"{st}.png"
            members = [(rec["config"]["method"], sca_png, rec)]
            for method, d in comps:
                cp = OUT / d / size / f"{st}.png"
                crec = record(cp)
                if crec is None:
                    missing.append(f"{g}/{row['id']}/{method}")
                else:
                    members.append((method, cp, crec))
            nrec = record(native_png)
            if nrec is None:
                missing.append(f"{g}/{row['id']}/lanczos")
            else:
                lz = OUT / f"{model}_lanczos" / size / f"{st}.png"
                if not a.dry and not a.no_lanczos:
                    lanczos(native_png, size, lz)
                members.append(("lanczos", lz, nrec))
            boxes = old_boxes.get((g, row["id"]))
            img_sca = None
            if boxes is None or g in want and row["id"] in want[g]:
                with Image.open(sca_png) as im:
                    img_sca = im.convert("RGB")
                boxes = boxes or crop_boxes(img_sca)
            for method, png, r in members:
                cfg = r["config"]
                e = {"group": g, "prompt_id": row["id"], "title": row["title"], "category": row["category"],
                     "orient": row["orient"], "prompt": row["prompt"], "model": MODELS[model], "method": method,
                     "method_name": NAMES.get(method, method), "size": size, "seed": int(cfg.get("seed", 0)),
                     "png": str(png), "crops": [list(b) for b in boxes]}
                if method == "lanczos":
                    e.update(native_png=str(native_png), native_size=cfg["res"], native_seconds=r.get("seconds"),
                             native_method=cfg["method"], resample="PIL LANCZOS")
                else:
                    e.update(seconds=r.get("seconds"), sidecar=str(png.with_suffix(".json")), sha256=r.get("png_sha256"),
                             peak_memory_gb=None if r.get("peak_memory_bytes") is None else round(r["peak_memory_bytes"] / 1e9, 1),
                             config_method=cfg["method"], steps=cfg.get("steps"), stage_seconds=r.get("stage_seconds"))
                    e["gpu_shared"] = shared.get((str(png.parent.resolve()), row["id"]))   # None: no task log
                if g in want and row["id"] in want[g] and not a.dry:
                    if method == sca_dir or png == sca_png:
                        img = img_sca
                    else:
                        with Image.open(png) as im:
                            img = im.convert("RGB")
                    e["web"], b = export(e, img, g, a.quality_overview, a.quality_crop)
                    web_bytes += b
                images.append(e)
    out = {"schema_version": 1, "description": "GPU showcase (HANDOVER Section 32): one entry per image; seconds are the "
           "samplers' per-image wall times (denoising and VAE decode, model loading excluded) and include slowdowns from "
           "other projects' processes sharing the GPU; clean batch-1 timings are in timings", "seed": 0,
           "crop_rule": f"{NCROP} non-overlapping {CROP} px windows with the most gradient energy in the group's SCA image",
           "overview_px": OVERVIEW, "images": images, "missing": missing, "web_bytes": web_bytes,
           "timings": old.get("timings", {})}
    if a.dry:
        print(json.dumps({"images": len(images), "missing": len(missing)}, indent=1))
        return
    tmp = MANIFEST.with_name(MANIFEST.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    os.replace(tmp, MANIFEST)
    total = sum(f.stat().st_size for f in WEB.rglob("*.webp")) if WEB.exists() else 0
    print(f"manifest: {len(images)} images, {len(missing)} missing; web/assets/img/showcase {total / 1e6:.1f} MB")
    if total > a.budget_mb * 1e6:
        print(f"WARNING over the {a.budget_mb} MB budget", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
