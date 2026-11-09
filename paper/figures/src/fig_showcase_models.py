"""New-model showcase (fig:showcase_models): single-pass SCA on FLUX.2-dev and Qwen-Image at 4096^2 and above.

Input: web/build/showcase_manifest.json. Two shapes are read:
  * the showcase builder's (web/build/build_showcase.py, HANDOVER Section 32): {"images": [{"group", "prompt_id",
    "prompt", "model", "method", "method_name", "size": "WxH", "seed", "png", "crops": [[x, y, s], ...]}, ...]};
    for each of FLUX.2-dev and Qwen-Image the SCA images at the model's largest size are taken for the first
    PER_MODEL prompts in the order of web/build/showcase_prompts.jsonl (the product prompts come first), and each
    image's window is the builder's first crop box (its highest gradient energy), so no image is chosen by eye;
  * a generic list {"items": [{"model", "file", "prompt", "seed", optional "method", "prompt_set",
    "crop_center": [x, y] fractions from the top left, "status"}]}, windows placed automatically otherwise.
Output: paper/figures/showcase_models/ (a downsampled image and a native 1:1 crop per item, provenance.json) and
paper/figures/snippets/showcase_models.tex. The paper inputs the snippet only when main.tex sets \\showcasetrue.
With --model FLUX.2-dev or --model Qwen-Image the script writes one model only, so that each model's slot can be
switched on as its images land: paper/figures/showcase_<key>/, snippets/showcase_<key>.tex (label
fig:showcase_<key>) and snippets/showcase_<key>_text.tex, which defines \\showcase<Key>Sizes (the sizes shown, for
the sentence of Section 4.4); key flux2 or qwen, Key FluxTwo or Qwen. main.tex inputs them when it sets
\\resfluxtwotrue or \\resqwentrue (Appendix S, "Newer models").
Qwen-Image's figure is a comparison (HANDOVER Section 38): each row is one prompt, and full attention at the target
size and AReA's released Qwen-Image code (scripts/run_area_qwen.py) stand left of SCA, each image with its crop of the
same window, whenever the manifest has those images for every row; the caption adds their steps and, once
results/qwen_5312_vs_area exists, the median seconds per image.
The caption states models, method, sizes, prompt set and seed from the manifest, nothing about selection.
Run with the sca env python:  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_showcase_models.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = ROOT / "web" / "build" / "showcase_manifest.json"
PROMPTS = ROOT / "web" / "build" / "showcase_prompts.jsonl"
MODELS = ("FLUX.2-dev", "Qwen-Image")
THUMB = 1400    # px, long side of the shipped full image
PER_ROW = 2     # items per row; each item is an image and its crop
PER_MODEL = 3   # builder schema: prompts per model
# comparison columns left of SCA, per model: (manifest method, header)
COMPARE = {"Qwen-Image": (("qwen_full", "Full attention"), ("qwen_area", "AReA"))}
QA_TIMING = ROOT / "results" / "qwen_5312_vs_area" / "timing_summary.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(file: str) -> Path:
    p = Path(file)
    for cand in ([p] if p.is_absolute() else [ROOT / p, ROOT / "web" / p, ROOT / "web" / "build" / p]):
        if cand.exists():
            return cand
    raise FileNotFoundError(file)


def auto_window(img: Image.Image, side: int) -> tuple[int, int]:
    """Top-left corner (px) of the side x side window with the highest mean gradient magnitude, 5% margins."""
    W, H = img.size
    f = 512 / max(W, H)
    small = img.convert("L").resize((max(1, round(W * f)), max(1, round(H * f))), Image.BICUBIC)
    g = np.asarray(small.filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    k = max(1, round(side * f))
    c = np.pad(g, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    mh, mw = round(0.05 * g.shape[0]), round(0.05 * g.shape[1])
    best, arg = -1.0, (0, 0)
    for y in range(mh, g.shape[0] - mh - k + 1, 2):
        for x in range(mw, g.shape[1] - mw - k + 1, 2):
            s = c[y + k, x + k] - c[y, x + k] - c[y + k, x] + c[y, x]
            if s > best:
                best, arg = s, (x, y)
    return round(arg[0] / f), round(arg[1] / f)


def builder_items(man: dict, prompts: Path) -> list[dict]:
    order = [json.loads(line)["id"] for line in prompts.read_text().splitlines() if line.strip()]
    best: dict[tuple[str, str], tuple[int, dict]] = {}
    for e in man["images"]:
        if e.get("model") not in MODELS or "SCA" not in str(e.get("method_name", "")):
            continue
        w, h = (int(v) for v in str(e["size"]).lower().split("x"))
        key = (e["model"], e["prompt_id"])
        if key not in best or w * h > best[key][0]:
            best[key] = (w * h, e)
    items = []
    for model in MODELS:
        ids = sorted({pid for (m, pid) in best if m == model}, key=lambda p: order.index(p) if p in order else 10**9)
        for pid in ids[:PER_MODEL]:
            e = best[(model, pid)][1]
            items.append(dict(model=model, file=e["png"], prompt=e["prompt"], seed=e.get("seed", 0),
                              method=e.get("method_name"), prompt_id=pid, group=e.get("group"), steps=e.get("steps"),
                              prompt_set="prompts %s of the %d written for these samples" % (
                                  ", ".join(str(order.index(p) + 1) for p in ids[:PER_MODEL]), len(order)),
                              crop_box=(e.get("crops") or [None])[0]))
    return items


KEYS = {"FLUX.2-dev": ("flux2", "FluxTwo"), "Qwen-Image": ("qwen", "Qwen")}


def generic_items(man: dict) -> list[dict]:
    items = [it for it in man.get("items", []) if str(it.get("status", "ok")).lower() not in ("missing", "failed")]
    for it in items:
        miss = [k for k in ("model", "file", "prompt", "seed") if it.get(k) in (None, "")]
        if miss:
            sys.exit(f"manifest item lacks {miss}: {it}")
    return items[:2 * PER_MODEL]


def tex(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def size_tex(sizes: list[str]) -> str:
    out = []
    for s in sorted(set(sizes), key=lambda s: int(s.split("x")[0]) * int(s.split("x")[1])):
        w, h = s.split("x")
        out.append(f"${w}^2$" if w == h else f"${h}\\times{w}$")
    return " and ".join(out)


def compare_figure(args, man, items, key, out_dir, snippet) -> bool:
    """Rows = prompts; per row, each comparison method of COMPARE[model] and then SCA, each a downsampled image with
    its window outlined and the native 1:1 crop of that window (the manifest's first crop box, shared by the group).
    Returns False when no comparison method has an image for every row (the SCA-only layout is used then)."""
    by = {(e["group"], e["prompt_id"], e["method"]): e for e in man["images"]}
    cols = [(m, h) for m, h in COMPARE[args.model] if all((it["group"], it["prompt_id"], m) in by for it in items)]
    if not cols:
        return False
    rows, recs, steps = [], [], {}
    for n, it in enumerate(items):
        x0, y0, side = (int(v) for v in it["crop_box"])
        cells = []
        for m, _ in cols + [("sca", None)]:
            e = by[(it["group"], it["prompt_id"], m)] if m != "sca" else None
            src = resolve(e["png"] if e else it["file"])
            img = Image.open(src).convert("RGB")
            W, H = img.size
            steps[m] = e.get("steps") if e else it.get("steps")
            tag = f"{n:02d}_{m}"
            full, crop = out_dir / f"{tag}.jpg", out_dir / f"{tag}_crop.jpg"
            scale = THUMB / max(W, H)
            img.resize((round(W * scale), round(H * scale)), Image.LANCZOS).save(full, quality=92, subsampling=0)
            img.crop((x0, y0, x0 + side, y0 + side)).save(crop, quality=95, subsampling=0)
            a, b = (x0 / W, 1 - (y0 + side) / H), ((x0 + side) / W, 1 - y0 / H)
            cells.append((full, crop, a, b))
            recs.append(dict(model=it["model"], prompt_id=it["prompt_id"], prompt=it["prompt"], seed=it["seed"],
                             method=m if e else "qwen_sca", resolution=f"{W}x{H}", source=str(src),
                             source_sha256=sha256(src), window_px=[x0, y0, side],
                             window_rule="manifest crop box (the builder's highest gradient energy in the SCA image)",
                             outputs={full.name: sha256(full), crop.name: sha256(crop)},
                             prompt_set=it.get("prompt_set")))
        rows.append(cells)
    k = len(cols) + 1
    gut = k * 1 + (k - 1) * 2
    head = [r"\scaHead{\dimexpr2\scaT+1pt\relax}{%s}" % h for _, h in cols]
    head.append(r"\scaHead{\dimexpr2\scaT+1pt\relax}{\scaOurs{SCA (ours)}}")
    lines = ["% Generated by paper/figures/src/fig_showcase_models.py from web/build/showcase_manifest.json;",
             "%% do not edit by hand. Input only when main.tex sets \\res%strue." % KEYS[args.model][1].lower(),
             r"\input{figures/snippets/_scagrid}%", r"\begin{figure}[!htbp]", r"\begingroup\scaGridStyle",
             r"\setlength{\scaT}{\dimexpr(\linewidth-%dpt)*1000/%d\relax}%%" % (gut, 2000 * k),
             r"\makebox[\linewidth][c]{" + r"\hspace{2pt}".join(head) + r"}\par\vskip2pt"]
    for cells in rows:
        lines.append(r"\makebox[\linewidth][c]{" + r"\hspace{2pt}".join(
            r"\scaFull{\scaT}{\scaT}{figures/showcase_%s/%s}{(%.5f,%.5f)}{(%.5f,%.5f)}\hspace{1pt}"
            r"\scaCrop{\scaT}{\scaT}{figures/showcase_%s/%s}" % (key, f.name, a[0], a[1], b[0], b[1], key, c.name)
            for f, c, a, b in cells) + r"}\par\vskip2pt")
    sizes = size_tex([r["resolution"] for r in recs])
    sets = sorted({r["prompt_set"] for r in recs if r.get("prompt_set")})
    seeds = sorted({int(r["seed"]) for r in recs})
    names = {"qwen_full": ("full attention at the target size", ""),
             "qwen_area": ("AReA's released Qwen-Image code at its own settings", ", its quality suffix appended to the prompt"),
             "sca": (r"single-pass \sca{}", "")}
    parts = []
    for m, h in cols + [("sca", None)]:
        name, extra = names.get(m, (tex(str(h)), ""))
        parts.append(f"{name} ({steps[m]} steps{extra})" if steps.get(m) else name)
    timing = ""
    if QA_TIMING.exists():
        tr = {r["method"]: r for r in csv.DictReader(open(QA_TIMING)) if r["res"] == "5312x5312"}
        bits = [f"{lab} {round(float(tr[m]['median_seconds_clean']))}~s" for m, lab in
                (("qwen_sca", r"\sca{}"), ("qwen_area", "AReA"), ("qwen_full", "full attention"))
                if m in tr and tr[m].get("median_seconds_clean")]
        if bits:
            timing = " Median seconds per image on one GH200 with no other process on it: " + ", ".join(bits) + "."
    caption = (r"\textbf{\sca{} on Qwen-Image beside AReA.} Qwen-Image at %s; %s, seed%s%s, the same initial noise "
               r"in every column. Left to right: %s. Each image is followed by a 1:1 crop of its outlined window at "
               r"native pixels, the same window in every column.%s"
               % (sizes, tex("; ".join(sets)) if sets else "", "s~" if len(seeds) > 1 else "~",
                  ", ".join(map(str, seeds)), "; ".join(parts), timing))
    lines += [r"\endgroup", r"\caption{%s}" % caption, r"\label{fig:showcase_%s}" % key, r"\end{figure}", ""]
    snippet.write_text("\n".join(lines))
    text = args.paper / "figures" / "snippets" / f"showcase_{key}_text.tex"
    text.write_text("%% Generated by paper/figures/src/fig_showcase_models.py --model %s; do not edit by hand.\n"
                    "\\providecommand{\\showcase%sSizes}{%s}\n" % (args.model, KEYS[args.model][1], sizes))
    prov = dict(figure=f"fig:showcase_{key}", script="paper/figures/src/fig_showcase_models.py",
                manifest=str(args.manifest), manifest_sha256=sha256(args.manifest),
                columns=[m for m, _ in cols] + ["qwen_sca"], items=recs, snippet_sha256=sha256(snippet),
                timing_source=str(QA_TIMING) if timing else None)
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"wrote {snippet}: {len(rows)} rows x {k} methods; caption: {caption}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--prompts", type=Path, default=PROMPTS)
    ap.add_argument("--paper", type=Path, default=ROOT / "paper", help="paper directory to write into (tests)")
    ap.add_argument("--model", choices=sorted(KEYS), help="write one model's slot (showcase_<key>) instead of both")
    args = ap.parse_args()
    key = KEYS[args.model][0] if args.model else "models"
    out_dir = args.paper / "figures" / f"showcase_{key}"
    snippet = args.paper / "figures" / "snippets" / f"showcase_{key}.tex"
    if not args.manifest.exists():
        sys.exit(f"no manifest at {args.manifest}; leave \\showcasefalse in main.tex")
    man = json.loads(args.manifest.read_text())
    items = builder_items(man, args.prompts) if "images" in man else generic_items(man)
    if args.model:
        items = [it for it in items if it["model"] == args.model]
    if not items:
        sys.exit(f"manifest has no usable SCA images of {args.model or 'FLUX.2-dev or Qwen-Image'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.model in COMPARE and "images" in man and compare_figure(args, man, items, key, out_dir, snippet):
        return
    rows, recs = [], []
    for n, it in enumerate(items):
        src = resolve(it["file"])
        img = Image.open(src).convert("RGB")
        W, H = img.size
        if it.get("crop_box"):
            x0, y0, side = (int(v) for v in it["crop_box"])
            rule = "manifest crop box (the builder's highest gradient energy)"
        else:
            side = min(W, H) // 8
            if it.get("crop_center"):
                cx, cy = it["crop_center"]
                x0 = int(round(min(max(cx * W - side / 2, 0), W - side)))
                y0 = int(round(min(max(cy * H - side / 2, 0), H - side)))
                rule = "manifest crop_center"
            else:
                x0, y0 = auto_window(img, side)
                rule = "highest mean gradient magnitude, 1/8 side, 5% margins"
        tag = f"{n:02d}_{''.join(ch for ch in it['model'] if ch.isalnum()).lower()}"
        full, crop = out_dir / f"{tag}.jpg", out_dir / f"{tag}_crop.jpg"
        scale = THUMB / max(W, H)
        img.resize((round(W * scale), round(H * scale)), Image.LANCZOS).save(full, quality=92, subsampling=0)
        img.crop((x0, y0, x0 + side, y0 + side)).save(crop, quality=95, subsampling=0)
        a = (x0 / W, 1 - (y0 + side) / H)   # window corners as tile fractions, origin bottom left (scaFull)
        b = ((x0 + side) / W, 1 - y0 / H)
        rows.append((full, crop, a, b, H / W))
        recs.append(dict(model=it["model"], prompt_id=it.get("prompt_id"), prompt=it["prompt"], seed=it["seed"],
                         method=it.get("method"), prompt_set=it.get("prompt_set"), resolution=f"{W}x{H}",
                         source=str(src), source_sha256=sha256(src), window_px=[x0, y0, side], window_rule=rule,
                         outputs={full.name: sha256(full), crop.name: sha256(crop)}))

    clauses = []
    for model in dict.fromkeys(r["model"] for r in recs):
        rs = [r for r in recs if r["model"] == model]
        clauses.append(f"{tex(model)} at {size_tex([r['resolution'] for r in rs])}")
    methods = sorted({r["method"] for r in recs if r["method"]})
    sets = sorted({r["prompt_set"] for r in recs if r["prompt_set"]})
    seeds = sorted({int(r["seed"]) for r in recs})
    method = "single-pass \\sca{}" if methods and all("single" in m.lower() for m in methods) else \
        (tex(", ".join(methods)) if methods else "\\sca{}")
    title = args.model or "FLUX.2-dev and Qwen-Image"
    caption = (r"\textbf{\sca{} on %s.} " % title + method[0].upper() + method[1:] + ", "
               + " and ".join(clauses)
               + (f"; {tex('; '.join(sets))}" if sets else "")
               + ", seed" + ("s~" if len(seeds) > 1 else "~") + ", ".join(map(str, seeds))
               + r". Each image is followed by a 1:1 crop of its outlined window at native pixels.")
    lines = ["% Generated by paper/figures/src/fig_showcase_models.py from web/build/showcase_manifest.json;",
             "%% do not edit by hand. Input only when main.tex sets \\%s." % (
                 "showcasetrue" if not args.model else "res%strue" % KEYS[args.model][1].lower()),
             r"\input{figures/snippets/_scagrid}%", r"\begin{figure}[!htbp]", r"\begingroup\scaGridStyle",
             r"\setlength{\scaT}{\dimexpr(\linewidth-%dpt)*1000/%d\relax}%%" % (2 * PER_ROW, 2000 * PER_ROW)]
    for i in range(0, len(rows), PER_ROW):
        cells = []
        for full, crop, a, b, ar in rows[i:i + PER_ROW]:
            h = r"\scaT" if abs(ar - 1) < 1e-6 else r"\dimexpr\scaT*%d/1000\relax" % round(1000 * ar)
            cells.append(r"\scaFull{\scaT}{%s}{figures/showcase_%s/%s}{(%.5f,%.5f)}{(%.5f,%.5f)}\hspace{1pt}"
                         r"\scaCrop{\scaT}{\scaT}{figures/showcase_%s/%s}"
                         % (h, key, full.name, a[0], a[1], b[0], b[1], key, crop.name))
        lines.append(r"\makebox[\linewidth][c]{" + r"\hspace{2pt}".join(cells) + r"}\par\vskip2pt")
    lines += [r"\endgroup", r"\caption{%s}" % caption, r"\label{fig:showcase_%s}" % key, r"\end{figure}", ""]
    snippet.write_text("\n".join(lines))
    if args.model:  # the sizes shown, for the sentence of Section 4.4
        text = args.paper / "figures" / "snippets" / f"showcase_{key}_text.tex"
        text.write_text("%% Generated by paper/figures/src/fig_showcase_models.py --model %s; do not edit by hand.\n"
                        "\\providecommand{\\showcase%sSizes}{%s}\n"
                        % (args.model, KEYS[args.model][1], size_tex([r["resolution"] for r in recs])))
    prov = dict(figure=f"fig:showcase_{key}", script="paper/figures/src/fig_showcase_models.py",
                manifest=str(args.manifest), manifest_sha256=sha256(args.manifest), items=recs,
                snippet_sha256=sha256(snippet))
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"wrote {snippet} with {len(rows)} items; caption: {caption}")


if __name__ == "__main__":
    main()
