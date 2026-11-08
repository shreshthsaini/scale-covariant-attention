"""Ideogram 4 at 8192^2 (fig:ideogram4): single-pass SCA, full attention and SEGA side by side, one prompt per row,
each full image followed by its 1:1 native crop (the layout of fig:gallery_flux2, short prompt names as row labels).

Prompts: the first --rows prompts of web/build/ideogram4_prompts.jsonl (the campaign's fixed order) that have all three
methods at seed 0; nothing about the images enters the choice. Crop: one window per prompt, the --crop native-pixel
square with the most edge energy averaged over the three methods (fig_qual.auto_box, the rule of every grid in the
paper), the same window for every method. Full images are Lanczos-downscaled to GRID_DPI of their printed size, crops
are native pixels; JPEG quality as fig_qual (JPEG_Q). Sources, hashes and windows go to figures/ideogram4/provenance.json.
Writes figures/ideogram4/*.jpg and snippets/ideogram4_fig.tex; never writes snippets/_scagrid.tex.
Run with the sca env python from paper/figures/src:  <python> fig_ideogram4.py [--rows 3] [--crop 512]
"""
import argparse
import json
import os
from pathlib import Path

import imgsel
from fig_qual import G_GROUP, G_PAIR, G_ROW, JPEG_Q, SNIP, Geo, auto_box, g_row, method_head, t_crop, t_full

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = Path("/scratch/09032/saini_2/sca/out/ideogram4")
PROMPTS = REPO / "web" / "build" / "ideogram4_prompts.jsonl"
NAME, RES = "ideogram4", "8192x8192"
METHODS = ("sca", "full", "sega")
HEADS = {"sca": "SCA (ours)", "full": "Full attention", "sega": "SEGA"}
SCRIPT = "paper/figures/src/fig_ideogram4.py"


def short(pid):
    """Row label from the prompt id (sc_p01_skeleton_watch -> Skeleton watch): short enough for a rotated label one
    tile high."""
    words = pid.split("_", 2)[-1].replace("_", " ")
    return words[:1].upper() + words[1:]


def source(out, m, pid):
    d = out / f"ideogram4_{m}" / RES
    for meta in sorted(d.glob(f"{pid}_*seed0.json")):
        r = json.loads(meta.read_text())
        if r.get("complete") and r["config"]["prompt_id"] == pid:
            return meta.with_suffix(".png"), r
    return None, None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--outputs", type=Path, default=OUT)
    p.add_argument("--prompts", type=Path, default=PROMPTS)
    a = p.parse_args(argv)
    rows = [json.loads(line) for line in open(a.prompts) if line.strip()]
    picked = []
    for r in rows:
        srcs = {m: source(a.outputs, m, r["id"]) for m in METHODS}
        if all(s[0] is not None for s in srcs.values()):
            picked.append((r, srcs))
        if len(picked) == a.rows:
            break
    if not picked:
        raise SystemExit("no prompt has SCA, full attention and SEGA at 8192^2 yet")
    geo = Geo(units=2 * len(METHODS), gaps=len(METHODS) * G_PAIR + (len(METHODS) - 1) * G_GROUP, label=True)
    px = geo.px()
    fig = os.path.join(imgsel.FIGDIR, NAME)
    os.makedirs(fig, exist_ok=True)
    prov = {"script": SCRIPT, "res": RES, "seed": 0, "crop_size": a.crop, "full_px": px, "jpeg_quality": JPEG_Q,
            "rule": "first prompts of web/build/ideogram4_prompts.jsonl with all three methods; crop = fig_qual.auto_box "
                    "over the three methods", "prompts": []}
    head = []
    for k, m in enumerate(METHODS):
        head += ([G_GROUP] if k else []) + [method_head(m, 2, G_PAIR, names=HEADS)]
    tex_rows = [(g_row(head, indent=True), G_ROW)]
    for r, srcs in picked:
        imgs = {m: imgsel.load(str(srcs[m][0])) for m in METHODS}
        box = auto_box([imgs[m] for m in METHODS], a.crop)
        cells, entry = [], {"id": r["id"], "title": r["title"], "prompt": r["prompt"], "crop_xy_size": list(box),
                            "sources": {}}
        for k, m in enumerate(METHODS):
            full = os.path.join(fig, f"{r['id']}_{m}.jpg")
            crop = os.path.join(fig, f"{r['id']}_{m}_crop.jpg")
            imgsel.save_jpeg(imgs[m], full, JPEG_Q, px)
            imgsel.save_jpeg(imgsel.crop(imgs[m], box), crop, JPEG_Q, a.crop)
            entry["sources"][m] = {"png": str(srcs[m][0]), "sha256": imgsel.sha256(str(srcs[m][0])),
                                   "seconds": srcs[m][1].get("seconds"), "full": os.path.basename(full),
                                   "crop": os.path.basename(crop)}
            cells += ([G_GROUP] if k else []) + [t_full(full, box, RES), G_PAIR, t_crop(crop)]
        tex_rows.append((g_row(cells, label=short(r["id"])), G_ROW))
        prov["prompts"].append(entry)
    with open(os.path.join(fig, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=1)
    titles = ", ".join(short(e["id"]).lower() for e in prov["prompts"])
    caption = (r"\textbf{Ideogram 4 at $8192^2$, twice its native $4096^2$.} Single-pass \sca{}, full attention and "
               rf"SEGA on the first {len(picked)} of the 10 showcase prompts in their fixed order ({titles}), seed 0. "
               rf"A 1:1 crop of {a.crop} native pixels follows each image, at the window with the most edge energy "
               r"averaged over the three methods.")
    lines = [f"% Generated by {SCRIPT}; do not edit by hand."]
    lines += [f"% {e['id']} ({RES}): {e['prompt'][:150]}" for e in prov["prompts"]]
    lines += [r"\input{figures/snippets/_scagrid}%", r"\begin{figure*}[t]", r"\begingroup\scaGridStyle", geo.tex()]
    lines += [t + (r"\par\vskip%gpt" % g if i < len(tex_rows) - 1 else r"\par") for i, (t, g) in enumerate(tex_rows)]
    lines += [r"\endgroup", rf"\caption{{{caption}}}", r"\label{fig:ideogram4}", r"\end{figure*}", ""]
    path = os.path.join(SNIP, "ideogram4_fig.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", path, "and", fig)


if __name__ == "__main__":
    main()
