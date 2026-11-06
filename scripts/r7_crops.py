#!/usr/bin/env python3
"""Adjudication crops (round 7): for each (method, prompt_id) the kept person/animal detector boxes of the image
(dup_rate.py's counting rule: threshold, NMS, containment) plus optional extra boxes, cropped from the full-resolution
image with a margin and written as JPEGs, with a 1024 px overview marking every box.
Usage: r7_crops.py --dets DIR [DIR ...] --pairs method:prompt_id [...] --out DIR [--extra-json FILE]
"""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dup_rate as dr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dets", nargs="+", required=True)
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--subjects", type=Path, default=ROOT / "results/dup_audit_20261027/subjects_a4k195.json")
    p.add_argument("--thr", type=float, default=0.35)
    p.add_argument("--margin", type=float, default=0.6, help="crop margin as a fraction of the box size")
    a = p.parse_args()
    Image.MAX_IMAGE_PIXELS = None
    subj = {x["id"]: x for x in json.loads(a.subjects.read_text())}
    rows = dr.load(a.dets)
    a.out.mkdir(parents=True, exist_ok=True)
    for item in a.pairs:
        m, pid = item.split(":", 1)
        r = rows[(m, pid)]
        im = Image.open(r["png"]).convert("RGB")
        W, H = im.size
        boxes = []
        for s in subj[pid]["subjects"]:
            if s.get("kind") not in ("person", "animal"):
                continue
            kept = dr.count_boxes(r["dets"], s["query"], a.thr, (800, 1333), 0.5, 0.9, 0.3, 0.9, W * H)
            boxes += [(s["query"], d["b"]) for d in kept]
        ov = im.resize((W // 4, H // 4))
        dw = ImageDraw.Draw(ov)
        for i, (q, b) in enumerate(boxes):
            x0, y0, x1, y1 = b
            dw.rectangle([x0 / 4, y0 / 4, x1 / 4, y1 / 4], outline=(255, 0, 0), width=2)
            dw.text((x0 / 4 + 2, y0 / 4 + 2), f"{i}", fill=(255, 0, 0))
            mx, my = a.margin * (x1 - x0), a.margin * (y1 - y0)
            c = im.crop((max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)))
            if max(c.size) < 384:
                c = c.resize((c.size[0] * 384 // max(c.size), c.size[1] * 384 // max(c.size)), Image.LANCZOS)
            c.save(a.out / f"{m}_{pid[-6:]}_box{i}_{q}.jpg", quality=92)
        ov.save(a.out / f"{m}_{pid[-6:]}_overview.jpg", quality=88)
        print(m, pid, len(boxes), "boxes:", [(q, [int(v) for v in b]) for q, b in boxes])


if __name__ == "__main__":
    main()
