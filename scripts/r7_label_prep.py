#!/usr/bin/env python3
"""Blind human-label packets for round 7 (protocol results/dup_audit_20261027/human_labels_95_protocol.md).

Input: JSONL rows {"label", "prompt_id", "png"} (any methods, mixed). The rows are shuffled (seed), given anonymous ids
img001.., and written as OUT/TILES/imgNNN/overview.jpg (whole image at 1024 px) plus r{R}c{C}.jpg (the 16 native
1024 x 1024 tiles of a 4096 x 4096 image), OUT/batch_NN.json (per labeler: img, prompt, person/animal subjects with
implied counts, note), and OUT/key.json (img -> label, prompt_id, png; never shown to labelers).
Usage: r7_label_prep.py --rows rows.jsonl --out DIR [--per 4] [--seed 7]
"""

import argparse
import json
from pathlib import Path
import random

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = ROOT / "results/dup_audit_20261027/subjects_a4k195.json"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per", type=int, default=4, help="images per labeler batch")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    Image.MAX_IMAGE_PIXELS = None
    subj = {x["id"]: x for x in json.loads(SUBJECTS.read_text())}
    rows = [json.loads(ln) for ln in a.rows.read_text().splitlines() if ln.strip()]
    random.Random(a.seed).shuffle(rows)
    key, items = {}, []
    for i, r in enumerate(rows, 1):
        img = f"img{i:03d}"
        d = a.out / "TILES" / img
        d.mkdir(parents=True, exist_ok=True)
        im = Image.open(r["png"]).convert("RGB")
        W, H = im.size
        im.resize((W // 4, H // 4), Image.LANCZOS).save(d / "overview.jpg", quality=90)
        for R in range(H // 1024):
            for C in range(W // 1024):
                im.crop((1024 * C, 1024 * R, 1024 * C + 1024, 1024 * R + 1024)).save(d / f"r{R}c{C}.jpg", quality=92)
        s = subj[r["prompt_id"]]
        key[img] = r
        items.append({"img": img, "prompt": s["prompt"],
                      "subjects": [{"query": t["query"], "implied": t["implied"]} for t in s["subjects"]
                                   if t.get("kind") in ("person", "animal")],
                      "note": s.get("note", "")})
    for b in range(0, len(items), a.per):
        (a.out / f"batch_{b // a.per + 1:02d}.json").write_text(json.dumps(items[b:b + a.per], indent=1))
    (a.out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(items)} images, {-(-len(items) // a.per)} batches -> {a.out}")


if __name__ == "__main__":
    main()
