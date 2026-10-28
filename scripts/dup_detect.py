#!/usr/bin/env python3
"""Duplicate-subject detector: open-vocabulary detection of each prompt subject (Grounding DINO tiny).

For every PNG of every `label=dir` pair, runs Grounding DINO once per scale with all of the prompt's subject
queries ("person . bear ."), scores each box per query (max token probability over the query's tokens) and
appends the raw boxes (score >= --floor) to a per-shard JSONL. Counting, NMS, thresholds and duplication flags
are applied afterwards by scripts/dup_rate.py, so thresholds can be validated without re-running detection.
Resumable: an image already present in the shard file is skipped. CPU or CUDA.
"""

import argparse
import glob
import json
import os
from pathlib import Path
import time

GDINO = "/scratch/09032/saini_2/tmp/hf/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71"


def prompt_id_of(png):
    meta = Path(png).with_suffix(".json")
    if meta.exists():
        c = json.loads(meta.read_text())
        c = c.get("config", c)
        if c.get("prompt_id"):
            return str(c["prompt_id"])
    stem = Path(png).stem  # a4k_eval_4096_<id>_<hash>_seed<k>
    return "_".join(stem.split("_")[:-2])


def work_list(pairs, ids):
    out = []
    for pair in pairs:
        label, d = pair.split("=", 1)
        for png in sorted(glob.glob(os.path.join(d, "*.png"))):
            pid = prompt_id_of(png)
            if pid in ids:
                out.append((label, pid, os.path.realpath(png)))
    return out


def phrase_spans(tokenizer, text, phrases, input_ids):
    enc = tokenizer(text, return_offsets_mapping=True)
    assert enc["input_ids"] == input_ids, "tokenization mismatch"
    spans, pos = [], 0
    for p in phrases:
        a = text.index(p, pos)
        b, pos = a + len(p), a + len(p)
        spans.append([i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > s and s >= a and e <= b])
    return spans


def detect(model, proc, image, phrases, sides, floor, device):
    import torch
    from PIL import Image
    text = " . ".join(phrases) + " ."
    W, H = image.size
    dets = []
    for side in sides:
        sc = side / max(W, H)
        im = image.resize((max(1, round(W * sc)), max(1, round(H * sc))), Image.BICUBIC) if sc < 1 else image
        inputs = proc(images=im, text=text, return_tensors="pt", do_resize=False).to(device)
        spans = phrase_spans(proc.tokenizer, text, phrases, inputs["input_ids"][0].tolist())
        with torch.inference_mode():
            out = model(**inputs)
        prob = out.logits[0].float().sigmoid()           # [Q, 256]
        boxes = out.pred_boxes[0].float()                 # [Q, 4] cx, cy, w, h relative to the unpadded image
        for j, sp in enumerate(spans):
            s = prob[:, sp].max(1).values
            keep = (s >= floor).nonzero().flatten().tolist()
            for q in keep:
                cx, cy, w, h = boxes[q].tolist()
                x0, y0, x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
                dets.append({"q": phrases[j], "s": round(float(s[q]), 4), "side": side,
                             "b": [round(max(0, x0)), round(max(0, y0)), round(min(W, x1)), round(min(H, y1))]})
    return dets


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=Path, required=True, help="JSON list of {id, subjects:[{query, implied}]}")
    p.add_argument("--dirs", nargs="+", required=True, metavar="LABEL=DIR")
    p.add_argument("--out", type=Path, required=True, help="output directory (one JSONL per shard)")
    p.add_argument("--sides", default="1333", help="comma-separated longest-side sizes for detection")
    p.add_argument("--floor", type=float, default=0.2)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)
    import torch
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    subj = {r["id"]: r for r in json.loads(args.subjects.read_text())}
    queries = {i: sorted({s["query"] for s in r["subjects"]}) for i, r in subj.items()}
    todo = [w for w in work_list(args.dirs, set(subj)) if queries[w[1]]]
    todo = todo[args.shard::args.nshards][:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    shard = args.out / f"dets_{args.shard:03d}of{args.nshards:03d}.jsonl"
    done = set()
    if shard.exists():
        for line in shard.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["method"], r["png"]))
            except json.JSONDecodeError:
                pass
    proc = AutoProcessor.from_pretrained(GDINO)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).eval().to(args.device)
    sides = [int(s) for s in args.sides.split(",")]
    t0, n = time.time(), 0
    with shard.open("a") as fh:
        for label, pid, png in todo:
            if (label, png) in done:
                continue
            image = Image.open(png).convert("RGB")
            dets = detect(model, proc, image, queries[pid], sides, args.floor, args.device)
            fh.write(json.dumps({"method": label, "prompt_id": pid, "png": png, "size": list(image.size),
                                 "sides": sides, "queries": queries[pid], "dets": dets}) + "\n")
            fh.flush()
            n += 1
            if n % 25 == 0:
                print(f"{n} images, {(time.time() - t0) / n:.2f} s/image", flush=True)
    print(f"done shard {args.shard}: {n} new images in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
