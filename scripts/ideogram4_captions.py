#!/usr/bin/env python3
"""Ideogram 4 prompt upsampling, once per prompt, shared by every method and size.

Ideogram 4 is conditioned on a structured JSON caption that its pipeline writes from the user's prompt with the
Qwen3-VL text encoder plus a prompt-enhancer LM head (pipeline_ideogram4.upsample_prompt: the caption's system message,
the target aspect ratio, temperature 1, sampling seeded from the call's generator). Every method of a comparison must
see the same caption, so captions are written here once and read by scripts/sample_ideogram4.py and
scripts/calibrate_ideogram4.py. The upsampler sees only the aspect ratio, so one caption serves 4096^2 and 8192^2.

The generator is torch.Generator("cpu").manual_seed(image_seed(id, seed)), as for the image noise. `outlines` (the
pipeline's optional schema-constrained decoding) is not installed, so decoding is unconstrained; a caption that is not
valid JSON of the caption schema is redrawn with the seed of attempt k (image_seed(f"{id}#{k}", seed)), at most
--attempts times, and the attempt is recorded. Output: one JSON line per prompt (resumable, keyed by id)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sca.runtime import image_seed, read_prompts, resolution, runtime_environment  # noqa: E402


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, nargs="+", required=True, help="Prompt JSONL files (id, prompt)")
    p.add_argument("--count", type=int, nargs="*", default=[], help="Use only the first N prompts of each file")
    p.add_argument("--out", type=Path, required=True, help="Caption JSONL (appended, resumable)")
    p.add_argument("--res", type=resolution, default=(8192, 8192), metavar="HxW", help="Target size (aspect only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0, help="Pipeline default PROMPT_UPSAMPLE_TEMPERATURE")
    p.add_argument("--max-new-tokens", type=int, default=2048, help="Pipeline default (max_sequence_length)")
    p.add_argument("--attempts", type=int, default=4)
    p.add_argument("--device", default="cuda")
    return p


def load(path):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def main(argv=None):
    args = parser().parse_args(argv)
    runtime_environment()
    import torch
    from sca.ideogram4 import (HEAD_REPO, MAX_TEXT, REPO, aspect_of, load_text_encoder, sega_modules, snapshot,
                               valid_caption)
    prompts = []
    for i, path in enumerate(args.prompts):
        rows = read_prompts(path)
        n = args.count[i] if i < len(args.count) else len(rows)
        prompts += [dict(r, source=str(path)) for r in rows[:n]]
    ids = [r["id"] for r in prompts]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate prompt ids across files")
    h, w = args.res
    aspect = aspect_of(h, w)
    have = load(args.out)
    for r in prompts:
        old = have.get(r["id"])
        if old is not None and (old["prompt"] != r["prompt"] or old["aspect"] != aspect or old["seed"] != args.seed):
            raise SystemExit(f"{args.out} holds a different caption setting for {r['id']}; choose a new --out")
    todo = [r for r in prompts if r["id"] not in have]
    if not todo:
        print(f"all {len(prompts)} captions present in {args.out}", flush=True)
        return
    _, P = sega_modules()
    tok, te = load_text_encoder(args.device)
    head = P.Ideogram4PromptEnhancerHead.from_pretrained(str(snapshot(HEAD_REPO)), torch_dtype=torch.bfloat16,
                                                         local_files_only=True).to(args.device)
    enhancer = P._build_prompt_enhancer(te, head, torch.device(args.device))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for r in todo:
        t0 = time.perf_counter()
        for k in range(args.attempts):
            key = r["id"] if k == 0 else f"{r['id']}#{k}"
            gen = torch.Generator(device="cpu").manual_seed(image_seed(key, args.seed))
            with torch.inference_mode():
                cap = P._generate_captions(enhancer, tok, None, r["prompt"], h, w, temperature=args.temperature,
                                           max_new_tokens=args.max_new_tokens, generator=gen,
                                           device=torch.device(args.device))[0]
            ok = valid_caption(cap)
            if ok:
                break
            print(f"{r['id']}: attempt {k} is not a valid caption ({len(cap)} chars); redrawing", flush=True)
        msgs = [{"role": "user", "content": [{"type": "text", "text": cap}]}]
        n_tok = len(tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                        add_special_tokens=False)["input_ids"])
        row = {"id": r["id"], "prompt": r["prompt"], "source": r["source"], "aspect": aspect, "seed": args.seed,
               "attempt": k, "valid": ok, "caption": cap, "caption_sha256": hashlib.sha256(cap.encode()).hexdigest(),
               "text_tokens": n_tok, "fits": n_tok <= MAX_TEXT, "temperature": args.temperature,
               "model": REPO, "snapshot": snapshot().name, "head": HEAD_REPO, "head_snapshot": snapshot(HEAD_REPO).name,
               "seconds": round(time.perf_counter() - t0, 2)}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"caption {r['id']} attempt={k} valid={ok} tokens={n_tok} seconds={row['seconds']}", flush=True)
    bad = [r["id"] for r in load(args.out).values() if not (r["valid"] and r["fits"])]
    if bad:
        raise SystemExit(f"captions without a valid schema or over {MAX_TEXT} tokens: {bad}")


if __name__ == "__main__":
    main()
