#!/usr/bin/env python3
"""Per-denoising-step FLUX.1-dev latency, full attention vs SCA, timed interleaved in one process.

One pipeline, one installed SCAContext whose mode is switched between "full" and "sca" (and the
local implementation between gather and flex), so every method sees the same GPU conditions.
Each timing is a pipeline call with --steps denoising steps on precomputed prompt embeddings and
output_type="latent" (no text encoders, no VAE). Reports the minimum over --rounds per method,
divided by the step count, and that times 28 as the denoising cost of one image.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sca.runtime import load_pipeline, resolution, runtime_environment, set_mu  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--res", type=resolution, nargs="+", default=[(2048, 2048), (4096, 4096)])
    p.add_argument("--methods", nargs="+", default=["full", "sca_flex", "sca_gather"])
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--guidance", type=float, default=4.5)
    p.add_argument("--tile", type=int, default=8)
    args = p.parse_args()
    runtime_environment()
    from sca.attn import PiTable, SCAContext, install
    pipe = load_pipeline("cuda")
    ctx = SCAContext(mode="full", tile=args.tile, pi=PiTable("const", value=0.5))
    uninstall = install(pipe.transformer, ctx)
    ctx.scheduler = pipe.scheduler
    with torch.inference_mode():
        pe, pooled, _ = pipe.encode_prompt(prompt="A lighthouse on a cliff at sunset", prompt_2=None,
                                           device="cuda", max_sequence_length=512)
    out = []
    for h, w in args.res:
        set_mu(pipe, h, w, "max_shift")
        best = {m: float("inf") for m in args.methods}

        def run(method):
            ctx.mode = "full" if method == "full" else "sca"
            ctx.local_impl = "gather" if method == "sca_gather" else "flex"
            g = torch.Generator(device="cpu").manual_seed(0)
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.inference_mode():
                pipe(prompt_embeds=pe, pooled_prompt_embeds=pooled, height=h, width=w,
                     num_inference_steps=args.steps, guidance_scale=args.guidance, generator=g,
                     output_type="latent")
            torch.cuda.synchronize()
            return time.perf_counter() - t

        for m in args.methods:  # warmup (compiles flex)
            run(m)
        for _ in range(args.rounds):
            for m in args.methods:
                best[m] = min(best[m], run(m))
        row = {"res": f"{h}x{w}", "tile": args.tile, "steps_timed": args.steps}
        for m in args.methods:
            row[f"{m}_s_per_step"] = best[m] / args.steps
            row[f"{m}_denoise_28_s"] = 28 * best[m] / args.steps
        print(json.dumps(row), flush=True)
        out.append(row)
    uninstall()


if __name__ == "__main__":
    main()
