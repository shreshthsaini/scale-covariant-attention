#!/usr/bin/env python3
"""Attention probe for the paper's attention-focus figure: FLUX.1-dev at 1024^2 with full attention, capturing every
head's queries and keys at chosen denoising steps.

The run is the native first stage of progressive SCA for one prompt and seed: HRDiT's shared 30-step schedule
(mu = 1.15), guidance 3.5, FLUX.1-dev's stock attention, the same per-prompt CPU generator and Euler update as
scripts/sample_progressive.py (generate, stage 0). At each step in --capture-steps a capturing processor stores, for
all 57 blocks (19 double-stream, then 38 single-stream), the pre-rotation queries and keys after RMSNorm, text first
([T + N, 24, 128], bf16, exactly the tensors the stock processor rotates and attends with); the attention output is
the stock processor's, so the trajectory is that of unmodified FLUX.1-dev. With the position ids (saved alongside),
any attention row can be recomputed exactly: full attention, its near and far parts, and SCA's emulated window and
block-centre branches (paper/figures/src/fig_attn_focus.py).

Outputs in --out:
  qk_step<j>.pt   {"q": [57, T+N, 24, 128] bf16, "k": same, "sigma": float, "step": j, "txt_ids", "img_ids"}
  image_1024.png  the decoded 1024^2 image at the end of the stage (VAE tiling on, as in the progressive runner)
  latents.pt      final packed latents (bf16)
  meta.json       prompt, seed, derived seed, sigmas, captured steps, per-step seconds, versions, code revision

Runs on CPU (bf16) at low priority, e.g.
  nice -n 19 <sca env python> scripts/attn_focus.py --device cpu --threads 96 \
      --prompts /scratch/09032/saini_2/sca/eval100.jsonl --id a4k_eval_4096_minus8343371271909676658 --seed 0 \
      --capture-steps 8,20,26 --out /scratch/09032/saini_2/sca/out/attn_focus/sheep_church_seed0
"""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from sca.runtime import (HF_HOME, atomic_json, code_revision, file_sha256, image_seed, implementation_hash,  # noqa: E402
                         read_prompts)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--id", required=True, help="prompt id in --prompts")
    p.add_argument("--seed", type=int, default=0, help="seed label (hashed with the prompt id, as every runner does)")
    p.add_argument("--steps", type=int, default=30, help="HRDiT base steps")
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--capture-steps", default="8,20,26", help="0-based step indices of the 30-step schedule")
    p.add_argument("--stop-after-last-capture", action="store_true",
                   help="skip the remaining steps and the decode (no image_1024.png)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=96)
    p.add_argument("--out", type=Path, required=True)
    return p


def capture_processor_class():
    import torch
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor, _get_qkv_projections

    class CaptureProcessor(FluxAttnProcessor):
        """The stock processor, plus a copy of the pre-rotation q and k (after RMSNorm, text first) when active."""

        def __init__(self, store, index):
            super().__init__()
            self.store, self.index, self.active = store, index, False

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                     image_rotary_emb=None):
            if self.active:
                q, k, _, eq, ek, _ = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
                q = attn.norm_q(q.unflatten(-1, (attn.heads, -1)))
                k = attn.norm_k(k.unflatten(-1, (attn.heads, -1)))
                if attn.added_kv_proj_dim is not None:
                    eq = attn.norm_added_q(eq.unflatten(-1, (attn.heads, -1)))
                    ek = attn.norm_added_k(ek.unflatten(-1, (attn.heads, -1)))
                    q, k = torch.cat([eq, q], dim=1), torch.cat([ek, k], dim=1)
                self.store[self.index] = (q[0].to(torch.bfloat16).clone(), k[0].to(torch.bfloat16).clone())
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

    return FluxAttnProcessor, CaptureProcessor


def main(argv=None):
    args = parser().parse_args(argv)
    os.environ.setdefault("HF_HOME", HF_HOME)
    import diffusers
    import torch
    torch.set_num_threads(args.threads)
    import sample_progressive as sp
    from sca.runtime import load_pipeline

    rows = {r["id"]: r for r in read_prompts(args.prompts)}
    row = rows[args.id]
    capture = sorted({int(x) for x in args.capture_steps.split(",") if x.strip()})
    if not capture or capture[0] < 0 or capture[-1] >= args.steps:
        raise SystemExit("--capture-steps must lie in [0, steps)")
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"prompt_id": row["id"], "prompt": row["prompt"], "prompts_file": str(args.prompts), "seed": args.seed,
            "derived_seed": image_seed(row["id"], args.seed), "res": args.res, "steps": args.steps,
            "guidance": args.guidance, "capture_steps": capture, "device": args.device, "threads": args.threads,
            "code_revision": code_revision(), "implementation_hash": implementation_hash(),
            "versions": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                         "python": platform.python_version()},
            "host": platform.node(), "step_seconds": [], "files": {}}

    t_load = time.perf_counter()
    pipe = load_pipeline(args.device, "dev")
    pipe.vae.enable_tiling()  # as in sample_progressive.load
    meta["load_seconds"] = time.perf_counter() - t_load
    tr = pipe.transformer
    stock_cls, capture_cls = capture_processor_class()
    blocks = list(tr.transformer_blocks) + list(tr.single_transformer_blocks)
    for b in blocks:
        if type(b.attn.processor) is not stock_cls:
            raise SystemExit(f"unexpected processor {type(b.attn.processor).__name__}; the probe wraps the stock one")
    store = {}
    procs = [capture_cls(store, i) for i in range(len(blocks))]
    for b, p in zip(blocks, procs):
        b.attn.set_processor(p)

    derived = meta["derived_seed"]
    torch.manual_seed(derived)
    generator = torch.Generator(device="cpu").manual_seed(derived)
    device = pipe._execution_device
    t0 = time.perf_counter()
    with torch.inference_mode():
        prompt_embeds, pooled, text_ids = sp.encode_text(pipe, row["prompt"], device)
        meta["text_seconds"] = time.perf_counter() - t0
        timesteps, sigmas, mu = sp.hrdit_schedule(pipe, args.steps, (args.res, args.res), device)
        meta.update(mu=mu, sigmas=[float(s) for s in sigmas])
        C = tr.config.in_channels // 4
        latents, ids = pipe.prepare_latents(1, C, args.res, args.res, prompt_embeds.dtype, device, generator)
        guidance = torch.full([1], args.guidance, device=device, dtype=torch.float32) \
            if tr.config.guidance_embeds else None
        last = capture[-1] if args.stop_after_last_capture else args.steps - 1
        for j in range(last + 1):
            t1 = time.perf_counter()
            active = j in capture
            for p in procs:
                p.active = active
            store.clear()
            t = timesteps[j]
            v = tr(hidden_states=latents, timestep=t.expand(1).to(latents.dtype) / 1000, guidance=guidance,
                   pooled_projections=pooled, encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=ids,
                   joint_attention_kwargs=None, return_dict=False)[0]
            latents, x0 = sp.euler(latents, v, sigmas[j], sigmas[j + 1])
            if active:
                if len(store) != len(blocks):
                    raise RuntimeError(f"captured {len(store)} of {len(blocks)} blocks at step {j}")
                q = torch.stack([store[i][0] for i in range(len(blocks))])
                k = torch.stack([store[i][1] for i in range(len(blocks))])
                path = args.out / f"qk_step{j:02d}.pt"
                tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
                torch.save({"q": q, "k": k, "sigma": float(sigmas[j]), "step": j, "txt_ids": text_ids.cpu(),
                            "img_ids": ids.cpu(), "text_len": int(prompt_embeds.shape[1]),
                            "x0": x0.to(torch.bfloat16).cpu()}, tmp)
                os.replace(tmp, path)
                meta["files"][path.name] = {"sigma": float(sigmas[j]), "bytes": path.stat().st_size}
                store.clear()
            meta["step_seconds"].append(time.perf_counter() - t1)
            print(f"step {j:2d} sigma {float(sigmas[j]):.4f} {meta['step_seconds'][-1]:.1f}s"
                  + (" captured" if active else ""), flush=True)
            atomic_json(args.out / "meta.json", meta)
        if not args.stop_after_last_capture:
            torch.save(latents.to(torch.bfloat16).cpu(), args.out / "latents.pt")
            t2 = time.perf_counter()
            image = sp.decode(pipe, latents, args.res, args.res)
            pil = pipe.image_processor.postprocess(image, output_type="pil")[0]
            png = args.out / "image_1024.png"
            pil.save(png, format="PNG")
            meta["decode_seconds"] = time.perf_counter() - t2
            meta["files"]["image_1024.png"] = {"sha256": file_sha256(png)}
    meta["total_seconds"] = time.perf_counter() - t0
    meta["complete"] = True
    atomic_json(args.out / "meta.json", meta)
    print(json.dumps({k: meta[k] for k in ("prompt_id", "capture_steps", "total_seconds")}), flush=True)


if __name__ == "__main__":
    main()
