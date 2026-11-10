#!/usr/bin/env python3
"""Progressive SCA on Qwen-Image: the FLUX.1-dev progressive design of scripts/sample_progressive.py (HRDiT's stage
schedule, upsampler, re-noising and FFT structure guidance, only the attention operator swapped) on Qwen-Image.

Stage 0 is Qwen-Image's own sampling at the native 1328^2 with its stock attention processors: true CFG (default 4.0
with the negative prompt " ", the pipeline's norm-preserving combination), Qwen's scheduler at the native sequence
(mu 0.834, shift_terminal 0.02) over --base-steps steps; its latents equal QwenImagePipeline's for the same seed,
steps and mu. Each later stage (default 2656^2, then 5312^2) decodes the previous stage with the Qwen VAE, upsamples
bicubically, sharpens (HRDiT's Gaussian unsharp mask, blur sigma from the global Torch RNG seeded per prompt), encodes
(latent mode, the img2img pipeline's normalization), re-noises to the sigma of its start step of the shared base
schedule with noise from the per-prompt generator, and runs the remaining steps of that schedule with SCA attention
(sca/qwen.py, one QwenSCAContext per stage, the reach table entry nearest the stage's pool factor) and HRDiT's
structure guidance from the previous stage's x0 prediction at the same step (alphas, betas and Butterworth low-pass
as sample_progressive.py). The default stage starts sit at the noise levels of HRDiT's FLUX.1 schedule (sigma 0.81
at 2K and 0.62 at 4K): steps 10 and 17 of 30, so 20 and 13 refinement steps.

Output: PNG + JSON sidecar per prompt with sample_qwen.py's stems, seeds and fields (prompt = benchmark prompt), so
the evaluators read --out unchanged. Timing scope as sample_qwen.py: every stage, every transition and the final
VAE decode; prompt embeddings are precomputed (sca/qwen.py cache, text encoder not resident).
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_progressive as sp  # noqa: E402  (HRDiT operators: Plan, butterworth, lowpass, sharpen, Clock)
from sca import dedup  # noqa: E402
from sca.runtime import (atomic_json, code_revision, file_sha256, fingerprint, image_seed, image_stem,  # noqa: E402
                         implementation_hash, read_prompts, resolution, runtime_environment)

QWEN_PROG = dict(stages="1328x1328,2656x2656,5312x5312", base_steps=30, highres_steps="20,13", true_cfg="4.0",
                 alphas="1.0,0.25", betas="0.5,0.5", filter_ratio="0.2,0.2")


# ----------------------------------------------------------------------------- Qwen latents and VAE

def latent_hw(pipe, h, w):
    s = pipe.vae_scale_factor
    return 2 * (h // (s * 2)), 2 * (w // (s * 2))


def unpack(pipe, x, h, w):
    """Packed [B, N, 4C] -> [B, C, lh, lw] (QwenImagePipeline._unpack_latents without the frame axis)."""
    return pipe._unpack_latents(x, h, w, pipe.vae_scale_factor)[:, :, 0]


def pack(pipe, z):
    """[B, C, lh, lw] -> packed [B, N, 4C] (QwenImagePipeline._pack_latents)."""
    B, C, lh, lw = z.shape
    return pipe._pack_latents(z.contiguous(), B, C, lh, lw)


def decode(pipe, latents, h, w):
    """QwenImagePipeline's decode with the same arithmetic: image [B, 3, H, W] in [-1, 1]."""
    import torch
    x = pipe._unpack_latents(latents, h, w, pipe.vae_scale_factor).to(pipe.vae.dtype)
    c = pipe.vae.config
    mean = torch.tensor(c.latents_mean).view(1, c.z_dim, 1, 1, 1).to(x.device, x.dtype)
    inv_std = 1.0 / torch.tensor(c.latents_std).view(1, c.z_dim, 1, 1, 1).to(x.device, x.dtype)
    return pipe.vae.decode(x / inv_std + mean, return_dict=False)[0][:, :, 0]


def encode(pipe, image):
    """QwenImageImg2ImgPipeline._encode_vae_image (latent mode, (z - mean) / std), packed."""
    import torch
    z = pipe.vae.encode(image[:, :, None].to(pipe.vae.dtype)).latent_dist.mode()
    c = pipe.vae.config
    mean = torch.tensor(c.latents_mean).view(1, c.z_dim, 1, 1, 1).to(z.device, z.dtype)
    std = torch.tensor(c.latents_std).view(1, c.z_dim, 1, 1, 1).to(z.device, z.dtype)
    return pack(pipe, ((z - mean) / std)[:, :, 0])


def velocity(tr, latents, t, cond, neg, img_shapes, cfg):
    """QwenImagePipeline's per-step prediction: conditional pass, and with true CFG the unconditional pass combined as
    neg + cfg (cond - neg), rescaled to the conditional prediction's norm per token."""
    import torch
    timestep = t.expand(latents.shape[0]).to(latents.dtype)

    def call(text, tag):
        ctx = tr.cache_context(tag) if hasattr(tr, "cache_context") else _Null()
        with ctx:
            return tr(hidden_states=latents, timestep=timestep / 1000, guidance=None, encoder_hidden_states_mask=None,
                      encoder_hidden_states=text, img_shapes=img_shapes, attention_kwargs={}, return_dict=False)[0]
    v = call(cond, "cond")
    if neg is None:
        return v
    vn = call(neg, "uncond")
    comb = vn + cfg * (v - vn)
    return comb * (torch.norm(v, dim=-1, keepdim=True) / torch.norm(comb, dim=-1, keepdim=True))


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def guided_step(pipe, sample, v, sigma, sigma_next, ref, hw, mask, alpha, beta, state):
    """sample_progressive.guided_step (HRDiT flowmatch_step, v_theta, latent upsampling, FFT structure guidance) on
    Qwen-Image's packing. ref = (packed x0 of the previous stage at this step, its pixel (h, w))."""
    import torch.nn.functional as F
    h, w = hw
    lh, lw = latent_hw(pipe, h, w)
    x = sample.float()
    x0 = x - v * sigma
    original = x0
    x0u = unpack(pipe, x0, h, w)
    prev = unpack(pipe, ref[0], ref[1][0], ref[1][1])
    refu = F.interpolate(prev, (lh, lw), mode="bicubic", align_corners=False)
    x0u = x0u + alpha * (sp.lowpass(refu, mask) - sp.lowpass(x0u, mask))
    x0 = pack(pipe, x0u)
    refp = pack(pipe, refu)
    out = (x - x0) / (sigma + 1e-6)
    out_ref = (x - refp) / (sigma + 1e-6)
    if state.get("high") is not None:
        out = out + beta * (state["high"] + out_ref - state["ref"] - out)
    state["high"], state["ref"] = out, out_ref
    return (x + (sigma_next - sigma) * out).to(out.dtype), original


def base_schedule(pipe, base_steps, native_hw, device):
    """Qwen's schedule at the native stage: sigmas linspace(1, 1/N, N) shifted by mu of the native sequence (sca/qwen.py
    mu_for, capped at 83 x 83) with the scheduler's terminal stretch; the later stages run its tail (HRDiT)."""
    import numpy as np
    from sca.qwen import mu_for
    lh, lw = latent_hw(pipe, *native_hw)
    mu = mu_for((lh // 2) * (lw // 2), "native", dict(pipe.scheduler.config))
    pipe.scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / base_steps, base_steps), device=device, mu=mu)
    return pipe.scheduler.timesteps.clone(), pipe.scheduler.sigmas.clone(), float(mu)


def generate(pipe, cond, neg, generator, plan, contexts, sync=None, record=None, keep=None):
    """Run every stage; return the decoded image tensor [B, 3, H, W] in [-1, 1]. cond / neg: prompt embeddings
    [1, L, D] on the device (neg None: no CFG). contexts[s] is the QwenSCAContext of stage s >= 1 (index 0 unused:
    the native stage runs the stock processors). plan.guidance holds the true-CFG scale per stage. The caller seeds
    the global Torch RNG (blur sigma). keep (dict, tests): final packed latents of every stage."""
    import torch
    from diffusers.utils.torch_utils import randn_tensor
    from sca.qwen import install
    record = {} if record is None else record
    sync = sync or (lambda: None)
    tr = pipe.transformer
    device, dtype = cond.device, cond.dtype
    timesteps, sigmas, mu = base_schedule(pipe, plan.base_steps, plan.stages[0], device)
    record.update(mu=mu, sigmas=[float(s) for s in sigmas], stage_seconds=[], transition_seconds=[], blur_sigmas=[],
                  stage_shapes=[], stage_sigmas=[])
    clock = sp.Clock(sigmas)
    C = tr.config.in_channels // 4
    h, w = plan.stages[0]
    latents = pipe.prepare_latents(1, C, h, w, dtype, device, generator)
    pred_x0 = {}
    for s, (h, w) in enumerate(plan.stages):
        sync()
        t0 = time.perf_counter()
        if s > 0:  # HRDiT transition: decode, bicubic upsample, sharpen, encode, re-noise at the stage's start sigma
            ph, pw = plan.stages[s - 1]
            image = decode(pipe, latents, ph, pw)
            image = torch.nn.functional.interpolate(image, (h, w), mode="bicubic", align_corners=False)
            image, blur = sp.sharpen(image, plan.sharpen_kernel, plan.sharpen_sigma, plan.sharpen_alpha)
            record["blur_sigmas"].append(blur)
            latents = encode(pipe, image)
            k = plan.start_index(s)
            noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
            sig = sigmas.to(device=latents.device, dtype=latents.dtype)[k]  # scheduler.scale_noise arithmetic
            latents = (sig * noise + (1.0 - sig) * latents).to(dtype)
            sync()
            record["transition_seconds"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
        start = plan.start_index(s)
        lh, lw = latent_hw(pipe, h, w)
        img_shapes = [[(1, lh // 2, lw // 2)]]
        record["stage_shapes"].append({"res": f"{h}x{w}", "latents": list(latents.shape), "grid": [lh // 2, lw // 2]})
        record["stage_sigmas"].append([float(x) for x in sigmas[start:plan.base_steps]])
        ctx = contexts[s] if s > 0 else None
        uninstall = None
        if ctx is not None:
            ctx.known_grid = (lh // 2, lw // 2)
            ctx.scheduler = clock
            uninstall = install(tr, ctx)
        cfg = float(plan.guidance[s])
        mask = sp.butterworth([1, C, lh, lw], plan.filter_ratio[s - 1], plan.butterworth_order, device) if s else None
        state = {}
        n = plan.base_steps - start
        try:
            for i in range(n):
                j = start + i
                clock.step_index = j
                v = velocity(tr, latents, timesteps[j], cond, neg, img_shapes, cfg)
                if s == 0:
                    latents, x0 = sp.euler(latents, v, sigmas[j], sigmas[j + 1])
                else:
                    a = plan.alphas[s - 1] * (n - i) / n
                    b = plan.betas[s - 1] * (n - i) / n
                    latents, x0 = guided_step(pipe, latents, v, sigmas[j], sigmas[j + 1], pred_x0[j], (h, w), mask, a,
                                              b, state)
                    latents = latents.to(dtype)
                pred_x0[j] = (x0, (h, w))
        finally:
            if uninstall is not None:
                uninstall()
        if not torch.isfinite(latents).all():
            raise FloatingPointError(f"Non-finite latents at the end of stage {s} ({h}x{w})")
        if keep is not None:
            keep[s] = latents.clone()
        sync()
        record["stage_seconds"].append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    image = decode(pipe, latents, *plan.stages[-1])
    sync()
    record["decode_seconds"] = time.perf_counter() - t0
    return image


# ----------------------------------------------------------------------------- CLI

def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="qwen_sca_prog")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stages", default=QWEN_PROG["stages"], help="Comma-separated HxW per stage, native first")
    p.add_argument("--base-steps", type=int, default=QWEN_PROG["base_steps"])
    p.add_argument("--highres-steps", default=QWEN_PROG["highres_steps"],
                   help="Steps per higher stage: the last k of the shared base schedule")
    p.add_argument("--true-cfg", default=QWEN_PROG["true_cfg"], help="True CFG scale: one value or one per stage")
    p.add_argument("--negative", default=" ", help="Negative prompt (model card: one space; --negative= for empty)")
    p.add_argument("--prompt-suffix", choices=("none", "area"), default="none",
                   help="area: append AReA's Qwen-Image QWEN_SUFFIX to the conditioning prompt (sample_qwen.py)")
    p.add_argument("--alphas", default=QWEN_PROG["alphas"])
    p.add_argument("--betas", default=QWEN_PROG["betas"])
    p.add_argument("--filter-ratio", default=QWEN_PROG["filter_ratio"])
    p.add_argument("--pi-table", type=Path, help="npz from scripts/calibrate_qwen.py (required unless --pi-const)")
    p.add_argument("--pi-const", type=float, help="Constant pi (tests and ablations)")
    p.add_argument("--gate-sigma", default="0.85799",
                   help="Global-only above this sigma; one value or one per higher stage (inactive below the stage "
                   "start, as in FLUX.1 progressive SCA)")
    p.add_argument("--n0", type=int, default=83)
    p.add_argument("--bridge-tokens", type=int, default=0, help="Bridge size per stage: this many tokens over the "
                   "largest cell count <= 8 dividing the stage grid (1024: 2 x 256 at 2656^2, 4 x 64 at 5312^2); "
                   "0 = --bridge-cells / --bridge-k at every stage")
    dedup.add_args(p)
    p.add_argument("--tile", type=int, default=8)
    p.add_argument("--window", type=int)
    p.add_argument("--pool", choices=("center", "mean"), default="center")
    p.add_argument("--chunk", type=int)
    p.add_argument("--local-impl", choices=("flex", "gather"), default="flex")
    p.add_argument("--embedding-cache", type=Path, help="Prompt-embedding cache directory (default sca/qwen.py)")
    p.add_argument("--device", default="cuda")
    return p


def make_plan(args):
    stages = [resolution(x) for x in args.stages.split(",")]
    cfg = sp.floats(args.true_cfg)
    if len(cfg) == 1:
        cfg = cfg * len(stages)
    return sp.Plan(stages=stages, base_steps=args.base_steps, highres_steps=sp.ints(args.highres_steps), guidance=cfg,
                   alphas=sp.floats(args.alphas), betas=sp.floats(args.betas), filter_ratio=sp.floats(args.filter_ratio),
                   sharpen_kernel=sp.HRDIT["sharpen_kernel"], sharpen_sigma=tuple(sp.floats(sp.HRDIT["sharpen_sigma"])),
                   sharpen_alpha=sp.HRDIT["sharpen_alpha"], butterworth_order=sp.HRDIT["butterworth_order"])


def bridge_for(grid, tokens, max_cells=8):
    """(cells, per_cell) for a bridge of `tokens` tokens: the largest cell count <= max_cells that divides both grid
    axes and splits the tokens evenly, with no more tokens per cell than the cell holds."""
    ok = [c for c in range(1, max_cells + 1) if grid[0] % c == 0 and grid[1] % c == 0 and tokens % (c * c) == 0
          and tokens // (c * c) <= (grid[0] // c) * (grid[1] // c)]
    if not ok:
        raise ValueError(f"no cell layout for {tokens} bridge tokens on a {grid} grid")
    return ok[-1], tokens // (ok[-1] ** 2)


def build_contexts(args, plan):
    """One QwenSCAContext per stage above native (index 0: None, stock attention)."""
    from sca.attn import PiTable, pool_factors
    from sca.qwen import QwenSCAContext
    gates = sp.gate_values(args, len(plan.stages) - 1)
    contexts, info = [None], []
    for s, (h, w) in enumerate(plan.stages[1:], start=1):
        grid = (h // 16, w // 16)
        target = max(pool_factors(grid, args.n0))
        if args.pi_const is not None:
            pi = PiTable("const", value=args.pi_const)
        else:
            pi = PiTable.load(args.pi_table, "head", None, None, target)
        kw = dedup.context_kwargs(args)
        if args.bridge_tokens:
            kw["bridge_cells"], kw["bridge_k"] = bridge_for(grid, args.bridge_tokens)
        contexts.append(QwenSCAContext(mode="sca", known_grid=grid, n0=args.n0, tile=args.tile, window=args.window,
                                       pool=args.pool, chunk=args.chunk, local_impl=args.local_impl, pi=pi,
                                       gate_sigma=gates[s - 1], **kw))
        d = {"res": f"{h}x{w}", "n0": args.n0, "tile": args.tile, "window": args.window or args.n0, "pool": args.pool,
             "chunk": args.chunk, "gate_sigma": gates[s - 1], "pool_factors": list(pool_factors(grid, args.n0)),
             "pi_variant": pi.variant, "pi_const": args.pi_const, "pi_scale": getattr(pi, "scale", None),
             **dedup.info(args)}
        if args.bridge_tokens:
            d.update(bridge_cells=kw["bridge_cells"], bridge_k=kw["bridge_k"])
        info.append(d)
    table = file_sha256(args.pi_table) if args.pi_const is None else None
    return contexts, {"stages": info, "pi_table_sha256": table, "native_attention": "stock",
                      "bridge_tokens": args.bridge_tokens or None}


def describe(plan):
    d = plan.describe()
    d["true_cfg"] = d.pop("guidance")
    d["structure_guidance_space"] = "Qwen-Image VAE latents (16 channels), unpacked per stage"
    d["schedule"] = "Qwen-Image scheduler at the native stage's sequence (mu native), later stages run its tail"
    return d


def main(argv=None):
    args = parser().parse_args(argv)
    runtime_environment()
    plan = make_plan(args)
    prompts = read_prompts(args.prompts)
    end = len(prompts) if args.end is None else args.end
    if not 0 <= args.start < end <= len(prompts):
        raise ValueError("Require 0 <= start < end <= number of prompts")
    if args.pi_const is None and (args.pi_table is None or not args.pi_table.is_file()):
        raise SystemExit("--pi-table must name an existing npz unless --pi-const is given")
    if args.anchor_lo:
        raise SystemExit("--anchor-* works in scripts/sample.py (FLUX.1) only")
    rows = prompts[args.start:end]
    from sample_qwen import prompt_suffix
    suffix = prompt_suffix(args.prompt_suffix)
    cond_text = {r["id"]: r["prompt"] + suffix for r in rows}
    import diffusers
    import torch
    from sca.qwen import REPO, encode_prompts, free_cuda, load_pipeline, snapshot
    cache = {"cache": args.embedding_cache} if args.embedding_cache is not None else {}
    cfg_on = any(g > 1 for g in plan.guidance)
    if cfg_on and not all(g > 1 for g in plan.guidance):
        raise SystemExit("--true-cfg: CFG on in every stage or in none")
    contexts, info = build_contexts(args, plan)

    def paths(row):
        stem = image_stem(row["id"], args.seed)
        return args.out / f"{stem}.png", args.out / f"{stem}.json"

    def done(row):
        png, meta = paths(row)
        return png.exists() and meta.exists() and json.loads(meta.read_text()).get("complete")
    todo = [r for r in rows if not done(r)]
    enc = encode_prompts([cond_text[r["id"]] for r in todo] + ([args.negative] if cfg_on else []), args.device,
                         **cache)
    neg = enc.pop() if cfg_on else None
    embeds = dict(zip([r["id"] for r in todo], enc))
    free_cuda()
    pipe = load_pipeline(args.device)
    pipe.vae.enable_tiling()
    revision, impl = code_revision(), implementation_hash()
    cuda = str(args.device).startswith("cuda")
    sync = torch.cuda.synchronize if cuda else None
    h, w = plan.stages[-1]
    for row in rows:
        png, meta = paths(row)
        config = {"schema_version": 1, "method": args.label, "res": f"{h}x{w}", "prompt_id": str(row["id"]),
                  "prompt": row["prompt"], "seed": int(args.seed), "derived_seed": image_seed(row["id"], args.seed),
                  "steps": plan.total_steps(), "true_cfg_scale": list(plan.guidance),
                  "negative_prompt": args.negative if cfg_on else None, "model": REPO, "snapshot": snapshot().name,
                  "dtype": "bfloat16", "batch_size": 1, "attention": "stock@native+sca@highres",
                  "progressive": describe(plan), "mu_mode": "native",
                  "global_torch_seed": "derived_seed (blur sigma RNG, as in the HRDiT runner)",
                  "diffusers": diffusers.__version__, "torch": torch.__version__, "implementation_sha256": impl,
                  "grid": [h // 16, w // 16], "sca": info, "scheduler": dict(pipe.scheduler.config),
                  "text_encoder": "Qwen2.5-VL last hidden state, template tokens dropped, precomputed"}
        if suffix:
            config.update(prompt_suffix=args.prompt_suffix, conditioning_prompt=cond_text[row["id"]])
        digest = fingerprint(config)
        if png.exists() and meta.exists():
            saved = json.loads(meta.read_text())
            if saved.get("complete"):
                if saved.get("config_fingerprint") != digest:
                    raise ValueError(f"Existing result has different settings: {meta}. Choose a new --out.")
                if file_sha256(png) != saved["png_sha256"]:
                    raise ValueError(f"Existing PNG checksum differs: {png}")
                print(f"skip {row['id']} {args.label} {config['res']}", flush=True)
                continue
        for ctx in contexts[1:]:
            ctx.pi_trace.clear()
            ctx.log.clear()
        derived = image_seed(row["id"], args.seed)
        torch.manual_seed(derived)  # global RNG: GaussianBlur sigma, as in the HRDiT runner
        if cuda:
            torch.cuda.manual_seed_all(derived)
        generator = torch.Generator(device="cpu").manual_seed(derived)
        record = {}
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            tensor = generate(pipe, embeds[row["id"]].to(args.device), None if neg is None else neg.to(args.device),
                              generator, plan, contexts, sync, record)
            image = pipe.image_processor.postprocess(tensor, output_type="pil")[0]
        if cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        args.out.mkdir(parents=True, exist_ok=True)
        tmp = png.with_name(png.name + f".tmp.{os.getpid()}")
        image.save(tmp, format="PNG")
        os.replace(tmp, png)
        trace = [dict(t, stage=s) for s, ctx in enumerate(contexts) if ctx is not None for t in ctx.pi_trace]
        gate_steps = [sum(1 for x in sig if x > ctx.gate_sigma) for sig, ctx in
                      zip(record["stage_sigmas"][1:], contexts[1:])]
        atomic_json(meta, {"complete": True, "config": config, "config_fingerprint": digest,
                           "code_revision": revision, "image": str(png.resolve()), "png_sha256": file_sha256(png),
                           "seconds": seconds, "seconds_scope": "all stages (two transformer calls per step with true "
                           "CFG), stage transitions (VAE decode, upsample, sharpen, VAE encode, re-noise) and the final "
                           "VAE decode; prompt embeddings precomputed (text encoder not resident); excludes model "
                           "loading and PNG I/O",
                           "stage_seconds": record["stage_seconds"],
                           "stage_seconds_scope": "denoising steps of each stage (index 0 = native)",
                           "transition_seconds": record["transition_seconds"],
                           "decode_seconds": record["decode_seconds"],
                           "peak_memory_bytes": torch.cuda.max_memory_allocated() if cuda else None,
                           "mu": record["mu"], "sigmas": record["sigmas"][:-1], "stage_sigmas": record["stage_sigmas"],
                           "stage_shapes": record["stage_shapes"], "blur_sigmas": record["blur_sigmas"],
                           "gate_active_steps": gate_steps, "pi_trace": trace,
                           "sca_log": sorted({x for c in contexts[1:] for x in c.log}), "local_impl": args.local_impl})
        print(f"saved {row['id']} {args.label} {config['res']} seconds={seconds:.3f} "
              f"stages={[round(x, 2) for x in record['stage_seconds']]} "
              f"transitions={[round(x, 2) for x in record['transition_seconds']]} "
              f"peak_gb={(torch.cuda.max_memory_allocated() / 1e9) if cuda else 0:.1f}", flush=True)


if __name__ == "__main__":
    main()
