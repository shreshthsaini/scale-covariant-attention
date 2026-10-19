#!/usr/bin/env python3
"""Progressive SCA: HRDiT's released 1K -> 2K -> 4K schedule with only the attention operator swapped.

The stage schedule, upsampler, re-noising, structure guidance, guidance per stage, scheduler settings and
seeding follow the frozen HRDiT runner (profile hrdit_dev_author) and its vendored pipeline
(/scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/vendor/hrdit/hrdit/pipeline.py, __call__
and flowmatch_step). The native first stage uses FLUX.1-dev's stock attention processors. Every stage above
native uses SCA instead of HRDiT's NTK-scaled RoPE, proportional attention scale, head-adaptive pruning (HAP)
and spatial position alignment (SPA). Output PNG + JSON sidecar per prompt with the stems, seeds and fields
of scripts/sample.py, so the existing evaluators read --out unchanged.
"""

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sca.runtime import (atomic_json, code_revision, file_sha256, fingerprint, image_seed, image_stem,  # noqa: E402
                         implementation_hash, read_prompts, resolution, runtime_environment)

# HRDiT released protocol as run here (frozen run.py, profile hrdit_dev_author).
HRDIT = dict(stages="1024x1024,2048x2048,4096x4096", base_steps=30, highres_steps="17,10", guidance="3.5,4.5,6.0",
             alphas="1.0,0.25", betas="0.5,0.5", filter_ratio="0.2,0.2", sharpen_kernel=3,
             sharpen_sigma="0.1,2.0", sharpen_alpha=1.0, butterworth_order=4)


def floats(text):
    return [float(x) for x in str(text).split(",") if x.strip()]


def ints(text):
    return [int(x) for x in str(text).split(",") if x.strip()]


@dataclass
class Plan:
    """Stage schedule. stages[0] is the native stage; stage s >= 1 runs the last highres_steps[s-1]
    timesteps of the shared base_steps schedule (HRDiT dlfg_timesteps)."""
    stages: list
    base_steps: int = 30
    highres_steps: list = field(default_factory=lambda: [17, 10])
    guidance: list = field(default_factory=lambda: [3.5, 4.5, 6.0])
    alphas: list = field(default_factory=lambda: [1.0, 0.25])
    betas: list = field(default_factory=lambda: [0.5, 0.5])
    filter_ratio: list = field(default_factory=lambda: [0.2, 0.2])
    sharpen_kernel: int = 3
    sharpen_sigma: tuple = (0.1, 2.0)
    sharpen_alpha: float = 1.0
    butterworth_order: int = 4

    def __post_init__(self):
        n = len(self.stages) - 1
        if n < 1:
            raise ValueError("Need a native stage and at least one higher-resolution stage")
        for name in ("highres_steps", "alphas", "betas", "filter_ratio"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} needs one value per higher-resolution stage ({n})")
        if len(self.guidance) != n + 1:
            raise ValueError("guidance needs one value per stage")
        if any(not 0 < k <= self.base_steps for k in self.highres_steps) or self.base_steps <= 0:
            raise ValueError("Each higher-resolution stage needs 1..base_steps steps")

    def start_index(self, s):
        return 0 if s == 0 else self.base_steps - self.highres_steps[s - 1]

    def total_steps(self):
        return self.base_steps + sum(self.highres_steps)

    def describe(self):
        return {"stages": [f"{h}x{w}" for h, w in self.stages], "base_steps": self.base_steps,
                "highres_steps": list(self.highres_steps), "guidance": list(self.guidance),
                "structure_guidance": "fft", "upsampling_choice": "latent", "flow_choice": "v_theta",
                "alphas": list(self.alphas), "betas": list(self.betas), "filter_ratio": list(self.filter_ratio),
                "butterworth_order": self.butterworth_order, "sharpen_kernel": self.sharpen_kernel,
                "sharpen_sigma": list(self.sharpen_sigma), "sharpen_alpha": self.sharpen_alpha,
                "stage_upsampler": "vae_decode, bicubic pixel upsample, gaussian-blur unsharp mask, vae_encode mode",
                "renoise": "sigma*noise + (1-sigma)*x at the stage start sigma, noise from the per-prompt generator"}


class Clock:
    """Stands in for the scheduler in sca.attn.install's pre-hook: exact sigma at the shared step index."""

    def __init__(self, sigmas):
        self.sigmas, self.step_index = sigmas, None


def butterworth(shape, ratio, order=4, device=None):
    """Vectorized hrdit.attention.butterworth_low_pass_filter_2d (same float64 arithmetic, float32 mask)."""
    import torch
    H, W = shape[-2], shape[-1]
    if ratio == 0:
        return torch.zeros(shape, device=device)
    h = 2 * torch.arange(H, dtype=torch.float64) / H - 1
    w = 2 * torch.arange(W, dtype=torch.float64) / W - 1
    d = h[:, None] ** 2 + w[None, :] ** 2
    mask = 1 / (1 + (d / ratio ** 2) ** order)
    return mask.to(torch.float32).expand(shape).contiguous().to(device)


def lowpass(x, mask):
    """hrdit.attention.split_frequency_components_fft(x, mask, is_low=True)."""
    import torch.fft as fft
    f = fft.fftshift(fft.fft2(x.to(mask.dtype)))
    return fft.ifft2(fft.ifftshift(f * mask)).real


def latent_hw(pipe, h, w):
    s = pipe.vae_scale_factor
    return 2 * (h // (s * 2)), 2 * (w // (s * 2))


def euler(sample, v, sigma, sigma_next):
    """hrdit flowmatch_step without guidance (native stage): float32 update, cast back to v's dtype."""
    x = sample.float()
    return (x + (sigma_next - sigma) * v).to(v.dtype), x - v * sigma


def guided_step(pipe, sample, v, sigma, sigma_next, ref, hw, mask, alpha, beta, state):
    """hrdit flowmatch_step with flow_choice=v_theta, upsampling_choice=latent, structure_guidance=fft.
    ref = (packed pred_x0 of the previous stage at this timestep, its pixel (h, w))."""
    import torch.nn.functional as F
    B, _, D = sample.shape
    C = D // 4
    h, w = hw
    lh, lw = latent_hw(pipe, h, w)
    x = sample.float()
    x0 = x - v * sigma
    original = x0
    x0u = pipe._unpack_latents(x0, h, w, pipe.vae_scale_factor)
    prev = pipe._unpack_latents(ref[0], ref[1][0], ref[1][1], pipe.vae_scale_factor)
    refu = F.interpolate(prev, (lh, lw), mode="bicubic", align_corners=False)
    x0u = x0u + alpha * (lowpass(refu, mask) - lowpass(x0u, mask))
    x0 = pipe._pack_latents(x0u, B, C, lh, lw)
    refp = pipe._pack_latents(refu, B, C, lh, lw)
    out = (x - x0) / (sigma + 1e-6)
    out_ref = (x - refp) / (sigma + 1e-6)
    if state.get("high") is not None:
        out = out + beta * (state["high"] + out_ref - state["ref"] - out)
    state["high"], state["ref"] = out, out_ref
    return (x + (sigma_next - sigma) * out).to(out.dtype), original


def sharpen(image, kernel, sigma_range, alpha):
    """hrdit gaussian_blur_image_sharpening; draws the blur sigma from the global Torch RNG exactly as
    torchvision GaussianBlur.forward does, and returns it for the sidecar."""
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    s = T.GaussianBlur.get_params(*sigma_range)
    blurred = TF.gaussian_blur(image, [kernel, kernel], [s, s])
    return (alpha + 1) * image - alpha * blurred, float(s)


def decode(pipe, latents, h, w):
    x = pipe._unpack_latents(latents, h, w, pipe.vae_scale_factor)
    x = x / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    return pipe.vae.decode(x.to(pipe.vae.dtype), return_dict=False)[0]


def encode(pipe, image, h, w, ids_dtype):
    lh, lw = latent_hw(pipe, h, w)
    z = pipe.vae.encode(image.to(pipe.vae.dtype)).latent_dist.mode()
    z = (z - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    packed = pipe._pack_latents(z, z.shape[0], z.shape[1], lh, lw)
    ids = pipe._prepare_latent_image_ids(z.shape[0], lh // 2, lw // 2, z.device, ids_dtype)
    return packed, ids


def hrdit_schedule(pipe, base_steps, base_hw, device):
    """HRDiT's shared schedule. Its pipeline passes sigmas=linspace(1, 1/N, N) and mu from the 1K sequence
    length; inference.py assigns scheduler.use_dynamic_shifting=False as an instance attribute, but
    set_timesteps reads config.use_dynamic_shifting (True for FLUX.1-dev), so the effective schedule is the
    dynamic exponential shift with mu(1K) = 1.15. The recorded HRDiT sidecar sigmas confirm this."""
    import numpy as np
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift
    cfg = pipe.scheduler.config
    lh, lw = latent_hw(pipe, *base_hw)
    mu = calculate_shift((lh // 2) * (lw // 2), cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                         cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
    pipe.scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / base_steps, base_steps), device=device, mu=mu)
    return pipe.scheduler.timesteps.clone(), pipe.scheduler.sigmas.clone(), float(mu)


def generate(pipe, embeds, generator, plan, contexts, native="stock", sync=None, record=None):
    """Run every stage and return the decoded image tensor in [-1, 1] (before postprocessing).

    embeds = (prompt_embeds, pooled_prompt_embeds, text_ids). contexts[s] is the SCAContext for stage s >= 1
    (contexts[0] is used only with native='sca_full'). The caller seeds the global Torch RNG (blur sigma)."""
    import torch
    from diffusers.utils.torch_utils import randn_tensor
    from sca.attn import install
    record = {} if record is None else record
    sync = sync or (lambda: None)
    prompt_embeds, pooled, text_ids = embeds
    tr = pipe.transformer
    device, dtype = prompt_embeds.device, tr.dtype
    timesteps, sigmas, mu = hrdit_schedule(pipe, plan.base_steps, plan.stages[0], device)
    record.update(mu=mu, sigmas=[float(s) for s in sigmas], stage_seconds=[], transition_seconds=[],
                  blur_sigmas=[], stage_shapes=[], stage_sigmas=[])
    clock = Clock(sigmas)
    C = tr.config.in_channels // 4
    B = prompt_embeds.shape[0]
    h, w = plan.stages[0]
    latents, ids = pipe.prepare_latents(B, C, h, w, prompt_embeds.dtype, device, generator)
    pred_x0 = {}
    for s, (h, w) in enumerate(plan.stages):
        sync()
        t0 = time.perf_counter()
        if s > 0:  # HRDiT stage transition: decode, bicubic upsample, sharpen, encode, re-noise
            ph, pw = plan.stages[s - 1]
            image = decode(pipe, latents, ph, pw)
            image = torch.nn.functional.interpolate(image, (h, w), mode="bicubic", align_corners=False)
            image, blur = sharpen(image, plan.sharpen_kernel, plan.sharpen_sigma, plan.sharpen_alpha)
            record["blur_sigmas"].append(blur)
            latents, ids = encode(pipe, image, h, w, prompt_embeds.dtype)
            k = plan.start_index(s)
            noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
            sig = sigmas.to(device=latents.device, dtype=latents.dtype)[k]  # scheduler.scale_noise arithmetic
            latents = (sig * noise + (1.0 - sig) * latents).to(dtype)
            sync()
            record["transition_seconds"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
        start = plan.start_index(s)
        lh, lw = latent_hw(pipe, h, w)
        record["stage_shapes"].append({"res": f"{h}x{w}", "latents": list(latents.shape), "grid": [lh // 2, lw // 2]})
        record["stage_sigmas"].append([float(x) for x in sigmas[start:plan.base_steps]])
        ctx = contexts[s] if (s > 0 or native == "sca_full") else None
        uninstall = None
        if ctx is not None:
            ctx.known_grid = (lh // 2, lw // 2)
            ctx.scheduler = clock
            uninstall = install(tr, ctx)
        g = plan.guidance[s]
        guidance = torch.full([B], g, device=device, dtype=torch.float32) if tr.config.guidance_embeds else None
        mask = None
        if s > 0:
            mask = butterworth([B, C, lh, lw], plan.filter_ratio[s - 1], plan.butterworth_order, device)
        state = {}
        n = plan.base_steps - start
        try:
            for i in range(n):
                j = start + i
                clock.step_index = j
                t = timesteps[j]
                v = tr(hidden_states=latents, timestep=t.expand(B).to(latents.dtype) / 1000, guidance=guidance,
                       pooled_projections=pooled, encoder_hidden_states=prompt_embeds, txt_ids=text_ids,
                       img_ids=ids, joint_attention_kwargs=None, return_dict=False)[0]
                if s == 0:
                    latents, x0 = euler(latents, v, sigmas[j], sigmas[j + 1])
                else:
                    a = plan.alphas[s - 1] * (n - i) / n
                    b = plan.betas[s - 1] * (n - i) / n
                    latents, x0 = guided_step(pipe, latents, v, sigmas[j], sigmas[j + 1], pred_x0[j], (h, w),
                                              mask, a, b, state)
                    latents = latents.to(dtype)
                pred_x0[j] = (x0, (h, w))
        finally:
            if uninstall is not None:
                uninstall()
        if not torch.isfinite(latents).all():
            raise FloatingPointError(f"Non-finite latents at the end of stage {s} ({h}x{w})")
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
    p.add_argument("--label", default="sca_prog")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stages", default=HRDIT["stages"], help="Comma-separated HxW per stage, native first")
    p.add_argument("--base-steps", type=int, default=HRDIT["base_steps"])
    p.add_argument("--highres-steps", default=HRDIT["highres_steps"])
    p.add_argument("--guidance", default=HRDIT["guidance"], help="Per-stage distilled guidance")
    p.add_argument("--alphas", default=HRDIT["alphas"])
    p.add_argument("--betas", default=HRDIT["betas"])
    p.add_argument("--filter-ratio", default=HRDIT["filter_ratio"])
    p.add_argument("--native-attention", choices=("stock", "sca_full"), default="stock",
                   help="stock = FLUX.1-dev processors untouched at the native stage")
    p.add_argument("--pi-table", type=Path, help="npz from scripts/calibrate.py (required unless --pi-const)")
    p.add_argument("--pi-const", type=float, help="Constant pi (tests and ablations)")
    p.add_argument("--gate-sigma", default="0.89",
                   help="Global-only above this sigma; one value, or one per higher-resolution stage")
    p.add_argument("--n0", type=int, default=64)
    p.add_argument("--tile", type=int, default=8)
    p.add_argument("--window", type=int)
    p.add_argument("--pool", choices=("center", "mean"), default="center")
    p.add_argument("--chunk", type=int)
    p.add_argument("--local-impl", choices=("flex", "gather"), default="flex")
    p.add_argument("--model", choices=("dev",), default="dev")
    p.add_argument("--device", default="cuda")
    p.add_argument("--wandb", action="store_true", help="Log per-image seconds (no images) to WandB")
    p.add_argument("--check", action="store_true",
                   help="Verify completed outputs for the slice (no model load); nonzero exit if any is missing")
    return p


def make_plan(args):
    stages = [resolution(x) for x in args.stages.split(",")]
    return Plan(stages=stages, base_steps=args.base_steps, highres_steps=ints(args.highres_steps),
                guidance=floats(args.guidance), alphas=floats(args.alphas), betas=floats(args.betas),
                filter_ratio=floats(args.filter_ratio), sharpen_kernel=HRDIT["sharpen_kernel"],
                sharpen_sigma=tuple(floats(HRDIT["sharpen_sigma"])), sharpen_alpha=HRDIT["sharpen_alpha"],
                butterworth_order=HRDIT["butterworth_order"])


def gate_values(args, n):
    g = floats(args.gate_sigma)
    if len(g) == 1:
        g = g * n
    if len(g) != n:
        raise ValueError("--gate-sigma needs one value or one per higher-resolution stage")
    return g


def build_contexts(args, plan, vae_scale_factor=8):
    """One SCAContext per stage above native; pi from the table entry nearest the stage pool factor
    (scale 2 at 2K, scale 4 at 4K with n0=64). Index 0 is a full-attention context or None."""
    from sca.attn import PiTable, SCAContext, pool_factors
    gates = gate_values(args, len(plan.stages) - 1)
    contexts = [SCAContext(mode="full", known_grid=None) if args.native_attention == "sca_full" else None]
    info = []
    for s, (h, w) in enumerate(plan.stages[1:], start=1):
        grid = (h // (2 * vae_scale_factor), w // (2 * vae_scale_factor))
        target = max(pool_factors(grid, args.n0))
        if args.pi_const is not None:
            pi = PiTable("const", value=args.pi_const)
        elif args.pi_table is None:
            raise SystemExit("--pi-table is required unless --pi-const is given")
        else:
            pi = PiTable.load(args.pi_table, "head", None, None, target)
        contexts.append(SCAContext(mode="sca", known_grid=grid, n0=args.n0, tile=args.tile, window=args.window,
                                   pool=args.pool, chunk=args.chunk, local_impl=args.local_impl, pi=pi,
                                   gate_sigma=gates[s - 1]))
        info.append({"res": f"{h}x{w}", "n0": args.n0, "tile": args.tile, "window": args.window or args.n0,
                     "pool": args.pool, "chunk": args.chunk, "gate_sigma": gates[s - 1], "band": 0, "route": 0.0,
                     "pool_factors": list(pool_factors(grid, args.n0)), "pi_variant": pi.variant,
                     "pi_const": args.pi_const, "pi_scale": getattr(pi, "scale", None)})
    table = file_sha256(args.pi_table) if args.pi_const is None else None
    return contexts, {"stages": info, "pi_table_sha256": table, "native_attention": args.native_attention}


def make_config(args, plan, row, info, versions, scheduler_config, impl):
    h, w = plan.stages[-1]
    return {"schema_version": 1, "method": args.label, "res": f"{h}x{w}", "prompt_id": str(row["id"]),
            "prompt": row["prompt"], "seed": int(args.seed), "derived_seed": image_seed(row["id"], args.seed),
            "steps": plan.total_steps(), "guidance": list(plan.guidance), "model": "black-forest-labs/FLUX.1-dev",
            "dtype": "bfloat16", "batch_size": 1, "attention": f"{args.native_attention}@native+sca@highres",
            "progressive": plan.describe(), "schedule": "hrdit_effective_dynamic_shift_mu_from_native_stage",
            "global_torch_seed": "derived_seed (blur sigma RNG, as in the HRDiT runner)",
            "diffusers": versions[0], "torch": versions[1], "implementation_sha256": impl,
            "sca": info, "scheduler": scheduler_config}


def check_outputs(args, plan, rows):
    """Completion barrier used by the smoke and evaluation tasks; loads no model."""
    from PIL import Image
    h, w = plan.stages[-1]
    bad = []
    for row in rows:
        stem = image_stem(row["id"], args.seed)
        png, meta = args.out / f"{stem}.png", args.out / f"{stem}.json"
        try:
            saved = json.loads(meta.read_text())
            cfg = saved["config"]
            want = {"method": args.label, "res": f"{h}x{w}", "prompt_id": str(row["id"]), "prompt": row["prompt"],
                    "seed": int(args.seed), "steps": plan.total_steps(), "guidance": list(plan.guidance)}
            if saved.get("complete") is not True or any(cfg.get(k) != v for k, v in want.items()):
                raise ValueError("incomplete or different identity")
            if cfg["progressive"]["stages"] != plan.describe()["stages"]:
                raise ValueError("different stage schedule")
            if file_sha256(png) != saved["png_sha256"]:
                raise ValueError("PNG checksum differs")
            with Image.open(png) as im:
                if im.size != (w, h):
                    raise ValueError(f"PNG size {im.size}")
                im.verify()
            sec = saved["seconds"]
            if not (isinstance(sec, (int, float)) and math.isfinite(sec) and sec > 0):
                raise ValueError("invalid seconds")
            if len(saved["stage_seconds"]) != len(plan.stages):
                raise ValueError("missing per-stage timing")
        except (OSError, KeyError, ValueError, TypeError) as err:
            bad.append(f"{row['id']}: {err}")
    return bad


def load(args):
    from sca.runtime import load_pipeline
    pipe = load_pipeline(args.device, args.model)
    pipe.vae.enable_tiling()  # as in the HRDiT runner and sample.py at 4K
    return pipe


def encode_text(pipe, prompt, device):
    return pipe.encode_prompt(prompt=prompt, prompt_2=None, device=device, num_images_per_prompt=1,
                              max_sequence_length=512)


def main(argv=None):
    args = parser().parse_args(argv)
    runtime_environment()
    plan = make_plan(args)
    prompts = read_prompts(args.prompts)
    end = len(prompts) if args.end is None else args.end
    if not 0 <= args.start < end <= len(prompts):
        raise ValueError("Require 0 <= start < end <= number of prompts")
    rows = prompts[args.start:end]
    if args.check:
        bad = check_outputs(args, plan, rows)
        for line in bad:
            print("MISSING " + line, flush=True)
        print(f"check {len(rows) - len(bad)}/{len(rows)} complete under {args.out}", flush=True)
        if bad:
            raise SystemExit(1)
        return
    if args.pi_const is None and (args.pi_table is None or not args.pi_table.is_file()):
        raise SystemExit("--pi-table must name an existing npz unless --pi-const is given")
    gate_values(args, len(plan.stages) - 1)
    import diffusers
    import torch
    pipe = load(args)
    contexts, info = build_contexts(args, plan, pipe.vae_scale_factor)
    revision, impl = code_revision(), implementation_hash()
    cuda = str(args.device).startswith("cuda")
    sync = torch.cuda.synchronize if cuda else None
    tracker = None
    if args.wandb:
        import wandb
        tracker = wandb.init(project="scale-covariant-attention", group="progressive-sca", job_type="generation",
                             dir=os.environ.get("WANDB_DIR"), config={"label": args.label, "start": args.start,
                                                                       "end": end, **plan.describe()})
    try:
        for row in rows:
            stem = image_stem(row["id"], args.seed)
            png, meta = args.out / f"{stem}.png", args.out / f"{stem}.json"
            config = make_config(args, plan, row, info, (diffusers.__version__, torch.__version__),
                                 dict(pipe.scheduler.config), impl)
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
            for ctx in contexts:
                if ctx is not None:
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
                embeds = encode_text(pipe, row["prompt"], pipe._execution_device)
                if cuda:
                    torch.cuda.synchronize()
                text_seconds = time.perf_counter() - start
                tensor = generate(pipe, embeds, generator, plan, contexts, args.native_attention, sync, record)
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
                               "seconds": seconds, "seconds_scope": "prompt encoding, all stages, stage transitions "
                               "(VAE decode, upsample, sharpen, VAE encode, re-noise) and final VAE decode; CUDA "
                               "synchronized; excludes model loading and PNG I/O",
                               "text_seconds": text_seconds, "stage_seconds": record["stage_seconds"],
                               "stage_seconds_scope": "denoising steps of each stage (index 0 = native)",
                               "transition_seconds": record["transition_seconds"],
                               "decode_seconds": record["decode_seconds"],
                               "peak_memory_bytes": torch.cuda.max_memory_allocated() if cuda else None,
                               "mu": record["mu"], "sigmas": record["sigmas"][:-1],
                               "stage_sigmas": record["stage_sigmas"], "stage_shapes": record["stage_shapes"],
                               "blur_sigmas": record["blur_sigmas"], "gate_active_steps": gate_steps,
                               "pi_trace": trace, "sca_log": sorted({x for c in contexts if c for x in c.log}),
                               "local_impl": args.local_impl})
            print(f"saved {row['id']} {args.label} {config['res']} seconds={seconds:.3f} "
                  f"stages={[round(x, 2) for x in record['stage_seconds']]} "
                  f"transitions={[round(x, 2) for x in record['transition_seconds']]}", flush=True)
            if tracker:
                tracker.log({"prompt_id": str(row["id"]), "seconds": seconds,
                             **{f"stage{s}_seconds": x for s, x in enumerate(record["stage_seconds"])}})
    finally:
        if tracker:
            tracker.finish()


if __name__ == "__main__":
    main()
