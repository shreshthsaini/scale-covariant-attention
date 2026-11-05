"""Native opening (round 6 candidate d; .agent/notes/duplication_fix_round6.md).

The first K steps run at the native canvas (1024^2 for FLUX.1-dev) with full attention, on the same sigma schedule
as the target (mu = max shift at both sizes). The x0 estimate of the last native step is upsampled (bicubic, latent
space) to the target and renoised at the next sigma with the target run's own initial noise:
    x_K = (1 - sigma_K) up(x0) + sigma_K eps_target,
then steps K .. 27 run at the target with SCA. One pass: no second schedule, no return to higher noise. K = 0 is the
plain target run (same noise, same steps). Used by scripts/sample.py --native-open K.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def upsample_packed(lat, small, big, vae_scale=8):
    """Packed FLUX latents [B, (h/2)(w/2), 4C] of an image of size `small` (H, W px) -> packed latents of `big`
    (bicubic in the unpacked [B, C, h, w] latent, fp32)."""
    from diffusers import FluxPipeline
    B = lat.shape[0]
    x = FluxPipeline._unpack_latents(lat.float(), small[0], small[1], vae_scale)        # [B, C, h, w]
    hb, wb = big[0] // vae_scale, big[1] // vae_scale
    x = F.interpolate(x, size=(hb, wb), mode="bicubic", align_corners=False)
    return FluxPipeline._pack_latents(x, B, x.shape[1], hb, wb)


def renoise(x0_up, eps, sigma):
    """Flow-matching state at sigma from a clean estimate and unit noise (same packed shape)."""
    return (1.0 - sigma) * x0_up + sigma * eps


def target_noise(pipe, h, w, seed, device, dtype):
    """The initial packed noise the target-size pipeline call would draw for this seed (FluxPipeline.prepare_latents)."""
    from diffusers.utils.torch_utils import randn_tensor
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = pipe.transformer.config.in_channels // 4
    hh, ww = 2 * (h // (pipe.vae_scale_factor * 2)), 2 * (w // (pipe.vae_scale_factor * 2))
    n = randn_tensor((1, c, hh, ww), generator=g, device=torch.device(device), dtype=dtype)
    return pipe._pack_latents(n, 1, c, hh, ww)


def generate(pipe, ctx, prompt, h, w, seed, k, steps=28, guidance=4.5, native=(1024, 1024), log=None):
    """Image for one prompt with a native opening of k steps (see module doc)."""
    if k <= 0:
        g = torch.Generator(device="cpu").manual_seed(seed)
        return pipe(prompt=prompt, height=h, width=w, num_inference_steps=steps, guidance_scale=guidance,
                    generator=g, max_sequence_length=512).images[0]
    if not 0 < k < steps:
        raise ValueError(f"native opening needs 0 < k < steps, got {k}")
    seq_cfg = pipe.scheduler.config.get("max_image_seq_len", 4096)
    mode, known = ctx.mode, ctx.known_grid
    rec = {}

    def hook(module, args, kwargs, out):
        rec["x"] = kwargs["hidden_states"].detach()
        rec["v"] = (out[0] if isinstance(out, (tuple, list)) else out.sample).detach()

    def stop(p, i, t, kw):
        if i == k - 1:
            p._interrupt = True
        return kw
    nh, nw = native
    try:
        ctx.mode, ctx.known_grid = "full", (nh // 16, nw // 16)
        ctx._img_ids = None
        pipe.scheduler.register_to_config(max_image_seq_len=(nh // 16) * (nw // 16))    # mu = max shift at native
        handle = pipe.transformer.register_forward_hook(hook, with_kwargs=True)
        try:
            g1 = torch.Generator(device="cpu").manual_seed(seed)
            pipe(prompt=prompt, height=nh, width=nw, num_inference_steps=steps, guidance_scale=guidance, generator=g1,
                 max_sequence_length=512, callback_on_step_end=stop, output_type="latent")
        finally:
            handle.remove()
        s_prev = float(pipe.scheduler.sigmas[k - 1])
        s_k = float(pipe.scheduler.sigmas[k])
        x0 = rec["x"].float() - s_prev * rec["v"].float()
    finally:
        ctx.mode, ctx.known_grid = mode, known
        ctx._img_ids = None
        pipe.scheduler.register_to_config(max_image_seq_len=seq_cfg)
    dev, dt = rec["x"].device, rec["x"].dtype
    eps = target_noise(pipe, h, w, seed, dev, dt).float()          # bit-identical to the target run's noise
    xk = renoise(upsample_packed(x0, native, (h, w), pipe.vae_scale_factor), eps, s_k).to(dt)
    base = np.linspace(1.0, 1 / steps, steps)[k:]
    if log is not None:
        log.append(f"native opening: {k} steps at {nh}x{nw}, sigma {s_prev:.4f} -> renoise at {s_k:.4f}")
    g2 = torch.Generator(device="cpu").manual_seed(seed)
    img = pipe(prompt=prompt, height=h, width=w, sigmas=list(base), num_inference_steps=steps - k, guidance_scale=guidance,
               generator=g2, latents=xk, max_sequence_length=512).images[0]
    s_first = float(pipe.scheduler.sigmas[0])
    if abs(s_first - s_k) > 1e-4:
        raise RuntimeError(f"target stage starts at sigma {s_first:.5f}, expected {s_k:.5f}")
    return img
