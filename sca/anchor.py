"""Latent low-band anchor for the late-duplication fix (.agent/notes/duplication_fix_20261027.md, round two).

Sampler-level option, off by default (scripts/sample.py --anchor-lo). FLUX flow matching: x_t = (1 - sigma) x0 +
sigma eps, the transformer predicts v, and x0_hat = x_t - sigma v. At the last global-only step (the largest step
index whose sigma > gate_sigma: step 7 of 28 at 4096^2 with gate 0.89) the hook stores the anchor L = x0_hat, the
packed latent tokens [B, H*W, C] (C = 64 for FLUX). At every later step with sigma > lo it replaces the low band of
x0_hat by L's and returns the matching velocity:

    x0' = x0_hat + w(sigma) (LP(L) - LP(x0_hat)),    v' = (x_t - x0') / sigma = v - (w / sigma) LP(L - x0_hat).

LP averages p x p tokens on the H x W token grid (all C channels) and upsamples bilinearly (sca.attn.band_low); the
default p is the global block size max(pool_factors(grid, n0)), 4 at 4096^2. The weight falls from 1 at the gate to 0
at lo: u = clip((sigma - lo) / (gate - lo), 0, 1), linear w = u, cosine w = sin^2(pi u / 2). With refresh K > 0 the
anchor is recomputed every K steps after the capture step (while sigma > lo) from one extra transformer forward in
global-only attention (SCAContext.force_global): L = x_t - sigma v_glob, so one extra global-only forward per K steps.

The hook runs after each forward that sca.attn.install's pre-hook prepared (ctx.sigma, ctx.step). Guidance-distilled
FLUX runs one forward per step with batch 1, which the hook asserts; call reset() before every image.
"""

from __future__ import annotations

import math

import torch

from sca.attn import band_low, pool_factors

SHAPES = ("linear", "cosine")


def anchor_weight(sigma, gate, lo, shape="linear"):
    """w(sigma), 1 at sigma >= gate and 0 at sigma <= lo: u or sin^2(pi u / 2), u = clip((sigma - lo) / (gate - lo), 0, 1)."""
    u = min(max((float(sigma) - lo) / (gate - lo), 0.0), 1.0)
    if shape == "linear":
        return u
    if shape == "cosine":
        return math.sin(0.5 * math.pi * u) ** 2
    raise ValueError(f"anchor shape must be one of {SHAPES}, got {shape}")


def _wide(t):
    """At least fp32 (bf16 latents are computed in fp32, fp64 test inputs stay fp64)."""
    return t.to(torch.promote_types(t.dtype, torch.float32))


def latent_low(x, grid, p):
    """Low band of packed latent tokens x [B, H*W, C]: p x p token average on the (H, W) grid, bilinear upsample."""
    return band_low(x[..., None], grid, p)[..., 0]


def anchor_velocity(x_t, v, sigma, anchor, grid, p, w):
    """v' = (x_t - x0') / sigma for x0' = x0_hat + w (LP(anchor) - LP(x0_hat)), x0_hat = x_t - sigma v.

    Evaluated as v - (w / sigma) LP(anchor - x0_hat) (LP is linear: the same quantity without the cancellation in
    x_t - x0'), in at least fp32 and cast back to v's dtype. w = 0 returns v itself."""
    if w == 0:
        return v
    x0 = _wide(x_t) - sigma * _wide(v)
    return (_wide(v) - (w / sigma) * latent_low(_wide(anchor) - x0, grid, p)).to(v.dtype)


def _sample(output):
    return output[0] if isinstance(output, tuple) else output.sample


def _with_sample(output, v):
    return (v,) + tuple(output[1:]) if isinstance(output, tuple) else type(output)(sample=v)


class LatentAnchor:
    """Transformer forward hook (with_kwargs) applying the anchor for one SCAContext; see the module docstring."""

    def __init__(self, ctx, lo, shape="linear", pool=0, refresh=0):
        if ctx.mode != "sca" or not 0 < lo < ctx.gate_sigma:
            raise ValueError("the latent anchor needs SCA with a global-only opening phase: 0 < anchor_lo < gate_sigma")
        if shape not in SHAPES:
            raise ValueError(f"anchor shape must be one of {SHAPES}, got {shape}")
        if pool < 0 or refresh < 0:
            raise ValueError("anchor pool and refresh must be >= 0")
        self.ctx, self.lo, self.shape, self.pool, self.refresh = ctx, float(lo), shape, int(pool), int(refresh)
        self._busy = False
        self.reset()

    def reset(self):
        """Forget the previous image; call before every image."""
        self.anchor = None          # L, [B, H*W, C] in at least fp32
        self.capture_step = None    # step L was taken at: the last global-only step
        self.p = None               # LP pool size in tokens, resolved from the grid at the first anchored step
        self.last_step = -1
        self.refresh_steps = []
        self.applied = []           # [step, sigma, w] of every anchored step

    def weight(self, sigma):
        return anchor_weight(sigma, self.ctx.gate_sigma, self.lo, self.shape)

    def install(self, transformer):
        """Register on the transformer (after sca.attn.install); returns the remover."""
        return transformer.register_forward_hook(self, with_kwargs=True).remove

    def trace(self):
        """Sidecar record of the last image."""
        return {"capture_step": self.capture_step, "pool_tokens": self.p, "refresh_steps": list(self.refresh_steps),
                "applied": [list(a) for a in self.applied]}

    def global_velocity(self, module, args, kwargs):
        """v_glob from one extra transformer forward in global-only attention (the opening-phase path)."""
        self._busy, self.ctx.force_global = True, True
        try:
            out = module(*args, **kwargs)
        finally:
            self._busy, self.ctx.force_global = False, False
        return _sample(out)

    def __call__(self, module, args, kwargs, output):
        if self._busy:  # the extra global-only forward of a refresh
            return None
        ctx = self.ctx
        if ctx.scheduler is None:
            raise RuntimeError("latent anchor needs ctx.scheduler (sigma and step index of every forward)")
        x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        v = _sample(output)
        if x.shape[0] != 1:
            raise AssertionError(f"latent anchor expects batch 1 (one guidance-distilled forward per step), got {x.shape[0]}")
        if ctx.step != self.last_step + 1:
            raise AssertionError(f"latent anchor expects one forward per step, in order: step {ctx.step} after "
                                 f"{self.last_step} (call reset() before every image)")
        self.last_step = ctx.step
        sigma, grid = float(ctx.sigma), ctx.grid
        if x.shape[1] != grid[0] * grid[1] or v.shape != x.shape:
            raise AssertionError(f"packed latents {tuple(x.shape)} / {tuple(v.shape)} do not match grid {grid}")
        if sigma > ctx.gate_sigma:  # global-only phase: keep x0_hat; the last such step's is the anchor
            self.anchor, self.capture_step = _wide(x) - sigma * _wide(v), ctx.step
            return None
        if sigma <= self.lo:
            return None
        if self.anchor is None:
            raise RuntimeError("latent anchor: no global-only step before the anchored steps")
        if self.p is None:
            self.p = self.pool or max(pool_factors(grid, ctx.n0))
        if self.refresh and (ctx.step - self.capture_step) % self.refresh == 0:
            self.anchor = _wide(x) - sigma * _wide(self.global_velocity(module, args, kwargs))
            self.refresh_steps.append(ctx.step)
        w = self.weight(sigma)
        self.applied.append([ctx.step, sigma, w])
        return _with_sample(output, anchor_velocity(x, v, sigma, self.anchor, grid, self.p, w))
