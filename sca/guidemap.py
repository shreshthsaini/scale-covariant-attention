"""Spatially varying guidance (lead, 2026-11-07, round 7 lever 1).

FLUX.1-dev is guidance-distilled: one forward pass per step, with the guidance scale as an embedding. There is no
uncond pass whose difference could be scaled per token, so on the window steps with gm_lo < sigma <= gate_sigma the
transformer runs a second time at guidance gm_low and the two velocities are blended per image token by the opening's
subject map g in [0, 1] (SCAContext.sa_map, sca/subjgate.py):

    v = v_low + g (v_high - v_low)

Full guidance on the subject, gm_low elsewhere. Attention is untouched: both passes run the installed SCA operator at the
same step (the pre-hook that sets sigma runs once, before the first pass). Steps without a map (the opening, prompts
without subject words) keep the single pass.
"""
import torch


def blend(v_high, v_low, g):
    """[B, N, C] velocities, [N] map -> v_low + g (v_high - v_low), computed in float32."""
    w = g.to(device=v_high.device, dtype=torch.float32).view(1, -1, 1)
    return (v_low.float() + w * (v_high.float() - v_low.float())).to(v_high.dtype)


def install(transformer, ctx):
    """Forward hook on the FLUX transformer; returns its remover. No-op unless ctx.gm_low > 0."""

    def hook(module, args, kwargs, output):
        sig = ctx.sigma
        if ctx.gm_low <= 0 or sig is None or not (ctx.gm_lo < float(sig) <= ctx.gate_sigma):
            return output
        if kwargs.get("guidance") is None:
            raise ValueError("spatially varying guidance needs a guidance-embedded model (FLUX.1-dev)")
        tup = isinstance(output, tuple)
        v_high = output[0] if tup else output.sample
        g = ctx.sa_map(v_high.shape[1], v_high.device)
        if g is None:
            return output
        kw = dict(kwargs)
        kw["guidance"] = torch.full_like(kwargs["guidance"], float(ctx.gm_low))
        low = module.forward(*args, **kw)   # forward, not __call__: no second pre-hook, no recursion into this hook
        v_low = low[0] if isinstance(low, tuple) else low.sample
        v = blend(v_high, v_low, g)
        ctx.log.append(f"guidemap low={ctx.gm_low} on sigma > {ctx.gm_lo}")
        if tup:
            return (v,) + tuple(output[1:])
        output.sample = v
        return output

    handle = transformer.register_forward_hook(hook, with_kwargs=True)
    return handle.remove
