"""Scale-covariant attention (SCA) for FLUX joint attention.

One processor class serves every FluxAttention module (double and single stream blocks).
Modes (SCAContext.mode):
  full   exact SDPA attention, numerically identical to the stock FluxAttnProcessor
  sca    image queries: pi[h] * O_loc + (1 - pi[h]) * O_glob; text queries: O_glob
  calib  exact attention for the forward pass, plus O_loc / O_glob statistics for emulated scales

Tensor layout inside the processor is [B, S, heads, D] with text tokens first, as in diffusers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import _get_qkv_projections


# ----------------------------------------------------------------------------- geometry

def nearest_divisor(n: int, target: float) -> int:
    """Divisor of n closest to target (ties go to the smaller divisor)."""
    divs = [d for d in range(1, n + 1) if n % d == 0]
    return min(divs, key=lambda d: (abs(d - target), d))


def pool_factors(grid, n0, log=None):
    """Per-axis integer pooling factors fh = H / n0, fw = W / n0, each dividing its axis."""
    out = []
    for n in grid:
        ideal = max(1.0, n / n0)
        f = max(1, round(ideal))
        if n % f:
            g = nearest_divisor(n, ideal)
            if log is not None:
                log.append(f"pool factor {f} does not divide grid axis {n}; using divisor {g}")
            f = g
        out.append(f)
    return tuple(out)


def fit_divisor(n, t, log=None, what="tile"):
    t = max(1, min(int(t), n))
    if n % t:
        g = nearest_divisor(n, t)
        if log is not None:
            log.append(f"{what} {t} does not divide grid axis {n}; using {g}")
        t = g
    return t


# ----------------------------------------------------------------------------- primitives

def rope(x, cs):
    """Apply FLUX RoPE to x [B, S, h, D] with (cos, sin) each [S, D]."""
    return apply_rotary_emb(x, cs, sequence_dim=1)


def sdpa(q, k, v):
    """Attention on [B, S, h, D] tensors, same call pattern as the diffusers native backend."""
    return F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                                          v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)


def pool_tokens(x, grid, f, pool):
    """Pool image tokens x [B, H*W, h, D] by (fh, fw) blocks -> [B, (H/fh)*(W/fw), h, D]."""
    B, N, h, D = x.shape
    (H, W), (fh, fw) = grid, f
    if fh == fw == 1:
        return x
    g = x.view(B, H, W, h, D)
    if pool == "center":
        g = g[:, fh // 2::fh, fw // 2::fw]
    elif pool == "mean":
        g = g.view(B, H // fh, fh, W // fw, fw, h, D).float().mean((2, 4)).to(x.dtype)
    else:
        raise ValueError(pool)
    return g.reshape(B, -1, h, D)


def global_branch(q, k, v, L, grid, f, pool, rope_txt, rope_q, rope_k):
    """Pooled image K/V on the coarse grid plus all text K/V; all queries (text and image).

    q, k, v are pre-RoPE [B, S, h, D], text first. rope_q are image-query RoPE tables at
    coarse coordinates (floor(row / fh), floor(col / fw)); rope_k the pooled-key tables.
    """
    kp = pool_tokens(k[:, L:], grid, f, pool)
    vp = pool_tokens(v[:, L:], grid, f, pool)
    qr = torch.cat([rope(q[:, :L], rope_txt), rope(q[:, L:], rope_q)], 1)
    kr = torch.cat([rope(k[:, :L], rope_txt), rope(kp, rope_k)], 1)
    return sdpa(qr, kr, torch.cat([v[:, :L], vp], 1))


def window_index(L, grid, tile, window, device):
    """Key indices [T, L + Wh*Ww] (text first) for every t x t query tile, row-major tiles."""
    (H, W), (th, tw), (Wh, Ww) = grid, tile, window
    rs = (torch.arange(H // th, device=device) * th + th // 2 - Wh // 2).clamp(0, H - Wh)
    cs = (torch.arange(W // tw, device=device) * tw + tw // 2 - Ww // 2).clamp(0, W - Ww)
    rows = rs[:, None, None, None] + torch.arange(Wh, device=device)[None, None, :, None]
    cols = cs[None, :, None, None] + torch.arange(Ww, device=device)[None, None, None, :]
    win = (rows * W + cols).reshape(-1, Wh * Ww) + L
    txt = torch.arange(L, device=device).expand(win.shape[0], L)
    return torch.cat([txt, win], 1)


def auto_chunk(n_keys, h, D, itemsize, budget=2 << 30):
    return max(1, int(budget // (2 * n_keys * h * D * itemsize)))


def local_branch(qr, kr, v, L, grid, tile, window, chunk=None):
    """Blocked neighborhood attention for image queries at full resolution.

    qr, kr are RoPE-applied with original (full-resolution) coordinates, [B, S, h, D].
    Each th x tw query tile attends to all text keys plus a Wh x Ww key window centred on
    the tile and clamped inside the grid. Returns [B, H*W, h, D].
    """
    B, S, h, D = qr.shape
    (H, W), (th, tw) = grid, tile
    Th, Tw = H // th, W // tw
    T, tq = Th * Tw, th * tw
    idx = window_index(L, grid, tile, window, qr.device)
    chunk = chunk or auto_chunk(idx.shape[1], h, D, kr.element_size())
    qt = qr[:, L:].view(B, Th, th, Tw, tw, h, D).permute(0, 1, 3, 2, 4, 5, 6).reshape(B, T, tq, h, D)
    out = qt.new_empty(B, T, tq, h, D)
    for c0 in range(0, T, chunk):
        ii = idx[c0:c0 + chunk]
        c = ii.shape[0]
        kk, vv = kr[:, ii].flatten(0, 1), v[:, ii].flatten(0, 1)  # [B*c, M, h, D]
        out[:, c0:c0 + c] = sdpa(qt[:, c0:c0 + c].flatten(0, 1), kk, vv).view(B, c, tq, h, D)
    return out.view(B, Th, Tw, th, tw, h, D).permute(0, 1, 3, 2, 4, 5, 6).reshape(B, H * W, h, D)


# ----------------------------------------------------------------------------- pi tables

class PiTable:
    """pi(layer, head, sigma). variant: head | layer | uniform | switch | const."""

    def __init__(self, variant="const", table=None, centers=None, value=1.0, sigma_star=None):
        self.variant, self.value, self.sigma_star = variant, value, sigma_star
        self.table = None if table is None else np.asarray(table, dtype=np.float64)
        self.centers = None if centers is None else np.asarray(centers, dtype=np.float64)

    @classmethod
    def load(cls, path, variant="head", scale=None, sigma_star=None, target_scale=None):
        z = np.load(path)
        scales = [int(s) for s in z["scales"]]
        if scale is None:
            t = target_scale or scales[0]
            scale = min(scales, key=lambda s: (abs(math.log(s / t)), -s))
        p = f"s{scale}_"
        if variant == "switch":
            obj = cls("switch", sigma_star=float(z[p + "sigma_star"]) if sigma_star is None else sigma_star)
        else:
            table = z[p + {"head": "pi_head", "layer": "pi_layer", "uniform": "pi_uniform"}[variant]]
            table = table[:, None] if variant == "layer" else table.reshape((1,) * (3 - table.ndim) + table.shape)
            obj = cls(variant, table, z["sigma_centers"])
        obj.scale = scale
        return obj

    def __call__(self, sigma):
        if self.variant == "const":
            return np.full((1, 1), float(self.value))
        if self.variant == "switch":
            return np.full((1, 1), 0.0 if sigma > self.sigma_star else 1.0)
        c, t = self.centers, self.table
        j = np.searchsorted(c, sigma)
        if j <= 0:
            return t[..., 0]
        if j >= len(c):
            return t[..., -1]
        a = (sigma - c[j - 1]) / (c[j] - c[j - 1])
        return (1 - a) * t[..., j - 1] + a * t[..., j]


# ----------------------------------------------------------------------------- context + processor

@dataclass
class SCAContext:
    mode: str = "full"                  # full | sca | calib
    n0: int = 64                        # native grid side in tokens
    tile: int = 8                       # query tile side t
    window: int | None = None           # local key window side Wn (default n0)
    pool: str = "center"                # center | mean
    chunk: int | None = None            # tiles per local-branch chunk (None = auto, 2 GiB budget)
    pi: PiTable = field(default_factory=PiTable)
    # calibration
    calib: object | None = None         # sca.calib.CalibStats
    calib_scales: tuple = (2, 4)
    calib_window: int | None = None     # override emulated window side (tests)
    calib_tile: int | None = None       # override emulated tile side (default t * Wc / Wn)
    # runtime state (set by the transformer pre-hook)
    grid: tuple | None = None
    sigma: float = 1.0
    step: int = 0
    scheduler: object | None = None
    n_layers: int = 0
    pos_embed: object | None = None
    log: list = field(default_factory=list)
    pi_trace: list = field(default_factory=list)

    def __post_init__(self):
        self._rope_cache, self._pi_now, self._img_ids = {}, None, None

    # -- state
    def set_inputs(self, img_ids, txt_ids=None, sigma=None):
        img_ids = img_ids[0] if img_ids.ndim == 3 else img_ids
        if self._img_ids is None or self._img_ids.shape != img_ids.shape or not torch.equal(self._img_ids, img_ids):
            self._img_ids, self._rope_cache = img_ids.detach().clone(), {}
            r, c = img_ids[:, 1], img_ids[:, 2]
            self.grid = (int(r.max() - r.min()) + 1, int(c.max() - c.min()) + 1)
            assert self.grid[0] * self.grid[1] == img_ids.shape[0], "image ids must form a full grid"
        if sigma is not None:
            self.sigma = float(sigma)
        self._pi_now = None

    def pi_now(self, layer, heads, device):
        if self._pi_now is None:
            p = np.atleast_2d(np.asarray(self.pi(self.sigma), dtype=np.float64))  # [layers|1, heads|1]
            if p.shape[0] not in (1, self.n_layers) or p.shape[1] not in (1, heads):
                raise ValueError(f"pi table shape {p.shape} does not match {self.n_layers} layers x {heads} heads")
            self._pi_now = torch.as_tensor(p, dtype=torch.float32, device=device)
            self.pi_trace.append({"step": self.step, "sigma": self.sigma, "pi_mean": float(p.mean())})
        p = self._pi_now
        return p[layer if p.shape[0] > 1 else 0].expand(heads)

    def coarse_rope(self, f, device):
        """(rope_q, rope_k): image queries at floor(id / f), pooled keys at block indices."""
        key = (f, str(device))
        if key not in self._rope_cache:
            ids = self._img_ids.to(device).float()
            div = torch.tensor([1.0, f[0], f[1]], device=device)
            q_ids = torch.cat([ids[:, :1], torch.floor((ids[:, 1:] - ids[:, 1:].amin(0)) / div[1:])], 1)
            H, W = self.grid
            k_ids = q_ids.view(H, W, 3)[f[0] // 2::f[0], f[1] // 2::f[1]].reshape(-1, 3)
            self._rope_cache[key] = (self.pos_embed(q_ids), self.pos_embed(k_ids))
        return self._rope_cache[key]

    def geometry(self, scale=None):
        """(f, tile, window) for sca mode (scale None) or a calibration emulation at scale s_c."""
        H, W = self.grid
        Wn = self.window or self.n0
        if scale is None:
            f = pool_factors(self.grid, self.n0, self.log)
            win = (min(Wn, H), min(Wn, W))
            tile = (fit_divisor(H, self.tile, self.log), fit_divisor(W, self.tile, self.log))
        else:
            f = (fit_divisor(H, scale, self.log, "calib pool"), fit_divisor(W, scale, self.log, "calib pool"))
            cw = self.calib_window
            win = (min(cw or H // scale, H), min(cw or W // scale, W))
            t = self.calib_tile or max(1, round(self.tile * win[0] / Wn))
            tile = (fit_divisor(H, t, self.log), fit_divisor(W, t, self.log))
        return f, tile, win


class SCAProcessor:
    """Drop-in replacement for FluxAttnProcessor; returns exactly the same structure."""

    def __init__(self, ctx: SCAContext, layer: int):
        self.ctx, self.layer = ctx, layer

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 image_rotary_emb=None):
        ctx = self.ctx
        q, k, v, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        q, k, v = (t.unflatten(-1, (attn.heads, -1)) for t in (q, k, v))
        q, k = attn.norm_q(q), attn.norm_k(k)
        if attn.added_kv_proj_dim is not None:
            eq, ek, ev = (t.unflatten(-1, (attn.heads, -1)) for t in (eq, ek, ev))
            q = torch.cat([attn.norm_added_q(eq), q], 1)
            k = torch.cat([attn.norm_added_k(ek), k], 1)
            v = torch.cat([ev, v], 1)

        if ctx.mode == "full" or ctx.grid is None:
            if ctx.mode != "full":
                raise RuntimeError("SCA context has no image grid; install the transformer pre-hook")
            out = sdpa(rope(q, image_rotary_emb), rope(k, image_rotary_emb), v)
        else:
            if attention_mask is not None:
                raise NotImplementedError("attention_mask is not supported by SCA")
            out = self.sca(q, k, v, image_rotary_emb) if ctx.mode == "sca" else self.calibrate(q, k, v, image_rotary_emb)

        hs = out.flatten(2, 3).to(q.dtype)
        if encoder_hidden_states is not None:
            txt, img = hs.split_with_sizes([encoder_hidden_states.shape[1], hs.shape[1] - encoder_hidden_states.shape[1]], 1)
            img = attn.to_out[1](attn.to_out[0](img.contiguous()))
            return img, attn.to_add_out(txt.contiguous())
        return hs

    def _split(self, q, image_rotary_emb):
        H, W = self.ctx.grid
        L = q.shape[1] - H * W
        assert L >= 0
        cos, sin = image_rotary_emb
        return L, (cos[:L], sin[:L])

    def branches(self, q, k, v, image_rotary_emb, f, tile, win, qr=None, kr=None):
        ctx = self.ctx
        L, rope_txt = self._split(q, image_rotary_emb)
        rq, rk = ctx.coarse_rope(f, q.device)
        o_glob = global_branch(q, k, v, L, ctx.grid, f, ctx.pool, rope_txt, rq, rk)
        if qr is None:
            qr, kr = rope(q, image_rotary_emb), rope(k, image_rotary_emb)
        o_loc = local_branch(qr, kr, v, L, ctx.grid, tile, win, ctx.chunk)
        return L, o_loc, o_glob

    def sca(self, q, k, v, image_rotary_emb):
        f, tile, win = self.ctx.geometry()
        L, o_loc, o_glob = self.branches(q, k, v, image_rotary_emb, f, tile, win)
        pi = self.ctx.pi_now(self.layer, q.shape[2], q.device).view(1, 1, -1, 1)
        img = (pi * o_loc.float() + (1 - pi) * o_glob[:, L:].float()).to(o_glob.dtype)
        return torch.cat([o_glob[:, :L], img], 1)

    def calibrate(self, q, k, v, image_rotary_emb):
        ctx = self.ctx
        qr, kr = rope(q, image_rotary_emb), rope(k, image_rotary_emb)
        full = sdpa(qr, kr, v)
        for i, s in enumerate(ctx.calib_scales):
            f, tile, win = ctx.geometry(s)
            L, o_loc, o_glob = self.branches(q, k, v, image_rotary_emb, f, tile, win, qr, kr)
            ctx.calib.add(self.layer, i, ctx.sigma, full[:, L:], o_loc, o_glob[:, L:])
        ctx.calib.add_mass(self.layer, ctx.sigma, ctx.step, qr, kr, L, ctx.grid)
        return full


# ----------------------------------------------------------------------------- install

def install(transformer, ctx: SCAContext):
    """Install SCAProcessor on every FluxAttention and a pre-hook that feeds grid and sigma."""
    ctx.pos_embed = transformer.pos_embed
    blocks = list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks)
    ctx.n_layers = len(blocks)
    original = dict(transformer.attn_processors)
    for i, block in enumerate(blocks):
        block.attn.set_processor(SCAProcessor(ctx, i))

    def pre_hook(module, args, kwargs):
        img_ids, txt_ids = kwargs.get("img_ids"), kwargs.get("txt_ids")
        sigma = None
        if ctx.scheduler is not None and getattr(ctx.scheduler, "sigmas", None) is not None:
            j = ctx.scheduler.step_index
            ctx.step = 0 if j is None else int(j)
            sigma = float(ctx.scheduler.sigmas[ctx.step])
        elif kwargs.get("timestep") is not None:
            sigma = float(kwargs["timestep"].flatten()[0])
        ctx.set_inputs(img_ids, txt_ids, sigma)
        if ctx.calib is not None:
            ctx.calib.add_sigma(ctx.sigma)

    handle = transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def uninstall():
        handle.remove()
        transformer.set_attn_processor(original)
    return uninstall


# ----------------------------------------------------------------------------- cost model

def attention_flops(height, width, n0=64, tile=8, window=None, text=512, heads=24, head_dim=128,
                    layers=57, steps=28, params=11.9e9, tflops=400.0, gather_tbs=3.0):
    """Attention FLOPs (QK^T + PV, 2 FLOP per MAC) per image for full attention and SCA.

    Also returns a rough latency estimate: attention time at `tflops` effective throughput,
    the local-branch K/V gather traffic at `gather_tbs` TB/s (bf16), and the linear layers
    (2 * params FLOP per token) for an end-to-end speedup estimate.
    """
    H, W = height // 16, width // 16
    N, S = H * W, H * W + text
    fh, fw = pool_factors((H, W), n0)
    Wn = window or n0
    keys_loc = text + min(Wn, H) * min(Wn, W)
    per = 4 * heads * head_dim * layers * steps
    full = per * S * S
    glob = per * S * (text + N // (fh * fw))
    loc = per * N * keys_loc
    tiles = N // (tile * tile)
    gather_bytes = tiles * keys_loc * heads * head_dim * 2 * 2 * 2 * layers * steps  # K,V write+read
    linear = 2 * params * S * steps
    t = lambda flops: flops / (tflops * 1e12)
    sca_s = t(glob + loc) + gather_bytes / (gather_tbs * 1e12)
    return {"res": f"{height}x{width}", "grid": [H, W], "pool": [fh, fw], "tile": tile, "window": Wn,
            "attn_flops_full": full, "attn_flops_sca": glob + loc, "attn_flops_sca_global": glob,
            "attn_flops_sca_local": loc, "attn_flop_ratio": full / (glob + loc),
            "linear_flops": linear, "est_attn_seconds_full": t(full), "est_attn_seconds_sca": sca_s,
            "est_total_seconds_full": t(full + linear), "est_total_seconds_sca": sca_s + t(linear),
            "est_speedup_total": t(full + linear) / (sca_s + t(linear)),
            "assumptions": f"{tflops} TFLOP/s effective bf16, gather at {gather_tbs} TB/s, {layers} layers, {steps} steps"}


if __name__ == "__main__":
    import json
    for hw in ((1024, 1024), (2048, 2048), (4096, 4096), (2048, 4096)):
        r = attention_flops(*hw)
        print(json.dumps({k: (f"{v:.3e}" if isinstance(v, float) and v > 1e4 else v) for k, v in r.items()}))
