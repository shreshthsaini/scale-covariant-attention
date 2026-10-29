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


def sdpa(q, k, v, attn_mask=None):
    """Attention on [B, S, h, D] tensors, same call pattern as the diffusers native backend."""
    return F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                                          v.permute(0, 2, 1, 3), attn_mask=attn_mask).permute(0, 2, 1, 3)


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


def global_branch(q, k, v, L, grid, f, pool, rope_txt, rope_q, rope_k, q_txt=None, k_txt=None):
    """Pooled image K/V on the coarse grid plus all text K/V; all queries (text and image).

    q, k, v are pre-RoPE [B, S, h, D], text first. rope_q are image-query RoPE tables at
    coarse coordinates (floor(row / fh), floor(col / fw)); rope_k the pooled-key tables.
    q_txt / k_txt optionally pass the RoPE'd text queries / keys when the caller has them.
    """
    kp = pool_tokens(k[:, L:], grid, f, pool)
    vp = pool_tokens(v[:, L:], grid, f, pool)
    q_txt = rope(q[:, :L], rope_txt) if q_txt is None else q_txt
    k_txt = rope(k[:, :L], rope_txt) if k_txt is None else k_txt
    qr = torch.cat([q_txt, rope(q[:, L:], rope_q)], 1)
    kr = torch.cat([k_txt, rope(kp, rope_k)], 1)
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


def local_branch(qr, kr, v, L, grid, tile, window, chunk=None, text_bias=0.0):
    """Blocked neighborhood attention for image queries at full resolution.

    qr, kr are RoPE-applied with original (full-resolution) coordinates, [B, S, h, D]
    (qr may also hold the image queries alone, [B, H*W, h, D]).
    Each th x tw query tile attends to all text keys plus a Wh x Ww key window centred on
    the tile and clamped inside the grid. text_bias is added to the logits of the text keys.
    Returns [B, H*W, h, D].
    """
    (H, W), (th, tw) = grid, tile
    B, _, h, D = qr.shape
    qr = qr if qr.shape[1] == H * W else qr[:, L:]
    Th, Tw = H // th, W // tw
    T, tq = Th * Tw, th * tw
    idx = window_index(L, grid, tile, window, qr.device)
    chunk = chunk or auto_chunk(idx.shape[1], h, D, kr.element_size())
    qt = qr.view(B, Th, th, Tw, tw, h, D).permute(0, 1, 3, 2, 4, 5, 6).reshape(B, T, tq, h, D)
    out = qt.new_empty(B, T, tq, h, D)
    bias = None
    if text_bias:
        bias = torch.zeros(1, idx.shape[1], dtype=qr.dtype, device=qr.device)  # [1, M], broadcast over queries
        bias[:, :L] = float(text_bias)
    for c0 in range(0, T, chunk):
        ii = idx[c0:c0 + chunk]
        c = ii.shape[0]
        kk, vv = kr[:, ii].flatten(0, 1), v[:, ii].flatten(0, 1)  # [B*c, M, h, D]
        out[:, c0:c0 + c] = sdpa(qt[:, c0:c0 + c].flatten(0, 1), kk, vv, bias).view(B, c, tq, h, D)
    return out.view(B, Th, Tw, th, tw, h, D).permute(0, 1, 3, 2, 4, 5, 6).reshape(B, H * W, h, D)


# ----------------------------------------------------------------------------- flex local branch
#
# Same keys as local_branch, computed with FlexAttention over a block-sparse mask. Image queries
# and image keys are permuted into tile-major order so that a sparse block is a small spatial
# tile; mask_mod states the gather-window membership per (query, key) pair, so the result does
# not depend on the block layout (only the amount of skipped work does). Text keys come first.

FLEX_Q_TILE = (8, 16)     # query permutation tile (rows, cols) = one sparse Q block (128 queries)
FLEX_K_TILE = (8, 8)      # key permutation tile = one sparse KV block (64 keys)
FLEX_KERNEL = None        # kernel_options for flex_attention (None = derived from the block sizes)
# Layout choice (GH200, 24 heads, d 128, bf16, interleaved min timings, scripts/bench_attn.py):
# 8x16 query blocks with 8x8 window-aligned key blocks and TMA beat 8x8/16x16 query blocks and
# 8x16/16x16 key blocks at 2048^2, 4096^2 and 6144x2048 for both t = 8 and t = 16.
_FLEX = {}


def _envfix():
    """torch.compile needs zero-byte modules a scratch purge removed from the env; see sca/envfix.py."""
    from sca.envfix import install
    install()


def _raise_recompile_limit(n=64):
    """Each BlockMask carries its own mask_mod closure, so every new (grid, tile, window) layout
    recompiles flex_attention and create_block_mask. Past dynamo's default limit (8) it would
    silently fall back to the eager path, which materializes the full score matrix."""
    from torch._dynamo import config as cfg
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(cfg, name):
            setattr(cfg, name, max(getattr(cfg, name), n))
    if hasattr(cfg, "accumulated_recompile_limit"):
        cfg.accumulated_recompile_limit = max(cfg.accumulated_recompile_limit, 8 * n)


def flex_fn(device):
    """flex_attention compiled once for CUDA; the eager reference elsewhere (CPU tests)."""
    from torch.nn.attention.flex_attention import flex_attention
    _envfix()
    if torch.device(device).type != "cuda":
        return flex_attention
    if "fn" not in _FLEX:
        _raise_recompile_limit()
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=False)
    return _FLEX["fn"]


def tile_order(grid, tile, offset=(0, 0), device=None):
    """Row-major image index of every slot of a tile-major ordering, -1 for padding slots.

    The grid is shifted by `offset` (padding rows/cols before the image) and padded at the end
    to whole tiles; tiles are visited row-major and tokens inside a tile row-major.
    """
    (H, W), (th, tw), (oh, ow) = grid, tile, offset
    Hp, Wp = -(-(H + oh) // th) * th, -(-(W + ow) // tw) * tw
    y = torch.arange(Hp, device=device)[:, None] - oh
    x = torch.arange(Wp, device=device)[None, :] - ow
    idx = (y * W + x).masked_fill((y < 0) | (y >= H) | (x < 0) | (x >= W), -1)
    return idx.view(Hp // th, th, Wp // tw, tw).permute(0, 2, 1, 3).reshape(-1)


@dataclass
class FlexLayout:
    q_gather: torch.Tensor      # [Nq] image index per query slot (padding -> 0)
    q_inv: torch.Tensor         # [N] query slot per image token
    k_gather: torch.Tensor      # [Nk] image index per key slot (padding -> 0)
    block_mask: object
    kernel_options: dict | None


def flex_layout(L, grid, tile, window, device, q_tile=None, k_tile=None, kernel_options=None):
    """Permutations and BlockMask reproducing window_index() membership exactly."""
    from torch.nn.attention.flex_attention import create_block_mask
    _envfix()
    _raise_recompile_limit()
    (H, W), (th, tw), (Wh, Ww) = grid, tile, window
    q_tile, k_tile = q_tile or FLEX_Q_TILE, k_tile or FLEX_K_TILE
    # window start per query tile, identical to window_index()
    r_start = (torch.arange(H, device=device) // th * th + th // 2 - Wh // 2).clamp(0, H - Wh)
    c_start = (torch.arange(W, device=device) // tw * tw + tw // 2 - Ww // 2).clamp(0, W - Ww)
    # shift key tiles so that unclamped windows start on a key-tile boundary
    k_off = ((Wh // 2 - th // 2) % k_tile[0], (Ww // 2 - tw // 2) % k_tile[1])
    qo = tile_order(grid, q_tile, device=device)
    ko = tile_order(grid, k_tile, k_off, device=device)
    q_gather, k_gather = qo.clamp(min=0), ko.clamp(min=0)
    q_inv = torch.empty(H * W, dtype=torch.long, device=device)
    valid = qo >= 0
    q_inv[qo[valid]] = torch.arange(qo.numel(), device=device)[valid]
    i32 = torch.int32
    qr0 = r_start[q_gather // W].to(i32)
    qc0 = c_start[q_gather % W].to(i32)
    far = -(1 << 30)
    krow = torch.where(ko >= 0, k_gather // W, far).to(i32)
    kcol = torch.where(ko >= 0, k_gather % W, far).to(i32)

    def mask_mod(b, h, q, kv):
        j = (kv - L).clamp(min=0)
        r, c, r0, c0 = krow[j], kcol[j], qr0[q], qc0[q]
        return (kv < L) | ((r >= r0) & (r < r0 + Wh) & (c >= c0) & (c < c0 + Ww))

    nq, nk = qo.numel(), L + ko.numel()
    bs = (q_tile[0] * q_tile[1], k_tile[0] * k_tile[1])
    from torch._dynamo import config as dynamo_config
    # static shapes: with automatic dynamic shapes the second layout traces the captured ints
    # (L, Wh, Ww) as SymInts and the mask_mod fails to trace
    with dynamo_config.patch(automatic_dynamic_shapes=False):
        bm = create_block_mask(mask_mod, None, None, nq, nk, device=device, BLOCK_SIZE=bs,
                               _compile=torch.device(device).type == "cuda")
    ko = kernel_options if kernel_options is not None else FLEX_KERNEL
    if ko is None and torch.device(device).type == "cuda":  # kernel tiles must divide the sparse blocks
        ko = {"BLOCK_M": math.gcd(bs[0], 128), "BLOCK_N": math.gcd(bs[1], 64), "USE_TMA": True}
    return FlexLayout(q_gather, q_inv, k_gather, bm, ko)


def text_score_mod(L, text_bias):
    """score_mod adding text_bias to the logits of the first L (text) keys; None when text_bias is 0."""
    if not text_bias:
        return None
    tb = float(text_bias)

    def score_mod(score, b, h, q, kv):
        return torch.where(kv < L, score + tb, score)
    return score_mod


def local_branch_flex(qi, kr, v, L, layout, permuted=False, text_bias=0.0, text_len=None, return_lse=False):
    """Flex version of local_branch. qi: RoPE'd image queries [B, N, h, D]; kr, v: [B, L+N, h, D].

    Returns [B, N, h, D], or with permuted=True the raw flex output [B, h, Nq, D] in query-slot
    order (un-permute with .transpose(1, 2).index_select(1, layout.q_inv)). The first L keys are
    visible to every query (text, plus bridge tokens when text_len < L); text_bias applies to the
    first text_len (default L) keys. return_lse=True also returns the log-sum-exp [B, h, Nq].
    """
    qp = qi.index_select(1, layout.q_gather)
    kp = torch.cat([kr[:, :L], kr[:, L:].index_select(1, layout.k_gather)], 1)
    vp = torch.cat([v[:, :L], v[:, L:].index_select(1, layout.k_gather)], 1)
    kw = {"kernel_options": layout.kernel_options} if layout.kernel_options else {}
    sm = text_score_mod(L if text_len is None else text_len, text_bias)
    if sm is not None:
        kw["score_mod"] = sm
    if return_lse:
        kw["return_lse"] = True
    o = flex_fn(qi.device)(qp.transpose(1, 2), kp.transpose(1, 2), vp.transpose(1, 2),
                           block_mask=layout.block_mask, **kw)
    if return_lse:
        o, lse = o
        return (o if permuted else o.transpose(1, 2).index_select(1, layout.q_inv)), lse
    return o if permuted else o.transpose(1, 2).index_select(1, layout.q_inv)


def band_low(x, grid, p):
    """Low band of image tokens [B, N, h, D] on the (gh, gw) grid: average-pool by p, bilinear upsample."""
    B, N, H, D = x.shape
    gh, gw = grid
    y = x.reshape(B, gh, gw, H * D).permute(0, 3, 1, 2)
    y = torch.nn.functional.avg_pool2d(y, p, stride=p, ceil_mode=True)
    y = torch.nn.functional.interpolate(y, size=(gh, gw), mode="bilinear", align_corners=False)
    return y.permute(0, 2, 3, 1).reshape(B, N, H, D)


def band_mix(o_glob, o_loc, pi, L, grid, p):
    """Image output og + pi * (d - LP_p(d)), d = O_loc - O_glob: the pi mix with its low band
    replaced by O_glob's (LP is linear and pi is constant over space, so one LP call suffices)."""
    og = o_glob[:, L:].float()
    d = o_loc.float() - og
    img = og + pi * (d - band_low(d, grid, p))
    return torch.cat([o_glob[:, :L], img.to(o_glob.dtype)], 1)


def band_mix_fn(device):
    """band_mix compiled once for CUDA (static grid and band per process)."""
    if torch.device(device).type != "cuda":
        return band_mix
    if "band" not in _FLEX:
        _envfix()
        _FLEX["band"] = torch.compile(band_mix, dynamic=False)
    return _FLEX["band"]


def mix_branches(o_glob, o_loc, pi, L, inv=None):
    """[text: O_glob | image: pi * O_loc + (1 - pi) * O_glob] in fp32, cast to O_glob's dtype.

    o_loc is [B, N, h, D], or the permuted flex output [B, h, Nq, D] when inv is given.
    """
    if inv is not None:
        o_loc = o_loc.transpose(1, 2).index_select(1, inv)
    img = (pi * o_loc.float() + (1 - pi) * o_glob[:, L:].float()).to(o_glob.dtype)
    return torch.cat([o_glob[:, :L], img], 1)


def mix_fn(device):
    """mix_branches fused into one kernel on CUDA (same arithmetic, fp32)."""
    if torch.device(device).type != "cuda":
        return mix_branches
    if "mix" not in _FLEX:
        _envfix()
        _FLEX["mix"] = torch.compile(mix_branches, dynamic=False)
    return _FLEX["mix"]


def sca_flex_forward(q, k, v, rot, rope_q, rope_k, pi, L, grid, f, pool, layout, text_bias=0.0):
    """Whole SCA attention (both branches and the pi mix) with the flex local branch.

    Same arithmetic as SCAProcessor.branches + mix_branches with local_impl="flex"; written as
    one function so that torch.compile fuses RoPE, pooling, the tile permutations and the mix
    around the two attention calls. q, k, v pre-RoPE [B, S, h, D]; returns [B, S, h, D].
    """
    cos, sin = rot
    kr = rope(k, rot)
    qi = rope(q[:, L:], (cos[L:], sin[L:]))
    q_txt = rope(q[:, :L], (cos[:L], sin[:L]))
    o_glob = global_branch(q, k, v, L, grid, f, pool, (cos[:L], sin[:L]), rope_q, rope_k, q_txt, kr[:, :L])
    o_loc = local_branch_flex(qi, kr, v, L, layout, permuted=True, text_bias=text_bias)
    return mix_branches(o_glob, o_loc, pi, L, layout.q_inv)


def sca_flex_fn(device):
    """sca_flex_forward compiled once for CUDA (flex_attention lowers inside the graph)."""
    if torch.device(device).type != "cuda":
        return sca_flex_forward
    if "sca" not in _FLEX:
        _envfix()
        _raise_recompile_limit()
        flex_fn(device)  # create the standalone compiled flex here: a torch.compile call reached
        # while dynamo traces sca_flex_forward would cache the eager flex_attention instead
        _FLEX["sca"] = torch.compile(sca_flex_forward, dynamic=False)
    return _FLEX["sca"]


# ----------------------------------------------------------------------------- late-duplication forward
#
# Options that change the published computation (.agent/notes/duplication_fix_20261027.md) live in a separate
# function so that sca_flex_forward, the published path, stays byte-for-byte the same.

def bridge_index(k_img, grid, cells, per_cell):
    """[B, cells_h * cells_w * per_cell] image-token indices: in every cell of a cells_h x cells_w grid, the
    per_cell tokens whose unrotated keys (summed over heads) lie farthest from the mean key (AReA Eq. 17)."""
    B, N, h, D = k_img.shape
    (H, W), (ch, cw) = grid, cells
    kf = k_img.float()
    score = (kf - kf.mean(1, keepdim=True)).pow(2).sum((2, 3))                      # [B, N]
    sh, sw = H // ch, W // cw
    s = score.view(B, ch, sh, cw, sw).permute(0, 1, 3, 2, 4).reshape(B, ch * cw, sh * sw)
    top = s.topk(per_cell, dim=-1).indices                                          # [B, C, k]
    cell = torch.arange(ch * cw, device=k_img.device)
    r = ((cell // cw) * sh)[None, :, None] + top // sw
    c = ((cell % cw) * sw)[None, :, None] + top % sw
    return (r * W + c).reshape(B, -1)


def global_branch_lse(q, k, v, L, grid, f, pool, rope_q, rope_k, q_txt, k_txt):
    """global_branch through flex_attention, also returning the log-sum-exp [B, h, S]."""
    kp = pool_tokens(k[:, L:], grid, f, pool)
    vp = pool_tokens(v[:, L:], grid, f, pool)
    qr = torch.cat([q_txt, rope(q[:, L:], rope_q)], 1)
    kr = torch.cat([k_txt, rope(kp, rope_k)], 1)
    vv = torch.cat([v[:, :L], vp], 1)
    o, lse = flex_fn(q.device)(qr.transpose(1, 2), kr.transpose(1, 2), vv.transpose(1, 2), return_lse=True)
    return o.transpose(1, 2), lse


def mix_lse(o_glob, lse_glob, o_loc, lse_loc, pi, L, tau, inv):
    """Per-query mix: logit(w) = logit(pi_h) + tau * (d - mean_i d), d = lse_loc - lse_glob per image query.

    The head's mean log-odds stays at its calibrated pi; queries whose window holds less softmax mass than the
    global branch (relative to the head's average) lean global, as in one shared softmax. o_loc / lse_loc are
    the permuted flex outputs [B, h, Nq, D] / [B, h, Nq]."""
    o_loc = o_loc.transpose(1, 2).index_select(1, inv)                              # [B, N, h, D]
    d = lse_loc.index_select(2, inv).float() - lse_glob[:, :, L:].float()           # [B, h, N]
    d = (d - d.mean(-1, keepdim=True)).transpose(1, 2)[..., None]                    # [B, N, h, 1]
    p = pi.float().clamp(1e-4, 1 - 1e-4)
    w = torch.sigmoid(torch.log(p) - torch.log1p(-p) + tau * d)
    img = (w * o_loc.float() + (1 - w) * o_glob[:, L:].float()).to(o_glob.dtype)
    return torch.cat([o_glob[:, :L], img], 1)


def sca_dedup_forward(q, k, v, rot, rope_q, rope_k, pi, L, grid, f, pool, layout, text_bias=0.0,
                      cells=None, per_cell=0, tau=0.0):
    """SCA with the late-duplication options: AReA-style bridge tokens (cells, per_cell) prepended to the
    local keys at their true positions, per-query LSE mixing (tau > 0), and the local text bias. layout
    must be built for the prefix length L + R, R = cells_h * cells_w * per_cell."""
    cos, sin = rot
    kr = rope(k, rot)
    qi = rope(q[:, L:], (cos[L:], sin[L:]))
    q_txt = rope(q[:, :L], (cos[:L], sin[:L]))
    if tau:
        o_glob, lse_glob = global_branch_lse(q, k, v, L, grid, f, pool, rope_q, rope_k, q_txt, kr[:, :L])
    else:
        o_glob = global_branch(q, k, v, L, grid, f, pool, (cos[:L], sin[:L]), rope_q, rope_k, q_txt, kr[:, :L])
    kl, vl, Lp = kr, v, L
    if per_cell:
        idx = bridge_index(k[:, L:], grid, cells, per_cell)                          # [B, R]
        g = idx[:, :, None, None].expand(-1, -1, k.shape[2], k.shape[3])
        kl = torch.cat([kr[:, :L], torch.gather(kr[:, L:], 1, g), kr[:, L:]], 1)
        vl = torch.cat([v[:, :L], torch.gather(v[:, L:], 1, g), v[:, L:]], 1)
        Lp = L + idx.shape[1]
    if tau:
        o_loc, lse_loc = local_branch_flex(qi, kl, vl, Lp, layout, True, text_bias, L, return_lse=True)
        return mix_lse(o_glob, lse_glob, o_loc, lse_loc, pi, L, tau, layout.q_inv)
    o_loc = local_branch_flex(qi, kl, vl, Lp, layout, True, text_bias, L)
    return mix_branches(o_glob, o_loc, pi, L, layout.q_inv)


def sca_dedup_fn(device):
    """sca_dedup_forward compiled once for CUDA."""
    if torch.device(device).type != "cuda":
        return sca_dedup_forward
    if "dedup" not in _FLEX:
        _envfix()
        _raise_recompile_limit()
        flex_fn(device)
        _FLEX["dedup"] = torch.compile(sca_dedup_forward, dynamic=False)
    return _FLEX["dedup"]


# ----------------------------------------------------------------------------- per-head apparent scale
#
# SCAContext.scale_sched (.agent/notes/duplication_fix_20261027.md, round two). Head h of block l attends a native
# n0 x n0 canvas of keys taken from the stride-(s / a) lattice around its query tile, queries and keys at coarse ids
# floor(p / d) as in the global branch, so every head attends T + n0^2 keys with offsets below n0 at every step. Its
# apparent scale a grows from 1 (the global branch itself) to max(s) (the published local branch) on the head's own
# timetable. Separate functions, so the published path stays byte-for-byte the same.

def scale_ladder(s_max):
    """Apparent scales: 1, 2, 4, ... below s_max, then s_max."""
    out, a = [], 1
    while a < s_max:
        out.append(a)
        a *= 2
    return out + [int(s_max)]


def apparent_scale(sigma, pi, gate, lo, s_max):
    """Apparent scale per head: 2 ** round(u * log2 s_max) capped at s_max (round half up), with
    u = clip((gate - sigma) / (gate - sigma_end), 0, 1) and sigma_end = lo + (gate - lo) * pi, pi the published table
    entry (target scale). sigma >= gate gives 1 everywhere; pi = 1 gives s_max anywhere below the gate."""
    pi = np.asarray(pi, dtype=np.float64)
    u = np.clip((gate - float(sigma)) / np.maximum((gate - lo) * (1 - pi), 1e-12), 0.0, 1.0)
    e = np.floor(u * math.log2(s_max) + 0.5).astype(np.int64)
    return np.minimum(2 ** e, int(s_max))


def strided_window_index(L, grid, tile, stride, window, device):
    """window_index() on the stride-(dh, dw) key lattice: indices [T, L + Wh*Ww] into [text | lattice] keys, the
    lattice holding (H / dh) x (W / dw) keys row-major. Each tile's window is centred on its coarse centre
    (tile centre // stride) and clamped inside the lattice. Equals window_index() at stride (1, 1)."""
    (H, W), (th, tw), (dh, dw), (Wh, Ww) = grid, tile, stride, window
    Hc, Wc = H // dh, W // dw
    rs = ((torch.arange(H // th, device=device) * th + th // 2) // dh - Wh // 2).clamp(0, Hc - Wh)
    cs = ((torch.arange(W // tw, device=device) * tw + tw // 2) // dw - Ww // 2).clamp(0, Wc - Ww)
    rows = rs[:, None, None, None] + torch.arange(Wh, device=device)[None, None, :, None]
    cols = cs[None, :, None, None] + torch.arange(Ww, device=device)[None, None, None, :]
    win = (rows * Wc + cols).reshape(-1, Wh * Ww) + L
    txt = torch.arange(L, device=device).expand(win.shape[0], L)
    return torch.cat([txt, win], 1)


def local_branch_strided(qc, kc, vc, L, grid, tile, stride, window, chunk=None, text_bias=0.0):
    """local_branch() with keys on the stride lattice (gather reference of the flex path).

    qc: image queries RoPE'd at coarse ids floor(p / stride) [B, H*W, h, D]; kc, vc: [text | lattice] keys (RoPE'd)
    and values [B, L + (H/dh)*(W/dw), h, D] from lattice_qkv(). Returns [B, H*W, h, D]."""
    (H, W), (th, tw) = grid, tile
    B, _, h, D = qc.shape
    Th, Tw = H // th, W // tw
    T, tq = Th * Tw, th * tw
    idx = strided_window_index(L, grid, tile, stride, window, qc.device)
    chunk = chunk or auto_chunk(idx.shape[1], h, D, kc.element_size())
    qt = qc.view(B, Th, th, Tw, tw, h, D).permute(0, 1, 3, 2, 4, 5, 6).reshape(B, T, tq, h, D)
    out = qt.new_empty(B, T, tq, h, D)
    bias = None
    if text_bias:
        bias = torch.zeros(1, idx.shape[1], dtype=qc.dtype, device=qc.device)
        bias[:, :L] = float(text_bias)
    for c0 in range(0, T, chunk):
        ii = idx[c0:c0 + chunk]
        c = ii.shape[0]
        kk, vv = kc[:, ii].flatten(0, 1), vc[:, ii].flatten(0, 1)
        out[:, c0:c0 + c] = sdpa(qt[:, c0:c0 + c].flatten(0, 1), kk, vv, bias).view(B, c, tq, h, D)
    return out.view(B, Th, Tw, th, tw, h, D).permute(0, 1, 3, 2, 4, 5, 6).reshape(B, H * W, h, D)


def lattice_qkv(q, k, v, L, grid, stride, rope_q, rope_k, k_txt):
    """Inputs of the strided local branch: image queries at coarse ids (rope_q), [text | lattice] keys with the
    lattice taken at block centres as pool_tokens 'center' does (rope_k at their coarse ids), and the matching
    values. q, k, v pre-RoPE [B, S, h, D]; k_txt the RoPE'd text keys."""
    qc = rope(q[:, L:], rope_q)
    kc = torch.cat([k_txt, rope(pool_tokens(k[:, L:], grid, stride, "center"), rope_k)], 1)
    vc = torch.cat([v[:, :L], pool_tokens(v[:, L:], grid, stride, "center")], 1)
    return qc, kc, vc


def strided_flex_layout(L, grid, tile, stride, window, device, q_tile=None, k_tile=None, kernel_options=None):
    """flex_layout() for keys on the stride-(dh, dw) lattice: image queries on the full H x W grid in tile-major order,
    lattice keys on the (H / dh) x (W / dw) grid in tile-major order, and a mask_mod stating strided_window_index()
    membership. Equals flex_layout() at stride (1, 1)."""
    from torch.nn.attention.flex_attention import create_block_mask
    _envfix()
    _raise_recompile_limit()
    (H, W), (th, tw), (dh, dw), (Wh, Ww) = grid, tile, stride, window
    Hc, Wc = H // dh, W // dw
    q_tile, k_tile = q_tile or FLEX_Q_TILE, k_tile or FLEX_K_TILE
    r_start = ((torch.arange(H, device=device) // th * th + th // 2) // dh - Wh // 2).clamp(0, Hc - Wh)
    c_start = ((torch.arange(W, device=device) // tw * tw + tw // 2) // dw - Ww // 2).clamp(0, Wc - Ww)
    # shift key tiles so that the unclamped window of the first query tile starts on a key-tile boundary
    k_off = ((Wh // 2 - (th // 2) // dh) % k_tile[0], (Ww // 2 - (tw // 2) // dw) % k_tile[1])
    qo = tile_order(grid, q_tile, device=device)
    ko = tile_order((Hc, Wc), k_tile, k_off, device=device)
    q_gather, k_gather = qo.clamp(min=0), ko.clamp(min=0)
    q_inv = torch.empty(H * W, dtype=torch.long, device=device)
    valid = qo >= 0
    q_inv[qo[valid]] = torch.arange(qo.numel(), device=device)[valid]
    i32 = torch.int32
    qr0 = r_start[q_gather // W].to(i32)
    qc0 = c_start[q_gather % W].to(i32)
    far = -(1 << 30)
    krow = torch.where(ko >= 0, k_gather // Wc, far).to(i32)
    kcol = torch.where(ko >= 0, k_gather % Wc, far).to(i32)

    def mask_mod(b, h, q, kv):
        j = (kv - L).clamp(min=0)
        r, c, r0, c0 = krow[j], kcol[j], qr0[q], qc0[q]
        return (kv < L) | ((r >= r0) & (r < r0 + Wh) & (c >= c0) & (c < c0 + Ww))

    nq, nk = qo.numel(), L + ko.numel()
    bs = (q_tile[0] * q_tile[1], k_tile[0] * k_tile[1])
    from torch._dynamo import config as dynamo_config
    with dynamo_config.patch(automatic_dynamic_shapes=False):  # as in flex_layout()
        bm = create_block_mask(mask_mod, None, None, nq, nk, device=device, BLOCK_SIZE=bs,
                               _compile=torch.device(device).type == "cuda")
    ko = kernel_options if kernel_options is not None else FLEX_KERNEL
    if ko is None and torch.device(device).type == "cuda":  # kernel tiles must divide the sparse blocks
        ko = {"BLOCK_M": math.gcd(bs[0], 128), "BLOCK_N": math.gcd(bs[1], 64), "USE_TMA": True}
    return FlexLayout(q_gather, q_inv, k_gather, bm, ko)


@dataclass
class HeadLayout:
    """FlexLayout whose BlockMask tables are expanded to every head. set_heads() rewrites the block counts in place:
    heads outside the group visit no key block (the kernel writes zeros for them, which the caller discards), so one
    compiled graph serves every grouping (no new mask_mod, no new shapes)."""
    layout: FlexLayout
    kv_num: torch.Tensor                # [B, 1, Q blocks] partial-block counts of the head-independent mask
    full_num: torch.Tensor | None       # [B, 1, Q blocks] full-block counts

    def set_heads(self, member):
        m = member.reshape(1, -1, 1).to(self.kv_num.dtype)
        bm = self.layout.block_mask
        bm.kv_num_blocks.copy_(self.kv_num * m)
        if self.full_num is not None:
            bm.full_kv_num_blocks.copy_(self.full_num * m)


def head_layout(layout, heads):
    """HeadLayout of a head-independent FlexLayout (tables made contiguous per head, as the flex kernel indexes them)."""
    from torch.nn.attention.flex_attention import BlockMask
    bm = layout.block_mask

    def exp(t):
        return None if t is None else t.expand(t.shape[0], heads, *t.shape[2:]).contiguous()
    hb = BlockMask.from_kv_blocks(exp(bm.kv_num_blocks), exp(bm.kv_indices), exp(bm.full_kv_num_blocks),
                                  exp(bm.full_kv_indices), BLOCK_SIZE=bm.BLOCK_SIZE, mask_mod=bm.mask_mod,
                                  seq_lengths=bm.seq_lengths)
    full = None if bm.full_kv_num_blocks is None else bm.full_kv_num_blocks.clone()
    return HeadLayout(FlexLayout(layout.q_gather, layout.q_inv, layout.k_gather, hb, layout.kernel_options),
                      bm.kv_num_blocks.clone(), full)


def sca_scale_forward(q, k, v, rot, rope_q, rope_k, pi, L, grid, f, pool, groups, text_bias=0.0):
    """SCA with per-head apparent scales (flex). groups: ((stride, rope_q_d, rope_k_d, layout, member), ...), one per
    apparent scale a > 1, layout a HeadLayout.layout whose block counts set_heads(member) has set; member [1, 1, h, 1]
    bool. Image queries of the heads in member take pi O_loc(stride) + (1 - pi) O_glob (pi [1, 1, h, 1], the weight
    at each head's apparent scale); heads in no group (a = 1) and all text queries take O_glob."""
    cos, sin = rot
    rot_txt = (cos[:L], sin[:L])
    k_txt = rope(k[:, :L], rot_txt)
    o_glob = global_branch(q, k, v, L, grid, f, pool, rot_txt, rope_q, rope_k, rope(q[:, :L], rot_txt), k_txt)
    og = o_glob[:, L:]
    img = og
    for stride, rqd, rkd, layout, member in groups:
        qc, kc, vc = lattice_qkv(q, k, v, L, grid, stride, rqd, rkd, k_txt)
        o_loc = local_branch_flex(qc, kc, vc, L, layout, text_bias=text_bias)
        img = torch.where(member, (pi * o_loc.float() + (1 - pi) * og.float()).to(og.dtype), img)
    return torch.cat([o_glob[:, :L], img], 1)


def sca_scale_fn(device):
    """sca_scale_forward compiled once for CUDA: groupings change only tensor values (block counts, member, pi)."""
    if torch.device(device).type != "cuda":
        return sca_scale_forward
    if "scale" not in _FLEX:
        _envfix()
        _raise_recompile_limit()
        flex_fn(device)
        _FLEX["scale"] = torch.compile(sca_scale_forward, dynamic=False)
    return _FLEX["scale"]


def sca_scale_gather(q, k, v, rot, rope_q, rope_k, pi, L, grid, f, pool, tile, groups, chunk=None, text_bias=0.0):
    """Gather reference of sca_scale_forward. groups: ((stride, window, rope_q_d, rope_k_d, heads), ...), heads a
    LongTensor of the heads at that apparent scale; each group runs local_branch_strided on its heads only
    (index_select on the head dimension), mixes with their pi and scatters back."""
    cos, sin = rot
    rot_txt = (cos[:L], sin[:L])
    k_txt = rope(k[:, :L], rot_txt)
    o_glob = global_branch(q, k, v, L, grid, f, pool, rot_txt, rope_q, rope_k, rope(q[:, :L], rot_txt), k_txt)
    img = o_glob[:, L:].clone()
    for stride, window, rqd, rkd, heads in groups:
        q_, k_, v_, kt_ = (t.index_select(2, heads) for t in (q, k, v, k_txt))
        qc, kc, vc = lattice_qkv(q_, k_, v_, L, grid, stride, rqd, rkd, kt_)
        o_loc = local_branch_strided(qc, kc, vc, L, grid, tile, stride, window, chunk, text_bias)
        p, og = pi.index_select(2, heads), o_glob[:, L:].index_select(2, heads)
        img.index_copy_(2, heads, (p * o_loc.float() + (1 - p) * og.float()).to(og.dtype))
    return torch.cat([o_glob[:, :L], img], 1)


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
        obj.scale, obj.path = scale, path
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
    local_impl: str = "flex"            # flex (FlexAttention block mask) | gather (K/V window copies)
    gate_sigma: float = 0.0             # sigma above which pi = 0 (global only, local branch skipped)
    band: int = 0                       # >1: low band (avgpool band, bilinear up) of each image output from O_glob; <0: adaptive, c=-band
    fused: bool = True                  # flex only: compile RoPE, pooling, both branches and the mix together
    # late-duplication options (.agent/notes/duplication_fix_20261027.md); defaults = published method
    ramp_sigma: float = 0.0             # >0: soft opening, pi handed over from 0 at gate_sigma to the table at ramp_sigma
    ramp_mode: str = "linear"           # linear: pi * r(sigma)^gamma; head: head h opens at ramp + (gate - ramp) * pi_h
    ramp_gamma: float = 1.0             # exponent of the linear ramp r = clip((gate - sigma) / (gate - ramp), 0, 1)
    glob_pos: str = "floor"             # floor: coarse ids floor(p / f) (published); frac: p / f (distinct per query)
    win_sched: tuple = ()               # ((sigma_k, window_k), ...): local window window_k while sigma > sigma_k
    text_local: float = 0.0             # logit bias on text keys in the local branch (0 = off, e.g. -2.77 = log 1/16)
    text_local_lo: float = 0.0          # text_local applies only while sigma > text_local_lo
    bridge_cells: int = 0               # >0: AReA bridge, per_cell tokens from each cell of a bridge_cells^2 grid
    bridge_k: int = 16                  # bridge tokens per cell (1024 tokens at 8 x 8 cells)
    bridge_lo: float = 0.0              # bridge active only while sigma > bridge_lo
    lse_tau: float = 0.0                # >0: per-query mix, logit w = logit pi + tau * (lse_loc - lse_glob - mean)
    scale_sched: bool = False           # per-head apparent scale: n0 x n0 keys on the stride-(s / a) lattice, a = 1 .. max(s)
    scale_lo: float = 0.5               # head h reaches a = max(s) at sigma_end = scale_lo + (gate - scale_lo) * pi_h
    scale_pi: dict = field(default_factory=dict)  # {a: PiTable} weight at apparent scale a (default: from pi's file)
    pi: PiTable = field(default_factory=PiTable)
    # calibration
    calib: object | None = None         # sca.calib.CalibStats
    calib_scales: tuple = (2, 4)
    calib_window: int | None = None     # override emulated window side (tests)
    calib_tile: int | None = None       # override emulated tile side (default t * Wc / Wn)
    # runtime state (set by the transformer pre-hook)
    grid: tuple | None = None
    route: float = 0.0                  # >0: hard per-head routing, pi -> 1[pi >= route] (local or global, no blend)
    known_grid: tuple | None = None     # exact (H, W) token grid from the sampler; FLUX builds img_ids in bf16, which rounds positions above 256
    sigma: float = 1.0
    step: int = 0
    scheduler: object | None = None
    n_layers: int = 0
    pos_embed: object | None = None
    log: list = field(default_factory=list)
    pi_trace: list = field(default_factory=list)

    def __post_init__(self):
        self._rope_cache, self._pi_now, self._img_ids = {}, None, None
        self._flex_cache = {}
        self._scale_now, self._scale_tables = None, {}
        if self.local_impl not in ("flex", "gather"):
            raise ValueError(f"local_impl must be flex or gather, got {self.local_impl}")
        if (self.bridge_cells or self.lse_tau) and (self.local_impl != "flex" or self.band):
            raise ValueError("bridge and lse mixing need local_impl flex and band 0")
        if self.ramp_mode not in ("linear", "head"):
            raise ValueError(f"ramp_mode must be linear or head, got {self.ramp_mode}")
        if self.ramp_sigma > 0 and not 0 < self.ramp_sigma < self.gate_sigma:
            raise ValueError("ramp_sigma needs 0 < ramp_sigma < gate_sigma")
        if self.glob_pos not in ("floor", "frac"):
            raise ValueError(f"glob_pos must be floor or frac, got {self.glob_pos}")
        if self.scale_sched:
            if self.band or self.bridge_cells or self.lse_tau or self.ramp_sigma or self.route:
                raise ValueError("scale_sched does not combine with band, bridge, lse mixing, pi ramp or routing")
            if self.pool != "center":
                raise ValueError("scale_sched takes its lattice keys at block centres (pool center)")
            if not 0 <= self.scale_lo < self.gate_sigma:
                raise ValueError("scale_sched needs 0 <= scale_lo < gate_sigma")
        self.win_sched = tuple(sorted(((float(a), int(b)) for a, b in self.win_sched), reverse=True))

    def flex_layout(self, L, tile, win, device):
        """BlockMask and permutations, cached per (text length, grid, tile, window, device)."""
        key = (L, self.grid, tuple(tile), tuple(win), str(device))
        if key not in self._flex_cache:
            self._flex_cache[key] = flex_layout(L, self.grid, tile, win, device)
        return self._flex_cache[key]

    # -- state
    def set_inputs(self, img_ids, txt_ids=None, sigma=None):
        img_ids = img_ids[0] if img_ids.ndim == 3 else img_ids
        if self._img_ids is None or self._img_ids.shape != img_ids.shape or not torch.equal(self._img_ids, img_ids):
            self._img_ids, self._rope_cache = img_ids.detach().clone(), {}
            if self.known_grid is not None:
                self.grid = tuple(self.known_grid)
            else:
                r, c = img_ids[:, 1], img_ids[:, 2]
                self.grid = (int(r.max() - r.min()) + 1, int(c.max() - c.min()) + 1)
            assert self.grid[0] * self.grid[1] == img_ids.shape[0], "image ids must form a full grid"
        if sigma is not None:
            self.sigma = float(sigma)
        self._pi_now = None
        self._scale_now = None

    def pi_now(self, layer, heads, device):
        if self._pi_now is None:
            p = np.atleast_2d(np.asarray(self.pi(self.sigma), dtype=np.float64))  # [layers|1, heads|1]
            if p.shape[0] not in (1, self.n_layers) or p.shape[1] not in (1, heads):
                raise ValueError(f"pi table shape {p.shape} does not match {self.n_layers} layers x {heads} heads")
            p = self.ramp(p)
            self._pi_now = torch.as_tensor(p, dtype=torch.float32, device=device)
            self.pi_trace.append({"step": self.step, "sigma": self.sigma, "pi_mean": float(p.mean())})
        p = self._pi_now
        return p[layer if p.shape[0] > 1 else 0].expand(heads)

    def ramp(self, p):
        """Soft opening phase: hand pi over from 0 (global only) at gate_sigma to the table at ramp_sigma."""
        if self.ramp_sigma <= 0 or self.gate_sigma <= 0 or self.sigma is None:
            return p
        s, g, r = float(self.sigma), float(self.gate_sigma), float(self.ramp_sigma)
        if self.ramp_mode == "head":  # heads that prefer the global branch (small pi) open later
            return np.where(s < r + (g - r) * p, p, 0.0)
        return p * float(np.clip((g - s) / (g - r), 0.0, 1.0)) ** self.ramp_gamma

    def window_now(self):
        """Local window side at the current sigma (win_sched, else window or n0)."""
        for s, w in self.win_sched:
            if self.sigma is not None and float(self.sigma) > s:
                return w
        return self.window or self.n0

    def text_bias_now(self):
        if self.text_local and (self.sigma is None or float(self.sigma) > self.text_local_lo):
            return float(self.text_local)
        return 0.0

    def coarse_rope(self, f, device):
        """(rope_q, rope_k): image queries at floor(id / f) (glob_pos frac: id / f), pooled keys at block centres."""
        key = (f, self.glob_pos, str(device))
        if key not in self._rope_cache:
            ids = self._img_ids.to(device).float()
            div = torch.tensor([1.0, f[0], f[1]], device=device)
            rel = (ids[:, 1:] - ids[:, 1:].amin(0)) / div[1:]
            q_ids = torch.cat([ids[:, :1], rel if self.glob_pos == "frac" else torch.floor(rel)], 1)
            H, W = self.grid
            k_ids = q_ids.view(H, W, 3)[f[0] // 2::f[0], f[1] // 2::f[1]].reshape(-1, 3)
            self._rope_cache[key] = (self.pos_embed(q_ids), self.pos_embed(k_ids))
        return self._rope_cache[key]

    def geometry(self, scale=None):
        """(f, tile, window) for sca mode (scale None) or a calibration emulation at scale s_c."""
        H, W = self.grid
        Wn = self.window or self.n0
        if scale is None:
            Wn = self.window_now()
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


    # -- per-head apparent scale (scale_sched)
    def scale_tables(self, s_max):
        """{a: PiTable} mix weight at apparent scale a > 1: scale_pi[a], else the calibrated scale nearest to a (log
        scale) in the file pi was loaded from, else pi itself."""
        if s_max not in self._scale_tables:
            out = {}
            for a in scale_ladder(s_max)[1:]:
                if a in self.scale_pi:
                    out[a] = self.scale_pi[a]
                elif getattr(self.pi, "path", None) is not None:
                    out[a] = PiTable.load(self.pi.path, self.pi.variant, None, self.pi.sigma_star, a)
                    self.log.append(f"scale_sched: apparent scale {a} mixes with the calibrated scale-{out[a].scale} pi")
                else:
                    out[a] = self.pi
                    self.log.append(f"scale_sched: apparent scale {a} mixes with the {self.pi.variant} pi")
            self._scale_tables[s_max] = out
        return self._scale_tables[s_max]

    def scale_now(self, layer, heads, device):
        """(a, a_t, w_t) of this layer at the current sigma: apparent scale per head (numpy and device tensor) and the
        mix weight at that scale (device float32, 0 at a = 1). Computed for every layer once per step, so the device
        copies cost one transfer per step."""
        if self._scale_now is None:
            n = max(self.n_layers, 1)

            def table(t):
                p = np.atleast_2d(np.asarray(t(self.sigma), dtype=np.float64))
                if p.shape[0] not in (1, n) or p.shape[1] not in (1, heads):
                    raise ValueError(f"pi table shape {p.shape} does not match {n} layers x {heads} heads")
                return np.broadcast_to(p, (n, heads))
            s_max = max(pool_factors(self.grid, self.n0, self.log))
            a = apparent_scale(self.sigma, table(self.pi), self.gate_sigma, self.scale_lo, s_max)
            w = np.zeros((n, heads))
            for s, t in self.scale_tables(s_max).items():
                w = np.where(a == s, table(t), w)
            self._scale_now = (a, torch.as_tensor(a, device=device), torch.as_tensor(w, dtype=torch.float32, device=device))
            self.pi_trace.append({"step": self.step, "sigma": self.sigma, "pi_mean": float(w.mean()),
                                  "scale_frac": {str(s): float((a == s).mean()) for s in scale_ladder(s_max)}})
        a, a_t, w_t = self._scale_now
        i = layer if a.shape[0] > 1 else 0
        return a[i], a_t[i], w_t[i]

    def scale_geometry(self, a):
        """(stride, window) of apparent scale a: stride max(1, s // a) per axis (a divisor of the axis), window
        min(Wn, lattice side) on the (H / dh) x (W / dw) lattice."""
        H, W = self.grid
        f = pool_factors(self.grid, self.n0, self.log)
        d = tuple(fit_divisor(n, max(1, s // a), self.log, "scale stride") for n, s in zip(self.grid, f))
        Wn = self.window_now()
        return d, (min(Wn, H // d[0]), min(Wn, W // d[1]))

    def scale_layout(self, L, tile, stride, win, heads, a, device):
        """HeadLayout for apparent scale a, cached per (text length, grid, tile, stride, window, heads, a, device);
        a is part of the key so that two groups never share block counts."""
        key = ("scale", L, self.grid, tuple(tile), tuple(stride), tuple(win), heads, a, str(device))
        if key not in self._flex_cache:
            self._flex_cache[key] = head_layout(strided_flex_layout(L, self.grid, tile, stride, win, device), heads)
        return self._flex_cache[key]


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

    def branches(self, q, k, v, image_rotary_emb, f, tile, win, qr=None, kr=None, permuted=False, text_bias=0.0):
        """(L, o_loc, o_glob, inv). Text q/k RoPE is shared by both branches. With permuted=True
        and the flex path, o_loc is the raw flex output and inv the un-permutation (else None)."""
        ctx = self.ctx
        L, rope_txt = self._split(q, image_rotary_emb)
        cos, sin = image_rotary_emb
        if kr is None:
            kr = rope(k, image_rotary_emb)
        if qr is None:
            q_txt, qi = rope(q[:, :L], rope_txt), rope(q[:, L:], (cos[L:], sin[L:]))
        else:
            q_txt, qi = qr[:, :L], qr[:, L:]
        rq, rk = ctx.coarse_rope(f, q.device)
        o_glob = global_branch(q, k, v, L, ctx.grid, f, ctx.pool, rope_txt, rq, rk, q_txt, kr[:, :L])
        if ctx.local_impl == "flex":
            layout = ctx.flex_layout(L, tile, win, q.device)
            o_loc = local_branch_flex(qi, kr, v, L, layout, permuted, text_bias)
            return L, o_loc, o_glob, (layout.q_inv if permuted else None)
        return L, local_branch(qi, kr, v, L, ctx.grid, tile, win, ctx.chunk, text_bias), o_glob, None

    def sca(self, q, k, v, image_rotary_emb):
        ctx = self.ctx
        f, tile, win = ctx.geometry()
        if ctx.gate_sigma > 0 and ctx.sigma is not None and float(ctx.sigma) > ctx.gate_sigma:
            L, rope_txt = self._split(q, image_rotary_emb)
            cos, sin = image_rotary_emb
            rq, rk = ctx.coarse_rope(f, q.device)
            kr_txt = rope(k[:, :L], rope_txt)
            return global_branch(q, k, v, L, ctx.grid, f, ctx.pool, rope_txt, rq, rk,
                                 rope(q[:, :L], rope_txt), kr_txt)
        if ctx.scale_sched:
            return self.scale_sched(q, k, v, image_rotary_emb, f, tile)
        pi = ctx.pi_now(self.layer, q.shape[2], q.device).view(1, 1, -1, 1)
        if ctx.route > 0:
            pi = (pi >= ctx.route).to(pi.dtype)
        band = ctx.band
        if band < 0:  # SNR-adaptive: p = c * sigma / (1 - sigma) tokens, off below 2
            sg = float(ctx.sigma) if ctx.sigma is not None else 0.0
            band = int(round(-band * sg / max(1e-3, 1 - sg)))
            band = min(band, min(ctx.grid))
        tb = ctx.text_bias_now()
        if ctx.bridge_cells or ctx.lse_tau:
            L, _ = self._split(q, image_rotary_emb)
            rq, rk = ctx.coarse_rope(f, q.device)
            on = ctx.bridge_cells and (ctx.sigma is None or float(ctx.sigma) > ctx.bridge_lo)
            H, W = ctx.grid
            cells = (fit_divisor(H, ctx.bridge_cells, ctx.log, "bridge cells"),
                     fit_divisor(W, ctx.bridge_cells, ctx.log, "bridge cells")) if on else None
            per = min(ctx.bridge_k, (H // cells[0]) * (W // cells[1])) if on else 0
            layout = ctx.flex_layout(L + (cells[0] * cells[1] * per if on else 0), tile, win, q.device)
            return sca_dedup_fn(q.device)(q, k, v, tuple(image_rotary_emb), rq, rk, pi, L, ctx.grid, f, ctx.pool,
                                          layout, tb, cells, per, float(ctx.lse_tau))
        if band > 1:
            L, o_loc, o_glob, inv = self.branches(q, k, v, image_rotary_emb, f, tile, win, permuted=True, text_bias=tb)
            if inv is not None:
                o_loc = o_loc.transpose(1, 2).index_select(1, inv)
            return band_mix_fn(q.device)(o_glob, o_loc, pi, L, ctx.grid, band)
        if ctx.local_impl == "flex" and ctx.fused:
            L, _ = self._split(q, image_rotary_emb)
            rq, rk = ctx.coarse_rope(f, q.device)
            layout = ctx.flex_layout(L, tile, win, q.device)
            args = (q, k, v, tuple(image_rotary_emb), rq, rk, pi, L, ctx.grid, f, ctx.pool, layout)
            return sca_flex_fn(q.device)(*args, tb) if tb else sca_flex_fn(q.device)(*args)
        L, o_loc, o_glob, inv = self.branches(q, k, v, image_rotary_emb, f, tile, win, permuted=True, text_bias=tb)
        return mix_fn(q.device)(o_glob, o_loc, pi, L, inv)

    def scale_sched(self, q, k, v, image_rotary_emb, f, tile):
        """Per-head apparent scale (SCAContext.scale_sched), below the gate: heads at a = 1 take O_glob, each group of
        heads at a > 1 mixes its strided local branch with the weight at its scale."""
        ctx = self.ctx
        heads = q.shape[2]
        L, rope_txt = self._split(q, image_rotary_emb)
        cos, sin = image_rotary_emb
        rq, rk = ctx.coarse_rope(f, q.device)
        a, a_t, w_t = ctx.scale_now(self.layer, heads, q.device)
        if (a <= 1).all():  # every head at apparent scale 1: its local view is the global branch
            return global_branch(q, k, v, L, ctx.grid, f, ctx.pool, rope_txt, rq, rk, rope(q[:, :L], rope_txt),
                                 rope(k[:, :L], rope_txt))
        tb, pi, groups = ctx.text_bias_now(), w_t.view(1, 1, -1, 1), []
        for s in scale_ladder(max(f))[1:]:
            d, win = ctx.scale_geometry(s)
            rqd, rkd = ((cos[L:], sin[L:]),) * 2 if d == (1, 1) else ctx.coarse_rope(d, q.device)
            if ctx.local_impl == "gather":
                if (a == s).any():
                    groups.append((d, win, rqd, rkd, torch.as_tensor(np.flatnonzero(a == s), device=q.device)))
                continue
            hl = ctx.scale_layout(L, tile, d, win, heads, s, q.device)  # every scale, so one graph serves all steps
            member = a_t == s
            hl.set_heads(member)
            groups.append((d, rqd, rkd, hl.layout, member.view(1, 1, -1, 1)))
        if ctx.local_impl == "gather":
            return sca_scale_gather(q, k, v, tuple(image_rotary_emb), rq, rk, pi, L, ctx.grid, f, ctx.pool, tile,
                                    tuple(groups), ctx.chunk, tb)
        args = (q, k, v, tuple(image_rotary_emb), rq, rk, pi, L, ctx.grid, f, ctx.pool, tuple(groups))
        return sca_scale_fn(q.device)(*args, tb) if tb else sca_scale_fn(q.device)(*args)

    def calibrate(self, q, k, v, image_rotary_emb):
        ctx = self.ctx
        qr, kr = rope(q, image_rotary_emb), rope(k, image_rotary_emb)
        full = sdpa(qr, kr, v)
        for i, s in enumerate(ctx.calib_scales):
            f, tile, win = ctx.geometry(s)
            L, o_loc, o_glob, _ = self.branches(q, k, v, image_rotary_emb, f, tile, win, qr, kr)
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


def scale_sched_cost(height, width, n0=64, tile=8, text=512, heads=24, head_dim=128, q_tile=None, k_tile=None):
    """Local-branch cost of every apparent scale of the scale schedule (SCAContext.scale_sched) at one resolution.

    keys: keys per image query per head (text + window); flops: QK^T + PV FLOPs of the local branch per layer over
    all heads; blocks: flex key blocks visited per query block (strided_flex_layout geometry, mean over query blocks,
    text blocks included; exact when the grid is a multiple of the query tile). Ratios are to the published local
    branch (stride 1). Apparent scale 1 runs no local branch (its output is O_glob)."""
    H, W = height // 16, width // 16
    s = pool_factors((H, W), n0)
    tl = (fit_divisor(H, tile), fit_divisor(W, tile))
    qt, kt = q_tile or FLEX_Q_TILE, k_tile or FLEX_K_TILE

    def axis_blocks(n, t, d, w, qb, kb):
        x = np.arange(-(-n // qb) * qb)
        x = np.where(x < n, x, 0)                                       # padding query slots read token 0
        start = np.clip(((x // t) * t + t // 2) // d - w // 2, 0, n // d - w).reshape(-1, qb)
        off = (w // 2 - (t // 2) // d) % kb
        return (start.max(1) + w - 1 + off) // kb - (start.min(1) + off) // kb + 1

    def cost(d, win):
        keys = text + win[0] * win[1]
        per_axis = [axis_blocks(n, t, dd, w, qb, kb) for n, t, dd, w, qb, kb in zip((H, W), tl, d, win, qt, kt)]
        blocks = float(np.outer(*per_axis).mean()) + -(-text // (kt[0] * kt[1]))
        return keys, 4 * heads * head_dim * H * W * keys, blocks

    pub = cost((1, 1), (min(n0, H), min(n0, W)))
    rows = []
    for a in scale_ladder(max(s)):
        d = tuple(fit_divisor(n, max(1, f // a)) for n, f in zip((H, W), s))
        win = (min(n0, H // d[0]), min(n0, W // d[1]))
        keys, flops, blocks = cost(d, win) if a > 1 else (0, 0, 0.0)
        rows.append({"a": a, "stride": list(d), "lattice": [H // d[0], W // d[1]], "window": list(win),
                     "keys": keys, "flops": flops, "blocks": blocks, "keys_ratio": keys / pub[0],
                     "flops_ratio": flops / pub[1], "blocks_ratio": blocks / pub[2]})
    return {"res": f"{height}x{width}", "grid": [H, W], "pool": list(s), "published_keys": pub[0],
            "published_flops": pub[1], "published_blocks": pub[2], "scales": rows}


if __name__ == "__main__":
    import json
    for hw in ((1024, 1024), (2048, 2048), (4096, 4096), (2048, 4096)):
        r = attention_flops(*hw)
        print(json.dumps({k: (f"{v:.3e}" if isinstance(v, float) and v > 1e4 else v) for k, v in r.items()}))
