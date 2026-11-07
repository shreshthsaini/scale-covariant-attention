"""Morph ladder between SCA (sca_v2) and AReA: single-component swaps (round 7, user directive 2026-11-06 evening;
.agent/notes/duplication_fix_round7.md, "Morph ladder").

SCA side (SCAContext.morph_local / morph_text, both off by default; AReA's own code from third_party AReA):
  morph_local "area_tiles"   sca+B: the local branch is AReA's Phase II window attention (area.attention.
                             windowed_attention): base tiles of SCA's current window side with AReA's per-step boundary
                             jitter, DWT splits, jagged Halo (8 patches) and raised-cosine merge, each window keyed by
                             the text, its tokens, its Halo and SCA's bridge count of AReA-selected outside tokens. SCA's
                             global branch, per-head pi mix, opening and text rule are unchanged.
  morph_text "window_mean"   sca+C: the text queries' output is AReA's rule, the mean over a partition of the image into
                             windows of SCA's current side (text keys, window keys and SCA's bridge), instead of SCA's
                             rule (text queries read only the global branch). Image outputs are sca_v2's.
  morph_text "all_coarse"    root-cause audit Fix 4: the text queries attend the text keys and every image key at its
                             coarse id floor(p / f) (the global branch's query ids), instead of only the 4,096 block
                             centres; offsets stay inside the native range. Image outputs are sca_v2's.
AReA side (scripts/run_area.py): patch_phase1() (SCA's opening, one pooling factor for every Phase I step),
patch_processors() (SCA's global branch mixed into AReA's image output with SCA's calibrated per-head pi, and/or SCA's
text rule), patch_fixed_tiles() (no jitter, no Halo, no DWT splits: a fixed partition, the closest AReA analogue of
SCA's deterministic windows).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

AREA_DIR = Path(os.environ.get("AREA_DIR", "/scratch/09032/saini_2/sca/third_party/AReA"))
MORPH_LOCAL = ("", "area_tiles")
MORPH_TEXT = ("", "window_mean")


def _area():
    if str(AREA_DIR) not in sys.path:
        sys.path.insert(0, str(AREA_DIR))
    from area import attention, tiling   # noqa: F401
    return attention, tiling


def _sdpa(q, k, v):
    """[B, S, h, D] attention."""
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


# ----------------------------------------------------------------------------- SCA side

def even_partition(n, size):
    """Start offsets of AReA's even base tiling of an axis of n tokens by windows of side size (last one flush)."""
    _, tiling = _area()
    return list(tiling.even_starts(n, size))


def text_window_mean(qr_txt, kr_txt, v_txt, kr_img, v_img, grid, window, bridge=None):
    """Mean over the windows of an even partition (side `window`, AReA's even_starts) of the text queries' attention
    to [text keys, the window's image keys, bridge keys]. qr_txt/kr_txt/v_txt [B, L, h, D]; kr_img/v_img [B, N, h, D]
    rotated at true positions; bridge [B, R] image indices or None. Returns [B, L, h, D]."""
    H, W = grid
    wh, ww = min(window[0], H), min(window[1], W)
    rows, cols = even_partition(H, wh), even_partition(W, ww)
    out = torch.zeros_like(qr_txt, dtype=torch.float32)
    extra_k = extra_v = None
    if bridge is not None and bridge.numel():
        g = bridge[:, :, None, None].expand(-1, -1, kr_img.shape[2], kr_img.shape[3])
        extra_k, extra_v = torch.gather(kr_img, 1, g), torch.gather(v_img, 1, g)
    for r0 in rows:
        for c0 in cols:
            idx = (torch.arange(r0, r0 + wh, device=kr_img.device)[:, None] * W
                   + torch.arange(c0, c0 + ww, device=kr_img.device)[None, :]).reshape(-1)
            ks = [kr_txt, kr_img[:, idx]] + ([extra_k] if extra_k is not None else [])
            vs = [v_txt, v_img[:, idx]] + ([extra_v] if extra_v is not None else [])
            out += _sdpa(qr_txt, torch.cat(ks, 1), torch.cat(vs, 1)).float()
    return (out / float(len(rows) * len(cols))).to(qr_txt.dtype)


def text_all_coarse(qr_txt, kr_txt, k_img, v, rope_q):
    """Audit Fix 4: the text queries' attention to [text keys, every image key rotated at its coarse id]. qr_txt/kr_txt
    [B, L, h, D] rotated text queries/keys; k_img [B, N, h, D] pre-RoPE image keys; v [B, L + N, h, D] (text first);
    rope_q the coarse-id tables of every image token (SCAContext.coarse_rope). Returns [B, L, h, D]."""
    from sca.attn import rope, sdpa
    return sdpa(qr_txt, torch.cat([kr_txt, rope(k_img, rope_q)], 1), v)


def install_area_tiles(transformer, ctx, halo=8, seed=0):
    """sca+B: per step after the gate, build AReA's Phase II layout for SCA's current window side from the embedded
    image tokens (x_embedder output, as AReA's FluxBackbone does) and keep it in ctx._area_layout. Returns uninstall."""
    attention, tiling = _area()

    def pre(module, args, kwargs):
        text = kwargs.get("encoder_hidden_states")
        ctx._txt_len = int(text.shape[1]) if text is not None else 0

    def after(module, args, output):
        ctx._area_layout = None
        if ctx.mode != "sca" or ctx.grid is None:
            return
        if ctx.gate_sigma > 0 and ctx.sigma is not None and float(ctx.sigma) > ctx.gate_sigma:
            return
        H, W = ctx.grid
        size = min(ctx.window_now(), max(H, W))
        energy = tiling.detail_map(output, H, W)
        windows = tiling.adaptive_tiling(energy, H, W, size, min(ctx.n0, max(H, W)), ctx._txt_len,
                                         rng=tiling.step_rng(seed, int(ctx.step), 0x5EA1))
        cells, per = ctx.bridge_now()
        registers = cells[0] * cells[1] * per if per else 0
        registers, h_used = attention.fit_extra_keys(windows, H, W, ctx._txt_len, registers, halo)
        ctx._area_layout = attention.build_fine_layout(windows, H, W, h_used, registers, seed, int(ctx.step),
                                                       output.dtype, output.device)

    hs = [transformer.register_forward_pre_hook(pre, with_kwargs=True), transformer.x_embedder.register_forward_hook(after)]

    def uninstall():
        for h in hs:
            h.remove()
    return uninstall


def morph_forward(proc, q, k, v, image_rotary_emb, f, tile, win, pi):
    """SCAProcessor post-gate step with the morph options (ctx.morph_local / ctx.morph_text). Returns [B, L + N, h, D]."""
    from sca.attn import bridge_index, global_branch, rope, sca_dedup_fn
    ctx = proc.ctx
    L, rope_txt = proc._split(q, image_rotary_emb)
    rq, rk = ctx.coarse_rope(f, q.device)
    cells, per = ctx.bridge_now()
    if ctx.morph_local == "area_tiles":
        attention, _ = _area()
        layout = getattr(ctx, "_area_layout", None)
        if layout is None:
            raise RuntimeError("morph_local area_tiles needs install_area_tiles() on the transformer")
        qr, kr = rope(q, image_rotary_emb), rope(k, image_rotary_emb)
        _, o_img = attention.windowed_attention(qr[:, :L], kr[:, :L], v[:, :L], qr[:, L:], kr[:, L:], v[:, L:],
                                                k[:, L:], layout)
        o_glob = global_branch(q, k, v, L, ctx.grid, f, ctx.pool, rope_txt, rq, rk, qr[:, :L], kr[:, :L])
        p = pi.float().view(1, 1, -1, 1)
        img = (p * o_img.float() + (1 - p) * o_glob[:, L:].float()).to(o_glob.dtype)
        out = torch.cat([o_glob[:, :L], img], 1)
    else:   # sca_v2's own operator (bridge path), image rows unchanged
        layout = ctx.flex_layout(L + (cells[0] * cells[1] * per if per else 0), tile, win, q.device)
        out = sca_dedup_fn(q.device)(q, k, v, tuple(image_rotary_emb), rq, rk, pi, L, ctx.grid, f, ctx.pool, layout,
                                     ctx.text_bias_now(), cells, per, float(ctx.lse_tau))
    if ctx.morph_text == "all_coarse":
        out = torch.cat([text_all_coarse(rope(q[:, :L], rope_txt), rope(k[:, :L], rope_txt), k[:, L:], v,
                                         rq).to(out.dtype), out[:, L:]], 1)
    if ctx.morph_text == "window_mean":
        qr, kr = rope(q, image_rotary_emb), rope(k, image_rotary_emb)
        bridge = bridge_index(k[:, L:], ctx.grid, cells, per) if per else None
        H, W = ctx.grid
        w = min(ctx.window_now(), H, W)
        out = torch.cat([text_window_mean(qr[:, :L], kr[:, :L], v[:, :L], kr[:, L:], v[:, L:], ctx.grid, (w, w),
                                          bridge).to(out.dtype), out[:, L:]], 1)
    return out


# ----------------------------------------------------------------------------- AReA side (scripts/run_area.py)

def patch_phase1(backbone, factors):
    """Every Phase I step pools by the given factors (e.g. SCA's opening: 4 for all 8 steps); Phase II unchanged."""
    from area.schedule import Schedule
    original = backbone.schedule

    def schedule(h, w, steps, txt_len=0):
        s = original(h, w, steps, txt_len)
        if s is None or not s.factors:
            return s
        fs = tuple(int(x) for x in factors)
        if len(fs) != len(s.factors):
            raise ValueError(f"{len(fs)} factors for {len(s.factors)} Phase I steps")
        return Schedule(s.grid, fs, s.windows)
    backbone.schedule = schedule
    return original


def pi_at_window(table2, table4, sigma, window, extent):
    """[layers, heads] SCA's calibrated pi for a window of side `window` on an `extent`-token canvas: log-linear in the
    emulated ratio extent / window between the scale-2 and scale-4 tables, clamped to [2, 4] (SCAContext.table_interp)."""
    r = min(max(extent / float(window), 2.0), 4.0)
    t = math.log(r / 2.0) / math.log(2.0)
    return (1 - t) * np.asarray(table2(sigma), dtype=np.float64) + t * np.asarray(table4(sigma), dtype=np.float64)


def patch_processors(backbone, pi_path=None, text_rule=False, pool_f=4, sigmas=None):
    """Wrap every FluxAReAProcessor so that on Phase II steps the image output becomes pi * AReA + (1 - pi) * SCA's
    global branch (pi_path: SCA's calibration npz) and/or the text output becomes the global branch's (text_rule).
    SCA's global branch = AReA's own coarse attention at pooling factor pool_f (block centre, coarse ids). `sigmas` maps a
    step to its sigma (for the pi table); the backbone's state gives the step and window side."""
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.embeddings import apply_rotary_emb
    from area import attention
    from area.backbones.flux import FluxAReAProcessor, _qkv
    tables = None
    if pi_path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from sca.attn import PiTable
        tables = (PiTable.load(pi_path, "head", 2), PiTable.load(pi_path, "head", 4))
    tr = backbone.pipe.transformer
    blocks = list(tr.transformer_blocks) + list(tr.single_transformer_blocks)

    class Swapped(FluxAReAProcessor):
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
            st = self.backbone.state
            if st.phase != "fine":
                return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
            query, key, value = _qkv(attn, hidden_states, encoder_hidden_states)
            t = st.txt_len
            q = apply_rotary_emb(query, st.data["rope"], sequence_dim=1)
            k = apply_rotary_emb(key, st.data["rope"], sequence_dim=1)
            out_txt, out_img = attention.windowed_attention(
                q[:, :t], k[:, :t], value[:, :t], q[:, t:], k[:, t:], value[:, t:], key[:, t:],
                st.data["layout"], backend=self._attention_backend)
            rope_q, rope_kv = self.backbone._coarse_ropes(st.h, st.w, pool_f, t, query.device)
            kp, vp = attention.pool_keys_values(key, value, t, st.h, st.w, pool_f, "subsample")
            glob = dispatch_attention_fn(apply_rotary_emb(query, rope_q, sequence_dim=1),
                                         apply_rotary_emb(kp, rope_kv, sequence_dim=1), vp, attn_mask=None,
                                         backend=self._attention_backend)
            if tables is not None:
                sched = backbone.pipe.scheduler
                sigma = float(sigmas[st.step]) if sigmas is not None else float(sched.sigmas[min(st.step, len(sched.sigmas) - 1)])
                p = pi_at_window(tables[0], tables[1], sigma, st.size, max(st.h, st.w))
                p = torch.as_tensor(p[self.block if p.shape[0] > 1 else 0], dtype=torch.float32,
                                    device=query.device).view(1, 1, -1, 1)
                out_img = (p * out_img.float() + (1 - p) * glob[:, t:].float()).to(out_img.dtype)
            if text_rule:
                out_txt = glob[:, :t].to(out_txt.dtype)
            out = torch.cat([out_txt, out_img], dim=1).flatten(2, 3).to(query.dtype)
            if encoder_hidden_states is not None:
                txt_out, img_out = out.split_with_sizes([t, out.shape[1] - t], dim=1)
                img_out = attn.to_out[1](attn.to_out[0](img_out.contiguous()))
                return img_out, attn.to_add_out(txt_out.contiguous())
            return out

    for i, block in enumerate(blocks):
        block.attn.set_processor(Swapped(backbone, i))


def patch_fixed_tiles(backbone):
    """AReA's Phase II with a fixed partition: even base tiles (no boundary jitter), no DWT splits, no Halo (pass
    --halo 0 too); bridge and merge as AReA's."""
    from area import attention, tiling

    def after(module, args, output):
        st = backbone.state
        if st.phase != "fine":
            return
        cfg = backbone.config
        rows, cols = tiling.even_starts(st.h, st.size), tiling.even_starts(st.w, st.size)
        windows = [tiling.Window(r, c, min(st.h - 1, r + st.size - 1), min(st.w - 1, c + st.size - 1))
                   for r in rows for c in cols]
        registers = max(1, int(round(st.size * cfg.bridge_ratio))) ** 2 if cfg.bridge_ratio > 0 else 0
        registers, halo = attention.fit_extra_keys(windows, st.h, st.w, st.txt_len, registers, 0)
        st.data["layout"] = attention.build_fine_layout(windows, st.h, st.w, 0, registers, cfg.seed, st.step,
                                                        output.dtype, output.device)
    backbone._after_embedding = after
    # re-register the x_embedder hook with the new function
    for hook in backbone._hooks[1:]:
        hook.remove()
    backbone._hooks = backbone._hooks[:1] + [backbone.pipe.transformer.x_embedder.register_forward_hook(after)]
