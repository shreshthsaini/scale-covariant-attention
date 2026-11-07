"""Text-side late-duplication options (round 6; .agent/notes/duplication_fix_round6.md).

The window branch of SCA reads the prompt through one softmax over the text keys, the bridge tokens and its window.
These options change only how much, and which, text a window may read; image keys, positions, the bridge and the pi
mix stay as in the adopted operator (sca_v2). To act on the text keys alone, the window softmax is computed in two
exact parts, the text part (dense, over the L text keys) and the image part (flex: bridge plus window, with its
log-sum-exp), and merged as one softmax: o = (e^a o_T + e^b o_I) / (e^a + e^b), a and b the two log-sum-exps.

Modes (SCAContext.text_mode; several may be joined with "+"):
  ctw  calibrated text weight. The text logits of a window get beta_lh(sigma, r) = log(share), share = the fraction of
       full attention's image mass that falls inside the emulated window at 1024^2 (calibration masses, ratio r of the
       current window). With it the window's text-to-image balance is the one full attention has at the target, per
       head and noise level: in full attention the text competes with every image key, in a window only with the
       near ones. Measured once at 1024^2, transferred by the ratio, as pi is.
  gtg  global text gate. The window's logit on text token j at query i gets gamma * min(0, log(m_ij / mean_i m_ij)),
       m_ij = the global branch's attention probability on token j at query i (mean over heads). A window may not
       attend to a word more, relative to that word's average over the image, than the global view does at that
       position: where the global layout puts no subject, the window cannot read the subject's words at full strength.
  gtx  global text read-out. The window's text part is the global branch's (coarse ids, global text mass a_g); the
       window only re-weights image context: O_loc = a_g O_T(coarse) + (1 - a_g) O_I.
  ffc  far-field calibrated bridge (image side). The window's image part is split into its window keys and its bridge
       keys (the far-field sample); their outputs are mixed so the far field carries the share of the image mass that
       full attention gives it, 1 - share_lh(sigma, r), from the same calibration masses as ctw; the text part and the
       text-to-image balance stay those of sca_v2 (raw partition of window plus bridge). Needs the bridge.
  ffl  as ffc, with the far field read from a uniform lattice (the block-centre token of every s x s block, 4,096 keys at
       4096^2, true positions) instead of the outlier-selected bridge, so the far-field sample covers every region,
       the main subject included, in proportion to its area. Does not need the bridge.
  sb   subject bridge: the bridge tokens are the block-centre tokens most attended by the subject words in the global
       branch (text queries of the subject tokens, SCAContext.subject_tokens, over the block keys), instead of AReA's
       farthest-from-mean keys; prompts without subject words keep AReA's selection. Combines with bw.
  bw   weighted bridge (image side, combines with the others); SCAContext.text_bw_scale scales its log(N / R). Each bridge key gets the logit offset log(N / R), the
       number of image tokens it stands for, so the far field enters the window softmax with the mass full attention
       would give it rather than as R single tokens.
Round-7 options of ffl (sca_text_forward keywords; SCAContext.ffl_*, .agent/notes/duplication_fix_round7.md):
  near="bridge"  the near part is sca_v2's window plus bridge (bw weights its bridge keys) instead of the window alone,
                 so a head or step whose far weight is 0 is exactly sca_v2.
  lat_w          [h] weights in [0, 1] (head set x sigma fade): the far field enters the image part with weight
                 lat_w * far and the window's partition with weight lat_w (lattice logits + log lat_w), so lat_w = 0
                 is the near part alone, partition included, and lat_w = 1 is round-6 ffl.
  lat            BlockMask from lattice_layout(): lattice keys inside the query's own window are left out of the far
                 field, which then holds only what the window cannot see.
  far_pos        "true": lattice keys and queries at their true positions (round 6); "coarse": at the global branch's
                 coarse ids (queries floor(p / f), keys their block ids), so every query-key offset stays within the
                 native grid (at most n0), as in the global branch, instead of reaching max(H, W).
Off by default; the published and sca_v2 paths (sca.attn.sca_flex_forward, sca_dedup_forward) are untouched.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from sca.attn import _FLEX, _envfix, _raise_recompile_limit, bridge_index, flex_fn, global_branch_lse, \
    local_branch_flex, rope

MODES = ("ctw", "gtg", "gtx", "bw", "ffc", "ffl", "sb")


def parse_mode(text):
    """'ctw+gtg' -> ('ctw', 'gtg'); '' -> (). gtx replaces the window's text part, so it combines only with bw. ffl+bw
    weights the bridge of ffl's near part and needs near "bridge" (checked by SCAContext)."""
    parts = tuple(p for p in str(text or "").split("+") if p)
    bad = [p for p in parts if p not in MODES]
    if bad:
        raise ValueError(f"unknown text mode(s) {bad}; choose from {MODES}")
    if "ffc" in parts and "ffl" in parts:
        raise ValueError("text modes ffc and ffl are alternatives")
    if ("ffc" in parts and "bw" in parts) or (("ffc" in parts or "ffl" in parts) and "gtx" in parts):
        raise ValueError("text mode ffc sets the bridge's share itself; it does not combine with bw or gtx")
    if "gtx" in parts and ("ctw" in parts or "gtg" in parts):
        raise ValueError("text mode gtx replaces the window's text part and does not combine with ctw or gtg")
    return parts


# ----------------------------------------------------------------------------- calibrated text weight (ctw)

_NPZ = {}


def _npz(path):
    key = str(path)
    if key not in _NPZ:
        z = np.load(key)
        _NPZ[key] = {k: z[k] for k in ("mass_mean", "radii", "sigma_centers", "populated")}
    return _NPZ[key]


def near_share(mass, radii, rho):
    """Fraction of the image-key mass within Chebyshev radius rho, [layers, heads, bins]: mass[1 + j] / (1 - mass[0]),
    log-linear in the radius between the stored radii (clamped to the stored range)."""
    radii = np.asarray(radii, dtype=np.float64)
    img = np.clip(1.0 - mass[0], 1e-6, None)
    logm = np.log(np.clip(mass[1:], 1e-12, None))                                  # [radii, layers, heads, bins]
    x = np.log(np.clip(rho, radii[0], radii[-1]))
    j = int(np.clip(np.searchsorted(np.log(radii), x), 1, len(radii) - 1))
    t = (x - math.log(radii[j - 1])) / (math.log(radii[j]) - math.log(radii[j - 1]))
    m = np.exp((1 - t) * logm[j - 1] + t * logm[j])
    return np.clip(m / img, 1e-6, 1.0)


def ctw_beta(path, sigma, window, grid, n0=64, floor=-6.0):
    """[layers, heads] text-logit offset for a window of side `window` tokens on a `grid` canvas at noise sigma:
    log(near-field share) at the emulated window of side n0 * window / max(grid) (radius half that), interpolated
    over the calibration's sigma bins as PiTable does (populated bins only)."""
    z = _npz(path)
    rho = 0.5 * n0 * window / max(grid)
    beta = np.maximum(np.log(near_share(z["mass_mean"], z["radii"], rho)), floor)  # [layers, heads, bins]
    c = np.asarray(z["sigma_centers"], dtype=np.float64)
    pop = np.asarray(z["populated"], dtype=bool)
    c, beta = c[pop], beta[..., pop]
    j = np.searchsorted(c, sigma)
    if j <= 0:
        return beta[..., 0]
    if j >= len(c):
        return beta[..., -1]
    a = (sigma - c[j - 1]) / (c[j] - c[j - 1])
    return (1 - a) * beta[..., j - 1] + a * beta[..., j]


def ffc_far(path, sigma, window, grid, n0=64, cap=0.95):
    """[layers, heads] share of full attention's image mass outside the emulated window (1 - near share), capped."""
    return np.minimum(1.0 - np.exp(ctw_beta(path, sigma, window, grid, n0, floor=-30.0)), cap)


# ----------------------------------------------------------------------------- exact split of the window softmax

def text_part(qr, kt, vt, head_bias=None, tok_bias=None):
    """Attention of image queries qr [B, N, h, D] (RoPE'd) over the text keys kt / values vt [B, L, h, D] alone, with
    optional additive logit offsets head_bias [h] and tok_bias [B, N, L]. Returns o [B, N, h, D], lse [B, h, N]."""
    kw = {"return_lse": True}
    if head_bias is not None or tok_bias is not None:
        hb, tb = head_bias, tok_bias

        def score_mod(score, b, h, q, kv):
            if hb is not None:
                score = score + hb[h]
            if tb is not None:
                score = score + tb[b, q, kv]
            return score
        kw["score_mod"] = score_mod
    o, lse = flex_fn(qr.device)(qr.transpose(1, 2), kt.transpose(1, 2), vt.transpose(1, 2), **kw)
    return o.transpose(1, 2), lse


def _hi(x):
    """x in fp32 at least (float64 stays float64, for the CPU tests)."""
    return x if x.dtype == torch.float64 else x.float()


def merge_lse(o1, l1, o2, l2):
    """One softmax from two disjoint key sets: o [B, N, h, D] with natural-log lse [B, h, N]; fp32 (or fp64) result."""
    a, b = _hi(l1).transpose(1, 2)[..., None], _hi(l2).transpose(1, 2)[..., None]
    m = torch.maximum(a, b)
    wa, wb = torch.exp(a - m), torch.exp(b - m)
    return (wa * _hi(o1) + wb * _hi(o2)) / (wa + wb)


def lattice_index(grid, f, device=None):
    """Row-major image index of the block-centre token of every f[0] x f[1] block (pool "center" positions)."""
    H, W = grid
    return torch.arange(H * W, device=device).view(H, W)[f[0] // 2::f[0], f[1] // 2::f[1]].reshape(-1)


def lattice_layout(grid, tile, window, f, device, block=(128, 128), inside=False):
    """(BlockMask, kernel_options) over image queries (row-major) x block-centre lattice keys (row-major): a lattice key
    is visible to a query iff it lies outside the query's local window (the window of its query tile, the same start
    rule as sca.attn.flex_layout), or inside it with inside=True. With it the lattice far field holds only what the window
    does not see."""
    from torch.nn.attention.flex_attention import create_block_mask
    from torch._dynamo import config as dynamo_config
    _envfix()
    _raise_recompile_limit()
    (H, W), (th, tw), (Wh, Ww) = grid, tile, window
    i32 = torch.int32
    rs = (torch.arange(H, device=device) // th * th + th // 2 - Wh // 2).clamp(0, H - Wh).to(i32)
    cs = (torch.arange(W, device=device) // tw * tw + tw // 2 - Ww // 2).clamp(0, W - Ww).to(i32)
    cen = lattice_index(grid, f, device)
    kr, kc = (cen // W).to(i32), (cen % W).to(i32)

    def mask_mod(b, h, q, kv):
        r0, c0, r, c = rs[q // W], cs[q % W], kr[kv], kc[kv]
        hit = (r >= r0) & (r < r0 + Wh) & (c >= c0) & (c < c0 + Ww)
        return hit if inside else ~hit

    N, Nb = H * W, cen.numel()
    bs = (math.gcd(N, block[0]), math.gcd(Nb, block[1]))
    cuda = torch.device(device).type == "cuda"
    with dynamo_config.patch(automatic_dynamic_shapes=False):
        bm = create_block_mask(mask_mod, None, None, N, Nb, device=device, BLOCK_SIZE=bs, _compile=cuda)
    ko = {"BLOCK_M": math.gcd(bs[0], 128), "BLOCK_N": math.gcd(bs[1], 64), "USE_TMA": True} if cuda else None
    return bm, ko


def lattice_part(qi, kc, vc, lat=None):
    """Attention of image queries qi [B, N, h, D] (RoPE'd, row-major) over the lattice keys kc / values vc
    [B, Nb, h, D] at their true positions, dense or under lattice_layout's (BlockMask, kernel_options). Returns
    o [B, N, h, D], lse [B, h, N]; a query that sees no lattice key gets o = 0 and lse = -inf."""
    if lat is None:
        return text_part(qi, kc, vc)
    bm, ko = lat
    kw = {"kernel_options": ko} if ko else {}
    o, lse = flex_fn(qi.device)(qi.transpose(1, 2), kc.transpose(1, 2), vc.transpose(1, 2), block_mask=bm,
                                return_lse=True, **kw)
    return o.transpose(1, 2), lse


def text_map(qg, kt, lse_img, chunk=8192):
    """Head-mean probability [B, N, L] that the global branch puts on each text key: mean_h exp(s_hij - lse_hi),
    s = coarse-RoPE'd image query . text key / sqrt(D) (fp32), lse_img [B, h, N] the global branch's log-sum-exp."""
    B, N, h, D = qg.shape
    kT = _hi(kt.permute(0, 2, 3, 1))                                              # [B, h, D, L]
    out = []
    for c0 in range(0, N, chunk):
        s = torch.matmul(_hi(qg[:, c0:c0 + chunk].permute(0, 2, 1, 3)), kT) * (D ** -0.5)   # [B, h, n, L]
        out.append(torch.exp(s - _hi(lse_img[:, :, c0:c0 + chunk, None])).mean(1))           # [B, n, L]
    return torch.cat(out, 1)


def gate_bias(m, gamma, clip):
    """gamma * min(0, log(m_ij / mean_i m_ij)), floored at -clip: [B, N, L] fp32."""
    r = m / m.mean(1, keepdim=True).clamp_min(1e-30)
    return (gamma * torch.log(r.clamp_min(1e-30))).clamp(min=-clip, max=0.0)


def subject_bridge(q_txt, k, lse_txt, w_subj, L, grid, f, pool, rope_k, R):
    """[B, R] image-token indices: the block-centre tokens with the largest global-branch attention from the subject
    words, sum_h sum_t w_t P(t -> block), P with the global branch's own normaliser (lse_txt [B, h, L])."""
    from sca.attn import pool_tokens
    H, W = grid
    kpr = rope(pool_tokens(k[:, L:], grid, f, pool), rope_k)                       # [B, Nb, h, D]
    D = q_txt.shape[-1]
    s = torch.einsum("bthd,bjhd->bhtj", _hi(q_txt), _hi(kpr)) * (D ** -0.5)        # [B, h, L, Nb]
    p = torch.exp(s - _hi(lse_txt)[..., None])
    score = (p * _hi(w_subj).view(1, 1, -1, 1)).sum((1, 2))                      # [B, Nb]
    top = score.topk(R, dim=-1).indices
    cen = torch.arange(H * W, device=k.device).view(H, W)[f[0] // 2::f[0], f[1] // 2::f[1]].reshape(-1)
    return cen[top]


def sca_text_forward(q, k, v, rot, rope_q, rope_k, pi, L, grid, f, pool, layout, cells=None, per_cell=0, mode=(),
                     beta=None, gamma=1.0, clip=6.0, far=None, bwscale=1.0, w_subj=None, near="window", lat_w=None,
                     lat=None, far_pos="true"):
    """SCA with the bridge (cells, per_cell; layout built for prefix R = cells_h * cells_w * per_cell, the bridge
    only) and the text modes of this module. q, k, v pre-RoPE [B, L + N, h, D]; returns [B, L + N, h, D].
    ffl round-7 keywords (module docstring): near "window" | "bridge" (ffc and ffl near "window" need a layout for
    prefix 0, ffl near "bridge" one for prefix R), lat_w [h] tensor or None (= 1), lat = lattice_layout() or None."""
    cos, sin = rot
    kr = rope(k, rot)
    qi = rope(q[:, L:], (cos[L:], sin[L:]))
    q_txt = rope(q[:, :L], (cos[:L], sin[:L]))
    o_glob, lse_glob = global_branch_lse(q, k, v, L, grid, f, pool, rope_q, rope_k, q_txt, kr[:, :L])
    lse_gi = lse_glob[:, :, L:]                                                    # [B, h, N]
    kt, vt = kr[:, :L], v[:, :L]
    kimg, vimg = kr[:, L:], v[:, L:]
    if per_cell:
        idx = bridge_index(k[:, L:], grid, cells, per_cell)                       # [B, R]
        if "sb" in mode and w_subj is not None:
            sb = subject_bridge(q_txt, k, lse_glob[:, :, :L], w_subj, L, grid, f, pool, rope_k, idx.shape[1])
            idx = torch.where(w_subj.sum() > 0, sb, idx)
        g = idx[:, :, None, None].expand(-1, -1, k.shape[2], k.shape[3])
        kl = torch.cat([torch.gather(kimg, 1, g), kimg], 1)
        vl = torch.cat([torch.gather(vimg, 1, g), vimg], 1)
        R = idx.shape[1]
    else:
        kl, vl, R = kimg, vimg, 0
    if "ffc" in mode or "ffl" in mode:   # the far field is attended densely (or outside each window, lat)
        if "ffl" in mode and near == "bridge" and R:   # near part = sca_v2's window plus bridge (layout prefix R)
            bw = bwscale * math.log(kimg.shape[1] / R) if "bw" in mode else 0.0
            o_w, lse_w = local_branch_flex(qi, kl, vl, R, layout, True, bw, None, return_lse=True)
        else:                                          # layout for prefix 0: window keys only
            o_w, lse_w = local_branch_flex(qi, kimg, vimg, 0, layout, True, 0.0, None, return_lse=True)
        o_w = o_w.transpose(1, 2).index_select(1, layout.q_inv)
        lse_w = lse_w.index_select(2, layout.q_inv)
        if "ffl" in mode:   # block-centre lattice at true positions (pool "center" ids)
            cen = lattice_index(grid, f, k.device)
            if far_pos == "coarse":   # the global branch's view: queries at floor(p / f), block keys at block ids
                o_b, lse_b = lattice_part(rope(q[:, L:], rope_q), rope(k[:, L:][:, cen], rope_k), vimg[:, cen], lat)
            else:
                o_b, lse_b = lattice_part(qi, kimg[:, cen], vimg[:, cen], lat)
        else:
            o_b, lse_b = text_part(qi, kl[:, :R], vl[:, :R])
        fr = far.to(o_w.device).view(1, 1, -1, 1)
        fr = fr if o_w.dtype == torch.float64 else fr.float()
        if lat_w is not None:
            fr = fr * lat_w.to(o_w.device).view(1, 1, -1, 1).to(fr.dtype)
        if lat is not None:   # a query whose window covers the whole lattice has no far field
            fr = fr * torch.isfinite(lse_b).transpose(1, 2)[..., None].to(fr.dtype)
        o_i = (1 - fr) * _hi(o_w) + fr * _hi(o_b)
        lse_b = _hi(lse_b)
        if lat_w is not None:     # the far field enters the partition with weight lat_w (0: the near part's own)
            lse_b = lse_b + torch.log(lat_w.to(o_w.device).view(1, -1, 1).to(lse_b.dtype))
        lse_i = torch.logaddexp(_hi(lse_w), lse_b)
    else:
        bw = bwscale * math.log(kimg.shape[1] / R) if ("bw" in mode and R) else 0.0   # bridge keys stand for N / R
        o_i, lse_i = local_branch_flex(qi, kl, vl, R, layout, True, bw, None, return_lse=True)
        o_i = o_i.transpose(1, 2).index_select(1, layout.q_inv)                   # [B, N, h, D]
        lse_i = lse_i.index_select(2, layout.q_inv)                                # [B, h, N]
    if "gtx" in mode:
        o_t, lse_t = text_part(rope(q[:, L:], rope_q), kt, vt)
        a = torch.exp(_hi(lse_t) - _hi(lse_gi)).clamp(max=1.0).transpose(1, 2)[..., None]   # [B, N, h, 1]
        o_loc = a * _hi(o_t) + (1 - a) * _hi(o_i)
    else:
        tb = None
        if "gtg" in mode:
            tb = gate_bias(text_map(rope(q[:, L:], rope_q), kt, lse_gi), gamma, clip).contiguous()
            # Inductor's flex template needs captured buffers with a fixed layout; a tensor computed in the same graph
            # has a flexible one ("convert FlexibleLayout to FixedLayout first"), so the gate enters the flex call as
            # an input of the next graph. No-op in eager mode.
            torch._dynamo.graph_break()
        o_t, lse_t = text_part(qi, kt, vt, beta if "ctw" in mode else None, tb)
        o_loc = merge_lse(o_t, lse_t, o_i, lse_i)
    img = (pi * o_loc + (1 - pi) * _hi(o_glob[:, L:])).to(o_glob.dtype)
    return torch.cat([o_glob[:, :L], img], 1)


def sca_text_fn(device):
    """sca_text_forward compiled once for CUDA."""
    if torch.device(device).type != "cuda":
        return sca_text_forward
    if "text" not in _FLEX:
        _envfix()
        _raise_recompile_limit()
        flex_fn(device)
        _FLEX["text"] = torch.compile(sca_text_forward, dynamic=False)
    return _FLEX["text"]
