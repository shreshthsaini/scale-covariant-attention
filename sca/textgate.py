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
  bw   weighted bridge (image side, combines with the others). Each bridge key gets the logit offset log(N / R), the
       number of image tokens it stands for, so the far field enters the window softmax with the mass full attention
       would give it rather than as R single tokens.
Off by default; the published and sca_v2 paths (sca.attn.sca_flex_forward, sca_dedup_forward) are untouched.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from sca.attn import _FLEX, _envfix, _raise_recompile_limit, bridge_index, flex_fn, global_branch_lse, \
    local_branch_flex, rope

MODES = ("ctw", "gtg", "gtx", "bw")


def parse_mode(text):
    """'ctw+gtg' -> ('ctw', 'gtg'); '' -> (). gtx replaces the window's text part, so it combines only with bw."""
    parts = tuple(p for p in str(text or "").split("+") if p)
    bad = [p for p in parts if p not in MODES]
    if bad:
        raise ValueError(f"unknown text mode(s) {bad}; choose from {MODES}")
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


def sca_text_forward(q, k, v, rot, rope_q, rope_k, pi, L, grid, f, pool, layout, cells=None, per_cell=0, mode=(),
                     beta=None, gamma=1.0, clip=6.0):
    """SCA with the bridge (cells, per_cell; layout built for prefix R = cells_h * cells_w * per_cell, the bridge
    only) and the text modes of this module. q, k, v pre-RoPE [B, L + N, h, D]; returns [B, L + N, h, D]."""
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
        g = idx[:, :, None, None].expand(-1, -1, k.shape[2], k.shape[3])
        kl = torch.cat([torch.gather(kimg, 1, g), kimg], 1)
        vl = torch.cat([torch.gather(vimg, 1, g), vimg], 1)
        R = idx.shape[1]
    else:
        kl, vl, R = kimg, vimg, 0
    bw = math.log(kimg.shape[1] / R) if ("bw" in mode and R) else 0.0              # bridge keys stand for N / R tokens
    o_i, lse_i = local_branch_flex(qi, kl, vl, R, layout, True, bw, None, return_lse=True)
    o_i = o_i.transpose(1, 2).index_select(1, layout.q_inv)                       # [B, N, h, D]
    lse_i = lse_i.index_select(2, layout.q_inv)                                    # [B, h, N]
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
