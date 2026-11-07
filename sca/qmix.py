"""Per-query mixing weight and one three-part softmax (root-cause audit 2026-11-06, Fix 1 and Fix 3;
.agent/notes/root_cause_audit_20261106.md). Off by default (SCAContext.qmix == "").

Lemma 1: O_full = m_i O_near + (1 - m_i) O_far with m_i the near set's softmax mass of query i. sca_v2 replaces m_i by
the per-head constant pi_h, and its window softmax is normalized without the far keys, so a strong local match is a
full-strength "draw it here" whatever the far mass.

qmix "f1" (Fix 1): image output w_i O_loc + (1 - w_i) O_glob with O_loc, O_glob sca_v2's branches and
    w_i = sigmoid(logit pi_h + tau (d_i - mean_i d_i)),   d_i = log Z_loc - log Z_far,
  Z_loc = the window softmax's image partition (bridge + window keys, text excluded; exact split of the window softmax),
  Z_far = s^2 (exp(lse_glob) - Z_T,glob - Z_in): the global branch's partition at the query minus its text keys and the
  block centres inside the query's window, every block centre counted for the s^2 tokens it stands for. Both partitions
  exclude the text; the head mean of the log-odds stays at logit pi_h (the calibrated weight), only its spread across
  queries is new; tau = 1 is Lemma 1's slope.
qmix "f13" (Fix 1 + Fix 3): one softmax over three disjoint parts, so the prompt is normalized once:
    O = (Z_T O_T + e^beta Z_near O_near + Z_far O_far) / (Z_T + e^beta Z_near + Z_far),
  T the text keys at the query's true position, near the bridge plus window keys (image only), far the block centres
  outside the window at coarse ids with the s^2 count correction, beta_h = logit pi_h - mean_i (log Z_near - log Z_far):
  pi enters only as the calibrated offset on log Z_near.
Text queries keep SCA's rule (the global branch's text output) in both.
"""

from __future__ import annotations

import math

import torch

from sca.attn import _FLEX, _envfix, _raise_recompile_limit, bridge_index, flex_fn, global_branch_lse, \
    local_branch_flex, rope
from sca.textgate import _hi, lattice_index, lattice_part, merge_lse, text_part

MODES = ("", "f1", "f13")


def _logit(p):
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p) - torch.log1p(-p)


def sca_qmix_forward(q, k, v, rot, rope_q, rope_k, pi, L, grid, f, pool, layout, cells=None, per_cell=0, lat_in=None,
                     lat_out=None, tau=1.0, three=False, log_gamma=0.0):
    """q, k, v pre-RoPE [B, L + N, h, D]; pi [1, 1, h, 1]; layout = flex layout for prefix R (the bridge only, no text),
    lat_in / lat_out = textgate.lattice_layout(inside=True / False). f13 only: log_gamma scales the whole image part
    (near and far, so their balance is unchanged) against the text part, log gamma = 0 being one plain softmax.
    Returns [B, L + N, h, D]."""
    cos, sin = rot
    kr = rope(k, rot)
    qi = rope(q[:, L:], (cos[L:], sin[L:]))                                       # image queries at true ids
    q_txt = rope(q[:, :L], (cos[:L], sin[:L]))
    kt, vt = kr[:, :L], v[:, :L]
    o_glob, lse_glob = global_branch_lse(q, k, v, L, grid, f, pool, rope_q, rope_k, q_txt, kt)
    lse_gi = _hi(lse_glob[:, :, L:])                                               # [B, h, N]
    qg = rope(q[:, L:], rope_q)                                                    # image queries at coarse ids
    o_t, lse_t = text_part(qi, kt, vt)                                             # text part at true ids
    kimg, vimg = kr[:, L:], v[:, L:]
    if per_cell:
        idx = bridge_index(k[:, L:], grid, cells, per_cell)
        g = idx[:, :, None, None].expand(-1, -1, k.shape[2], k.shape[3])
        kl, vl, R = torch.cat([torch.gather(kimg, 1, g), kimg], 1), torch.cat([torch.gather(vimg, 1, g), vimg], 1), idx.shape[1]
    else:
        kl, vl, R = kimg, vimg, 0
    o_n, lse_n = local_branch_flex(qi, kl, vl, R, layout, True, 0.0, None, return_lse=True)
    o_n = o_n.transpose(1, 2).index_select(1, layout.q_inv)                      # [B, N, h, D] image part (near)
    lse_n = _hi(lse_n.index_select(2, layout.q_inv))                               # [B, h, N] log Z_near
    cen = lattice_index(grid, f, k.device)
    kb, vb = rope(k[:, L:][:, cen], rope_k), v[:, L:][:, cen]                    # block centres at coarse ids
    log_s2 = math.log(f[0] * f[1])
    lp = _logit(_hi(pi).view(1, -1, 1))                                            # [1, h, 1]
    if not three:
        _, lse_tg = text_part(qg, kt, vt)                                          # global branch's text partition
        _, lse_in = lattice_part(qg, kb, vb, lat_in)                               # its block centres inside the window
        frac = 1 - torch.exp(_hi(lse_tg) - lse_gi) - torch.exp(_hi(lse_in) - lse_gi)
        log_far = log_s2 + lse_gi + torch.log(frac.clamp_min(1e-6))
        d = lse_n - log_far
        w = torch.sigmoid(lp + tau * (d - d.mean(-1, keepdim=True))).transpose(1, 2)[..., None]   # [B, N, h, 1]
        o_loc = merge_lse(o_t, lse_t, o_n, lse_n)                                  # sca_v2's local branch
        img = w * o_loc + (1 - w) * _hi(o_glob[:, L:])
    else:
        o_far, lse_out = lattice_part(qg, kb, vb, lat_out)                         # block centres outside the window
        log_far = log_s2 + _hi(lse_out)
        beta = lp - (lse_n - log_far).mean(-1, keepdim=True)                       # [B, h, 1]
        a, b, c = _hi(lse_t), lse_n + beta + log_gamma, log_far + log_gamma
        m = torch.maximum(torch.maximum(a, b), c)
        wa, wb, wc = (torch.exp(x - m).transpose(1, 2)[..., None] for x in (a, b, c))
        img = (wa * _hi(o_t) + wb * _hi(o_n) + wc * _hi(o_far)) / (wa + wb + wc)
    return torch.cat([o_glob[:, :L], img.to(o_glob.dtype)], 1)


def sca_qmix_fn(device):
    """sca_qmix_forward compiled once for CUDA."""
    if torch.device(device).type != "cuda":
        return sca_qmix_forward
    if "qmix" not in _FLEX:
        _envfix()
        _raise_recompile_limit()
        flex_fn(device)
        _FLEX["qmix"] = torch.compile(sca_qmix_forward, dynamic=False)
    return _FLEX["qmix"]
