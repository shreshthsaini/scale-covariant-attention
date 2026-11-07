"""Subject-anchored prompt gating (round 7, after the text-value screens; .agent/notes/duplication_fix_round7.md).

Copies on copyprone21 are driven by the prompt content image tokens read on the window steps: zeroing the text values
they read while sigma > 0.55 (r7_tv0m) leaves 1 copy of 21 but also strips the main subject of its prompted look.
The attention probe (r6/probe) says where the two separate: at the end of the opening (step 6, sigma 0.92) the future
copy sites carry background-level subject-word attention (global branch 0.0018 vs background 0.0016) while the main
subject carries 0.0085; the copy sites only overtake it during steps 8 to 12.

So the subject words keep their full content where the opening put the subject, and lose it elsewhere:
  opening (sigma > gate): per image token, the share of global-branch attention on the subject-word tokens, averaged
      over heads and summed over blocks, from the last opening step: m_i;
  map: g_i = clip((m_i - q_lo) / (q_hi - q_lo), 0, 1) over the image (quantiles of m), dilated by `dilate` tokens;
  window steps while sigma > lo: query i reads the subject-word values scaled by s_i = a + (1 - a) g_i in both
      branches (attention weights unchanged, so the prompt keeps its sink mass; the other text tokens untouched).
By linearity, per branch: O_i' = O_i - (1 - s_i) w_i Os_i, with w_i the subject tokens' share of that softmax and Os_i
their own softmax output. Off by default (SCAContext.sa_value = 1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

S_MAX = 8  # subject-word tokens per prompt (padded, so one compiled graph serves every prompt)


def _hi(x):
    """x in fp32 at least (float64 stays float64, for the CPU tests)."""
    return x if x.dtype == torch.float64 else x.float()


def subject_index(subject_tokens, L, device):
    """(idx [S_MAX] long, valid [S_MAX] bool) of the prompt's subject-word token positions (< L), padded."""
    idx = [int(t) for t in subject_tokens if 0 <= int(t) < L][:S_MAX]
    valid = [True] * len(idx) + [False] * (S_MAX - len(idx))
    idx = idx + [0] * (S_MAX - len(idx))
    return (torch.tensor(idx, dtype=torch.long, device=device), torch.tensor(valid, dtype=torch.bool, device=device))


def subject_part(qr, kt, vt, idx, valid, lse):
    """Subject tokens' contribution w_i Os_i to a softmax whose log-partition per query is lse [B, h, N]: queries qr
    [B, N, h, D] (RoPE'd as in that softmax), text keys kt / values vt [B, L, h, D]. Returns [B, N, h, D] (fp32) and
    the share w [B, h, N]."""
    ks, vs = _hi(kt[:, idx]), _hi(vt[:, idx])                                       # [B, S, h, D]
    sc = torch.einsum("bnhd,bshd->bhns", _hi(qr), ks) * qr.shape[-1] ** -0.5
    sc = sc.masked_fill(~valid.view(1, 1, 1, -1), float("-inf"))
    ls = torch.logsumexp(sc, -1)                                                    # [B, h, N]
    w = torch.exp(ls - _hi(lse))                                                    # subject share of that softmax
    o = torch.einsum("bhns,bshd->bnhd", torch.softmax(sc, -1), vs)
    return w.transpose(1, 2)[..., None] * o, w


def finalize_map(m, grid, q_lo=0.5, q_hi=0.95, dilate=2):
    """[N] accumulated subject share -> [N] map in [0, 1]: quantile-normalized and max-dilated by `dilate` tokens."""
    m = m.float()
    lo, hi = torch.quantile(m, q_lo), torch.quantile(m, q_hi)
    g = ((m - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1)
    if dilate > 0:
        H, W = grid
        g = F.max_pool2d(g.view(1, 1, H, W), 2 * dilate + 1, stride=1, padding=dilate).view(-1)
    return g


def text_all_part(qr, kt, vt, lse):
    """All text tokens' contribution w_i O_T,i to a softmax with log-partition lse [B, h, N] (the subject_part of the
    whole prompt), through textgate.text_part."""
    from sca.textgate import text_part
    o, lt = text_part(qr, kt, vt)
    w = torch.exp(_hi(lt) - _hi(lse))
    return w.transpose(1, 2)[..., None] * _hi(o), w


def sca_subject_forward(q, k, v, rot, rope_q, rope_k, pi, L, grid, f, pool, layout, text_bias, cells, per_cell,
                        idx, valid, s_img, all_text=False):
    """sca_v2's operator (sca.attn.sca_dedup_forward, bridge path, tau = 0) where image query i reads the subject-word
    values (all text values with all_text=True) scaled by s_img[:, i] in both branches. s_img [B, N] (1 = untouched).
    Returns [B, L + N, h, D]."""
    from sca.attn import bridge_index, global_branch_lse, local_branch_flex, rope
    cos, sin = rot
    kr = rope(k, rot)
    qi = rope(q[:, L:], (cos[L:], sin[L:]))
    q_txt = rope(q[:, :L], (cos[:L], sin[:L]))
    o_glob, lse_glob = global_branch_lse(q, k, v, L, grid, f, pool, rope_q, rope_k, q_txt, kr[:, :L])
    kl, vl, Lp = kr, v, L
    if per_cell:
        bi = bridge_index(k[:, L:], grid, cells, per_cell)
        g = bi[:, :, None, None].expand(-1, -1, k.shape[2], k.shape[3])
        kl = torch.cat([kr[:, :L], torch.gather(kr[:, L:], 1, g), kr[:, L:]], 1)
        vl = torch.cat([v[:, :L], torch.gather(v[:, L:], 1, g), v[:, L:]], 1)
        Lp = L + bi.shape[1]
    o_loc, lse_loc = local_branch_flex(qi, kl, vl, Lp, layout, True, text_bias, L, return_lse=True)
    o_loc = _hi(o_loc.transpose(1, 2).index_select(1, layout.q_inv))               # [B, N, h, D]
    lse_loc = lse_loc.index_select(2, layout.q_inv)                                # [B, h, N]
    cut = (1.0 - _hi(s_img))[:, :, None, None]                                      # [B, N, 1, 1]
    if all_text:
        sub_loc, _ = text_all_part(qi, kr[:, :L], v[:, :L], lse_loc)
        sub_glob, _ = text_all_part(rope(q[:, L:], rope_q), kr[:, :L], v[:, :L], lse_glob[:, :, L:])
    else:
        sub_loc, _ = subject_part(qi, kr[:, :L], v[:, :L], idx, valid, lse_loc)
        sub_glob, _ = subject_part(rope(q[:, L:], rope_q), kr[:, :L], v[:, :L], idx, valid, lse_glob[:, :, L:])
    o_loc = o_loc - cut * sub_loc
    img_g = _hi(o_glob[:, L:]) - cut * sub_glob
    img = _hi(pi) * o_loc + (1 - _hi(pi)) * img_g
    return torch.cat([o_glob[:, :L], img.to(o_glob.dtype)], 1)


def sca_subject_fn(device):
    """sca_subject_forward compiled once for CUDA."""
    from sca.attn import _FLEX, _envfix, _raise_recompile_limit, flex_fn
    if torch.device(device).type != "cuda":
        return sca_subject_forward
    if "subject" not in _FLEX:
        _envfix()
        _raise_recompile_limit()
        flex_fn(device)
        _FLEX["subject"] = torch.compile(sca_subject_forward, dynamic=False)
    return _FLEX["subject"]
