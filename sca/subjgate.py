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


def finalize_map(m, grid, q_lo=0.5, q_hi=0.95, dilate=2, smooth=0, rel=None, edge=0):
    """[N] accumulated subject share -> [N] map in [0, 1]. smooth > 0 first averages m over smooth x smooth tokens (the
    raw map is speckled: single background tokens with high subject attention). Ramp from lo to hi: quantiles q_lo,
    q_hi of m, or with rel = (a, b) lo = med + a (max - med), hi = med + b (max - med), so only regions near the
    strongest subject response get g = 1. Then max-dilated by `dilate` tokens. edge > 0 keeps g = 1 on a border band
    of that many tokens (the last token columns lose stability when they lose the prompt's content, as with round 6's
    ctw and r7_tv0m)."""
    m = m.float()
    H, W = grid
    if smooth and smooth > 1:
        m = F.avg_pool2d(m.view(1, 1, H, W), smooth, stride=1, padding=smooth // 2,
                         count_include_pad=False).view(-1)[: H * W]
    if rel is not None:
        med, mx = torch.quantile(m, 0.5), m.max()
        lo, hi = med + rel[0] * (mx - med), med + rel[1] * (mx - med)
    else:
        lo, hi = torch.quantile(m, q_lo), torch.quantile(m, q_hi)
    g = ((m - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1)
    if dilate > 0:
        g = F.max_pool2d(g.view(1, 1, H, W), 2 * dilate + 1, stride=1, padding=dilate).view(-1)
    if edge > 0:
        g = g.view(H, W).clone()
        g[:edge, :] = 1.0
        g[-edge:, :] = 1.0
        g[:, :edge] = 1.0
        g[:, -edge:] = 1.0
        g = g.view(-1)
    return g


def copy_map(S, M0, grid, smooth=9, rel=(0.5, 1.0), excl=3, grow=2, q=0.9, S0=None, new=(0.1, 0.25), z=None):
    """Copy-site map of one window step (dynamic copy-site gate, SCAContext.csg_*). S [N]: the global branch's
    subject-word share summed over blocks; M0 [N]: the opening's subject map (finalize_map). S is averaged over
    smooth x smooth tokens and divided by the main subject's level (quantile q of S inside M0 > 0.5); sites whose level
    reaches rel[0] .. rel[1] of the main subject's ramp to 1. The opening's subject region, max-dilated by excl tokens,
    is excluded, and the map is max-dilated by grow tokens to cover a forming figure. With S0 (the opening's raw share,
    the accumulation M0 was made from) only new subject attention counts: with o = (S0 - median) / (q99 - median) (0 at
    background, 1 at the subject), the map is multiplied by 1 - ramp(o, new[0], new[1]), so the subject's own body and
    scene parts that already read the subject words at the opening are never gated (copies start at background level
    there, r6/probe). Returns [N] in [0, 1]."""
    H, W = grid
    S = S.float()
    if smooth and smooth > 1:
        S = F.avg_pool2d(S.view(1, 1, H, W), smooth, stride=1, padding=smooth // 2,
                         count_include_pad=False).reshape(-1)[: H * W]
    inside = M0.float() > 0.5
    if z is not None:   # background z-score: copies stand out of their own image's background (median, p95)
        exm = F.max_pool2d(M0.float().view(1, 1, H, W), 2 * excl + 1, stride=1, padding=excl).reshape(-1) > 0.5
        bg = S[~exm] if bool((~exm).any()) else S
        med, p95 = torch.quantile(bg, 0.5), torch.quantile(bg, 0.95)
        zz = (S - med) / (p95 - med).clamp_min(1e-12)
        C = ((zz - z[0]) / max(z[1] - z[0], 1e-6)).clamp(0, 1)
    else:
        main = torch.quantile(S[inside], q) if bool(inside.any()) else S.max()
        r = S / main.clamp_min(1e-12)
        C = ((r - rel[0]) / max(rel[1] - rel[0], 1e-6)).clamp(0, 1)
    if S0 is not None:
        o = S0.float().to(S.device)
        if smooth and smooth > 1:
            o = F.avg_pool2d(o.view(1, 1, H, W), smooth, stride=1, padding=smooth // 2,
                             count_include_pad=False).reshape(-1)[: H * W]
        med = torch.quantile(o, 0.5)
        o = ((o - med) / (torch.quantile(o, 0.99) - med).clamp_min(1e-12)).clamp_min(0)   # background 0, subject 1
        C = C * (1 - ((o - new[0]) / max(new[1] - new[0], 1e-6)).clamp(0, 1))
    ex = M0.float()
    if excl > 0:
        ex = F.max_pool2d(ex.view(1, 1, H, W), 2 * excl + 1, stride=1, padding=excl).reshape(-1)
    C = C * (1 - ex.clamp(0, 1))
    if grow > 0:
        C = F.max_pool2d(C.view(1, 1, H, W), 2 * grow + 1, stride=1, padding=grow).reshape(-1)
    return C


def text_all_part(qr, kt, vt, lse):
    """All text tokens' contribution w_i O_T,i to a softmax with log-partition lse [B, h, N] (the subject_part of the
    whole prompt), through textgate.text_part."""
    from sca.textgate import text_part
    o, lt = text_part(qr, kt, vt)
    w = torch.exp(_hi(lt) - _hi(lse))
    return w.transpose(1, 2)[..., None] * _hi(o), w


def sca_subject_forward(q, k, v, rot, rope_q, rope_k, pi, L, grid, f, pool, layout, text_bias, cells, per_cell,
                        idx, valid, s_img, all_text=False, branch=0, return_share=False):
    """sca_v2's operator (sca.attn.sca_dedup_forward, bridge path, tau = 0) where image query i reads the subject-word
    values (all text values with all_text=True) scaled by s_img[:, i] in both branches. s_img [B, N] (1 = untouched).
    branch: 0 both branches, 1 the local (window) branch only, 2 the global branch only. Returns [B, L + N, h, D];
    with return_share=True also the global branch's subject-word share per image query, head mean [B, N] (the copy-site
    gate's detector, SCAContext.csg_*)."""
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
    o_loc = o_loc - cut * sub_loc if branch != 2 else o_loc
    img_g = _hi(o_glob[:, L:]) - cut * sub_glob if branch != 1 else _hi(o_glob[:, L:])
    img = _hi(pi) * o_loc + (1 - _hi(pi)) * img_g
    out = torch.cat([o_glob[:, :L], img.to(o_glob.dtype)], 1)
    if return_share:
        _, w = subject_part(rope(q[:, L:], rope_q), kr[:, :L], v[:, :L], idx, valid, lse_glob[:, :, L:])
        return out, w.mean(1)
    return out


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
