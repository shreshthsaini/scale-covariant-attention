"""Round-6 text modes (sca/textgate.py): exact split of the window softmax, calibrated text weight (ctw), global text
gate (gtg) and global text read-out (gtx). All CPU, float64, against brute force."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sca.attn import PiTable, SCAContext, bridge_index, flex_layout, global_branch, rope, window_index  # noqa: E402
from sca import textgate as tg  # noqa: E402
from test_dedup import _qkv, _setup, flat  # noqa: E402
from test_sca import attn_outputs, model  # noqa: E402

TABLE = Path("/scratch/09032/saini_2/sca/calib/k1_1024.npz")
H, W, L, F, TILE, WIN = 8, 8, 4, (2, 2), (2, 2), (4, 4)


def _window_mask(R):
    """[N, R + N] keys visible to each image query: R bridge tokens, then its 4 x 4 window (tiles of 2)."""
    mask = torch.zeros(H * W, R + H * W, dtype=torch.bool)
    mask[:, :R] = True
    for r in range(H):
        for c in range(W):
            r0 = min(max((r // 2) * 2 + 1 - 2, 0), H - 4)
            c0 = min(max((c // 2) * 2 + 1 - 2, 0), W - 4)
            for a in range(r0, r0 + 4):
                for b in range(c0, c0 + 4):
                    mask[r * W + c, R + a * W + b] = True
    return mask


def _brute_local(qi, keys_txt, vals_txt, keys_img, vals_img, mask, txt_bias, img_bias=0.0):
    """One softmax over text (logits + txt_bias [h, N, L]) and masked image keys (+ img_bias); qi [N, h, D]."""
    d = qi.shape[-1] ** -0.5
    st = torch.einsum("nhd,lhd->hnl", qi, keys_txt) * d + txt_bias
    si = torch.einsum("nhd,mhd->hnm", qi, keys_img) * d + img_bias
    si = si.masked_fill(~mask[None], float("-inf"))
    p = torch.cat([st, si], -1).softmax(-1)
    return torch.einsum("hnm,mhd->nhd", p, torch.cat([vals_txt, vals_img], 0))


def _bridge(k, cells=(2, 2), per=2):
    idx = bridge_index(k[:, L:], (H, W), cells, per)[0]
    return cells, per, idx


def test_parse_mode():
    assert tg.parse_mode("") == () and tg.parse_mode("ctw+gtg") == ("ctw", "gtg")
    with pytest.raises(ValueError):
        tg.parse_mode("gtx+ctw")
    with pytest.raises(ValueError):
        tg.parse_mode("foo")
    assert tg.parse_mode("gtx+bw") == ("gtx", "bw")


def test_text_part_and_merge_match_bruteforce():
    g = torch.Generator().manual_seed(3)
    B, N, h, D, Lt = 1, 12, 2, 8, 5
    q, kt, vt = (torch.randn(B, n, h, D, generator=g, dtype=torch.float64) for n in (N, Lt, Lt))
    hb = torch.tensor([0.4, -1.3], dtype=torch.float64)
    tb = -torch.rand(B, N, Lt, generator=g, dtype=torch.float64) * 3
    o, lse = tg.text_part(q, kt, vt, hb, tb)
    s = torch.einsum("bnhd,blhd->bhnl", q, kt) / math.sqrt(D) + hb.view(1, -1, 1, 1) + tb[:, None]
    assert torch.allclose(lse, s.logsumexp(-1), atol=1e-10)                       # natural-log lse
    assert torch.allclose(o, torch.einsum("bhnl,blhd->bnhd", s.softmax(-1), vt), atol=1e-10)
    # merge of two disjoint key sets equals one softmax over their union
    k2, v2 = (torch.randn(B, 7, h, D, generator=g, dtype=torch.float64) for _ in range(2))
    o2, lse2 = tg.text_part(q, k2, v2)
    s2 = torch.einsum("bnhd,blhd->bhnl", q, k2) / math.sqrt(D)
    ref = torch.einsum("bhnl,blhd->bnhd", torch.cat([s, s2], -1).softmax(-1), torch.cat([vt, v2], 1))
    assert torch.allclose(tg.merge_lse(o, lse, o2, lse2), ref, atol=1e-10)


def test_split_without_offsets_equals_sca_v2_forward():
    """ctw with beta = 0 is the bridge forward of sca_v2 (sca_dedup_forward), computed through the split."""
    from sca.attn import sca_dedup_forward
    q, k, v = _qkv(H, W, L, seed=7)
    _, _, rot, rq, rk = _setup(H, W, L, F)
    cells, per, idx = _bridge(k)
    R = len(idx)
    pi = torch.tensor([0.3, 0.8], dtype=torch.float64).view(1, 1, -1, 1)
    ref = sca_dedup_forward(q, k, v, rot, rq, rk, pi, L, (H, W), F, "center",
                            flex_layout(L + R, (H, W), TILE, WIN, "cpu", (2, 2), (2, 2)), 0.0, cells, per)
    lay = flex_layout(R, (H, W), TILE, WIN, "cpu", (2, 2), (2, 2))
    out = tg.sca_text_forward(q, k, v, rot, rq, rk, pi, L, (H, W), F, "center", lay, cells, per, ("ctw",),
                              torch.zeros(2, dtype=torch.float64))
    assert (out - ref).abs().max() < 1e-6         # the reference mixes in fp32 (mix_branches)
    # no bridge either
    ref0 = sca_dedup_forward(q, k, v, rot, rq, rk, pi, L, (H, W), F, "center",
                             flex_layout(L, (H, W), TILE, WIN, "cpu", (2, 2), (2, 2)))
    out0 = tg.sca_text_forward(q, k, v, rot, rq, rk, pi, L, (H, W), F, "center",
                               flex_layout(0, (H, W), TILE, WIN, "cpu", (2, 2), (2, 2)), None, 0, ("ctw",),
                               torch.zeros(2, dtype=torch.float64))
    assert (out0 - ref0).abs().max() < 1e-6


@pytest.mark.parametrize("modes", [("ctw",), ("gtg",), ("ctw", "gtg"), ("gtx",), ("bw",), ("gtx", "bw")])
def test_modes_match_bruteforce(modes):
    """pi = 1 (window branch only), bridge on: every mode against a direct one-softmax computation."""
    q, k, v = _qkv(H, W, L, seed=11)
    _, _, rot, rq, rk = _setup(H, W, L, F)
    cells, per, idx = _bridge(k)
    R = len(idx)
    lay = flex_layout(R, (H, W), TILE, WIN, "cpu", (2, 2), (2, 2))
    pi = torch.ones(1, 1, 2, 1, dtype=torch.float64)
    beta = torch.tensor([-0.7, -2.1], dtype=torch.float64)
    gamma, clip = 1.5, 4.0
    out = tg.sca_text_forward(q, k, v, rot, rq, rk, pi, L, (H, W), F, "center", lay, cells, per, modes, beta,
                              gamma, clip)
    qr, kr = rope(q, rot), rope(k, rot)
    qi = qr[0, L:]
    keys_img = torch.cat([kr[0, L + idx], kr[0, L:]], 0)
    vals_img = torch.cat([v[0, L + idx], v[0, L:]], 0)
    mask = _window_mask(R)
    img_bias = torch.zeros(H * W, R + H * W, dtype=torch.float64)
    if "bw" in modes:
        img_bias[:, :R] = math.log(H * W / R)
    # global branch probabilities on the text keys (coarse query ids, block-centre keys)
    d = q.shape[-1] ** -0.5
    cen = [(a * 2 + 1) * W + b * 2 + 1 for a in range(H // 2) for b in range(W // 2)]
    qg = rope(q[:, L:], rq)[0]
    kg = torch.cat([kr[0, :L], rope(k[:, L:][:, cen], rk)[0]], 0)
    sg = torch.einsum("nhd,mhd->hnm", qg, kg) * d
    pg = sg.softmax(-1)                                                           # [h, N, L + blocks]
    if "gtx" in modes:
        a = pg[:, :, :L].sum(-1).transpose(0, 1)[..., None]                       # [N, h, 1] global text mass
        st = torch.einsum("nhd,lhd->hnl", qg, kr[0, :L]) * d
        o_t = torch.einsum("hnl,lhd->nhd", st.softmax(-1), v[0, :L])
        si = torch.einsum("nhd,mhd->hnm", qi, keys_img) * d + img_bias[None]
        o_i = torch.einsum("hnm,mhd->nhd", si.masked_fill(~mask[None], float("-inf")).softmax(-1), vals_img)
        ref = a * o_t + (1 - a) * o_i
    else:
        bias = torch.zeros(2, H * W, L, dtype=torch.float64)
        if "ctw" in modes:
            bias = bias + beta.view(-1, 1, 1)
        if "gtg" in modes:
            m = pg[:, :, :L].mean(0)                                              # [N, L] head mean
            r = m / m.mean(0, keepdim=True)
            bias = bias + (gamma * torch.log(r)).clamp(min=-clip, max=0.0)[None]
        ref = _brute_local(qi, kr[0, :L], v[0, :L], keys_img, vals_img, mask, bias, img_bias)
    assert (out[0, L:] - ref).abs().max() < 1e-9
    o_glob = global_branch(q, k, v, L, (H, W), F, "center", (rot[0][:L], rot[1][:L]), rq, rk, qr[:, :L], kr[:, :L])
    assert (out[:, :L] - o_glob[:, :L]).abs().max() < 1e-9                       # text queries: global branch


def test_gate_is_neutral_for_uniform_tokens_and_bounded():
    m = torch.full((1, 6, 3), 0.1, dtype=torch.float64)
    assert torch.equal(tg.gate_bias(m, 1.0, 6.0), torch.zeros_like(m))
    m[0, :, 1] = torch.tensor([1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1.0])
    b = tg.gate_bias(m, 2.0, 5.0)
    assert b.max() <= 0 and b.min() >= -5.0 and b[0, 5, 1] == 0 and b[0, 0, 1] == -5.0


def test_near_share_interpolation():
    mass = np.zeros((5, 1, 1, 1))
    mass[0] = 0.2                                     # text
    mass[1:, 0, 0, 0] = [0.08, 0.16, 0.32, 0.64]      # radii 2, 4, 8, 16: doubles with the radius
    radii = (2, 4, 8, 16)
    assert np.isclose(tg.near_share(mass, radii, 8)[0, 0, 0], 0.32 / 0.8)
    assert np.isclose(tg.near_share(mass, radii, 16)[0, 0, 0], 0.64 / 0.8)
    assert np.isclose(tg.near_share(mass, radii, 8 * 2 ** 0.5)[0, 0, 0], 0.32 * 2 ** 0.5 / 0.8)   # log-linear
    assert np.isclose(tg.near_share(mass, radii, 64)[0, 0, 0], 0.64 / 0.8)                       # clamped


@pytest.mark.skipif(not TABLE.exists(), reason="calibration table not on this machine")
def test_ctw_table_from_calibration():
    z = np.load(TABLE)
    b64 = tg.ctw_beta(TABLE, 0.6, 64, (256, 256))
    b128 = tg.ctw_beta(TABLE, 0.6, 128, (256, 256))
    assert b64.shape == (57, 24) and np.isfinite(b64).all() and (b64 <= 0).all()
    assert (b128 >= b64 - 1e-9).all()                 # a wider window holds more of the mass
    # at a populated bin centre, window 64 on 256 tokens is radius 8 exactly
    c = z["sigma_centers"][z["populated"]]
    j = int(np.argmin(np.abs(c - 0.6)))
    col = np.flatnonzero(z["populated"])[j]
    m = z["mass_mean"]
    ref = np.log(np.clip(m[3][..., col] / (1 - m[0][..., col]), 1e-6, 1))
    assert np.allclose(tg.ctw_beta(TABLE, float(c[j]), 64, (256, 256)), np.maximum(ref, -6.0))


def _ctx(**kw):
    base = dict(mode="sca", n0=4, tile=2, window=4, local_impl="flex", pi=PiTable("const", value=0.7))
    base.update(kw)
    return SCAContext(**base)


def test_processor_text_modes_gate_by_sigma_and_validate():
    m = model()
    base = attn_outputs(m, _ctx(bridge_cells=2, bridge_k=2), H, W)
    off = attn_outputs(m, _ctx(bridge_cells=2, bridge_k=2, text_mode="gtg", text_mode_lo=0.6), H, W)  # sigma .5
    on = attn_outputs(m, _ctx(bridge_cells=2, bridge_k=2, text_mode="gtg", text_gamma=2.0), H, W)
    gtx = attn_outputs(m, _ctx(text_mode="gtx"), H, W)
    assert torch.allclose(flat(off), flat(base), atol=1e-5)
    assert not torch.allclose(flat(on), flat(base), atol=1e-4) and torch.isfinite(flat(on)).all()
    assert torch.isfinite(flat(gtx)).all()
    with pytest.raises(ValueError):
        _ctx(text_mode="ctw", local_impl="gather")
    with pytest.raises(ValueError):
        _ctx(text_mode="ctw", lse_tau=1.0)
    with pytest.raises(ValueError):
        _ctx(text_mode="gtg", text_clip=0.0)
    with pytest.raises(ValueError):            # ctw needs a calibration npz behind pi
        attn_outputs(m, _ctx(text_mode="ctw"), H, W)


def test_sample_info_lists_text_mode_only_when_set():
    from scripts.sample import parser
    from sca import dedup
    base = ["--method", "sca", "--res", "64x64", "--prompts", "x.jsonl", "--out", "o"]
    assert "text_mode" not in dedup.info(parser().parse_args(base))
    a = parser().parse_args(base + ["--text-mode", "ctw+gtg", "--text-mode-lo", "0.5"])
    info = dedup.info(a)
    assert info["text_mode"] == "ctw+gtg" and info["text_mode_lo"] == 0.5
    kw = dedup.context_kwargs(a)
    assert kw["text_mode"] == "ctw+gtg" and kw["text_gamma"] == 1.0


def test_probe_measure_reduces_to_full_attention():
    """scripts/r6_attn_probe.measure: with one window covering the grid and no pooling, the window and global
    operators are full attention, so all three rows agree; masses are probabilities."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import r6_attn_probe as pr
    m = model().double()
    Hh = Ww = 8
    Lt = 4
    ids = __import__("test_sca").inputs(Hh, Ww)["img_ids"].double()
    rot = m.pos_embed(torch.cat([torch.zeros(Lt, 3, dtype=torch.float64), ids]))
    c = SCAContext(mode="sca", n0=8, tile=2, window=8, pi=PiTable("const", value=0.5))
    c.pos_embed = m.pos_embed
    c.set_inputs(ids)
    q, k, _ = _qkv(Hh, Ww, Lt, seed=5)
    regions = [pr.patch_indices((40, 40), (Hh, Ww), tile=4, side=2), pr.patch_indices((100, 20), (Hh, Ww), tile=4, side=2)]
    main = pr.box_mask((0, 0, 60, 60), (Hh, Ww))
    out = pr.measure(q, k, rot, c, regions, [1, 2], main)
    assert out.shape == (2, 3, 3, 2) and float(out.min()) >= 0 and float(out.max()) <= 1 + 1e-9
    assert torch.allclose(out[:, 1], out[:, 0], atol=1e-9) and torch.allclose(out[:, 2], out[:, 0], atol=1e-9)
    assert pr.box_mask((0, 0, 60, 60), (Hh, Ww)).sum() == 16          # token centres 8, 24, 40, 56 px
    assert pr.window_keys(torch.tensor([0]), (Hh, Ww), (2, 2), (4, 4)).tolist()[:4] == [0, 1, 2, 3]
