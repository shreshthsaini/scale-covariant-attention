"""Per-head apparent scale (SCAContext.scale_sched; sample.py --scale-sched, --scale-lo).

Head h attends text keys plus an n0 x n0 window of the stride-(s / a) lattice around its query tile, queries and keys
at coarse ids floor(p / d); a = 1 is the global branch, a = max(s) the published local branch."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sca.attn import (PiTable, SCAContext, apparent_scale, flex_layout, global_branch, head_layout,  # noqa: E402
                      lattice_qkv, local_branch, local_branch_flex, local_branch_strided, pool_factors, rope, sca_flex_forward,
                      sca_scale_forward, sca_scale_gather, scale_ladder, scale_sched_cost, sdpa, strided_flex_layout)
from test_dedup import TABLE, flat, head_table, sample  # noqa: E402,F401  (sample is a fixture)
from test_sca import attn_outputs, inputs, model  # noqa: E402

QT, KT = (2, 4), (2, 2)  # small flex tiles for the CPU tests


def _qkv(H, W, L, h=2, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, L + H * W, h, D, generator=g, dtype=torch.float64) for _ in range(3)]


def _setup(H, W, L):
    m = model().double()
    ids = inputs(H, W)["img_ids"].double()
    rot = m.pos_embed(torch.cat([torch.zeros(L, 3, dtype=torch.float64), ids]))
    c = SCAContext(mode="sca")
    c.pos_embed = m.pos_embed
    c.set_inputs(ids)
    return m, rot, c


def brute_strided(q, k, v, L, grid, tile, stride, window, pos_embed, text_bias=0.0):
    """Masked attention of the image queries over [text | every lattice key], ids and mask built element by element."""
    (H, W), (th, tw), (dh, dw), (Wh, Ww) = grid, tile, stride, window
    Hc, Wc = H // dh, W // dw
    dt = q.dtype
    q_ids = torch.tensor([[0, r // dh, c // dw] for r in range(H) for c in range(W)], dtype=dt)
    k_ids = torch.tensor([[0, i, j] for i in range(Hc) for j in range(Wc)], dtype=dt)
    pos = [(i * dh + dh // 2) * W + j * dw + dw // 2 for i in range(Hc) for j in range(Wc)]
    txt = pos_embed(torch.zeros(L, 3, dtype=dt))
    qr = rope(q[:, L:], pos_embed(q_ids))
    kr = torch.cat([rope(k[:, :L], txt), rope(k[:, L:][:, pos], pos_embed(k_ids))], 1)
    vv = torch.cat([v[:, :L], v[:, L:][:, pos]], 1)
    mask = torch.full((H * W, L + Hc * Wc), float("-inf"), dtype=dt)
    mask[:, :L] = text_bias
    for r in range(H):
        for c in range(W):
            r0 = min(max(((r // th) * th + th // 2) // dh - Wh // 2, 0), Hc - Wh)
            c0 = min(max(((c // tw) * tw + tw // 2) // dw - Ww // 2, 0), Wc - Ww)
            for i in range(r0, r0 + Wh):
                mask[r * W + c, L + i * Wc + c0:L + i * Wc + c0 + Ww] = 0.0
    return sdpa(qr, kr, vv, mask)


@pytest.mark.parametrize("grid,stride,tile,window", [((8, 12), (2, 2), (2, 3), (3, 4)), ((8, 16), (1, 2), (4, 4), (4, 6)),
                                                     ((12, 8), (4, 2), (2, 2), (3, 3)), ((8, 8), (1, 1), (2, 2), (4, 4))])
@pytest.mark.parametrize("bias", [0.0, -1.5])
def test_strided_local_matches_bruteforce(grid, stride, tile, window, bias):
    (H, W), L = grid, 3
    q, k, v = _qkv(H, W, L, seed=7)
    m, rot, c = _setup(H, W, L)
    rqd, rkd = c.coarse_rope(stride, "cpu")
    qc, kc, vc = lattice_qkv(q, k, v, L, grid, stride, rqd, rkd, rope(k[:, :L], (rot[0][:L], rot[1][:L])))
    assert kc.shape[1] == L + (H // stride[0]) * (W // stride[1])
    out = local_branch_strided(qc, kc, vc, L, grid, tile, stride, window, text_bias=bias)
    ref = brute_strided(q, k, v, L, grid, tile, stride, window, m.pos_embed, bias)
    assert (out - ref).abs().max() < 1e-10
    lay = strided_flex_layout(L, grid, tile, stride, window, "cpu", QT, KT)
    assert (local_branch_flex(qc, kc, vc, L, lay, text_bias=bias) - out).abs().max() < 1e-12
    if stride == (1, 1):  # the native lattice is the published local branch
        qr, kr = rope(q, rot), rope(k, rot)
        assert (local_branch(qr, kr, v, L, grid, tile, window, text_bias=bias) - out).abs().max() < 1e-12
    # small chunks give the same result
    assert torch.equal(local_branch_strided(qc, kc, vc, L, grid, tile, stride, window, chunk=1, text_bias=bias), out)


def test_strided_layout_stride1_is_flex_layout():
    L, grid, tile, win = 5, (16, 12), (4, 4), (8, 8)
    a = strided_flex_layout(L, grid, tile, (1, 1), win, "cpu", QT, KT)
    b = flex_layout(L, grid, tile, win, "cpu", QT, KT)
    for x, y in ((a.q_gather, b.q_gather), (a.q_inv, b.q_inv), (a.k_gather, b.k_gather)):
        assert torch.equal(x, y)
    for name in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices"):
        assert torch.equal(getattr(a.block_mask, name), getattr(b.block_mask, name))


def test_head_layout_equals_head_restricted_mask():
    from torch.nn.attention.flex_attention import create_block_mask
    L, grid = 4, (16, 16)
    lay = strided_flex_layout(L, grid, (2, 2), (2, 2), (4, 4), "cpu", QT, KT)
    hl = head_layout(lay, 3)
    member = torch.tensor([True, False, True])
    hl.set_heads(member)
    base = lay.block_mask.mask_mod
    ref = create_block_mask(lambda b, h, q, kv: member[h] & base(b, h, q, kv), None, 3, *lay.block_mask.seq_lengths,
                            device="cpu", BLOCK_SIZE=lay.block_mask.BLOCK_SIZE)
    bm = hl.layout.block_mask
    for num, idx, rnum, ridx in ((bm.kv_num_blocks, bm.kv_indices, ref.kv_num_blocks, ref.kv_indices),
                                 (bm.full_kv_num_blocks, bm.full_kv_indices, ref.full_kv_num_blocks, ref.full_kv_indices)):
        assert torch.equal(num, rnum)
        for h in range(3):
            for i in range(num.shape[-1]):
                n = int(num[0, h, i])
                assert sorted(idx[0, h, i, :n].tolist()) == sorted(ridx[0, h, i, :n].tolist())
    assert int(bm.kv_num_blocks[0, 1].sum()) == 0 and int(bm.full_kv_num_blocks[0, 1].sum()) == 0
    hl.set_heads(torch.ones(3, dtype=torch.bool))  # all heads back on: the head-independent tables
    assert torch.equal(bm.kv_num_blocks, lay.block_mask.kv_num_blocks.expand(1, 3, -1))
    assert torch.equal(bm.full_kv_num_blocks, lay.block_mask.full_kv_num_blocks.expand(1, 3, -1))


def _forward_setup(heads=4, H=16, W=16, L=4, n0=4, tile=(2, 2), seed=3):
    """Function-level inputs on a 16 x 16 grid with n0 4: pool (4, 4), apparent scales 1, 2, 4."""
    q, k, v = _qkv(H, W, L, h=heads, seed=seed)
    m, rot, c = _setup(H, W, L)
    f = pool_factors((H, W), n0)
    geo = {}
    for a in scale_ladder(max(f))[1:]:
        d = (f[0] // a, f[1] // a)
        win = (min(n0, H // d[0]), min(n0, W // d[1]))
        rr = ((rot[0][L:], rot[1][L:]),) * 2 if d == (1, 1) else c.coarse_rope(d, "cpu")
        geo[a] = (d, win, rr, head_layout(strided_flex_layout(L, (H, W), tile, d, win, "cpu", QT, KT), heads))
    return q, k, v, rot, c.coarse_rope(f, "cpu"), f, geo, (L, (H, W), tile)


def _run(scales, pi, setup, impl):
    q, k, v, rot, (rq, rk), f, geo, (L, grid, tile) = setup
    a = np.asarray(scales)
    groups = []
    for s, (d, win, (rqd, rkd), hl) in geo.items():
        if impl == "gather":
            if (a == s).any():
                groups.append((d, win, rqd, rkd, torch.as_tensor(np.flatnonzero(a == s))))
            continue
        member = torch.as_tensor(a == s)
        hl.set_heads(member)
        groups.append((d, rqd, rkd, hl.layout, member.view(1, 1, -1, 1)))
    if impl == "gather":
        return sca_scale_gather(q, k, v, rot, rq, rk, pi, L, grid, f, "center", tile, tuple(groups))
    return sca_scale_forward(q, k, v, rot, rq, rk, pi, L, grid, f, "center", tuple(groups))


@pytest.mark.parametrize("impl", ["flex", "gather"])
def test_scale_forward_extremes(impl):
    st = _forward_setup()
    q, k, v, rot, (rq, rk), f, geo, (L, grid, tile) = st
    pi = torch.tensor([0.3, 0.6, 0.8, 0.4]).view(1, 1, -1, 1)
    txt = (rot[0][:L], rot[1][:L])
    o_glob = global_branch(q, k, v, L, grid, f, "center", txt, rq, rk)
    assert torch.equal(_run([1, 1, 1, 1], pi, st, impl), o_glob)                       # a = 1: the global branch
    win = geo[4][1]
    if impl == "flex":                                                                   # a = max(s): published sca
        ref = sca_flex_forward(q, k, v, rot, rq, rk, pi, L, grid, f, "center", flex_layout(L, grid, tile, win, "cpu", QT, KT))
    else:
        qr, kr = rope(q, rot), rope(k, rot)
        o_loc = local_branch(qr[:, L:], kr, v, L, grid, tile, win)
        ref = torch.cat([o_glob[:, :L], (pi * o_loc.float() + (1 - pi) * o_glob[:, L:].float()).double()], 1)
    assert (_run([4, 4, 4, 4], pi, st, impl) - ref).abs().max() < 1e-12


def test_grouping_equals_each_head_alone():
    st = _forward_setup()
    q, k, v, rot, (rq, rk), f, geo, (L, grid, tile) = st
    scales, pis = [1, 2, 4, 2], [0.3, 0.6, 0.8, 0.4]
    pi = torch.tensor(pis).view(1, 1, -1, 1)
    out_flex, out_gather = _run(scales, pi, st, "flex"), _run(scales, pi, st, "gather")
    txt = (rot[0][:L], rot[1][:L])
    for h, (a, p) in enumerate(zip(scales, pis)):
        qh, kh, vh = (t[:, :, h:h + 1] for t in (q, k, v))
        og = global_branch(qh, kh, vh, L, grid, f, "center", txt, rq, rk)
        ref = og
        if a > 1:
            d, win, (rqd, rkd), _ = geo[a]
            qc, kc, vc = lattice_qkv(qh, kh, vh, L, grid, d, rqd, rkd, rope(kh[:, :L], txt))
            o_loc = local_branch_strided(qc, kc, vc, L, grid, tile, d, win)
            pt = torch.tensor(p)
            ref = torch.cat([og[:, :L], (pt * o_loc.float() + (1 - pt) * og[:, L:].float()).double()], 1)
        assert (out_flex[:, :, h:h + 1] - ref).abs().max() < 1e-12, h
        assert (out_gather[:, :, h:h + 1] - ref).abs().max() < 1e-12, h
    # the stride-2 heads differ from both extremes
    g = _run([1, 1, 1, 1], pi, st, "flex")
    n = _run([4, 4, 4, 4], pi, st, "flex")
    assert not torch.allclose(out_flex[:, L:, 1], g[:, L:, 1]) and not torch.allclose(out_flex[:, L:, 1], n[:, L:, 1])


def test_timetable_hand_values():
    assert scale_ladder(4) == [1, 2, 4] and scale_ladder(2) == [1, 2] and scale_ladder(1) == [1]
    assert scale_ladder(3) == [1, 2, 3] and scale_ladder(8) == [1, 2, 4, 8]
    pi = [0.0, 0.5, 0.9, 1.0]  # sigma_end = 0.5 + 0.39 pi = 0.5, 0.695, 0.851, 0.89
    # sigma 0.8: u = 0.09 / (0.39, 0.195, 0.039, 0) = 0.231, 0.462, 1, 1 -> round(2u) = 0, 1, 2, 2
    assert apparent_scale(0.8, pi, 0.89, 0.5, 4).tolist() == [1, 2, 4, 4]
    # sigma 0.6: u = 0.744, 1, 1, 1 -> round(2u) = 1, 2, 2, 2
    assert apparent_scale(0.6, pi, 0.89, 0.5, 4).tolist() == [2, 4, 4, 4]
    assert apparent_scale(0.5, pi, 0.89, 0.5, 4).tolist() == [4, 4, 4, 4]
    assert apparent_scale(0.89, pi, 0.89, 0.5, 4).tolist() == [1, 1, 1, 1]       # at the gate
    assert apparent_scale(0.95, pi, 0.89, 0.5, 4).tolist() == [1, 1, 1, 1]       # above the gate
    # two-step ladder (s 2): round(u) with u = 0.231, 0.462, 1, 1
    assert apparent_scale(0.8, pi, 0.89, 0.5, 2).tolist() == [1, 1, 2, 2]
    # s 3 (not a power of two): 2 ** round(u log2 3) capped at 3
    assert apparent_scale(0.6, pi, 0.89, 0.5, 3).tolist() == [2, 3, 3, 3]


def _sched_ctx(impl, **kw):
    base = dict(mode="sca", n0=4, tile=2, window=4, local_impl=impl, gate_sigma=0.9, scale_sched=True,
                scale_pi={2: head_table([0.35, 0.65]), 4: head_table([0.2, 0.9])}, pi=head_table([0.2, 0.9]))
    base.update(kw)
    return SCAContext(**base)


@pytest.mark.parametrize("impl", ["flex", "gather"])
def test_processor_extremes_and_mixed(impl):
    m, H, W = model(), 16, 16  # n0 4: pool (4, 4), apparent scales 1, 2, 4; attn_outputs runs at sigma 0.5
    pub = dict(mode="sca", n0=4, tile=2, window=4, local_impl=impl, pi=head_table([0.2, 0.9]))
    # scale_lo 0.6 > sigma: every head at its native window, weights from the scale-4 table = published sca
    native = attn_outputs(m, _sched_ctx(impl, scale_lo=0.6), H, W)
    ref = attn_outputs(m, SCAContext(gate_sigma=0.9, **pub), H, W)
    assert torch.allclose(flat(native), flat(ref), atol=1e-5)
    # gate 0.55, scale_lo 0, small pi: u < 0.25 for both heads, apparent scale 1 = global only (gate below sigma)
    glob = attn_outputs(m, _sched_ctx(impl, gate_sigma=0.55, scale_lo=0.0, pi=head_table([0.1, 0.4])), H, W)
    ref_g = attn_outputs(m, SCAContext(gate_sigma=0.4, **pub), H, W)
    assert torch.allclose(flat(glob), flat(ref_g), atol=1e-5)
    # scale_lo 0.2, pi (0, 0.9): head 0 at u = 0.57 -> a = 2 (weight 0.35 from the scale-2 table), head 1 at a = 4
    c = _sched_ctx(impl, scale_lo=0.2, pi=head_table([0.0, 0.9]))
    mixed = attn_outputs(m, c, H, W)
    a, _, w = c.scale_now(0, 2, "cpu")
    assert a.tolist() == [2, 4] and np.allclose(w.numpy(), [0.35, 0.9])
    assert c.pi_trace[-1]["scale_frac"] == {"1": 0.0, "2": 0.5, "4": 0.5}
    assert not torch.allclose(flat(mixed), flat(native), atol=1e-4) and not torch.allclose(flat(mixed), flat(glob), atol=1e-4)
    other = attn_outputs(m, _sched_ctx("gather" if impl == "flex" else "flex", scale_lo=0.2, pi=head_table([0.0, 0.9])), H, W)
    assert torch.allclose(flat(mixed), flat(other), atol=1e-5)


def test_scale_sched_validates():
    SCAContext(mode="sca", scale_sched=True, gate_sigma=0.89, local_impl="gather")
    for kw in (dict(band=2), dict(bridge_cells=8), dict(lse_tau=1.0), dict(ramp_sigma=0.5), dict(route=0.5),
               dict(pool="mean"), dict(scale_lo=0.9), dict(gate_sigma=0.0)):
        with pytest.raises(ValueError):
            SCAContext(**{"mode": "sca", "scale_sched": True, "gate_sigma": 0.89, **kw})


@pytest.mark.skipif(not TABLE.exists(), reason="calibrated table not on this machine")
def test_sample_info_and_tables(sample):  # noqa: F811
    base = ["--method", "sca", "--pi-table", str(TABLE), "--gate-sigma", "0.89", "--res", "4096x4096",
            "--prompts", "x.jsonl", "--out", "o"]
    c0, info = sample.build_context(sample.parser().parse_args(base), (256, 256))
    assert not c0.scale_sched and not {"scale_sched", "scale_lo"} & set(info)
    c, info2 = sample.build_context(sample.parser().parse_args(base + ["--scale-sched", "--scale-lo", "0.4"]), (256, 256))
    assert set(info2) - set(info) == {"scale_sched", "scale_lo"} and info2["scale_sched"] is True and info2["scale_lo"] == 0.4
    assert c.scale_sched and c.scale_lo == 0.4 and c.gate_sigma == 0.89
    t = c.scale_tables(4)
    assert sorted(t) == [2, 4] and t[2].scale == 2 and t[4].scale == 4
    assert np.array_equal(t[4](0.6), c.pi(0.6)) and not np.array_equal(t[2](0.6), c.pi(0.6))
    c.grid = (256, 256)
    assert [c.scale_geometry(a) for a in (1, 2, 4)] == [((4, 4), (64, 64)), ((2, 2), (64, 64)), ((1, 1), (64, 64))]
    c.grid = (128, 256)  # 2048 x 4096: pool (2, 4)
    assert [c.scale_geometry(a) for a in (1, 2, 4)] == [((2, 4), (64, 64)), ((1, 2), (64, 64)), ((1, 1), (64, 64))]


def test_cost_never_exceeds_published_local():
    for hw in ((4096, 4096), (2048, 4096)):
        r = scale_sched_cost(*hw)
        rows = r["scales"]
        assert [x["a"] for x in rows] == [1, 2, 4] and rows[0]["keys"] == 0 and rows[0]["flops"] == 0
        assert rows[-1]["keys_ratio"] == 1 and rows[-1]["blocks_ratio"] == 1
        for x in rows:
            assert x["keys"] <= r["published_keys"] and x["flops"] <= r["published_flops"]
            assert x["blocks_ratio"] <= 1.1  # flex tile alignment only


@pytest.mark.parametrize("px", [(512, 512), (512, 1024)])
def test_cost_blocks_match_block_mask(px):
    """The analytic block count equals the BlockMask of strided_flex_layout (text = one key block)."""
    r = scale_sched_cost(*px, n0=8, tile=4, text=16, heads=1, head_dim=1, q_tile=(4, 8), k_tile=(4, 4))
    grid = tuple(r["grid"])
    for x in r["scales"][1:]:
        bm = strided_flex_layout(16, grid, (4, 4), tuple(x["stride"]), tuple(x["window"]), "cpu", (4, 8), (4, 4)).block_mask
        assert abs(float((bm.kv_num_blocks + bm.full_kv_num_blocks).double().mean()) - x["blocks"]) < 1e-9, x


@pytest.mark.parametrize("grid", [(256, 256), (128, 256)])
def test_keys_and_offsets_at_every_scale(grid):
    """At 4096^2 and 2048 x 4096 every image query sees n0^2 lattice keys whose coarse offsets stay below n0 per axis
    (Proposition 1 at every apparent scale)."""
    from sca.attn import strided_window_index
    H, W = grid
    f = pool_factors(grid, 64)
    for a in scale_ladder(max(f))[1:]:
        d = (max(1, f[0] // a), max(1, f[1] // a))
        Wc = W // d[1]
        win = (min(64, H // d[0]), min(64, Wc))
        idx = strided_window_index(0, grid, (8, 8), d, win, "cpu")             # [tiles, keys], no text
        assert idx.shape[1] == 64 * 64
        t = torch.arange(idx.shape[0])
        q_rows = ((t // (W // 8)) * 8)[:, None] + torch.arange(8)              # full-resolution rows of the tile
        q_cols = ((t % (W // 8)) * 8)[:, None] + torch.arange(8)
        for k, q, s in ((idx // Wc, q_rows, d[0]), (idx % Wc, q_cols, d[1])):
            off = torch.maximum(k.amax(1) - (q // s).amin(1), (q // s).amax(1) - k.amin(1))
            assert int(off.max()) < 64, (a, d)
