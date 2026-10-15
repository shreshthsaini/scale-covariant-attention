"""CPU tests on a tiny random FluxTransformer2DModel (float32)."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusers import FluxTransformer2DModel  # noqa: E402

from sca.attn import (PiTable, SCAContext, attention_flops, install, nearest_divisor,  # noqa: E402
                      pool_factors)
from sca.calib import CalibStats, evaluate, centers_of, fit, merge, rel_err  # noqa: E402

L_TXT, HEADS, DH = 6, 2, 16


def model():
    torch.manual_seed(0)
    return FluxTransformer2DModel(patch_size=1, in_channels=4, num_layers=2, num_single_layers=2,
                                  attention_head_dim=DH, num_attention_heads=HEADS, joint_attention_dim=24,
                                  pooled_projection_dim=8, guidance_embeds=False, axes_dims_rope=(4, 6, 6)).eval()


def inputs(H, W, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(H, W, 3)
    ids[..., 1] = torch.arange(H)[:, None]
    ids[..., 2] = torch.arange(W)[None, :]
    return dict(hidden_states=torch.randn(1, H * W, 4, generator=g),
                encoder_hidden_states=torch.randn(1, L_TXT, 24, generator=g),
                pooled_projections=torch.randn(1, 8, generator=g), timestep=torch.tensor([0.7]),
                img_ids=ids.reshape(-1, 3), txt_ids=torch.zeros(L_TXT, 3), return_dict=False)


@torch.no_grad()
def run(m, x, ctx=None):
    if ctx is None:
        return m(**x)[0]
    undo = install(m, ctx)
    try:
        return m(**x)[0]
    finally:
        undo()


@torch.no_grad()
def attn_outputs(m, ctx, H, W):
    """Per-module outputs (double: (img, txt); single: [txt|img]) for a fixed random input."""
    x = inputs(H, W, seed=3)
    undo = install(m, ctx) if ctx is not None else None
    if ctx is not None:
        ctx.set_inputs(x["img_ids"], sigma=0.5)
    rot = m.pos_embed(torch.cat([x["txt_ids"], x["img_ids"]]))
    g = torch.Generator().manual_seed(5)
    img, txt = torch.randn(1, H * W, m.inner_dim, generator=g), torch.randn(1, L_TXT, m.inner_dim, generator=g)
    outs = [m.transformer_blocks[0].attn(img, txt, image_rotary_emb=rot),
            m.single_transformer_blocks[0].attn(torch.cat([txt, img], 1), image_rotary_emb=rot)]
    if undo:
        undo()
    return outs


def test_full_matches_stock():
    m = model()
    for H, W in ((8, 8), (6, 10)):
        x = inputs(H, W)
        ref = run(m, x)
        out = run(m, x, SCAContext(mode="full"))
        assert torch.allclose(out, ref, atol=1e-5, rtol=0), (out - ref).abs().max()


def test_local_full_window_equals_full_attention():
    m = model()
    H = W = 8
    ref = attn_outputs(m, None, H, W)
    for tile in (2, 8):  # several tiles whose clamped windows cover the grid, and one tile
        ctx = SCAContext(mode="sca", n0=4, tile=tile, window=8, pi=PiTable("const", value=1.0))
        out = attn_outputs(m, ctx, H, W)
        assert ctx.geometry()[0] == (2, 2)
        assert torch.allclose(out[0][0], ref[0][0], atol=1e-5)          # double block image stream
        assert torch.allclose(out[1][:, L_TXT:], ref[1][:, L_TXT:], atol=1e-5)  # single block image part


def test_global_f1_equals_full():
    m = model()
    x = inputs(8, 8)
    ref = run(m, x)
    for pi in (0.0, 0.37):  # f = 1 and a window covering the grid make both branches exact
        ctx = SCAContext(mode="sca", n0=8, tile=4, window=8, pi=PiTable("const", value=pi))
        assert torch.allclose(run(m, x, ctx), ref, atol=1e-5)
    ctx = SCAContext(mode="sca", n0=8, tile=4, window=2, pi=PiTable("const", value=0.0))
    assert torch.allclose(run(m, x, ctx), ref, atol=1e-5)  # pi = 0: local window size irrelevant


def test_nonsquare_and_text_queries():
    m = model()
    H, W = 8, 16
    ctx = SCAContext(mode="sca", n0=4, tile=2, window=4, pi=PiTable("const", value=0.5))
    out = run(m, inputs(H, W), ctx)
    assert out.shape == (1, H * W, 4) and torch.isfinite(out).all()
    assert ctx.geometry()[0] == (2, 4)
    # text-query outputs come from the global branch only: independent of pi
    a = attn_outputs(m, SCAContext(mode="sca", n0=4, tile=2, window=4, pi=PiTable("const", value=0.0)), H, W)
    b = attn_outputs(m, SCAContext(mode="sca", n0=4, tile=2, window=4, pi=PiTable("const", value=1.0)), H, W)
    assert torch.equal(a[0][1], b[0][1]) and torch.equal(a[1][:, :L_TXT], b[1][:, :L_TXT])
    assert not torch.allclose(a[0][0], b[0][0])
    # mean pooling and per-layer / per-head tables run
    t = PiTable("head", np.random.rand(4, HEADS, 3), np.array([0.2, 0.5, 0.8]))
    out = run(m, inputs(H, W), SCAContext(mode="sca", n0=4, tile=2, window=4, pool="mean", pi=t))
    assert torch.isfinite(out).all()


def test_divisor_rules():
    assert pool_factors((128, 256), 64) == (2, 4)
    assert pool_factors((384, 128), 64) == (6, 2)
    assert pool_factors((9, 9), 4) == (3, 3) and nearest_divisor(10, 2.5) == 2
    log = []
    ctx = SCAContext(mode="sca", n0=4, tile=4, window=4, log=log)
    ctx.grid = (9, 9)
    f, tile, win = ctx.geometry()
    assert f == (3, 3) and tile == (3, 3) and win == (4, 4) and log


def test_calib_pi_is_one_when_local_is_exact():
    m = model()
    H = W = 8
    x = inputs(H, W)
    ref = run(m, x)
    ctx = SCAContext(mode="calib", n0=4, tile=2, calib_scales=(2,), calib_window=8)
    ctx.calib = CalibStats(4, HEADS, (2,), bins=4, mass_queries=16)
    out = run(m, x, ctx)
    assert torch.allclose(out, ref, atol=1e-5)  # sampling unchanged in calib mode
    S = ctx.calib.state()["stats"][0]
    b = ctx.calib.bin(0.7)
    A, B = S[0, ..., b], S[1, ..., b]
    assert (A > 0).all() and np.allclose(B / A, 1.0, atol=1e-4)
    assert np.allclose(S[3, ..., b], 0, atol=1e-8)  # E_full_loc
    mass = ctx.calib.state()["mass"][:, :, :, b] / ctx.calib.state()["mass_n"][None, :, None, b]
    assert (mass[1:] <= 1 + 1e-6).all() and (np.diff(mass[1:], axis=0) >= -1e-6).all()  # radii nested


def test_chunking_identical():
    m = model()
    x = inputs(8, 8)
    outs = [run(m, x, SCAContext(mode="sca", n0=4, tile=2, window=4, chunk=c, local_impl="gather",
                                 pi=PiTable("const", value=0.6))) for c in (1, 3, None)]
    assert torch.allclose(outs[0], outs[2], atol=1e-6) and torch.allclose(outs[1], outs[2], atol=1e-6)
    for fused in (False, True):
        flex = run(m, x, SCAContext(mode="sca", n0=4, tile=2, window=4, local_impl="flex", fused=fused,
                                    pi=PiTable("const", value=0.6)))
        assert torch.allclose(flex, outs[2], atol=1e-6), (fused, (flex - outs[2]).abs().max())


@pytest.mark.parametrize("H,W,L,tile,win,q_tile,k_tile", [
    (16, 24, 5, (8, 8), (8, 8), None, None),        # default 8x8 blocks, text length not a block multiple
    (16, 16, 7, (4, 4), (8, 8), None, None),        # several model tiles per query block
    (12, 20, 3, (2, 4), (6, 8), (4, 4), (4, 4)),    # grid not divisible by blocks (padding), odd window
    (24, 16, 0, (8, 8), (16, 16), (4, 8), (8, 4)),  # no text, non-square blocks
    (32, 40, 16, (8, 8), (16, 16), (8, 16), (8, 8)),
    (8, 12, 6, (8, 12), (8, 12), None, None),       # window covers the whole grid
])
def test_flex_local_matches_gather(H, W, L, tile, win, q_tile, k_tile):
    from sca.attn import flex_layout, local_branch, local_branch_flex
    g = torch.Generator().manual_seed(H * W + L)
    q, k, v = (torch.randn(2, L + H * W, 2, 16, generator=g, dtype=torch.float64) for _ in range(3))
    ref = local_branch(q, k, v, L, (H, W), tile, win)
    lay = flex_layout(L, (H, W), tile, win, "cpu", q_tile, k_tile)
    out = local_branch_flex(q[:, L:], k, v, L, lay)
    assert (out - ref).abs().max() < 1e-12
    o_perm = local_branch_flex(q[:, L:], k, v, L, lay, permuted=True)
    assert torch.equal(o_perm.transpose(1, 2).index_select(1, lay.q_inv), out)


def test_merge_and_report(tmp_path):
    rng = np.random.default_rng(0)
    parts = []
    for j in range(4):
        st = np.zeros((2, 6, 3, 2, 4))
        A = rng.random((2, 3, 2, 4)) + 0.1
        pi = rng.random((2, 3, 2, 4))
        st[:, 0], st[:, 1] = A, pi * A
        st[:, 2] = A + rng.random(A.shape)  # E_fg >= B^2 / A
        st[:, 3] = st[:, 2] - 2 * st[:, 1] + st[:, 0]
        st[:, 4], st[:, 5] = 10.0, 1.0
        st[..., 0] = 0  # sigma bin 0 is never visited
        sig = np.array([[0.0, 0.3, 0.6, 0.9], [0, 1, 1, 1]])
        parts.append({"stats": st, "mass": np.ones((5, 3, 2, 4)), "mass_n": np.ones((3, 4)), "sigma": sig,
                      "scales": np.array([2, 4]), "radii": np.array([2, 4, 8, 16])})
    rep = merge(parts, "64x64", tmp_path / "t.npz")
    r = rep["scales"]["2"]["rel_err"]
    assert r["per_head"] <= r["per_layer"] + 1e-12 <= r["uniform"] + 2e-12
    assert r["per_head"] <= r["switch"] and r["per_head"] <= min(r["glob"], r["loc"])
    t = PiTable.load(tmp_path / "t.npz", "head", target_scale=4)
    assert t.scale == 4 and t(0.5).shape == (3, 2)
    assert PiTable.load(tmp_path / "t.npz", "layer")(0.5).shape == (3, 1)
    assert PiTable.load(tmp_path / "t.npz", "uniform")(0.5).shape == (1, 1)
    assert PiTable.load(tmp_path / "t.npz", "switch").variant == "switch"


def test_flops_report():
    r = attention_flops(4096, 4096)
    print(r)
    assert r["pool"] == [4, 4] and r["attn_flop_ratio"] > 5


def test_branches_match_bruteforce():
    """Local windows and coarse-grid global branch against explicit masked / per-id references."""
    from sca.attn import global_branch, local_branch, rope
    m = model()
    H, W, L, h, D = 8, 12, L_TXT, HEADS, DH
    g = torch.Generator().manual_seed(7)
    q, k, v = (torch.randn(1, L + H * W, h, D, generator=g) for _ in range(3))
    ids = inputs(H, W)["img_ids"]
    rot = m.pos_embed(torch.cat([torch.zeros(L, 3), ids]))
    qr, kr = rope(q, rot), rope(k, rot)
    tile, win = (2, 3), (4, 6)
    mask = torch.zeros(H * W, L + H * W, dtype=torch.bool)
    mask[:, :L] = True
    for r in range(H):
        for c in range(W):
            r0 = min(max((r // 2) * 2 + 1 - 2, 0), H - 4)
            c0 = min(max((c // 3) * 3 + 1 - 3, 0), W - 6)
            for a in range(r0, r0 + 4):
                for b in range(c0, c0 + 6):
                    mask[r * W + c, L + a * W + b] = True
    ref = torch.nn.functional.scaled_dot_product_attention(
        qr[:, L:].transpose(1, 2), kr.transpose(1, 2), v.transpose(1, 2), attn_mask=mask).transpose(1, 2)
    out = local_branch(qr, kr, v, L, (H, W), tile, win, chunk=5)
    assert torch.allclose(out, ref, atol=1e-5)
    # global: pooled keys at block centres carry block-index ids; queries carry floor(id / f)
    f = (2, 3)
    q_ids = torch.stack([torch.zeros(H * W), ids[:, 1] // 2, ids[:, 2] // 3], 1)
    cen = [(a * 2 + 1) * W + b * 3 + 1 for a in range(H // 2) for b in range(W // 3)]
    k_ids = torch.stack([torch.zeros(len(cen)), torch.tensor([float(a) for a in range(H // 2) for b in range(W // 3)]),
                         torch.tensor([float(b) for a in range(H // 2) for b in range(W // 3)])], 1)
    txt = m.pos_embed(torch.zeros(L, 3))
    rq, rk = m.pos_embed(q_ids), m.pos_embed(k_ids)
    sel = torch.tensor(list(range(L)) + [L + i for i in cen])
    qg = torch.cat([rope(q[:, :L], txt), rope(q[:, L:], rq)], 1)
    kg = torch.cat([rope(k[:, :L], txt), rope(k[:, sel[L:]], rk)], 1)
    ref = torch.nn.functional.scaled_dot_product_attention(qg.transpose(1, 2), kg.transpose(1, 2),
                                                           v[:, sel].transpose(1, 2)).transpose(1, 2)
    ctx = SCAContext(mode="sca")
    ctx.pos_embed = m.pos_embed
    ctx.set_inputs(ids)
    out = global_branch(q, k, v, L, (H, W), f, "center", (txt[0], txt[1]), *ctx.coarse_rope(f, q.device))
    assert torch.allclose(out, ref, atol=1e-5)
