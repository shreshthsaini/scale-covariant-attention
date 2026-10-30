"""Duplication fix round three: matched window weights (E, --win-pi-match), per-head wide windows (F,
--win-heads-below), fixed apparent scale with the bridge (G, --scale-fixed). All off by default."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sca.attn import (SCAContext, bridge_index, flex_layout, head_layout, rope, sca_dedup_forward,  # noqa: E402
                      sca_headwin_forward, sca_scale_forward, sdpa, strided_flex_layout)
from test_dedup import TABLE, flat, head_table, sample  # noqa: E402,F401  (sample is a fixture)
from test_sca import attn_outputs, model  # noqa: E402
from test_scale_sched import KT, QT, _qkv, _setup  # noqa: E402

T4, T2 = head_table([0.2, 0.9]), head_table([0.35, 0.65])   # published (target scale 4) and scale-2 weights
BRIDGE = dict(bridge_cells=2, bridge_k=2)
GRID = (16, 16)  # n0 4: pool (4, 4), base window 4 (ratio 4), window 8 has ratio 2


def _ctx(impl="flex", **kw):
    base = dict(mode="sca", n0=4, tile=2, window=4, local_impl=impl, gate_sigma=0.9, pi=T4, scale_pi={2: T2})
    base.update(kw)
    return SCAContext(**base)


def _close(a, b, tol=1e-5):
    return torch.allclose(flat(a), flat(b), atol=tol)


def test_win_pi_match_reads_the_matched_table():
    m = model()  # attn_outputs runs at sigma 0.5
    e = attn_outputs(m, _ctx(win_sched=((0.4, 8),), win_pi_match=True, **BRIDGE), *GRID)
    assert _close(e, attn_outputs(m, _ctx(window=8, pi=T2, **BRIDGE), *GRID))           # window 8 with scale-2 weights
    plain = attn_outputs(m, _ctx(win_sched=((0.4, 8),), **BRIDGE), *GRID)
    assert _close(plain, attn_outputs(m, _ctx(window=8, **BRIDGE), *GRID)) and not _close(e, plain, 1e-4)
    below = attn_outputs(m, _ctx(win_sched=((0.6, 8),), win_pi_match=True, **BRIDGE), *GRID)  # base window this step
    assert _close(below, attn_outputs(m, _ctx(**BRIDGE), *GRID))
    c = _ctx(win_sched=((0.4, 8),), win_pi_match=True)
    c.grid, c.sigma = GRID, 0.5
    assert c.pi_table_now() is T2
    c.sigma = 0.3
    assert c.pi_table_now() is T4


def test_headwin_forward_per_head_and_limits():
    H, W, L, f, tile, heads = 16, 16, 4, (4, 4), (2, 2), 3
    q, k, v = _qkv(H, W, L, h=heads, seed=5)
    _, rot, c = _setup(H, W, L)
    rq, rk = c.coarse_rope(f, "cpu")
    cells, per = (2, 2), 2
    P = L + cells[0] * cells[1] * per
    p4, p2 = torch.tensor([0.2, 0.9, 0.6]), torch.tensor([0.35, 0.65, 0.5])
    ref = {}
    for name, win, p in (("native", (4, 4), p4), ("wide", (8, 8), p2)):  # every head at one window: the bridge path
        ref[name] = sca_dedup_forward(q, k, v, rot, rq, rk, p.view(1, 1, -1, 1), L, (H, W), f, "center",
                                      flex_layout(P, (H, W), tile, win, "cpu", QT, KT), 0.0, cells, per)
    native = head_layout(flex_layout(P, (H, W), tile, (4, 4), "cpu", QT, KT), heads)
    wide = head_layout(flex_layout(P, (H, W), tile, (8, 8), "cpu", QT, KT), heads)
    assert torch.equal(native.layout.q_inv, wide.layout.q_inv)
    for tau in (0.0, 0.5, 0.7, 1.0):
        member = p4 < tau
        native.set_heads(~member)
        wide.set_heads(member)
        out = sca_headwin_forward(q, k, v, rot, rq, rk, torch.where(member, p2, p4).view(1, 1, -1, 1), L, (H, W), f,
                                  "center", native.layout, wide.layout, member.view(1, 1, -1, 1), 0.0, cells, per)
        for h in range(heads):
            r = ref["wide" if member[h] else "native"]
            assert (out[:, :, h] - r[:, :, h]).abs().max() < 1e-12, (tau, h)
        if tau in (0.0, 1.0):  # all native (published bridge) or all wide (bridge + wide window + matched weight)
            assert torch.equal(out, ref["native" if tau == 0.0 else "wide"])
    assert not torch.allclose(ref["native"], ref["wide"])


@pytest.mark.parametrize("tau", [0.0, 0.5, 1.0])
def test_processor_head_windows(tau):
    m = model()
    out = attn_outputs(m, _ctx(win_sched=((0.4, 8),), win_heads_below=tau, **BRIDGE), *GRID)
    native = attn_outputs(m, _ctx(**BRIDGE), *GRID)                       # base window, published weights
    wide = attn_outputs(m, _ctx(window=8, pi=T2, **BRIDGE), *GRID)        # variant E on every head
    if tau == 0.0:
        assert _close(out, native)
    elif tau == 1.0:
        assert _close(out, wide)
    else:  # head 0 (pi 0.2) wide with weight 0.35, head 1 (pi 0.9) native with weight 0.9
        assert not _close(out, native, 1e-4) and not _close(out, wide, 1e-4)
        c = _ctx(win_sched=((0.4, 8),), win_heads_below=tau, **BRIDGE)
        attn_outputs(m, c, *GRID)
        member, w = c.headwin_now(0, 2, "cpu")
        assert member.tolist() == [True, False] and np.allclose(w.numpy(), [0.35, 0.9])
        assert c.pi_trace[-1]["wide_frac"] == 0.5
    # off (None) is the plain window schedule: every head wide with the published weight
    off = attn_outputs(m, _ctx(win_sched=((0.4, 8),), **BRIDGE), *GRID)
    assert _close(off, attn_outputs(m, _ctx(window=8, **BRIDGE), *GRID))


def test_strided_bridge_matches_bruteforce():
    """G's local branch: text, then the bridge keys at coarse ids floor(p / 2), then the stride-2 window; one softmax."""
    H, W, L, f, tile, d, win = 16, 16, 4, (4, 4), (2, 2), (2, 2), (4, 4)
    q, k, v = _qkv(H, W, L, seed=9)
    m, rot, c = _setup(H, W, L)
    rq, rk = c.coarse_rope(f, "cpu")
    rqd, rkd = c.coarse_rope(d, "cpu")
    cells, per = (2, 2), 2
    idx = bridge_index(k[:, L:], (H, W), cells, per)[0]
    P = L + len(idx)
    lay = strided_flex_layout(P, (H, W), tile, d, win, "cpu", QT, KT)
    every = torch.ones(1, 1, 2, 1, dtype=torch.bool)
    out = sca_scale_forward(q, k, v, rot, rq, rk, torch.ones(1, 1, 2, 1), L, (H, W), f, "center",
                            ((d, rqd, rkd, lay, every),), 0.0, cells, per)
    dt, Hc, Wc = torch.float64, H // 2, W // 2
    coarse = lambda n: [0, (n // W) // 2, (n % W) // 2]  # noqa: E731
    pos = [(i * 2 + 1) * W + j * 2 + 1 for i in range(Hc) for j in range(Wc)]
    pe = lambda ids: m.pos_embed(torch.tensor(ids, dtype=dt))  # noqa: E731
    qr = rope(q[:, L:], pe([coarse(n) for n in range(H * W)]))
    kk = torch.cat([rope(k[:, :L], pe([[0, 0, 0]] * L)), rope(k[:, L:][:, idx], pe([coarse(int(n)) for n in idx])),
                    rope(k[:, L:][:, pos], pe([[0, i, j] for i in range(Hc) for j in range(Wc)]))], 1)
    vv = torch.cat([v[:, :L], v[:, L:][:, idx], v[:, L:][:, pos]], 1)
    mask = torch.zeros(H * W, P + Hc * Wc, dtype=torch.bool)
    mask[:, :P] = True
    for r in range(H):
        for cc in range(W):
            r0 = min(max(((r // 2) * 2 + 1) // 2 - 2, 0), Hc - 4)
            c0 = min(max(((cc // 2) * 2 + 1) // 2 - 2, 0), Wc - 4)
            for i in range(r0, r0 + 4):
                mask[r * W + cc, P + i * Wc + c0:P + i * Wc + c0 + 4] = True
    ref = sdpa(qr, kk, vv, mask)
    assert (out[:, L:] - ref).abs().max() < 1e-6  # the mix runs in fp32
    # at stride 1 the strided path with the bridge is the published bridge path (true ids)
    pi = torch.tensor([0.3, 0.8]).view(1, 1, -1, 1)
    img = (rot[0][L:], rot[1][L:])
    one = sca_scale_forward(q, k, v, rot, rq, rk, pi, L, (H, W), f, "center",
                            (((1, 1), img, img, strided_flex_layout(P, (H, W), tile, (1, 1), win, "cpu", QT, KT), every),),
                            0.0, cells, per)
    pub = sca_dedup_forward(q, k, v, rot, rq, rk, pi, L, (H, W), f, "center",
                            flex_layout(P, (H, W), tile, win, "cpu", QT, KT), 0.0, cells, per)
    assert (one - pub).abs().max() < 1e-12


def test_processor_scale_fixed():
    m = model()
    c = _ctx(scale_fixed=((0.4, 2),), **BRIDGE)  # sigma 0.5 > 0.4: every head at a = 2 with scale-2 weights
    out = attn_outputs(m, c, *GRID)
    assert c.fixed_scale_now() == 2 and c.pi_table_now() is T2
    below = attn_outputs(m, _ctx(scale_fixed=((0.6, 2),), **BRIDGE), *GRID)  # sigma 0.5 < 0.6: published bridge path
    assert _close(below, attn_outputs(m, _ctx(**BRIDGE), *GRID)) and not _close(out, below, 1e-4)
    # without the bridge flex equals gather; a = max(s) is the published path
    g = attn_outputs(m, _ctx("gather", scale_fixed=((0.4, 2),)), *GRID)
    assert _close(g, attn_outputs(m, _ctx("flex", scale_fixed=((0.4, 2),)), *GRID))
    assert _close(attn_outputs(m, _ctx(scale_fixed=((0.4, 4),)), *GRID), attn_outputs(m, _ctx(), *GRID))
    assert _close(attn_outputs(m, _ctx(scale_fixed=((0.4, 1),)), *GRID), attn_outputs(m, _ctx(gate_sigma=0.4), *GRID))


def test_round3_options_validate():
    ok = dict(mode="sca", gate_sigma=0.89)
    SCAContext(**ok, bridge_cells=8, scale_fixed=((0.8, 2),))
    SCAContext(**ok, bridge_cells=8, win_sched=((0.8, 128),), win_heads_below=0.5, win_pi_match=True)
    assert SCAContext(**ok, scale_fixed=((0.7, 4), (0.8, 2))).scale_fixed == ((0.8, 2), (0.7, 4))
    for kw in (dict(scale_fixed=((0.8, 2),), scale_sched=True), dict(scale_fixed=((0.8, 2),), win_sched=((0.8, 128),)),
               dict(scale_fixed=((0.8, 0),)), dict(scale_fixed=((0.8, 2),), pool="mean"),
               dict(win_heads_below=0.5), dict(win_heads_below=0.5, win_sched=((0.8, 128),), local_impl="gather"),
               dict(win_heads_below=0.5, win_sched=((0.8, 128),), lse_tau=1.0)):
        with pytest.raises(ValueError):
            SCAContext(**ok, **kw)


@pytest.mark.skipif(not TABLE.exists(), reason="calibrated table not on this machine")
def test_sample_info_round3(sample):  # noqa: F811
    base = ["--method", "sca", "--pi-table", str(TABLE), "--gate-sigma", "0.89", "--res", "4096x4096",
            "--prompts", "x.jsonl", "--out", "o", "--bridge-cells", "8", "--bridge-k", "16"]
    build = lambda extra: sample.build_context(sample.parser().parse_args(base + extra), (256, 256))  # noqa: E731
    c0, info = build([])
    assert c0.win_heads_below is None and not c0.win_pi_match and c0.scale_fixed == ()
    for extra, keys in ((["--win-sched", "0.80:128", "--win-pi-match"], {"win_sched", "win_pi_match"}),
                        (["--win-sched", "0.80:128", "--win-heads-below", "0.5"], {"win_sched", "win_heads_below"}),
                        (["--scale-fixed", "0.80:2"], {"scale_fixed"})):
        c, info2 = build(extra)
        assert set(info2) - set(info) == keys, extra
    assert info2["scale_fixed"] == [[0.8, 2]] and c.scale_fixed == ((0.8, 2),)
    c.grid = (256, 256)
    assert [c.table_at(s).scale for s in (256 / 128, 256 / 96, 4.0)] == [2, 2, 4]
    c.sigma = 0.85
    assert c.fixed_scale_now() == 2 and c.pi_table_now().scale == 2
    c.sigma = 0.7
    assert c.fixed_scale_now() == 4 and c.pi_table_now() is c.pi
