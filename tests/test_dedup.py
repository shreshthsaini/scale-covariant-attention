"""Late-duplication options (pi ramp, per-head opening, fractional global ids, window schedule, text bias).

Every option defaults to off and must leave the published method (and its sidecar fingerprint) unchanged."""

import glob
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sca.attn import PiTable, SCAContext, flex_layout, local_branch, local_branch_flex, rope  # noqa: E402
from test_sca import HEADS, L_TXT, attn_outputs, inputs, model  # noqa: E402

TABLE = Path("/scratch/09032/saini_2/sca/calib/k1_1024.npz")


def flat(o):
    return torch.cat([t.flatten() for t in (o[0] if isinstance(o[0], tuple) else (o[0],))] + [o[1].flatten()])


def ctx(**kw):
    base = dict(mode="sca", n0=4, tile=2, window=4, local_impl="gather")
    base.update(kw)
    return SCAContext(**base)


def head_table(values):
    """PiTable 'head' with per-head pi constant in sigma, shape [1 layer, heads, 2 bins]."""
    t = np.repeat(np.asarray(values, dtype=np.float64)[None, :, None], 2, axis=2)
    return PiTable("head", t, [0.1, 0.9])


def test_ramp_linear_scales_pi():
    m, H, W = model(), 8, 8
    # attn_outputs evaluates at sigma 0.5: r = (0.9 - 0.5) / (0.9 - 0.3) = 2/3
    ramped = attn_outputs(m, ctx(pi=PiTable("const", value=0.7), gate_sigma=0.9, ramp_sigma=0.3), H, W)
    ref = attn_outputs(m, ctx(pi=PiTable("const", value=0.7 * 2 / 3), gate_sigma=0.9), H, W)
    assert torch.allclose(flat(ramped), flat(ref), atol=1e-5)
    sq = attn_outputs(m, ctx(pi=PiTable("const", value=0.7), gate_sigma=0.9, ramp_sigma=0.3, ramp_gamma=2.0), H, W)
    ref2 = attn_outputs(m, ctx(pi=PiTable("const", value=0.7 * 4 / 9), gate_sigma=0.9), H, W)
    assert torch.allclose(flat(sq), flat(ref2), atol=1e-5)
    # below the ramp end the table is untouched
    c = ctx(pi=PiTable("const", value=0.7), gate_sigma=0.9, ramp_sigma=0.6)
    c.sigma = 0.5
    assert np.allclose(c.ramp(np.full((1, 1), 0.7)), 0.7)
    c.sigma = 0.75
    assert np.allclose(c.ramp(np.full((1, 1), 0.7)), 0.35)


def test_ramp_head_opens_heads_by_pi():
    m, H, W = model(), 8, 8
    # sigma 0.5, gate 0.9, ramp 0.3: head 0 (pi 0.2) opens at 0.42 (still global), head 1 (pi 0.9) at 0.84 (open)
    out = attn_outputs(m, ctx(pi=head_table([0.2, 0.9]), gate_sigma=0.9, ramp_sigma=0.3, ramp_mode="head"), H, W)
    ref = attn_outputs(m, ctx(pi=head_table([0.0, 0.9]), gate_sigma=0.9), H, W)
    assert torch.allclose(flat(out), flat(ref), atol=1e-5)
    c = ctx(pi=head_table([0.2, 0.9]), gate_sigma=0.9, ramp_sigma=0.3, ramp_mode="head")
    c.sigma = 0.35
    assert np.allclose(c.ramp(np.array([[0.2, 0.9]])), [[0.2, 0.9]])


def test_options_validate():
    with pytest.raises(ValueError):
        SCAContext(mode="sca", gate_sigma=0.89, ramp_sigma=0.95)
    with pytest.raises(ValueError):
        SCAContext(mode="sca", ramp_mode="cosine")
    with pytest.raises(ValueError):
        SCAContext(mode="sca", glob_pos="round")
    assert SCAContext(win_sched=((0.7, 96), (0.8, 128))).win_sched == ((0.8, 128), (0.7, 96))


def test_glob_pos_frac_matches_bruteforce():
    from sca.attn import global_branch
    m = model()
    H, W, L, h, D = 8, 12, L_TXT, HEADS, 16
    g = torch.Generator().manual_seed(11)
    q, k, v = (torch.randn(1, L + H * W, h, D, generator=g) for _ in range(3))
    ids = inputs(H, W)["img_ids"]
    f = (2, 3)
    q_ids = torch.stack([torch.zeros(H * W), ids[:, 1] / 2, ids[:, 2] / 3], 1)
    cen = [(a * 2 + 1) * W + b * 3 + 1 for a in range(H // 2) for b in range(W // 3)]
    k_ids = q_ids[cen]
    txt = m.pos_embed(torch.zeros(L, 3))
    sel = torch.tensor(list(range(L)) + [L + i for i in cen])
    qg = torch.cat([rope(q[:, :L], txt), rope(q[:, L:], m.pos_embed(q_ids))], 1)
    kg = torch.cat([rope(k[:, :L], txt), rope(k[:, sel[L:]], m.pos_embed(k_ids))], 1)
    ref = torch.nn.functional.scaled_dot_product_attention(qg.transpose(1, 2), kg.transpose(1, 2),
                                                           v[:, sel].transpose(1, 2)).transpose(1, 2)
    c = SCAContext(mode="sca", glob_pos="frac")
    c.pos_embed = m.pos_embed
    c.set_inputs(ids)
    out = global_branch(q, k, v, L, (H, W), f, "center", (txt[0], txt[1]), *c.coarse_rope(f, q.device))
    assert torch.allclose(out, ref, atol=1e-5)
    # fractional ids differ from the published floor ids, and offsets stay inside the coarse range
    c2 = SCAContext(mode="sca")
    c2.pos_embed = m.pos_embed
    c2.set_inputs(ids)
    assert not torch.allclose(c2.coarse_rope(f, q.device)[0][0], c.coarse_rope(f, q.device)[0][0])
    assert float(q_ids[:, 1].max()) <= H / 2 and float(q_ids[:, 2].max()) <= W / 3


def test_win_sched_switches_window():
    m, H, W = model(), 8, 8
    wide = attn_outputs(m, ctx(pi=PiTable("const", value=0.7), window=4, win_sched=((0.4, 8),)), H, W)  # sigma .5 > .4
    ref8 = attn_outputs(m, ctx(pi=PiTable("const", value=0.7), window=8), H, W)
    assert torch.allclose(flat(wide), flat(ref8), atol=1e-5)
    narrow = attn_outputs(m, ctx(pi=PiTable("const", value=0.7), window=4, win_sched=((0.6, 8),)), H, W)
    ref4 = attn_outputs(m, ctx(pi=PiTable("const", value=0.7), window=4), H, W)
    assert torch.allclose(flat(narrow), flat(ref4), atol=1e-5)
    assert not torch.allclose(flat(ref4), flat(ref8), atol=1e-4)
    c = SCAContext(mode="sca", n0=64, win_sched=((0.8, 128), (0.7, 96)))
    for s, w in ((0.85, 128), (0.75, 96), (0.7, 64), (0.3, 64)):
        c.sigma = s
        assert c.window_now() == w


@pytest.mark.parametrize("bias", [-2.0, 1.5])
def test_text_bias_flex_matches_gather(bias):
    H, W, L, tile, win = 16, 12, 5, (4, 4), (8, 8)
    g = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(2, L + H * W, 2, 16, generator=g, dtype=torch.float64) for _ in range(3))
    ref = local_branch(q, k, v, L, (H, W), tile, win, text_bias=bias)
    lay = flex_layout(L, (H, W), tile, win, "cpu", (4, 4), (4, 4))
    out = local_branch_flex(q[:, L:], k, v, L, lay, text_bias=bias)
    assert (out - ref).abs().max() < 1e-12
    assert not torch.allclose(ref, local_branch(q, k, v, L, (H, W), tile, win), atol=1e-6)
    # zero bias is the published branch; a very negative bias removes the text keys
    assert torch.equal(local_branch(q, k, v, L, (H, W), tile, win, text_bias=0.0), local_branch(q, k, v, L, (H, W), tile, win))
    no_txt = local_branch(q[:, L:], k[:, L:], v[:, L:], 0, (H, W), tile, win)
    assert (local_branch(q, k, v, L, (H, W), tile, win, text_bias=-200.0) - no_txt).abs().max() < 1e-9


def test_text_bias_only_above_lo():
    m, H, W = model(), 8, 8
    on = attn_outputs(m, ctx(pi=PiTable("const", value=0.7), text_local=-3.0, text_local_lo=0.4), H, W)  # sigma .5
    off = attn_outputs(m, ctx(pi=PiTable("const", value=0.7), text_local=-3.0, text_local_lo=0.6), H, W)
    base = attn_outputs(m, ctx(pi=PiTable("const", value=0.7)), H, W)
    assert torch.allclose(flat(off), flat(base), atol=1e-6)
    assert not torch.allclose(flat(on), flat(base), atol=1e-4)


@pytest.fixture
def sample():
    path = Path(__file__).resolve().parents[1] / "scripts/sample.py"
    spec = importlib.util.spec_from_file_location("sca_sample_dedup_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not TABLE.exists(), reason="calibrated table not on this machine")
def test_sample_info_unchanged_by_default(sample):
    base = ["--method", "sca", "--pi-table", str(TABLE), "--gate-sigma", "0.89", "--res", "4096x4096",
            "--prompts", "x.jsonl", "--out", "o"]
    _, info = sample.build_context(sample.parser().parse_args(base), (256, 256))
    published = glob.glob("/scratch/09032/saini_2/sca/out/main/sca_nb/4096x4096/*.json")
    if published:  # the sidecar block of the published sca_nb run must be reproduced key for key
        assert info == json.loads(Path(published[0]).read_text())["config"]["sca"]
    assert set(info) == {"n0", "tile", "window", "pool", "chunk", "gate_sigma", "band", "route", "pool_factors",
                         "pi_variant", "pi_const", "sigma_star", "pi_scale", "pi_table_sha256"}
    c, info2 = sample.build_context(sample.parser().parse_args(
        base + ["--pi-ramp", "0.6", "--glob-pos", "frac", "--win-sched", "0.7:96,0.8:128", "--text-local", "-2"]), (256, 256))
    assert info2["pi_ramp"] == 0.6 and info2["glob_pos"] == "frac" and info2["win_sched"] == [[0.8, 128], [0.7, 96]]
    assert info2["text_local"] == -2.0 and c.ramp_sigma == 0.6 and c.win_sched == ((0.8, 128), (0.7, 96))


def _qkv(H, W, L, h=2, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, L + H * W, h, D, generator=g, dtype=torch.float64) for _ in range(3)]


def _setup(H, W, L, f, glob_pos="floor"):
    m = model().double()
    ids = inputs(H, W)["img_ids"].double()
    rot = m.pos_embed(torch.cat([torch.zeros(L, 3, dtype=torch.float64), ids]))
    c = SCAContext(mode="sca", glob_pos=glob_pos)
    c.pos_embed = m.pos_embed
    c.set_inputs(ids)
    rq, rk = c.coarse_rope(f, "cpu")
    return m, ids, rot, rq, rk


def test_dedup_forward_reduces_to_published():
    from sca.attn import sca_dedup_forward, sca_flex_forward
    H, W, L, f, tile, win = 8, 8, 6, (2, 2), (2, 2), (4, 4)
    q, k, v = _qkv(H, W, L)
    _, _, rot, rq, rk = _setup(H, W, L, f)
    pi = torch.tensor([0.3, 0.8], dtype=torch.float64).view(1, 1, -1, 1)
    lay = flex_layout(L, (H, W), tile, win, "cpu", (2, 2), (2, 2))
    ref = sca_flex_forward(q, k, v, rot, rq, rk, pi, L, (H, W), f, "center", lay)
    out = sca_dedup_forward(q, k, v, rot, rq, rk, pi, L, (H, W), f, "center", lay)
    assert torch.equal(out, ref)


def test_bridge_index_bruteforce():
    from sca.attn import bridge_index
    g = torch.Generator().manual_seed(4)
    H, W = 8, 12
    k = torch.randn(1, H * W, 2, 4, generator=g, dtype=torch.float64)
    idx = bridge_index(k, (H, W), (2, 3), 3)[0].tolist()
    score = ((k - k.mean(1, keepdim=True)) ** 2).sum((2, 3))[0]
    ref = []
    for a in range(2):
        for b in range(3):
            cell = [r * W + c for r in range(a * 4, a * 4 + 4) for c in range(b * 4, b * 4 + 4)]
            ref += sorted(cell, key=lambda i: -float(score[i]))[:3]
    assert sorted(idx) == sorted(ref) and len(set(idx)) == 18


def test_bridge_local_matches_bruteforce():
    """pi = 1 (local only): text + bridge tokens at true positions + window, one softmax."""
    from sca.attn import bridge_index, sca_dedup_forward
    H, W, L, f, tile, win = 8, 8, 4, (2, 2), (2, 2), (4, 4)
    q, k, v = _qkv(H, W, L, seed=2)
    _, _, rot, rq, rk = _setup(H, W, L, f)
    cells, per = (2, 2), 2
    idx = bridge_index(k[:, L:], (H, W), cells, per)[0]
    lay = flex_layout(L + len(idx), (H, W), tile, win, "cpu", (2, 2), (2, 2))
    pi = torch.ones(1, 1, 2, 1, dtype=torch.float64)
    out = sca_dedup_forward(q, k, v, rot, rq, rk, pi, L, (H, W), f, "center", lay, 0.0, cells, per)
    qr, kr = rope(q, rot), rope(k, rot)
    keys = torch.cat([kr[:, :L], kr[:, L + idx], kr[:, L:]], 1)
    vals = torch.cat([v[:, :L], v[:, L + idx], v[:, L:]], 1)
    P = L + len(idx)
    mask = torch.zeros(H * W, P + H * W, dtype=torch.bool)
    mask[:, :P] = True
    for r in range(H):
        for c in range(W):
            r0 = min(max((r // 2) * 2 + 1 - 2, 0), H - 4)
            c0 = min(max((c // 2) * 2 + 1 - 2, 0), W - 4)
            for a in range(r0, r0 + 4):
                for b in range(c0, c0 + 4):
                    mask[r * W + c, P + a * W + b] = True
    ref = torch.nn.functional.scaled_dot_product_attention(qr[:, L:].transpose(1, 2), keys.transpose(1, 2),
                                                           vals.transpose(1, 2), attn_mask=mask).transpose(1, 2)
    assert (out[:, L:] - ref).abs().max() < 1e-6  # the mix runs in fp32 (mix_branches)


def test_lse_mix_matches_manual():
    from sca.attn import global_branch, local_branch, sca_dedup_forward
    H, W, L, f, tile, win = 8, 8, 4, (2, 2), (2, 2), (4, 4)
    q, k, v = _qkv(H, W, L, seed=5)
    _, _, rot, rq, rk = _setup(H, W, L, f)
    pi = torch.tensor([0.3, 0.8], dtype=torch.float64).view(1, 1, -1, 1)
    lay = flex_layout(L, (H, W), tile, win, "cpu", (2, 2), (2, 2))
    tau = 0.7
    out = sca_dedup_forward(q, k, v, rot, rq, rk, pi, L, (H, W), f, "center", lay, 0.0, None, 0, tau)
    # manual: both branches and their log-sum-exp by brute force
    txt = (rot[0][:L], rot[1][:L])
    qr, kr = rope(q, rot), rope(k, rot)
    o_glob = global_branch(q, k, v, L, (H, W), f, "center", txt, rq, rk, qr[:, :L], kr[:, :L])
    o_loc = local_branch(qr, kr, v, L, (H, W), tile, win)
    d = q.shape[-1] ** -0.5
    cen = [(a * 2 + 1) * W + b * 2 + 1 for a in range(H // 2) for b in range(W // 2)]
    kg = torch.cat([kr[:, :L], rope(k[:, L:][:, cen], rk)], 1)
    qg = rope(q[:, L:], rq)
    lse_g = torch.einsum("bnhd,bmhd->bhnm", qg, kg).mul(d).logsumexp(-1)                   # [B, h, N]
    idxw = __import__("sca.attn", fromlist=["window_index"]).window_index(L, (H, W), tile, win, "cpu")
    lse_l = torch.empty(1, 2, H * W, dtype=torch.float64)
    qt = qr[:, L:]
    for t in range(idxw.shape[0]):
        rows = [(t // (W // 2)) * 2 + i for i in range(2)]
        cols = [(t % (W // 2)) * 2 + j for j in range(2)]
        for rr in rows:
            for cc in cols:
                n = rr * W + cc
                lse_l[0, :, n] = torch.einsum("hd,mhd->hm", qt[0, n], kr[0, idxw[t]]).mul(d).logsumexp(-1)
    dd = lse_l - lse_g
    dd = (dd - dd.mean(-1, keepdim=True)).transpose(1, 2)[..., None]
    w = torch.sigmoid(torch.log(pi) - torch.log1p(-pi) + tau * dd)
    ref = w * o_loc + (1 - w) * o_glob[:, L:]
    assert (out[:, L:] - ref).abs().max() < 1e-6  # fp32 mix
    assert (out[:, :L] - o_glob[:, :L]).abs().max() < 1e-9


def test_processor_bridge_and_lse_run_and_gate():
    m, H, W = model(), 8, 8
    mk = lambda **kw: SCAContext(mode="sca", n0=4, tile=2, window=4, local_impl="flex", pi=PiTable("const", value=0.7), **kw)
    base = attn_outputs(m, mk(), H, W)
    off = attn_outputs(m, mk(bridge_cells=2, bridge_k=2, bridge_lo=0.6), H, W)   # sigma .5 < .6: bridge off
    on = attn_outputs(m, mk(bridge_cells=2, bridge_k=2), H, W)
    lse = attn_outputs(m, mk(lse_tau=1.0), H, W)
    assert torch.allclose(flat(off), flat(base), atol=1e-5)
    assert not torch.allclose(flat(on), flat(base), atol=1e-4)
    assert not torch.allclose(flat(lse), flat(base), atol=1e-4) and torch.isfinite(flat(lse)).all()
    with pytest.raises(ValueError):
        SCAContext(mode="sca", bridge_cells=8, local_impl="gather")
