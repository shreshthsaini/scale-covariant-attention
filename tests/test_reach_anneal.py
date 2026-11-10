"""Reach-annealed hand-off (experiment A) and the progressive table ablation (experiment B), HANDOVER Section 40.

A: --lattice-pool mean. On --scale-fixed steps at apparent scale 2 every query attends the text keys, the bridge and a
window of 2 x 2 block means of the pre-RoPE keys and of the values, keys at the block's coarse id floor(p / 2), queries
at floor(p / 2); one softmax, mixed with the global branch by the scale-2 weight. Checked against a float64 brute force
built element by element. B: PiTable headmean / layermean and sample_progressive.py --pi-variant. Also the measured
schedule of scripts/handoff_from_mass.py. Every option is off by default."""

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sca.attn import (PiTable, SCAContext, bridge_index, rope, sca_scale_forward, sca_scale_gather,  # noqa: E402
                      sdpa, strided_flex_layout)
from test_dedup import TABLE, flat, head_table, sample  # noqa: E402,F401  (sample is a fixture)
from test_sca import attn_outputs, model  # noqa: E402
from test_scale_sched import KT, QT, _qkv, _setup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T4, T2 = head_table([0.2, 0.9]), head_table([0.35, 0.65])
BRIDGE = dict(bridge_cells=2, bridge_k=2)
GRID = (16, 16)   # n0 4: pool (4, 4), base window 4 (ratio 4); stride 2 gives an 8 x 8 lattice, window 4 of it


def _ctx(impl="flex", **kw):
    base = dict(mode="sca", n0=4, tile=2, window=4, local_impl=impl, gate_sigma=0.9, pi=T4, scale_pi={2: T2})
    base.update(kw)
    return SCAContext(**base)


def _close(a, b, tol=1e-5):
    return torch.allclose(flat(a), flat(b), atol=tol)


def _brute_mean(q, k, v, L, H, W, idx, m, win=4, pi=None, og=None):
    """Element-by-element reference of the mean-pooled stride-2 local view with the bridge (float64)."""
    dt, Hc, Wc = torch.float64, H // 2, W // 2
    coarse = lambda n: [0, (n // W) // 2, (n % W) // 2]  # noqa: E731
    pe = lambda ids: m.pos_embed(torch.tensor(ids, dtype=dt))  # noqa: E731
    qr = rope(q[:, L:], pe([coarse(n) for n in range(H * W)]))
    blocks = [[(2 * i + a) * W + 2 * j + b for a in (0, 1) for b in (0, 1)] for i in range(Hc) for j in range(Wc)]
    kmean = torch.stack([k[:, L:][:, bl].mean(1) for bl in blocks], 1)            # pre-RoPE block means
    vmean = torch.stack([v[:, L:][:, bl].mean(1) for bl in blocks], 1)
    P = L + len(idx)
    kb = [rope(k[:, L:][:, idx], pe([coarse(int(n)) for n in idx]))] if len(idx) else []
    kk = torch.cat([rope(k[:, :L], pe([[0, 0, 0]] * L))] + kb +
                   [rope(kmean, pe([[0, i, j] for i in range(Hc) for j in range(Wc)]))], 1)
    vv = torch.cat([v[:, :L]] + ([v[:, L:][:, idx]] if len(idx) else []) + [vmean], 1)
    mask = torch.zeros(H * W, P + Hc * Wc, dtype=torch.bool)
    mask[:, :P] = True
    for r in range(H):
        for cc in range(W):
            r0 = min(max(((r // 2) * 2 + 1) // 2 - win // 2, 0), Hc - win)
            c0 = min(max(((cc // 2) * 2 + 1) // 2 - win // 2, 0), Wc - win)
            for i in range(r0, r0 + win):
                mask[r * W + cc, P + i * Wc + c0:P + i * Wc + c0 + win] = True
    return sdpa(qr, kk, vv, mask)


def test_mean_lattice_with_bridge_matches_bruteforce():
    H, W, L, f, tile, d, win = 16, 16, 4, (4, 4), (2, 2), (2, 2), (4, 4)
    q, k, v = _qkv(H, W, L, seed=11)
    m, rot, c = _setup(H, W, L)
    rq, rk = c.coarse_rope(f, "cpu")
    rqd, rkd = c.coarse_rope(d, "cpu")
    cells, per = (2, 2), 2
    idx = bridge_index(k[:, L:], (H, W), cells, per)[0]
    lay = strided_flex_layout(L + len(idx), (H, W), tile, d, win, "cpu", QT, KT)
    every = torch.ones(1, 1, 2, 1, dtype=torch.bool)
    out = sca_scale_forward(q, k, v, rot, rq, rk, torch.ones(1, 1, 2, 1), L, (H, W), f, "center",
                            ((d, rqd, rkd, lay, every),), 0.0, cells, per, "mean")
    ref = _brute_mean(q, k, v, L, H, W, idx, m)
    assert (out[:, L:] - ref).abs().max() < 1e-6
    centre = sca_scale_forward(q, k, v, rot, rq, rk, torch.ones(1, 1, 2, 1), L, (H, W), f, "center",
                               ((d, rqd, rkd, lay, every),), 0.0, cells, per)
    assert (centre[:, L:] - ref).abs().max() > 1e-3               # the block-centre lattice is a different view
    # mixing: pi O_loc + (1 - pi) O_glob, text queries take O_glob
    pi = torch.tensor([0.3, 0.8]).view(1, 1, -1, 1)
    mix = sca_scale_forward(q, k, v, rot, rq, rk, pi, L, (H, W), f, "center", ((d, rqd, rkd, lay, every),), 0.0,
                            cells, per, "mean")
    zero = sca_scale_forward(q, k, v, rot, rq, rk, torch.zeros(1, 1, 2, 1), L, (H, W), f, "center",
                             ((d, rqd, rkd, lay, every),), 0.0, cells, per, "mean")
    assert (mix[:, :L] - zero[:, :L]).abs().max() == 0
    assert (mix[:, L:] - (pi * ref + (1 - pi) * zero[:, L:])).abs().max() < 1e-6


def test_mean_lattice_gather_matches_bruteforce():
    """Gather reference without the bridge equals the brute force with an empty bridge."""
    H, W, L, f, tile, d, win = 16, 16, 4, (4, 4), (2, 2), (2, 2), (4, 4)
    q, k, v = _qkv(H, W, L, seed=12)
    m, rot, c = _setup(H, W, L)
    rq, rk = c.coarse_rope(f, "cpu")
    rqd, rkd = c.coarse_rope(d, "cpu")
    heads = torch.arange(2)
    out = sca_scale_gather(q, k, v, rot, rq, rk, torch.ones(1, 1, 2, 1), L, (H, W), f, "center", tile,
                           ((d, win, rqd, rkd, heads),), None, 0.0, "mean")
    ref = _brute_mean(q, k, v, L, H, W, torch.zeros(0, dtype=torch.long), m)
    assert (out[:, L:] - ref).abs().max() < 1e-6


def test_processor_lattice_pool():
    mdl = model()
    mean = attn_outputs(mdl, _ctx(scale_fixed=((0.4, 2),), lattice_pool="mean", **BRIDGE), *GRID)
    centre = attn_outputs(mdl, _ctx(scale_fixed=((0.4, 2),), **BRIDGE), *GRID)
    assert _close(centre, attn_outputs(mdl, _ctx(scale_fixed=((0.4, 2),), lattice_pool="center", **BRIDGE), *GRID))
    assert not _close(mean, centre, 1e-4)
    # flex equals gather (no bridge)
    g = attn_outputs(mdl, _ctx("gather", scale_fixed=((0.4, 2),), lattice_pool="mean"), *GRID)
    assert _close(g, attn_outputs(mdl, _ctx("flex", scale_fixed=((0.4, 2),), lattice_pool="mean"), *GRID))
    # below the threshold (sigma 0.5 < 0.6) and at a = max(s) the published bridge path, whatever the pooling
    pub = attn_outputs(mdl, _ctx(**BRIDGE), *GRID)
    assert _close(attn_outputs(mdl, _ctx(scale_fixed=((0.6, 2),), lattice_pool="mean", **BRIDGE), *GRID), pub)
    assert _close(attn_outputs(mdl, _ctx(scale_fixed=((0.4, 4),), lattice_pool="mean", **BRIDGE), *GRID), pub)
    # the gate still wins above it; gate 0 with scale_fixed puts a = 2 on every step above the threshold
    assert _close(attn_outputs(mdl, _ctx(gate_sigma=0.4, scale_fixed=((0.3, 2),), lattice_pool="mean", **BRIDGE), *GRID),
                  attn_outputs(mdl, _ctx(gate_sigma=0.4, **BRIDGE), *GRID))
    c = _ctx(gate_sigma=0.0, scale_fixed=((0.3, 2),), lattice_pool="mean", **BRIDGE)
    assert _close(attn_outputs(mdl, c, *GRID), mean)
    assert c.pi_table_now() is T2


def test_lattice_pool_validates():
    ok = dict(mode="sca", gate_sigma=0.89)
    SCAContext(**ok, scale_fixed=((0.8, 2),), lattice_pool="mean")
    SCAContext(**ok, scale_sched=True, lattice_pool="mean")
    for kw in (dict(lattice_pool="mean"), dict(scale_fixed=((0.8, 2),), lattice_pool="max")):
        with pytest.raises(ValueError):
            SCAContext(**ok, **kw)


def _npz(tmp_path):
    rng = np.random.default_rng(0)
    ph = rng.uniform(0, 1, size=(3, 4, 5))
    pl, pu = ph.mean(1) + 0.1, ph.mean((0, 1)) + 0.2
    p = tmp_path / "t.npz"
    np.savez(p, scales=np.array([2, 4]), sigma_centers=np.linspace(0.1, 0.9, 5),
             **{f"s{s}_{k}": v * (1 if s == 4 else 0.5) for s in (2, 4)
                for k, v in (("pi_head", ph), ("pi_layer", pl), ("pi_uniform", pu))})
    return p, ph


def test_pitable_headmean_layermean(tmp_path):
    p, ph = _npz(tmp_path)
    hm = PiTable.load(p, "headmean", target_scale=4)
    lm = PiTable.load(p, "layermean", target_scale=4)
    hd = PiTable.load(p, "head", target_scale=4)
    assert hm.variant == "headmean" and hm.scale == 4 and hm.table.shape == (1, 1, 5)
    assert lm.table.shape == (3, 1, 5)
    for s in (0.1, 0.33, 0.9):
        assert np.allclose(hm(s), hd(s).mean()), s
        assert np.allclose(lm(s)[:, 0], hd(s).mean(1)), s
    assert np.allclose(PiTable.load(p, "headmean", target_scale=2)(0.5), 0.5 * hd(0.5).mean())
    # pi_now broadcasts them to every head of a layer
    c = SCAContext(mode="sca", pi=lm)
    c.n_layers, c.sigma, c.grid = 3, 0.5, (8, 8)
    assert np.allclose(c.pi_now(1, 4, "cpu").numpy(), np.full(4, hd(0.5)[1].mean()), atol=1e-6)


def _prog():
    spec = importlib.util.spec_from_file_location("sca_prog_ra_test", ROOT / "scripts/sample_progressive.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not TABLE.exists(), reason="calibrated table not on this machine")
def test_progressive_pi_variant():
    mod = _prog()
    base = ["--prompts", "x.jsonl", "--out", "o", "--pi-table", str(TABLE), "--gate-sigma", "0.89",
            "--bridge-cells", "8", "--bridge-k", "16"]
    plan = None
    infos = {}
    for v in ("head", "headmean", "layermean"):
        args = mod.parser().parse_args(base + ([] if v == "head" else ["--pi-variant", v]))
        plan = mod.make_plan(args)
        ctxs, info = mod.build_contexts(args, plan)
        assert [c.pi.variant for c in ctxs[1:]] == [v, v] and [c.pi.scale for c in ctxs[1:]] == [2, 4]
        infos[v] = info
    assert [s["pi_variant"] for s in infos["head"]["stages"]] == ["head", "head"]
    z = np.load(TABLE)
    hm = PiTable.load(TABLE, "headmean", target_scale=4)
    assert np.allclose(hm(0.5), PiTable.load(TABLE, "head", target_scale=4)(0.5).mean())
    assert np.allclose(hm.table[0, 0], z["s4_pi_head"].mean((0, 1)))


@pytest.mark.skipif(not TABLE.exists(), reason="calibrated table not on this machine")
def test_sample_info_lattice_pool(sample):  # noqa: F811
    base = ["--method", "sca", "--pi-table", str(TABLE), "--gate-sigma", "0.89", "--res", "4096x4096",
            "--prompts", "x.jsonl", "--out", "o", "--bridge-cells", "8", "--bridge-k", "16", "--scale-fixed", "0.77:2"]
    c0, info0 = sample.build_context(sample.parser().parse_args(base), (256, 256))
    c1, info1 = sample.build_context(sample.parser().parse_args(base + ["--lattice-pool", "mean"]), (256, 256))
    assert "lattice_pool" not in info0 and c0.lattice_pool == "center"
    assert info1["lattice_pool"] == "mean" and c1.lattice_pool == "mean" and set(info1) - set(info0) == {"lattice_pool"}


def _handoff():
    spec = importlib.util.spec_from_file_location("handoff_ra_test", ROOT / "scripts/handoff_from_mass.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_handoff_sigmas_and_schedule():
    h = _handoff()
    sig = h.flux_sigmas(28, 1.15)
    ref = [1.0, 0.9884, 0.9762, 0.9634, 0.9499, 0.9356, 0.9205, 0.9045, 0.8876, 0.8696, 0.8504, 0.83, 0.8081, 0.7847,
           0.7595, 0.7324, 0.7031, 0.6714, 0.637, 0.5994, 0.5582, 0.5128, 0.4627, 0.4071, 0.3448, 0.2748, 0.1955, 0.1047]
    assert np.allclose(sig, ref, atol=6e-5)     # the sca_v2 sidecars' sigmas at 4096^2
    assert h.schedule(0.94, 0.79, sig) == [1] * 5 + [2] * 8 + [4] * 15
    assert h.schedule(math.inf, 0.79, sig) == [2] * 13 + [4] * 15
    assert h.schedule(0.59, -math.inf, sig) == [1] * 20 + [2] * 8


@pytest.mark.skipif(not TABLE.exists(), reason="calibrated table not on this machine")
def test_handoff_measured_values():
    h = _handoff()
    # head-averaged image share at the hand-off: the audit's 0.36 at ratio 4 (sigma 0.896)
    assert abs(h.head_mean_share(TABLE, 4, 0.8955)[0] - 0.359) < 0.005
    assert h.crossing(TABLE, 2, 0.4) == math.inf and h.crossing(TABLE, 4, 0.5) == -math.inf
    assert abs(h.crossing(TABLE, 2, 0.5) - 0.9404) < 2e-3 and abs(h.crossing(TABLE, 4, 0.4) - 0.7892) < 2e-3
    assert abs(h.crossing(TABLE, 2, 0.6) - 0.5896) < 2e-3
