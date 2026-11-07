"""Per-query mix (sca/qmix.py, root-cause audit Fix 1 and Fix 1+3): CPU, float64, against brute force."""

import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sca.attn import SCAContext, flex_layout, rope  # noqa: E402
from sca import qmix as qm  # noqa: E402
from sca import textgate as tg  # noqa: E402
from test_dedup import _qkv, _setup, flat  # noqa: E402
from test_sca import attn_outputs, model  # noqa: E402
from test_textgate import F, H, L, TILE, W, WIN, _bridge, _lattice, _window_mask  # noqa: E402


def _pieces(seed):
    q, k, v = _qkv(H, W, L, seed=seed)
    _, _, rot, rq, rk = _setup(H, W, L, F)
    cells, per, idx = _bridge(k)
    R = len(idx)
    layR = flex_layout(R, (H, W), TILE, WIN, "cpu", (2, 2), (2, 2))
    qr, kr = rope(q, rot), rope(k, rot)
    d = q.shape[-1] ** -0.5
    cen = _lattice()
    qi, kt, vt = qr[0, L:], kr[0, :L], v[0, :L]
    qg = rope(q[:, L:], rq)[0]
    kb, vb = rope(k[:, L:][:, cen], rk)[0], v[0, L + cen]
    keys_n = torch.cat([kr[0, L + idx], kr[0, L:]], 0)
    vals_n = torch.cat([v[0, L + idx], v[0, L:]], 0)
    sn = (torch.einsum("nhd,mhd->hnm", qi, keys_n) * d).masked_fill(~_window_mask(R)[None], float("-inf"))
    st = torch.einsum("nhd,lhd->hnl", qi, kt) * d
    sgt = torch.einsum("nhd,lhd->hnl", qg, kt) * d
    sgb = torch.einsum("nhd,mhd->hnm", qg, kb) * d
    inside = _window_mask(0)[:, cen]
    glob_keys = torch.cat([kt, kb], 0)
    glob_vals = torch.cat([vt, vb], 0)
    sg = torch.cat([sgt, sgb], -1)
    o_glob_img = torch.einsum("hnm,mhd->nhd", sg.softmax(-1), glob_vals)
    return dict(q=q, k=k, v=v, rot=rot, rq=rq, rk=rk, cells=cells, per=per, layR=layR, sn=sn, st=st, sgt=sgt, sgb=sgb,
                inside=inside, vals_n=vals_n, vt=vt, vb=vb, o_glob_img=o_glob_img, sg=sg, qr=qr, kr=kr)


def test_f1_matches_bruteforce():
    p = _pieces(37)
    pi = torch.tensor([0.6, 0.3], dtype=torch.float64).view(1, 1, 2, 1)
    lat_in = tg.lattice_layout((H, W), TILE, WIN, F, "cpu", inside=True)
    out = qm.sca_qmix_forward(p["q"], p["k"], p["v"], p["rot"], p["rq"], p["rk"], pi, L, (H, W), F, "center", p["layR"],
                              p["cells"], p["per"], lat_in, None, 1.0, False)
    lse_n = p["sn"].logsumexp(-1)                                                  # [h, N]
    lse_t = p["st"].logsumexp(-1)
    lse_g = p["sg"].logsumexp(-1)
    lse_tg = p["sgt"].logsumexp(-1)
    lse_in = p["sgb"].masked_fill(~p["inside"][None], float("-inf")).logsumexp(-1)
    log_far = math.log(F[0] * F[1]) + torch.log(torch.exp(lse_g) - torch.exp(lse_tg) - torch.exp(lse_in))
    dd = lse_n - log_far
    lp = torch.log(pi.view(-1, 1)) - torch.log1p(-pi.view(-1, 1))
    w = torch.sigmoid(lp + dd - dd.mean(-1, keepdim=True)).transpose(0, 1)[..., None]   # [N, h, 1]
    o_n = torch.einsum("hnm,mhd->nhd", p["sn"].softmax(-1), p["vals_n"])
    o_t = torch.einsum("hnl,lhd->nhd", p["st"].softmax(-1), p["vt"])
    a = torch.sigmoid(lse_t - lse_n).transpose(0, 1)[..., None]                   # text share of the local softmax
    o_loc = a * o_t + (1 - a) * o_n
    ref = w * o_loc + (1 - w) * p["o_glob_img"]
    assert (out[0, L:] - ref).abs().max() < 1e-9


@pytest.mark.parametrize("gamma,far_all", [(1.0, False), (0.25, False), (1.0, True)])
def test_f13_matches_bruteforce(gamma, far_all):
    p = _pieces(41)
    pi = torch.tensor([0.7, 0.2], dtype=torch.float64).view(1, 1, 2, 1)
    lat_out = tg.lattice_layout((H, W), TILE, WIN, F, "cpu")
    out = qm.sca_qmix_forward(p["q"], p["k"], p["v"], p["rot"], p["rq"], p["rk"], pi, L, (H, W), F, "center", p["layR"],
                              p["cells"], p["per"], None, lat_out, 1.0, True, math.log(gamma), far_all)
    lse_n = p["sn"].logsumexp(-1)
    lse_t = p["st"].logsumexp(-1)
    sfar = p["sgb"] if far_all else p["sgb"].masked_fill(p["inside"][None], float("-inf"))
    log_far = math.log(F[0] * F[1]) + sfar.logsumexp(-1)
    lp = torch.log(pi.view(-1, 1)) - torch.log1p(-pi.view(-1, 1))
    beta = lp - (lse_n - log_far).mean(-1, keepdim=True)
    o_n = torch.einsum("hnm,mhd->nhd", p["sn"].softmax(-1), p["vals_n"])
    o_t = torch.einsum("hnl,lhd->nhd", p["st"].softmax(-1), p["vt"])
    o_far = torch.einsum("hnm,mhd->nhd", sfar.softmax(-1), p["vb"])
    logits = torch.stack([lse_t, lse_n + beta + math.log(gamma), log_far + math.log(gamma)], 0)   # [3, h, N]
    wts = logits.softmax(0).permute(0, 2, 1)[..., None]                            # [3, N, h, 1]
    ref = wts[0] * o_t + wts[1] * o_n + wts[2] * o_far
    assert (out[0, L:] - ref).abs().max() < 1e-9


def test_f13_dose_interpolates_to_sca_v2():
    """dose 0 gives sca_v2's own image output (pi O_loc + (1 - pi) O_glob), dose 0.5 the midpoint."""
    from sca.attn import sca_dedup_forward
    p = _pieces(43)
    pi = torch.tensor([0.6, 0.3], dtype=torch.float64).view(1, 1, 2, 1)
    lat_out = tg.lattice_layout((H, W), TILE, WIN, F, "cpu")
    a = (p["q"], p["k"], p["v"], p["rot"], p["rq"], p["rk"], pi, L, (H, W), F, "center", p["layR"], p["cells"],
         p["per"], None, lat_out, 1.0, True, 0.0, False)
    o1 = qm.sca_qmix_forward(*a, 1.0)
    o0 = qm.sca_qmix_forward(*a, 0.0)
    oh = qm.sca_qmix_forward(*a, 0.5)
    layout = flex_layout(L + p["cells"][0] * p["cells"][1] * p["per"], (H, W), TILE, WIN, "cpu", (2, 2), (2, 2))
    ref = sca_dedup_forward(p["q"], p["k"], p["v"], p["rot"], p["rq"], p["rk"], pi, L, (H, W), F, "center", layout,
                            0.0, p["cells"], p["per"], 0.0)
    assert (o0[0, L:] - ref[0, L:]).abs().max() < 1e-6          # sca_dedup_forward mixes in float32
    assert (oh[0, L:] - 0.5 * (o0[0, L:] + o1[0, L:])).abs().max() < 1e-9


def test_processor_qmix_gates_and_runs():
    m = model()
    c = dict(mode="sca", n0=4, tile=2, window=4, local_impl="flex", bridge_cells=2, bridge_k=2)
    base = attn_outputs(m, SCAContext(**c), H, W)
    off = attn_outputs(m, SCAContext(**c, qmix="f1", qmix_lo=0.6), H, W)                # sigma .5: off
    f1 = attn_outputs(m, SCAContext(**c, qmix="f1"), H, W)
    f13 = attn_outputs(m, SCAContext(**c, qmix="f13"), H, W)
    assert torch.allclose(flat(off), flat(base), atol=1e-5)
    for o in (f1, f13):
        assert torch.isfinite(flat(o)).all() and not torch.allclose(flat(o), flat(base), atol=1e-4)
    with pytest.raises(ValueError):
        SCAContext(**c, qmix="f2")
    with pytest.raises(ValueError):
        SCAContext(**c, qmix="f1", text_mode="ffl")


TABLE = Path("/scratch/09032/saini_2/sca/calib/k1_1024.npz")


@pytest.mark.skipif(not TABLE.exists(), reason="calibration table not on this machine")
def test_pi_blend_moves_pi_to_the_near_share():
    """Audit Fix 2: pi' = (1 - lambda) pi + lambda m_h while sigma > pi_blend_lo, m_h = exp(ctw_beta) without its floor;
    unchanged at or below pi_blend_lo and with lambda 0."""
    import numpy as np
    from sca.attn import PiTable
    pi = PiTable.load(TABLE, "head", target_scale=4)
    grid, sched = (256, 256), ((0.88, 128), (0.86, 112), (0.84, 96), (0.82, 80))

    def pi_at(sigma, **kw):
        ctx = SCAContext(mode="sca", known_grid=grid, n0=64, tile=8, pi=pi, n_layers=57, win_sched=sched,
                         win_pi_match=True, win_pi_interp=True, **kw)
        ctx.grid, ctx.sigma = grid, sigma
        return ctx.pi_now(3, 24, "cpu").numpy().astype(np.float64), ctx

    for sigma in (0.888, 0.85, 0.81):
        base, ctx = pi_at(sigma)
        full, _ = pi_at(sigma, pi_blend=1.0)
        half, _ = pi_at(sigma, pi_blend=0.5)
        m = np.exp(tg.ctw_beta(TABLE, sigma, ctx.window_now(), grid, 64, floor=-30.0))[3]
        assert np.allclose(full, m, atol=1e-6) and np.allclose(half, 0.5 * (base + m), atol=1e-6)
    base, _ = pi_at(0.75)
    assert np.allclose(pi_at(0.75, pi_blend=1.0)[0], base) and np.allclose(pi_at(0.85, pi_blend=0.0)[0], pi_at(0.85)[0])
    # the audit's premise on the real table: at the hand-off the near share sits well below the calibrated pi
    b, _ = pi_at(0.85)
    f, _ = pi_at(0.85, pi_blend=1.0)
    assert float(np.median(b - f)) > 0.1


def test_pi_blend_validates():
    from sca.attn import PiTable
    with pytest.raises(ValueError):
        SCAContext(mode="sca", pi_blend=1.5)
    ctx = SCAContext(mode="sca", known_grid=(8, 8), n0=4, pi=PiTable("const", value=0.7), pi_blend=1.0)
    ctx.grid, ctx.sigma = (8, 8), 0.9
    with pytest.raises(ValueError):
        ctx.pi_now(0, 2, "cpu")
