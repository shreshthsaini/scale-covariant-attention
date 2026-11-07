"""Count-corrected global branch (--glob-count, round 7 afternoon): for image queries the global branch's block-centre
keys carry the weight (fh fw)^gamma against the text keys. Default off leaves sca_v2 unchanged."""
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sca.attn import PiTable, SCAContext, flex_layout, sca_dedup_forward  # noqa: E402
from test_dedup import flat  # noqa: E402
from test_qmix import _pieces  # noqa: E402
from test_sca import attn_outputs, model  # noqa: E402
from test_textgate import F, H, L, TILE, W, WIN  # noqa: E402


def _args(p, pi):
    layout = flex_layout(L + p["cells"][0] * p["cells"][1] * p["per"], (H, W), TILE, WIN, "cpu", (2, 2), (2, 2))
    return (p["q"], p["k"], p["v"], p["rot"], p["rq"], p["rk"], pi, L, (H, W), F, "center", layout, 0.0, p["cells"],
            p["per"], 0.0, None, 0, False, 0)


@pytest.mark.parametrize("gamma", [1.0, 0.5])
def test_glob_count_matches_bruteforce_global_branch(gamma):
    """pi = 0 (global branch alone): image rows equal a brute-force softmax over [text, block centres] with the centres'
    logits raised by gamma log(fh fw); text rows are unchanged."""
    p = _pieces(51)
    pi = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
    base = sca_dedup_forward(*_args(p, pi))
    out = sca_dedup_forward(*_args(p, pi), gamma)
    sg = torch.cat([p["sgt"], p["sgb"] + gamma * math.log(F[0] * F[1])], -1)
    ref = torch.einsum("hnm,mhd->nhd", sg.softmax(-1), torch.cat([p["vt"], p["vb"]], 0))
    assert (out[0, L:] - ref).abs().max() < 1e-6
    assert (base[0, L:] - p["o_glob_img"]).abs().max() < 1e-6
    assert (out[0, :L] - base[0, :L]).abs().max() < 1e-12
    assert (out[0, L:] - base[0, L:]).abs().max() > 1e-4


def test_glob_count_zero_is_sca_v2_and_mix_is_linear():
    """gamma 0 reproduces the plain forward; with pi > 0 the window part is untouched (difference = (1 - pi) dO_glob)."""
    p = _pieces(52)
    pi = torch.tensor([0.6, 0.3], dtype=torch.float64).view(1, 1, 2, 1)
    base = sca_dedup_forward(*_args(p, pi))
    assert (sca_dedup_forward(*_args(p, pi), 0.0) - base).abs().max() == 0
    zero = torch.zeros_like(pi)
    d_glob = sca_dedup_forward(*_args(p, zero), 1.0)[0, L:] - sca_dedup_forward(*_args(p, zero))[0, L:]
    d_mix = sca_dedup_forward(*_args(p, pi), 1.0)[0, L:] - base[0, L:]
    assert (d_mix - (1 - pi[0, 0, :, 0]).view(1, 2, 1) * d_glob).abs().max() < 1e-9


def test_processor_glob_count_gates_by_sigma_and_validates():
    m = model()
    c = dict(mode="sca", n0=4, tile=2, window=4, local_impl="flex", bridge_cells=2, bridge_k=2,
             pi=PiTable("const", value=0.5))   # the default table is pi = 1, which hides the global branch
    base = attn_outputs(m, SCAContext(**c), H, W)
    off = attn_outputs(m, SCAContext(**c, glob_count=1.0, glob_count_lo=0.6), H, W)          # sigma .5: off
    on = attn_outputs(m, SCAContext(**c, glob_count=1.0), H, W)
    assert torch.allclose(flat(off), flat(base), atol=1e-6)
    assert torch.isfinite(flat(on)).all() and not torch.allclose(flat(on), flat(base), atol=1e-4)
    for bad in (dict(glob_count=-1.0), dict(glob_count=1.0, bridge_cells=0), dict(glob_count=1.0, text_value=0.5),
                dict(glob_count=1.0, qmix="f1")):
        with pytest.raises(ValueError):
            SCAContext(**{**c, **bad})


def test_glob_count_cli_and_sidecar():
    import argparse
    from sca import dedup
    p = argparse.ArgumentParser()
    dedup.add_args(p)
    a = p.parse_args([])
    assert dedup.context_kwargs(a)["glob_count"] == 0.0 and "glob_count" not in dedup.info(a)
    a = p.parse_args(["--glob-count", "1", "--glob-count-lo", "0.55"])
    kw, info = dedup.context_kwargs(a), dedup.info(a)
    assert kw["glob_count"] == 1.0 and kw["glob_count_lo"] == 0.55
    assert info["glob_count"] == 1.0 and info["glob_count_lo"] == 0.55


def test_glob_count_combines_with_window_text_value():
    """glob_count + text_value on the window branch only: the window part equals the text_value (branch loc) run, the
    global part equals the glob_count run (the two corrections act on disjoint branches and add up)."""
    p = _pieces(53)
    pi = torch.tensor([0.6, 0.3], dtype=torch.float64).view(1, 1, 2, 1)
    a = list(_args(p, pi))
    base = sca_dedup_forward(*a)
    sv = torch.full((L,), 0.5, dtype=torch.float64)
    a_tv = a[:16] + [sv, 1] + a[18:]
    tv = sca_dedup_forward(*a_tv)                      # window text values x 0.5
    gc = sca_dedup_forward(*a, 1.0)                    # count-corrected global branch
    both = sca_dedup_forward(*a_tv, 1.0)
    assert ((both - base) - ((tv - base) + (gc - base))).abs().max() < 1e-9
    m = model()
    c = dict(mode="sca", n0=4, tile=2, window=4, local_impl="flex", bridge_cells=2, bridge_k=2,
             pi=PiTable("const", value=0.5))
    SCAContext(**c, glob_count=1.0, text_value=0.5, text_branch="loc")
    with pytest.raises(ValueError):
        SCAContext(**c, glob_count=1.0, text_value=0.5, text_branch="both")
    out = attn_outputs(m, SCAContext(**c, glob_count=1.0, text_value=0.5, text_branch="loc"), H, W)
    assert torch.isfinite(flat(out)).all()



@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_text_count_matches_bruteforce(gamma):
    """qmix tc: window = (Z_T A O_T + Z_near O_near) / (Z_T + Z_near) with A = (Z_T + Z_near) / (Z_T + Z_near + Z_far),
    Z_far = s^2 x the global branch's block centres outside the window; global = count-corrected (gamma); mix by pi."""
    from sca import qmix as qm
    from sca import textgate as tg
    p = _pieces(54)
    pi = torch.tensor([0.6, 0.3], dtype=torch.float64).view(1, 1, 2, 1)
    lat_in = tg.lattice_layout((H, W), TILE, WIN, F, "cpu", inside=True)
    out = qm.sca_qmix_forward(p["q"], p["k"], p["v"], p["rot"], p["rq"], p["rk"], pi, L, (H, W), F, "center", p["layR"],
                              p["cells"], p["per"], lat_in, None, 1.0, False, 0.0, False, 1.0, False, False, True, gamma)
    s2 = F[0] * F[1]
    z_t, z_n = p["st"].exp().sum(-1), p["sn"].exp().sum(-1)                        # [h, N]
    z_far = s2 * p["sgb"].masked_fill(p["inside"][None], float("-inf")).exp().sum(-1)
    a = (z_t + z_n) / (z_t + z_n + z_far)
    o_n = torch.einsum("hnm,mhd->nhd", p["sn"].softmax(-1), p["vals_n"])
    o_t = torch.einsum("hnl,lhd->nhd", p["st"].softmax(-1), p["vt"])
    tt = (z_t / (z_t + z_n)).transpose(0, 1)[..., None]
    o_loc = tt * a.transpose(0, 1)[..., None] * o_t + (1 - tt) * o_n
    sg = torch.cat([p["sgt"], p["sgb"] + gamma * math.log(s2)], -1)
    o_g = torch.einsum("hnm,mhd->nhd", sg.softmax(-1), torch.cat([p["vt"], p["vb"]], 0))
    pp = pi[0, 0, :, 0].view(1, 2, 1)
    ref = pp * o_loc + (1 - pp) * o_g
    assert (out[0, L:] - ref).abs().max() < 1e-8
    base = qm.sca_qmix_forward(p["q"], p["k"], p["v"], p["rot"], p["rq"], p["rk"], pi, L, (H, W), F, "center",
                               p["layR"], p["cells"], p["per"], lat_in, None, 1.0, False)
    assert (out[0, :L] - base[0, :L]).abs().max() < 1e-12      # text queries untouched


def test_processor_text_count_runs_and_validates():
    m = model()
    c = dict(mode="sca", n0=4, tile=2, window=4, local_impl="flex", bridge_cells=2, bridge_k=2,
             pi=PiTable("const", value=0.5))
    base = attn_outputs(m, SCAContext(**c), H, W)
    on = attn_outputs(m, SCAContext(**c, qmix="tc", glob_count=1.0), H, W)
    off = attn_outputs(m, SCAContext(**c, qmix="tc", qmix_lo=0.6, glob_count=1.0, glob_count_lo=0.6), H, W)
    assert torch.isfinite(flat(on)).all() and not torch.allclose(flat(on), flat(base), atol=1e-4)
    assert torch.allclose(flat(off), flat(base), atol=1e-6)
    with pytest.raises(ValueError):
        SCAContext(**c, qmix="f1", glob_count=1.0)


def test_glob_count_per_query_weights_match_bruteforce():
    """glob_w (glob_count_map): per image query the centre weight w_i; w_i = 1 rows equal the plain branch, rows with
    w_i = s^2 equal the uniform count correction."""
    p = _pieces(55)
    pi = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
    n = H * W
    g = torch.zeros(n, dtype=torch.float64)
    g[: n // 2] = 1.0                                             # first half: subject (no correction)
    s2 = F[0] * F[1]
    w = (1.0 + (s2 - 1.0) * (1.0 - g)).view(1, -1, 1, 1)
    out = sca_dedup_forward(*_args(p, pi), 1.0, w)
    plain = sca_dedup_forward(*_args(p, pi))
    full = sca_dedup_forward(*_args(p, pi), 1.0)
    assert (out[0, L: L + n // 2] - plain[0, L: L + n // 2]).abs().max() < 1e-9
    assert (out[0, L + n // 2:] - full[0, L + n // 2:]).abs().max() < 1e-9
    sg = torch.cat([p["sgt"], p["sgb"] + torch.log(w.view(1, -1, 1).expand(1, n, 1))], -1)  # [h, N, L + C]
    ref = torch.einsum("hnm,mhd->nhd", sg.softmax(-1), torch.cat([p["vt"], p["vb"]], 0))
    assert (out[0, L:] - ref).abs().max() < 1e-6      # the correction runs in float32


def test_glob_count_args_follow_the_subject_map():
    c = SCAContext(mode="sca", n0=4, tile=2, window=4, local_impl="flex", bridge_cells=2, bridge_k=2,
                   pi=PiTable("const", value=0.5), glob_count=1.0, glob_count_map=True, subject_tokens=(1, 2))
    c.sigma = 0.7
    assert c.sa_on()
    n = 16
    c._sa_map = torch.tensor([1.0] * 4 + [0.0] * 12)
    c._sa_acc = torch.zeros(n)
    gam, w = c.glob_count_args((2, 2), n, "cpu")
    assert gam == 1.0 and w.shape == (1, n, 1, 1)
    assert torch.allclose(w.view(-1)[:4], torch.ones(4)) and torch.allclose(w.view(-1)[4:], torch.full((12,), 4.0))
    c2 = SCAContext(mode="sca", n0=4, tile=2, window=4, local_impl="flex", bridge_cells=2, bridge_k=2,
                    pi=PiTable("const", value=0.5), glob_count=1.0, glob_count_map=True)   # no subject named
    c2.sigma = 0.7
    assert c2.glob_count_args((2, 2), n, "cpu") == (0.0, None)
    with pytest.raises(ValueError):
        SCAContext(mode="sca", bridge_cells=2, glob_count_map=True)
