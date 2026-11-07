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
