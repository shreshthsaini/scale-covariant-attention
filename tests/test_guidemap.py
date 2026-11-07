"""Spatially varying guidance (sca/guidemap.py, round 7 lever 1): blend, step gating, single pre-hook, and that the
attention operator itself is untouched (the map is recorded in the opening, sa_value stays 1)."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sca.attn import SCAContext  # noqa: E402
from sca import guidemap  # noqa: E402
from test_dedup import flat  # noqa: E402
from test_sca import attn_outputs, model  # noqa: E402
from test_textgate import H, W  # noqa: E402

G = torch.tensor([0.0, 1.0, 0.5, 0.25, 1.0, 0.0])


class Toy(torch.nn.Module):
    """Velocity = hidden_states * guidance + 1; counts forward calls."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states=None, guidance=None, return_dict=False):
        self.calls += 1
        out = hidden_states * guidance.view(-1, 1, 1) + 1.0
        return SimpleNamespace(sample=out) if return_dict else (out,)


def test_blend_is_affine_in_the_map():
    torch.manual_seed(0)
    hi, lo = torch.randn(1, 6, 4), torch.randn(1, 6, 4)
    out = guidemap.blend(hi, lo, G)
    assert torch.allclose(out, lo + G.view(1, -1, 1) * (hi - lo), atol=1e-6)
    assert torch.allclose(out[:, 1], hi[:, 1], atol=1e-6) and torch.equal(out[:, 0], lo[:, 0])


@pytest.mark.parametrize("return_dict", [False, True])
def test_hook_blends_on_window_steps_only(return_dict):
    m, x, guid, pre = Toy(), torch.randn(1, 6, 4), torch.full([1], 4.5), []
    m.register_forward_pre_hook(lambda mod, a, kw: pre.append(1), with_kwargs=True)

    def run(sigma, g=G):
        ctx = SimpleNamespace(sigma=sigma, gm_low=1.5, gm_lo=0.58, gate_sigma=0.89, log=[],
                              sa_map=lambda N, device: g)
        undo = guidemap.install(m, ctx)
        m.calls, pre[:] = 0, []
        try:
            out = m(hidden_states=x, guidance=guid, return_dict=return_dict)
        finally:
            undo()
        return out.sample if return_dict else out[0]

    hi, lo = x * 4.5 + 1, x * 1.5 + 1
    assert torch.allclose(run(0.75), lo + G.view(1, -1, 1) * (hi - lo), atol=1e-6)
    assert m.calls == 2 and len(pre) == 1                   # the second pass does not rerun the sigma pre-hook
    assert torch.equal(guid, torch.full([1], 4.5))          # the caller's guidance tensor is not modified
    for sigma in (0.95, 0.8901, 0.58, 0.3):                 # opening, and at or below gm_lo: one pass, unchanged
        assert torch.allclose(run(sigma), hi) and m.calls == 1
    assert torch.allclose(run(0.75, None), hi) and m.calls == 1   # no map (prompt without subject words)


def test_context_records_map_without_changing_attention():
    with pytest.raises(ValueError):
        SCAContext(mode="sca", gm_low=1.5)                  # needs an opening (gate_sigma > 0) to build the map
    with pytest.raises(ValueError):
        SCAContext(mode="sca", gate_sigma=0.6, gm_low=-1.0)
    m = model()
    c = dict(mode="sca", n0=4, tile=2, window=4, local_impl="flex", bridge_cells=2, bridge_k=2, gate_sigma=0.6)
    ctx, base = SCAContext(**c, gm_low=1.5, subject_tokens=(1, 2)), SCAContext(**c)
    assert ctx.sa_on() and not base.sa_on() and not SCAContext(**c, gm_low=1.5).sa_on()

    def at(cx, sigma):
        orig = SCAContext.set_inputs
        cx.set_inputs = lambda ids, txt_ids=None, sigma=None, _s=sigma: orig(cx, ids, txt_ids, _s)
        try:
            return attn_outputs(m, cx, H, W)
        finally:
            del cx.set_inputs

    hi_a, hi_b = at(ctx, 0.8), at(base, 0.8)                 # opening: same output, map recorded
    assert torch.allclose(flat(hi_a), flat(hi_b), atol=1e-5) and ctx._sa_acc is not None
    lo_a, lo_b = at(ctx, 0.5), at(base, 0.5)                 # window step: attention untouched
    assert torch.allclose(flat(lo_a), flat(lo_b), atol=1e-6)
    assert ctx.sa_scale(H * W, "cpu") is None
    g = ctx.sa_map(H * W, "cpu")
    assert g is not None and g.shape == (H * W,) and float(g.min()) >= 0 and float(g.max()) <= 1
    assert float(g.max()) == 1.0
