"""Morph ladder (sca/morph.py): SCA-side AReA components on a tiny FLUX transformer, CPU."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sca.attn import SCAContext  # noqa: E402
from sca import morph  # noqa: E402
from test_dedup import flat  # noqa: E402
from test_sca import attn_outputs, model  # noqa: E402

H, W = 8, 8
pytestmark = pytest.mark.skipif(not (morph.AREA_DIR / "area").exists(), reason="AReA checkout missing")


def _ctx(**kw):
    base = dict(mode="sca", n0=4, tile=2, window=4, local_impl="flex", bridge_cells=2, bridge_k=2)
    base.update(kw)
    return SCAContext(**base)


def test_text_window_mean_one_window_is_plain_attention():
    g = torch.Generator().manual_seed(5)
    B, L, h, D = 1, 3, 2, 8
    qt, kt, vt = (torch.randn(B, L, h, D, generator=g, dtype=torch.float64) for _ in range(3))
    ki, vi = (torch.randn(B, H * W, h, D, generator=g, dtype=torch.float64) for _ in range(2))
    bridge = torch.tensor([[1, 9, 30]])
    out = morph.text_window_mean(qt, kt, vt, ki, vi, (H, W), (H, W), bridge)
    k = torch.cat([kt, ki, ki[:, bridge[0]]], 1)
    v = torch.cat([vt, vi, vi[:, bridge[0]]], 1)
    s = torch.einsum("blhd,bmhd->bhlm", qt, k) / D ** 0.5
    ref = torch.einsum("bhlm,bmhd->blhd", s.softmax(-1), v)
    assert torch.allclose(out, ref, atol=1e-10)
    # four 4x4 windows: mean of four window attentions
    out4 = morph.text_window_mean(qt, kt, vt, ki, vi, (H, W), (4, 4), None)
    acc = 0
    for r0 in (0, 4):
        for c0 in (0, 4):
            idx = torch.tensor([r * W + c for r in range(r0, r0 + 4) for c in range(c0, c0 + 4)])
            kk, vv = torch.cat([kt, ki[:, idx]], 1), torch.cat([vt, vi[:, idx]], 1)
            ss = torch.einsum("blhd,bmhd->bhlm", qt, kk) / D ** 0.5
            acc = acc + torch.einsum("bhlm,bmhd->blhd", ss.softmax(-1), vv)
    assert torch.allclose(out4, acc / 4, atol=1e-10)


def test_processor_morph_text_changes_only_text_rows():
    m = model()
    base = attn_outputs(m, _ctx(), H, W)
    mt = attn_outputs(m, _ctx(morph_text="window_mean"), H, W)
    # dual block: (img, txt); single block: [txt | img]
    assert torch.isfinite(flat(mt)).all()
    assert not torch.allclose(mt[0][1], base[0][1], atol=1e-4)                # text output changed


def test_processor_morph_area_tiles_runs():
    attention, tiling = morph._area()
    m = model()
    c = _ctx(morph_local="area_tiles")
    windows = tiling.adaptive_tiling(torch.rand(H, W).numpy(), H, W, 4, 4, 0, rng=tiling.step_rng(0, 0, 0x5EA1))
    c._area_layout = attention.build_fine_layout(windows, H, W, 1, 4, 0, 0, torch.float32, "cpu")
    out = attn_outputs(m, c, H, W)
    assert torch.isfinite(flat(out)).all()
    base = attn_outputs(m, _ctx(), H, W)
    assert not torch.allclose(flat(out), flat(base), atol=1e-4)


def test_morph_validation_and_info():
    with pytest.raises(ValueError):
        _ctx(morph_local="tiles")
    with pytest.raises(ValueError):
        _ctx(morph_text="window_mean", text_mode="ffl")
    from scripts.sample import parser
    from sca import dedup
    base = ["--method", "sca", "--res", "64x64", "--prompts", "x.jsonl", "--out", "o"]
    assert "morph_text" not in dedup.info(parser().parse_args(base))
    a = parser().parse_args(base + ["--morph-text", "window_mean", "--morph-local", "area_tiles"])
    info, kw = dedup.info(a), dedup.context_kwargs(a)
    assert info["morph_text"] == "window_mean" and kw["morph_local"] == "area_tiles"


def test_pi_at_window_interpolates_between_scales():
    import numpy as np

    class T:
        def __init__(self, v):
            self.v = v

        def __call__(self, sigma):
            return np.full((2, 3), self.v)
    assert np.allclose(morph.pi_at_window(T(0.2), T(0.6), 0.5, 128, 256), 0.2)    # ratio 2 -> scale-2 table
    assert np.allclose(morph.pi_at_window(T(0.2), T(0.6), 0.5, 64, 256), 0.6)     # ratio 4 -> scale-4 table
    assert np.allclose(morph.pi_at_window(T(0.2), T(0.6), 0.5, 180, 256), 0.2)    # clamped
    mid = morph.pi_at_window(T(0.2), T(0.6), 0.5, 256 / 2 ** 1.5, 256)
    assert np.allclose(mid, 0.4)


def test_text_all_coarse_is_full_text_attention_at_scale_one_and_pools_ids_at_four():
    """Audit Fix 4: with f = 1 the coarse ids are the true ids, so the rule is the text rows of full attention; with
    f = 2 every image key sits at floor(p / 2) and all of them (not only block centres) are attended."""
    from sca.attn import rope
    from test_dedup import _qkv, _setup
    L = 3
    q, k, v = _qkv(H, W, L, seed=4)
    _, _, rot, rq1, _ = _setup(H, W, L, (1, 1))
    cos, sin = rot
    qt, kt = rope(q[:, :L], (cos[:L], sin[:L])), rope(k[:, :L], (cos[:L], sin[:L]))
    full = torch.cat([kt, rope(k[:, L:], (cos[L:], sin[L:]))], 1)
    s = torch.einsum("blhd,bmhd->bhlm", qt, full) / q.shape[-1] ** 0.5
    ref = torch.einsum("bhlm,bmhd->blhd", s.softmax(-1), v)
    assert torch.allclose(morph.text_all_coarse(qt, kt, k[:, L:], v, rq1), ref, atol=1e-10)
    _, _, _, rq2, _ = _setup(H, W, L, (2, 2))
    kc = rope(k[:, L:], rq2)
    s2 = torch.einsum("blhd,bmhd->bhlm", qt, torch.cat([kt, kc], 1)) / q.shape[-1] ** 0.5
    ref2 = torch.einsum("bhlm,bmhd->blhd", s2.softmax(-1), v)
    out2 = morph.text_all_coarse(qt, kt, k[:, L:], v, rq2)
    assert torch.allclose(out2, ref2, atol=1e-10) and not torch.allclose(out2, ref, atol=1e-6)


def test_processor_all_coarse_changes_only_text_rows():
    m = model()
    base = attn_outputs(m, _ctx(), H, W)
    mt = attn_outputs(m, _ctx(morph_text="all_coarse"), H, W)
    assert torch.isfinite(flat(mt)).all()
    assert not torch.allclose(mt[0][1], base[0][1], atol=1e-4)                # text output changed
    assert torch.allclose(mt[0][0], base[0][0], atol=1e-5)                    # image output of the first block unchanged
