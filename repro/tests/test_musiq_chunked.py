"""Bounded all-key attention equivalence, including masked query rows."""

import copy

import pytest
import torch
from pyiqa.archs.musiq_arch import MultiHeadAttention

from repro.musiq_chunked import install


@pytest.mark.parametrize("mask_mode", ["none", "partial", "all"])
@pytest.mark.parametrize("chunk", [1, 13, 100])
def test_attention_keeps_all_keys_and_finite_mask(mask_mode, chunk):
    torch.manual_seed(834)
    original = MultiHeadAttention(48, num_heads=6, bias=True).eval()
    adapted = copy.deepcopy(original)
    install(adapted, chunk)
    x = torch.randn(2, 39, 48)
    mask = None
    if mask_mode != "none":
        mask = torch.zeros(2, 39, dtype=torch.bool)
        if mask_mode == "partial":
            mask[0, :27] = True
            mask[1, ::3] = True
    with torch.inference_mode():
        expected = original(x, mask)
        actual = adapted(x, mask)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


def test_adapter_rejects_training_or_gradients():
    model = MultiHeadAttention(48).eval()
    install(model, 8)
    with pytest.raises(RuntimeError, match="inference/no_grad"):
        model(torch.randn(1, 12, 48))
    model.train()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="eval mode"):
        model(torch.randn(1, 12, 48))
