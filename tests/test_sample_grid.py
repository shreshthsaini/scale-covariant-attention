"""Full-attention sampling must not infer grid size from rounded bf16 positions."""

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sca.attn import SCAContext  # noqa: E402


@pytest.fixture
def sample():
    # Also exercise a separately patched historical sample without editing its freeze.
    path = Path(os.environ.get("SCA_SAMPLE_SOURCE", Path(__file__).resolve().parents[1] / "scripts/sample.py"))
    spec = importlib.util.spec_from_file_location("sca_sample_grid_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("grid", [(384, 384), (128, 384), (512, 512)])
def test_full_context_preserves_bf16_positions_and_exact_grid(sample, grid):
    height, width = grid
    ids = torch.zeros(height, width, 3, dtype=torch.bfloat16)
    ids[..., 1] = torch.arange(height, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(width, dtype=torch.float32)[None, :]
    ids = ids.reshape(-1, 3)
    original = ids.clone()

    # Reproduce the actual failure: a 384-token axis ends at bf16(383) = 384.
    assert int(ids[:, 2].max()) == width
    with pytest.raises(AssertionError, match="image ids must form a full grid"):
        SCAContext(mode="full").set_inputs(ids)

    context, info = sample.build_context(SimpleNamespace(method="full"), grid)
    context.set_inputs(ids)
    assert context.mode == "full" and context.grid == grid and info == {}
    assert torch.equal(ids, original)
    assert torch.equal(context._img_ids, original)
    assert context._img_ids.dtype == torch.bfloat16


def test_full_context_retains_token_count_validation(sample):
    context, _ = sample.build_context(SimpleNamespace(method="full"), (384, 384))
    with pytest.raises(AssertionError, match="image ids must form a full grid"):
        context.set_inputs(torch.zeros(384 * 384 - 1, 3, dtype=torch.bfloat16))
