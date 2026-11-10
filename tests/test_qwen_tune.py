"""CPU tests for the Qwen-Image tuning options (HANDOVER Section 39): the resolution time shift (sca/qwen.py), AReA's
conditioning suffix in scripts/sample_qwen.py, and progressive SCA on Qwen-Image (scripts/sample_progressive_qwen.py)
on a tiny random pipeline."""

import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_qwen import patch_loaders, tiny_pipeline, write_prompts  # noqa: E402

from sca.qwen import NATIVE_SEQ, mu_for, set_mu  # noqa: E402


def test_resolution_mu_rule():
    native = mu_for(NATIVE_SEQ)
    assert abs(mu_for(332 * 332, "resolution") - (native + math.log(4))) < 1e-9          # 5312^2: 2.22
    assert abs(mu_for(256 * 256, "resolution") - (native + 0.5 * math.log(65536 / 6889))) < 1e-9
    assert mu_for(83 * 83, "resolution") == native and mu_for(64 * 64, "resolution") == mu_for(64 * 64, "native")
    assert mu_for(332 * 332, "native") == native                                          # default protocol unchanged
    from diffusers.pipelines.qwenimage import pipeline_qwenimage as P
    cfg = (256, 8192, 0.5, 0.9)
    try:
        set_mu("resolution")
        assert abs(P.calculate_shift(332 * 332, *cfg) - mu_for(332 * 332, "resolution")) < 1e-9
        set_mu("native")
        assert abs(P.calculate_shift(332 * 332, *cfg) - native) < 1e-9
    finally:
        set_mu("default")


def test_sample_qwen_area_suffix_and_resolution_mu(tmp_path, monkeypatch):
    from scripts import run_area_qwen as R
    from scripts import sample_qwen
    pipe, calls = patch_loaders(monkeypatch, tmp_path)
    prompts = write_prompts(tmp_path / "p.jsonl", 2)
    out = tmp_path / "out"
    base = ["--prompts", str(prompts), "--res", "128x128", "--steps", "2", "--device", "cpu", "--method", "sca",
            "--pi-const", "0.5", "--n0", "4", "--tile", "2", "--gate-sigma", "0.9", "--label", "qt"]
    sample_qwen.main(base + ["--out", str(out), "--prompt-suffix", "area", "--negative=", "--mu-mode", "resolution"])
    suffix = R.upstream_defaults()["suffix"]
    rows = [json.loads(x) for x in prompts.read_text().splitlines()]
    assert calls[0] == [r["prompt"] + suffix for r in rows] + [""]       # conditioning prompts, then the empty negative
    m = json.loads(sorted(out.glob("*.json"))[0].read_text())
    c = m["config"]
    assert c["prompt"] == rows[0]["prompt"] and c["conditioning_prompt"] == rows[0]["prompt"] + suffix
    assert c["prompt_suffix"] == "area" and c["negative_prompt"] == "" and c["mu_mode"] == "resolution"
    out2 = tmp_path / "out2"                                              # no suffix: no new keys in the config
    sample_qwen.main(base + ["--out", str(out2)])
    c2 = json.loads(sorted(out2.glob("*.json"))[0].read_text())["config"]
    assert "conditioning_prompt" not in c2 and "prompt_suffix" not in c2 and c2["negative_prompt"] == " "
    set_mu("default")


# ----------------------------------------------------------------------------- progressive SCA on Qwen-Image

def test_bridge_for_stage_grids():
    from scripts.sample_progressive_qwen import bridge_for
    assert bridge_for((166, 166), 1024) == (2, 256)
    assert bridge_for((332, 332), 1024) == (4, 64)
    assert bridge_for((256, 256), 1024) == (8, 16)


@torch.no_grad()
def test_native_stage_equals_stock_pipeline():
    """Stage 0 (stock processors, Qwen's schedule at the native sequence, true CFG with the norm rescale) reproduces
    QwenImagePipeline's latents for the same seed, steps and prompt embeddings."""
    from scripts import sample_progressive_qwen as Q
    pipe = tiny_pipeline()
    g = torch.Generator().manual_seed(3)
    cond, neg = torch.randn(1, 5, 24, generator=g), torch.randn(1, 3, 24, generator=g)
    plan = Q.sp.Plan(stages=[(64, 64), (128, 128)], base_steps=4, highres_steps=[2], guidance=[4.0, 4.0],
                     alphas=[1.0], betas=[0.5], filter_ratio=[0.2])
    from sca.attn import PiTable
    from sca.qwen import QwenSCAContext
    ctx = QwenSCAContext(mode="sca", known_grid=(8, 8), n0=4, tile=2, pi=PiTable("const", value=0.5))
    keep = {}
    torch.manual_seed(0)
    img = Q.generate(pipe, cond, neg, torch.Generator().manual_seed(11), plan, [None, ctx], keep=keep)
    assert img.shape == (1, 3, 128, 128) and torch.isfinite(img).all()
    set_mu("native")
    ref = pipe(prompt=None, prompt_embeds=cond, negative_prompt_embeds=neg, height=64, width=64, num_inference_steps=4,
               true_cfg_scale=4.0, generator=torch.Generator().manual_seed(11), output_type="latent").images
    set_mu("default")
    assert torch.equal(keep[0], ref)
    assert keep[1].shape == (1, 64, 16) and not torch.equal(keep[1][:, :16], keep[0])
    assert type(pipe.transformer.transformer_blocks[0].attn.processor).__name__ == "QwenDoubleStreamAttnProcessor2_0"


@torch.no_grad()
def test_guided_step_without_guidance_is_euler():
    """alpha = beta = 0: the HRDiT step reduces to the Euler step (sample_progressive.guided_step's identity)."""
    from scripts import sample_progressive_qwen as Q
    pipe = tiny_pipeline()
    g = torch.Generator().manual_seed(5)
    x, v = torch.randn(1, 16, 16, generator=g), torch.randn(1, 16, 16, generator=g)
    ref = (torch.randn(1, 4, 16, generator=g), (32, 32))
    mask = Q.sp.butterworth([1, 4, 8, 8], 0.2)
    out, x0 = Q.guided_step(pipe, x, v, torch.tensor(0.6), torch.tensor(0.5), ref, (64, 64), mask, 0.0, 0.0, {})
    e, x0e = Q.sp.euler(x, v, torch.tensor(0.6), torch.tensor(0.5))
    assert torch.allclose(out, e.float(), atol=1e-5) and torch.allclose(x0, x0e)
    z = torch.randn(1, 4, 8, 8, generator=g)
    assert torch.equal(Q.unpack(pipe, Q.pack(pipe, z), 64, 64), z)


def test_progressive_script_cpu(tmp_path, monkeypatch):
    from scripts import run_area_qwen as R
    from scripts import sample_progressive_qwen as Q
    pipe, calls = patch_loaders(monkeypatch, tmp_path)
    prompts = write_prompts(tmp_path / "p.jsonl", 3)
    out = tmp_path / "out"
    args = ["--prompts", str(prompts), "--stages", "64x64,128x128,256x256", "--base-steps", "4", "--highres-steps",
            "3,2", "--pi-const", "0.5", "--n0", "4", "--tile", "2", "--device", "cpu", "--out", str(out), "--start",
            "0", "--end", "2", "--bridge-tokens", "16", "--label", "qt_prog"]
    Q.main(args)
    metas = sorted(out.glob("*.json"))
    assert len(metas) == 2 and len(list(out.glob("*.png"))) == 2
    m = json.loads(metas[0].read_text())
    c = m["config"]
    assert m["complete"] and c["method"] == "qt_prog" and c["res"] == "256x256" and c["steps"] == 4 + 3 + 2
    assert c["true_cfg_scale"] == [4.0, 4.0, 4.0] and c["negative_prompt"] == " " and c["prompt"].startswith("prompt")
    assert [s["grid"] for s in m["stage_shapes"]] == [[4, 4], [8, 8], [16, 16]]
    assert [len(s) for s in m["stage_sigmas"]] == [4, 3, 2] and len(m["stage_seconds"]) == 3
    assert len(m["transition_seconds"]) == 2 and len(m["blur_sigmas"]) == 2
    assert [d["pool_factors"] for d in c["sca"]["stages"]] == [[2, 2], [4, 4]]
    assert [(d["bridge_cells"], d["bridge_k"]) for d in c["sca"]["stages"]] == [(4, 1), (4, 1)]
    assert m["pi_trace"] and m["gate_active_steps"] == [0, 0]
    with Image_open(sorted(out.glob("*.png"))[0]) as im:
        assert im.size == (256, 256)
    n = len(calls)
    Q.main(args)                                                          # resume: nothing regenerated
    assert calls[n] == [" "]
    out2 = tmp_path / "out2"                                              # AReA's conditioning
    Q.main(args[:-1] + ["qt_prog_sfx", "--out", str(out2), "--prompt-suffix", "area", "--negative="])
    c2 = json.loads(sorted(out2.glob("*.json"))[0].read_text())["config"]
    assert c2["conditioning_prompt"] == c2["prompt"] + R.upstream_defaults()["suffix"] and c2["negative_prompt"] == ""
    assert type(pipe.transformer.transformer_blocks[0].attn.processor).__name__ == "QwenDoubleStreamAttnProcessor2_0"


def Image_open(path):
    from PIL import Image
    return Image.open(path)
