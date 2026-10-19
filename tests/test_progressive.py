"""CPU tests for scripts/sample_progressive.py on a tiny random FLUX pipeline (float32)."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline  # noqa: E402

import sample_progressive as sp  # noqa: E402
from sca.attn import PiTable, SCAContext, SCAProcessor  # noqa: E402
from sca.runtime import image_stem  # noqa: E402
from test_sca import L_TXT, model  # noqa: E402

FLUX_SNAPSHOT = Path("/scratch/09032/saini_2/tmp/hf/hub/models--black-forest-labs--FLUX.1-dev/snapshots/"
                     "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")
HRDIT_OUT = Path("/scratch/09032/saini_2/sca/out/baseline_expansion_20261017_v1/hrdit_dev_author/4096x4096")


def tiny_pipe():
    torch.manual_seed(0)
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2,
                        up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(8, 8), layers_per_block=1,
                        latent_channels=1, norm_num_groups=4, sample_size=32, scaling_factor=0.5,
                        shift_factor=0.1).eval()
    sched = FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
                                            base_image_seq_len=256, max_image_seq_len=4096)
    return FluxPipeline(scheduler=sched, vae=vae, text_encoder=None, tokenizer=None, text_encoder_2=None,
                        tokenizer_2=None, transformer=model())


def embeds(seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, L_TXT, 24, generator=g), torch.randn(1, 8, generator=g), torch.zeros(L_TXT, 3)


PLAN = dict(stages=[(16, 16), (32, 32), (64, 64)], base_steps=6, highres_steps=[4, 3])


def contexts(gate=0.89):
    return [None] + [SCAContext(mode="sca", n0=4, tile=2, local_impl="gather", pi=PiTable("const", value=0.5),
                                gate_sigma=gate) for _ in range(2)]


@torch.no_grad()
def run(seed=0, plan=None, ctxs=None, record=None, pipe=None):
    pipe = pipe or tiny_pipe()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    return sp.generate(pipe, embeds(), gen, plan or sp.Plan(**PLAN), ctxs or contexts(), record=record)


def test_plan_matches_hrdit_release():
    args = sp.parser().parse_args(["--prompts", "x", "--out", "y"])
    plan = sp.make_plan(args)
    assert plan.stages == [(1024, 1024), (2048, 2048), (4096, 4096)]
    assert plan.total_steps() == 57 and [plan.start_index(s) for s in range(3)] == [0, 13, 20]
    assert plan.guidance == [3.5, 4.5, 6.0] and plan.alphas == [1.0, 0.25] and plan.betas == [0.5, 0.5]
    with pytest.raises(ValueError):
        sp.Plan(stages=[(16, 16), (32, 32)], highres_steps=[4, 3])


@pytest.mark.skipif(not (FLUX_SNAPSHOT.is_dir() and HRDIT_OUT.is_dir()), reason="needs FLUX snapshot and HRDiT outputs")
def test_schedule_reproduces_recorded_hrdit_sigmas():
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(str(FLUX_SNAPSHOT), subfolder="scheduler")
    pipe = SimpleNamespace(scheduler=sched, vae_scale_factor=8)
    _, sigmas, mu = sp.hrdit_schedule(pipe, 30, (1024, 1024), "cpu")
    record = json.loads(next(HRDIT_OUT.glob("*.json")).read_text())
    assert mu == pytest.approx(1.15)
    np.testing.assert_allclose(sigmas.numpy(), record["scheduler_sigmas"], rtol=0, atol=1e-6)
    assert sigmas[13].item() == pytest.approx(0.8051, abs=1e-4) and sigmas[20].item() == pytest.approx(0.6123, abs=1e-4)


def test_butterworth_matches_upstream_loop():
    shape, ratio, n = [1, 2, 6, 10], 0.2, 4
    ref = torch.zeros(shape)
    for h in range(6):
        for w in range(10):
            d = (2 * h / 6 - 1) ** 2 + (2 * w / 10 - 1) ** 2
            ref[..., h, w] = 1 / (1 + (d / ratio ** 2) ** n)
    assert torch.equal(sp.butterworth(shape, ratio, n), ref)


def test_guided_step_without_guidance_is_euler():
    pipe = tiny_pipe()
    g = torch.Generator().manual_seed(0)
    x, v = torch.randn(1, 64, 4, generator=g), torch.randn(1, 64, 4, generator=g)
    ref = (torch.randn(1, 16, 4, generator=g), (16, 16))
    mask = sp.butterworth([1, 1, 16, 16], 0.2)
    state = {}
    a, x0a = sp.guided_step(pipe, x, v, torch.tensor(0.6), torch.tensor(0.5), ref, (32, 32), mask, 0.0, 0.0, state)
    b, x0b = sp.euler(x, v, torch.tensor(0.6), torch.tensor(0.5))
    torch.testing.assert_close(a, b, rtol=0, atol=1e-5)
    torch.testing.assert_close(x0a, x0b)
    assert state["high"] is not None


def test_stage_handoff_shapes_and_attention_per_stage():
    pipe = tiny_pipe()
    seen = []
    hook = pipe.transformer.transformer_blocks[0].attn.register_forward_pre_hook(
        lambda m, a: seen.append(type(m.processor).__name__))
    original = {k: type(v) for k, v in pipe.transformer.attn_processors.items()}
    rec, ctxs = {}, contexts()
    image = run(record=rec, ctxs=ctxs, pipe=pipe)
    hook.remove()
    assert image.shape == (1, 3, 64, 64) and torch.isfinite(image).all()
    assert [s["grid"] for s in rec["stage_shapes"]] == [[4, 4], [8, 8], [16, 16]]
    assert [s["latents"] for s in rec["stage_shapes"]] == [[1, 16, 4], [1, 64, 4], [1, 256, 4]]
    # stock processors for the 6 native steps, SCA for the 4 + 3 higher-resolution steps
    assert seen[:6] == ["FluxAttnProcessor"] * 6 and seen[6:] == ["SCAProcessor"] * 7
    assert {k: type(v) for k, v in pipe.transformer.attn_processors.items()} == original
    assert [t["step"] for t in ctxs[1].pi_trace] == [2, 3, 4, 5] and [t["step"] for t in ctxs[2].pi_trace] == [3, 4, 5]
    assert ctxs[1].grid == (8, 8) and ctxs[2].grid == (16, 16)
    assert [t["sigma"] for t in ctxs[2].pi_trace] == pytest.approx(rec["stage_sigmas"][2])
    assert len(rec["stage_seconds"]) == 3 and len(rec["transition_seconds"]) == 2 and len(rec["blur_sigmas"]) == 2
    assert all(0.1 <= s <= 2.0 for s in rec["blur_sigmas"])


def test_determinism_and_seed_dependence():
    a, b, c = run(0), run(0), run(1)
    assert torch.equal(a, b)
    assert not torch.allclose(a, c)


def test_sca_changes_only_highres_stages():
    """With the gate above every sigma (global-only branch) the result differs from pi=0.5 mixing,
    while the native stage is untouched: the first-stage latent is identical."""
    r1, r2 = {}, {}
    run(record=r1, ctxs=contexts(0.89))
    run(record=r2, ctxs=contexts(0.0))
    assert r1["stage_sigmas"] == r2["stage_sigmas"]
    a, b = run(ctxs=contexts(0.0)), run(ctxs=contexts(0.01))  # 0.01 gates every high-res step global-only
    assert not torch.allclose(a, b)


def test_cli_sidecar_resume_and_check(tmp_path, monkeypatch):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("".join(json.dumps({"id": f"p{i}", "prompt": f"prompt {i}"}) + "\n" for i in range(3)))
    out = tmp_path / "out"
    pipe = tiny_pipe()
    monkeypatch.setattr(sp, "load", lambda args: pipe)
    monkeypatch.setattr(sp, "encode_text", lambda p, prompt, device: embeds(len(prompt)))
    argv = ["--prompts", str(prompts), "--out", str(out), "--start", "0", "--end", "2", "--device", "cpu",
            "--stages", "64x64,128x128,256x256", "--base-steps", "4", "--highres-steps", "2,1",
            "--pi-const", "0.5", "--n0", "16", "--tile", "4", "--local-impl", "gather"]
    with pytest.raises(SystemExit):
        sp.main(argv + ["--check"])
    sp.main(argv)
    from PIL import Image
    for i in range(2):
        stem = image_stem(f"p{i}", 0)
        meta = json.loads((out / f"{stem}.json").read_text())
        cfg = meta["config"]
        assert meta["complete"] and cfg["method"] == "sca_prog" and cfg["res"] == "256x256"
        assert cfg["prompt_id"] == f"p{i}" and cfg["seed"] == 0 and cfg["steps"] == 7
        assert cfg["progressive"]["stages"] == ["64x64", "128x128", "256x256"]
        assert [x["pi_scale"] for x in cfg["sca"]["stages"]] == [None, None]
        assert [x["pool_factors"] for x in cfg["sca"]["stages"]] == [[2, 2], [4, 4]]
        assert meta["seconds"] > 0 and len(meta["stage_seconds"]) == 3 and meta["gate_active_steps"] == [0, 0]
        assert {t["stage"] for t in meta["pi_trace"]} == {1, 2}
        with Image.open(out / f"{stem}.png") as im:
            assert im.size == (256, 256)
    before = (out / f"{image_stem('p0', 0)}.json").read_bytes()
    sp.main(argv)  # resume: skips, rewrites nothing
    assert (out / f"{image_stem('p0', 0)}.json").read_bytes() == before
    sp.main(argv + ["--check"])
    with pytest.raises(SystemExit):
        sp.main(argv[:6] + ["--end", "3"] + argv[8:] + ["--check"])
    with pytest.raises(ValueError):
        sp.main(argv[:-6] + ["--pi-const", "0.4", "--n0", "16", "--tile", "4", "--local-impl", "gather"])


def test_contexts_pick_scale_per_stage():
    calib = Path("/scratch/09032/saini_2/sca/calib/k1_1024.npz")
    if not calib.is_file():
        pytest.skip("calibration table not available")
    args = sp.parser().parse_args(["--prompts", "x", "--out", "y", "--pi-table", str(calib)])
    ctxs, info = sp.build_contexts(args, sp.make_plan(args))
    assert ctxs[0] is None and [x["pi_scale"] for x in info["stages"]] == [2, 4]
    assert [x["gate_sigma"] for x in info["stages"]] == [0.89, 0.89]
