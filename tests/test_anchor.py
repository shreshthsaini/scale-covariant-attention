"""Latent low-band anchor (sca/anchor.py) and the noise-dependent bridge size (SCAContext.bridge_k_hi).

Both default to off and must leave the published method unchanged. The anchor runs inside the real FluxPipeline
denoising loop on a tiny random pipeline (CPU, float32) whose token grid is h // 16 x w // 16 as at full size."""

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline  # noqa: E402
from diffusers.models.modeling_outputs import Transformer2DModelOutput  # noqa: E402

from sca import dedup  # noqa: E402
from sca.anchor import LatentAnchor, _with_sample, anchor_velocity, anchor_weight, latent_low  # noqa: E402
from sca.attn import PiTable, SCAContext, install  # noqa: E402
from test_sca import L_TXT, inputs, model  # noqa: E402

GATE, LO, RES, STEPS = 0.8, 0.3, 128, 10   # tiny schedule: global-only steps 0-2, anchored steps 3-7


def flat(o):
    return torch.cat([t.flatten() for t in (o[0] if isinstance(o[0], tuple) else (o[0],))] + [o[1].flatten()])


def brute_lp(x, H, W, p):
    """Independent LP: mean over each p x p block of row-major tokens, then bilinear upsampling. x [B, H*W, C]."""
    B, N, C = x.shape
    pooled = torch.zeros(B, C, H // p, W // p, dtype=x.dtype)
    for a in range(H // p):
        for b in range(W // p):
            idx = [(a * p + i) * W + b * p + j for i in range(p) for j in range(p)]
            pooled[:, :, a, b] = x[:, idx].mean(1)
    up = torch.nn.functional.interpolate(pooled, size=(H, W), mode="bilinear", align_corners=False)
    return up.flatten(2).transpose(1, 2)


def tiny_pipe():
    """FluxPipeline with the tiny transformer of tests/test_sca.py and a 3-downsample VAE (vae_scale_factor 8, so the
    token grid is h // 16 x w // 16 as in scripts/sample.py); text encoders replaced by fixed embeddings."""
    torch.manual_seed(0)
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                        up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(4,) * 4, layers_per_block=1,
                        latent_channels=1, norm_num_groups=2, sample_size=32, scaling_factor=0.5,
                        shift_factor=0.1).eval()
    sched = FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
                                            base_image_seq_len=256, max_image_seq_len=4096)
    pipe = FluxPipeline(scheduler=sched, vae=vae, text_encoder=None, tokenizer=None, text_encoder_2=None,
                        tokenizer_2=None, transformer=model())
    g = torch.Generator().manual_seed(7)
    emb = (torch.randn(1, L_TXT, 24, generator=g), torch.randn(1, 8, generator=g), torch.zeros(L_TXT, 3))
    pipe.encode_prompt = lambda *a, **k: emb
    return pipe


def sca_ctx(**kw):
    base = dict(mode="sca", n0=4, tile=2, window=4, local_impl="gather", pi=PiTable("const", value=0.7),
                gate_sigma=GATE, known_grid=(RES // 16, RES // 16))
    base.update(kw)
    return SCAContext(**base)


class Recorder:
    """Forward hook recording (step, sigma, x_t, v) of every sampler forward; the anchor's extra forward
    (ctx.force_global) goes to .extra with its kwargs instead."""

    def __init__(self, ctx):
        self.ctx, self.rows, self.extra = ctx, [], []

    def __call__(self, module, args, kwargs, output):
        if self.ctx.force_global:
            self.extra.append((self.ctx.step, {k: (t.clone() if torch.is_tensor(t) else t) for k, t in kwargs.items()},
                               output[0].clone()))
        else:
            self.rows.append((self.ctx.step, self.ctx.sigma, kwargs["hidden_states"].clone(), output[0].clone()))


@torch.no_grad()
def denoise(pipe, ctx, anchor=None, seed=0):
    """Final packed latents of the real FluxPipeline loop with SCA (and the anchor) installed, plus recorders of the
    raw transformer outputs (before the anchor) and the outputs the scheduler received (after it)."""
    undo = install(pipe.transformer, ctx)
    ctx.scheduler = pipe.scheduler
    raw, final = Recorder(ctx), Recorder(ctx)
    handles = [pipe.transformer.register_forward_hook(raw, with_kwargs=True)]
    remove = anchor.install(pipe.transformer) if anchor is not None else None
    handles.append(pipe.transformer.register_forward_hook(final, with_kwargs=True))
    try:
        out = pipe(prompt="x", height=RES, width=RES, num_inference_steps=STEPS, guidance_scale=3.5,
                   generator=torch.Generator().manual_seed(seed), output_type="latent").images
    finally:
        for h in handles:
            h.remove()
        if remove is not None:
            remove()
        undo()
    return out, raw, final, [float(s) for s in pipe.scheduler.sigmas[:-1]]


def test_weight_schedule_by_hand():
    for shape in ("linear", "cosine"):
        for s in (1.0, 0.95, 0.89):
            assert anchor_weight(s, 0.89, 0.3, shape) == 1.0
        for s in (0.3, 0.2, 0.0):
            assert anchor_weight(s, 0.89, 0.3, shape) == 0.0
        assert anchor_weight(0.595, 0.89, 0.3, shape) == pytest.approx(0.5)   # midpoint: u = 1/2, sin^2(pi/4) = 1/2
    s = 0.3 + 0.25 * 0.59
    assert anchor_weight(s, 0.89, 0.3, "linear") == pytest.approx(0.25)
    assert anchor_weight(s, 0.89, 0.3, "cosine") == pytest.approx(math.sin(math.pi / 8) ** 2)
    assert anchor_weight(0.8876, 0.89, 0.3, "linear") == pytest.approx(0.5876 / 0.59)     # step 8 at 4096^2
    assert anchor_weight(0.4071, 0.89, 0.3, "cosine") == pytest.approx(math.sin(math.pi / 2 * 0.1071 / 0.59) ** 2)
    with pytest.raises(ValueError):
        anchor_weight(0.5, 0.89, 0.3, "step")


def test_low_band_matches_bruteforce_and_flux_packing():
    g = torch.Generator().manual_seed(1)
    H, W, p = 8, 12, 4
    x = torch.randn(2, H * W, 64, generator=g, dtype=torch.float64)
    assert torch.allclose(latent_low(x, (H, W), p), brute_lp(x, H, W, p), atol=1e-12)
    # FLUX packs 2 x 2 latent patches row-major: token r * W + c holds latent rows 2r, 2r + 1 and cols 2c, 2c + 1
    lat = torch.arange(2 * H, dtype=torch.float64)[None, None, :, None].expand(1, 16, 2 * H, 2 * W) // 2
    packed = FluxPipeline._pack_latents(lat, 1, 16, 2 * H, 2 * W)
    assert packed.shape == (1, H * W, 64)
    assert torch.equal(packed[0, :, 0], torch.arange(H, dtype=torch.float64).repeat_interleave(W))


def test_full_weight_replaces_low_band_and_zero_weight_is_identity():
    g = torch.Generator().manual_seed(2)
    H, W, p, sigma = 8, 12, 4, 0.6
    x_t, v, L = (torch.randn(1, H * W, 64, generator=g, dtype=torch.float64) for _ in range(3))
    x0 = x_t - sigma * v
    v1 = anchor_velocity(x_t, v, sigma, L, (H, W), p, 1.0)
    x0p = x_t - sigma * v1
    lp = lambda t: brute_lp(t, H, W, p)  # noqa: E731
    assert torch.allclose(x0p - (x0 - lp(x0)), lp(L), atol=1e-12)        # low band of x0' = LP(L)
    assert torch.allclose(x0p - lp(L), x0 - lp(x0), atol=1e-12)          # high band of x0' = x0_hat's
    assert torch.allclose(v1, (x_t - (x0 + lp(L) - lp(x0))) / sigma, atol=1e-12)   # v' = (x_t - x0') / sigma
    w = 0.37
    vw = anchor_velocity(x_t, v, sigma, L, (H, W), p, w)
    assert torch.allclose(x_t - sigma * vw, x0 + w * (lp(L) - lp(x0)), atol=1e-12)
    assert anchor_velocity(x_t, v, sigma, L, (H, W), p, 0.0) is v        # w = 0: v' == v
    # bf16 in, bf16 out (the scheduler keeps the latents in the model dtype)
    vb = anchor_velocity(x_t.bfloat16(), v.bfloat16(), sigma, L.float(), (H, W), p, 0.5)
    assert vb.dtype == torch.bfloat16
    out = _with_sample(Transformer2DModelOutput(sample=v), v1)
    assert out.sample is v1 and _with_sample((v,), v1)[0] is v1


def test_anchor_capture_and_application_in_flux_loop():
    pipe = tiny_pipe()
    ref, _, _, sig = denoise(pipe, sca_ctx())
    ctx = sca_ctx()
    anchor = LatentAnchor(ctx, LO)
    out, raw, final, sig2 = denoise(pipe, ctx, anchor)
    assert sig2 == sig and len(raw.rows) == len(final.rows) == STEPS and not raw.extra
    capture = max(j for j, s in enumerate(sig) if s > GATE)
    anchored = [j for j, s in enumerate(sig) if LO < s <= GATE]
    assert capture == 2 and anchored == [3, 4, 5, 6, 7]                   # the schedule this test was written for
    assert anchor.capture_step == capture and [a[0] for a in anchor.applied] == anchored
    assert anchor.p == 2                                                  # global block size: pool_factors((8, 8), 4)
    step, sigma, x, v = raw.rows[capture]
    assert step == capture and torch.equal(anchor.anchor, x - sigma * v)  # L = x0_hat of the last global-only step
    for (j, s, x, v), (_, _, x2, v2) in zip(raw.rows, final.rows):
        assert torch.equal(x, x2)
        if j in anchored:
            ref_v = anchor_velocity(x, v, s, anchor.anchor, ctx.grid, 2, anchor_weight(s, GATE, LO))
            assert torch.equal(v2, ref_v) and not torch.allclose(v2, v, atol=1e-6)
        else:
            assert torch.equal(v2, v)                                     # global-only phase and sigma <= lo: untouched
    assert not torch.allclose(out, ref, atol=1e-5)
    assert torch.equal(raw.rows[0][2], denoise(pipe, sca_ctx())[1].rows[0][2])   # same start, deterministic


def test_anchor_off_is_bit_identical():
    pipe = tiny_pipe()
    ref, raw0, _, sig = denoise(pipe, sca_ctx())
    # a capture-only anchor (no step with lo < sigma <= gate) computes L but changes nothing
    below = max(s for s in sig if s <= GATE)
    anchor = LatentAnchor(sca_ctx(), 0.5 * (below + GATE))
    ctx = anchor.ctx
    out, raw, final, _ = denoise(pipe, ctx, anchor)
    assert anchor.capture_step == 2 and anchor.applied == [] and torch.equal(out, ref)
    assert all(torch.equal(a[3], b[3]) for a, b in zip(raw0.rows, final.rows))
    assert not pipe.transformer._forward_hooks and not pipe.transformer._forward_pre_hooks   # all removed
    assert torch.equal(denoise(pipe, sca_ctx())[0], ref)
    # CLI defaults: no anchor and no sidecar keys
    args = sample_module().parser().parse_args(["--method", "sca", "--pi-const", "0.5", "--res", "4096x4096",
                                                "--prompts", "x.jsonl", "--out", "o"])
    assert args.anchor_lo == 0 and dedup.anchor(args, sca_ctx()) is None
    assert not [k for k in dedup.info(args) if k.startswith(("anchor", "bridge_k_hi"))]


@pytest.mark.parametrize("K", [0, 2, 3])
def test_refresh_runs_one_global_forward_every_k_steps(K):
    pipe = tiny_pipe()
    calls = []
    ctx = sca_ctx()
    count = pipe.transformer.register_forward_pre_hook(  # step from the scheduler (this hook runs before install's)
        lambda m, a, kw: calls.append((pipe.scheduler.step_index or 0, ctx.force_global)), with_kwargs=True)
    anchor = LatentAnchor(ctx, LO, refresh=K)
    try:
        _, raw, final, sig = denoise(pipe, ctx, anchor)
    finally:
        count.remove()
    capture = max(j for j, s in enumerate(sig) if s > GATE)
    anchored = [j for j, s in enumerate(sig) if LO < s <= GATE]
    expected = [j for j in anchored if (j - capture) % K == 0] if K else []
    assert expected == {0: [], 2: [4, 6], 3: [5]}[K]
    assert len(calls) == STEPS + len(expected)                            # one extra forward per refresh
    assert [s for s, forced in calls if forced] == expected and anchor.refresh_steps == expected
    assert [s for s, forced in calls if not forced] == list(range(STEPS))
    assert [e[0] for e in raw.extra] == expected and len(raw.rows) == STEPS
    if not K:
        return
    # the extra forward is the global-only (opening-phase) prediction, and the anchor is rebuilt from it
    j, kw, v_glob = raw.extra[-1]
    glob = sca_ctx(gate_sigma=1e-3)  # sigma from the timestep (no scheduler) > 1e-3: global only
    undo = install(pipe.transformer, glob)
    try:
        with torch.no_grad():
            assert torch.equal(pipe.transformer(**kw)[0], v_glob)
    finally:
        undo()
    _, s, x, v = raw.rows[j]
    assert torch.equal(anchor.anchor, x - s * v_glob) and not torch.allclose(v_glob, v, atol=1e-6)
    ref_v = anchor_velocity(x, v, s, anchor.anchor, ctx.grid, 2, anchor_weight(s, GATE, LO))
    assert torch.equal(final.rows[j][3], ref_v)


def test_anchor_validates():
    with pytest.raises(ValueError):
        LatentAnchor(SCAContext(mode="full"), 0.3)
    with pytest.raises(ValueError):
        LatentAnchor(sca_ctx(gate_sigma=0.0), 0.3)
    with pytest.raises(ValueError):
        LatentAnchor(sca_ctx(), 0.85)                                     # lo above the gate
    with pytest.raises(ValueError):
        LatentAnchor(sca_ctx(), 0.3, shape="step")
    a = LatentAnchor(sca_ctx(), 0.3)
    a.ctx.scheduler = object()
    x = torch.zeros(2, 64, 4)
    with pytest.raises(AssertionError):                                   # batch 2 (true CFG): not supported
        a(None, (), {"hidden_states": x}, (x,))


def sample_module():
    spec = importlib.util.spec_from_file_location("sca_sample_anchor_test", ROOT / "scripts/sample.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_cli_records_anchor(tmp_path, monkeypatch):
    sample = sample_module()
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("".join(json.dumps({"id": f"p{i}", "prompt": f"prompt {i}"}) + "\n" for i in range(2)))
    pipe = tiny_pipe()
    import sca.runtime
    monkeypatch.setattr(sca.runtime, "load_pipeline", lambda device, model: pipe)
    base = ["--method", "sca", "--pi-const", "0.7", "--res", f"{RES}x{RES}", "--prompts", str(prompts),
            "--steps", str(STEPS), "--n0", "4", "--tile", "2", "--window", "4", "--local-impl", "gather",
            "--gate-sigma", str(GATE), "--device", "cpu"]
    sample.main(base + ["--out", str(tmp_path / "off")])
    sample.main(base + ["--out", str(tmp_path / "on"), "--anchor-lo", str(LO), "--anchor-shape", "cosine",
                        "--anchor-refresh", "2"])
    assert not pipe.transformer._forward_hooks and not pipe.transformer._forward_pre_hooks
    for stem_json in sorted((tmp_path / "on").glob("*.json")):  # both prompts: reset() between images
        on = json.loads(stem_json.read_text())
        off = json.loads((tmp_path / "off" / stem_json.name).read_text())
        assert "anchor_trace" not in off and not [k for k in off["config"]["sca"] if k.startswith("anchor")]
        cfg = on["config"]["sca"]
        assert (cfg["anchor_lo"], cfg["anchor_shape"], cfg["anchor_pool"], cfg["anchor_refresh"]) == (LO, "cosine", 0, 2)
        sig = on["sigmas"]
        t = on["anchor_trace"]
        assert t["capture_step"] == 2 and t["pool_tokens"] == 2 and t["refresh_steps"] == [4, 6]
        assert [a[0] for a in t["applied"]] == [j for j, s in enumerate(sig) if LO < s <= GATE]
        assert all(a[2] == pytest.approx(anchor_weight(sig[a[0]], GATE, LO, "cosine")) for a in t["applied"])
        assert on["config_fingerprint"] != off["config_fingerprint"]


def test_sample_progressive_rejects_anchor(tmp_path):
    spec = importlib.util.spec_from_file_location("sca_prog_anchor_test", ROOT / "scripts/sample_progressive.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"id": "p0", "prompt": "x"}) + "\n")
    with pytest.raises(SystemExit, match="anchor"):
        sp.main(["--prompts", str(prompts), "--out", str(tmp_path / "o"), "--pi-const", "0.5", "--device", "cpu",
                 "--anchor-lo", "0.3"])


# ----------------------------------------------------------------------------- noise-dependent bridge size

@torch.no_grad()
def attn_at(m, ctx, H, W, sigma):
    """Per-module attention outputs at a given sigma (tests/test_sca.py attn_outputs with sigma as an argument)."""
    x = inputs(H, W, seed=3)
    undo = install(m, ctx)
    ctx.set_inputs(x["img_ids"], sigma=sigma)
    rot = m.pos_embed(torch.cat([x["txt_ids"], x["img_ids"]]))
    g = torch.Generator().manual_seed(5)
    img, txt = torch.randn(1, H * W, m.inner_dim, generator=g), torch.randn(1, L_TXT, m.inner_dim, generator=g)
    outs = [m.transformer_blocks[0].attn(img, txt, image_rotary_emb=rot),
            m.single_transformer_blocks[0].attn(torch.cat([txt, img], 1), image_rotary_emb=rot)]
    undo()
    return outs


def bridge_ctx(**kw):
    return SCAContext(mode="sca", n0=4, tile=2, window=4, local_impl="flex", pi=PiTable("const", value=0.7),
                      bridge_cells=2, **kw)


def test_bridge_k_hi_switches_at_sigma():
    m, H, W = model(), 8, 8                                               # 2 x 2 cells of 16 tokens
    c = bridge_ctx(bridge_k=2, bridge_k_hi=4, bridge_k_hi_lo=0.6)
    above, below = attn_at(m, c, H, W, 0.7), attn_at(m, c, H, W, 0.5)
    assert {k[0] for k in c._flex_cache} == {L_TXT + 4 * 4, L_TXT + 4 * 2}   # two prefix lengths, two layouts
    assert torch.equal(flat(above), flat(attn_at(m, bridge_ctx(bridge_k=4), H, W, 0.7)))
    assert torch.equal(flat(below), flat(attn_at(m, bridge_ctx(bridge_k=2), H, W, 0.5)))
    hi_only = bridge_ctx(bridge_k=2, bridge_k_hi=4, bridge_k_hi_lo=0.6)
    attn_at(m, hi_only, H, W, 0.61)
    assert {k[0] for k in hi_only._flex_cache} == {L_TXT + 16}
    lo_only = bridge_ctx(bridge_k=2, bridge_k_hi=4, bridge_k_hi_lo=0.6)
    attn_at(m, lo_only, H, W, 0.6)                                        # sigma == threshold: K
    assert {k[0] for k in lo_only._flex_cache} == {L_TXT + 8}
    for s, k in ((0.89, 64), (0.61, 64), (0.6, 16), (0.3, 16)):
        c2 = SCAContext(mode="sca", local_impl="flex", bridge_cells=8, bridge_k=16, bridge_k_hi=64, bridge_k_hi_lo=0.6)
        c2.sigma = s
        assert c2.bridge_k_now() == k
    with pytest.raises(ValueError):
        SCAContext(mode="sca", local_impl="flex", bridge_k_hi=64)


def test_bridge_k_hi_equal_k_is_plain_bridge():
    m, H, W = model(), 8, 8
    for s in (0.7, 0.5):
        same = attn_at(m, bridge_ctx(bridge_k=2, bridge_k_hi=2, bridge_k_hi_lo=0.6), H, W, s)
        assert torch.equal(flat(same), flat(attn_at(m, bridge_ctx(bridge_k=2), H, W, s)))


def test_bridge_k_hi_cli():
    sample = sample_module()
    base = ["--method", "sca", "--pi-const", "0.5", "--res", "4096x4096", "--prompts", "x.jsonl", "--out", "o",
            "--gate-sigma", "0.89", "--bridge-cells", "8", "--bridge-k", "16"]
    c, info = sample.build_context(sample.parser().parse_args(base), (256, 256))
    assert c.bridge_k_hi == 0 and "bridge_k_hi" not in info and info["bridge_k"] == 16
    c, info = sample.build_context(sample.parser().parse_args(base + ["--bridge-k-hi", "64", "--bridge-k-hi-lo", "0.6"]),
                                   (256, 256))
    assert (c.bridge_k_hi, c.bridge_k_hi_lo, info["bridge_k_hi"], info["bridge_k_hi_lo"]) == (64, 0.6, 64, 0.6)
    args = sample.parser().parse_args(base + ["--anchor-lo", "0.3", "--anchor-refresh", "4"])
    c, info = sample.build_context(args, (256, 256))
    assert info["anchor_lo"] == 0.3 and info["anchor_refresh"] == 4 and info["anchor_shape"] == "linear"
    assert isinstance(dedup.anchor(args, c), LatentAnchor)


class FakeFlux(torch.nn.Module):
    """Stands in for the FLUX transformer at 4096^2: packed bf16 latents [1, 65536, 64], output (v,)."""

    def __init__(self, ctx):
        super().__init__()
        self.ctx, self.calls = ctx, []

    def forward(self, hidden_states, timestep, return_dict=False):
        self.calls.append((self.ctx.step, self.ctx.force_global))
        g = 0.5 if self.ctx.force_global else 1.0
        return (torch.sin(hidden_states.float() * g + timestep.float()).to(hidden_states.dtype),)


@torch.no_grad()
def test_anchor_steps_at_4096():
    """The production schedule: 28 steps, mu = max_shift, gate 0.89, lo 0.3 -> capture step 7, anchored steps 8-24,
    LP over 4 x 4 tokens, refresh every 4 steps at 11, 15, 19, 23; bf16 latents stay bf16."""
    sched = FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
                                            base_image_seq_len=256, max_image_seq_len=256 * 256)
    sched.set_timesteps(sigmas=[1 - j * (1 - 1 / 28) / 27 for j in range(28)], mu=1.15)
    sig = sched.sigmas
    ctx = SCAContext(mode="sca", gate_sigma=0.89, known_grid=(256, 256))
    ctx.grid, ctx.scheduler = (256, 256), sched
    m = FakeFlux(ctx)

    def pre(module, args, kw):  # what sca.attn.install's pre-hook sets
        ctx.step = int(sched.step_index or 0)
        ctx.sigma = float(sig[ctx.step])
    hp = m.register_forward_pre_hook(pre, with_kwargs=True)
    anchor = LatentAnchor(ctx, 0.3, refresh=4)
    remove = anchor.install(m)
    x = torch.randn(1, 256 * 256, 64, generator=torch.Generator().manual_seed(0)).bfloat16()
    for j in range(28):
        sched._step_index = None if j == 0 else j
        v = m(hidden_states=x, timestep=sig[j].expand(1), return_dict=False)[0]
        assert v.dtype == torch.bfloat16 and v.shape == x.shape
        x = (x.float() + (sig[j + 1] - sig[j]) * v.float()).to(torch.bfloat16)
    remove()
    hp.remove()
    assert anchor.capture_step == 7 and anchor.p == 4
    assert [a[0] for a in anchor.applied] == list(range(8, 25)) and anchor.refresh_steps == [11, 15, 19, 23]
    assert [s for s, forced in m.calls if forced] == [11, 15, 19, 23] and len(m.calls) == 32
    assert anchor.applied[0][2] == pytest.approx((float(sig[8]) - 0.3) / 0.59)
