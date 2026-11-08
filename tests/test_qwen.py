"""CPU tests for the Qwen-Image port on a tiny random QwenImageTransformer2DModel (float32)."""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusers import QwenImageTransformer2DModel  # noqa: E402
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen  # noqa: E402

from sca.attn import PiTable, global_branch, local_branch, rope  # noqa: E402
from sca.calib import CalibStats  # noqa: E402
from sca.qwen import (QwenSCAContext as Ctx, gate_from_steps, image_ids, install, mu_for, qwen_freqs,  # noqa: E402
                      rot_from_freqs, sigma_schedule, text_ids, win_sched_from_steps)

L_TXT, HEADS, DH, AXES = 5, 2, 16, (4, 6, 6)


def model(layers=2):
    torch.manual_seed(0)
    return QwenImageTransformer2DModel(patch_size=2, in_channels=16, out_channels=4, num_layers=layers,
                                       attention_head_dim=DH, num_attention_heads=HEADS, joint_attention_dim=24,
                                       axes_dims_rope=AXES).eval()


def inputs(H, W, L=L_TXT, seed=1, mask=None):
    g = torch.Generator().manual_seed(seed)
    return dict(hidden_states=torch.randn(1, H * W, 16, generator=g),
                encoder_hidden_states=torch.randn(1, L, 24, generator=g), encoder_hidden_states_mask=mask,
                timestep=torch.tensor([0.7]), img_shapes=[[(1, H, W)]], return_dict=False)


@torch.no_grad()
def run(m, x, ctx=None):
    if ctx is None:
        return m(**x)[0]
    undo = install(m, ctx)
    try:
        return m(**x)[0]
    finally:
        undo()


@torch.no_grad()
def attn_outputs(m, ctx, H, W, sigma=0.5, L=L_TXT):
    """(img, txt) outputs of block 0's attention for a fixed random input."""
    undo = install(m, ctx) if ctx is not None else None
    if ctx is not None:
        ctx.known_grid = (H, W)
        ctx.set_inputs(ctx.grid_ids(H, W), sigma=sigma)
        ctx.forward_id += 1
    rot = m.pos_embed([(1, H, W)], max_txt_seq_len=L, device=torch.device("cpu"))
    g = torch.Generator().manual_seed(5)
    img, txt = torch.randn(1, H * W, m.inner_dim, generator=g), torch.randn(1, L, m.inner_dim, generator=g)
    out = m.transformer_blocks[0].attn(hidden_states=img, encoder_hidden_states=txt, image_rotary_emb=rot)
    if undo:
        undo()
    return out


def flat(o):
    return torch.cat([o[0].flatten(), o[1].flatten()])


def test_positions_match_model_tables():
    m = model()
    for H, W, L in ((8, 8, 5), (6, 10, 3), (7, 9, 4), (83, 83, 7), (256, 254, 2)):
        vid, txt = m.pos_embed([(1, H, W)], max_txt_seq_len=L, device=torch.device("cpu"))
        assert torch.equal(qwen_freqs(m.pos_embed, image_ids(H, W)), vid)
        assert torch.equal(qwen_freqs(m.pos_embed, text_ids(H, W, L)), txt)


def test_real_rope_equals_complex_rope():
    m = model()
    vid, txt = m.pos_embed([(1, 6, 10)], max_txt_seq_len=4, device=torch.device("cpu"))
    f = torch.cat([txt, vid])
    x = torch.randn(1, f.shape[0], HEADS, DH, generator=torch.Generator().manual_seed(3))
    ref = apply_rotary_emb_qwen(x, f, use_real=False)
    assert torch.allclose(rope(x, rot_from_freqs(f)), ref, atol=1e-6)


def test_full_matches_stock():
    m = model()
    for H, W in ((8, 8), (6, 10), (7, 9)):
        x = inputs(H, W)
        ref = run(m, x)
        out = run(m, x, Ctx(mode="full"))
        assert torch.allclose(out, ref, atol=1e-5, rtol=0), (out - ref).abs().max()
    # padded text in full mode uses the joint mask exactly as the stock processor
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    x = inputs(8, 8, mask=mask)
    assert torch.allclose(run(m, x, Ctx(mode="full")), run(m, x), atol=1e-5)
    assert type(m.transformer_blocks[0].attn.processor).__name__ == "QwenDoubleStreamAttnProcessor2_0"


def test_local_full_window_equals_full_attention():
    m = model()
    H = W = 8
    ref = attn_outputs(m, None, H, W)
    for impl in ("flex", "gather"):
        for tile in (2, 8):
            ctx = Ctx(mode="sca", n0=4, tile=tile, window=8, local_impl=impl, pi=PiTable("const", value=1.0))
            out = attn_outputs(m, ctx, H, W)
            assert ctx.geometry()[0] == (2, 2)
            assert torch.allclose(out[0], ref[0], atol=1e-5)    # image stream: window covers the grid


def test_global_f1_equals_full():
    m = model()
    for H, W in ((8, 8), (7, 9)):
        x = inputs(H, W)
        ref = run(m, x)
        for pi in (0.0, 0.37):   # f = 1 (coarse ids = model ids) and a window covering the grid: both exact
            ctx = Ctx(mode="sca", n0=max(H, W), tile=1, window=max(H, W), pi=PiTable("const", value=pi))
            assert torch.allclose(run(m, x, ctx), ref, atol=1e-5)


def test_coarse_rope_is_native_geometry():
    """The global branch (coarse ids shifted by M - Mc, text at full-resolution ids) equals attention over a native
    Hc x Wc image: queries at their block's centred native ids, pooled keys at block centres, text at Mc + j."""
    m = model()
    for (H, W), f in (((8, 12), (2, 3)), ((12, 8), (4, 2)), ((16, 16), (4, 4))):
        L, h, D = L_TXT, HEADS, DH
        Hc, Wc = H // f[0], W // f[1]
        g = torch.Generator().manual_seed(7)
        q, k, v = (torch.randn(1, L + H * W, h, D, generator=g) for _ in range(3))
        nat_img = image_ids(Hc, Wc)                                       # native Hc x Wc image
        qn = nat_img.view(Hc, Wc, 3).repeat_interleave(f[0], 0).repeat_interleave(f[1], 1).reshape(-1, 3)
        cen = [(a * f[0] + f[0] // 2) * W + b * f[1] + f[1] // 2 for a in range(Hc) for b in range(Wc)]
        r_txt = rot_from_freqs(qwen_freqs(m.pos_embed, text_ids(Hc, Wc, L)))
        rq, rk = rot_from_freqs(qwen_freqs(m.pos_embed, qn)), rot_from_freqs(qwen_freqs(m.pos_embed, nat_img))
        qg = torch.cat([rope(q[:, :L], r_txt), rope(q[:, L:], rq)], 1)
        kg = torch.cat([rope(k[:, :L], r_txt), rope(k[:, [L + i for i in cen]], rk)], 1)
        vg = torch.cat([v[:, :L], v[:, [L + i for i in cen]]], 1)
        ref = torch.nn.functional.scaled_dot_product_attention(qg.transpose(1, 2), kg.transpose(1, 2),
                                                               vg.transpose(1, 2)).transpose(1, 2)
        ctx = Ctx(mode="sca")
        ctx.pos_embed = m.pos_embed
        ctx.known_grid = (H, W)
        ctx.set_inputs(ctx.grid_ids(H, W))
        full_txt = rot_from_freqs(qwen_freqs(m.pos_embed, text_ids(H, W, L)))   # what the model passes for text
        out = global_branch(q, k, v, L, (H, W), f, "center", full_txt, *ctx.coarse_rope(f, q.device))
        assert torch.allclose(out, ref, atol=2e-5), (out - ref).abs().max()
        assert ctx.coarse_shift(f) == max(H // 2, W // 2) - max(Hc // 2, Wc // 2)


def test_local_branch_model_ids():
    """Local branch: model ids for queries, window keys and text, against an explicit window mask."""
    m = model()
    H, W, L = 8, 12, L_TXT
    g = torch.Generator().manual_seed(9)
    q, k, v = (torch.randn(1, L + H * W, HEADS, DH, generator=g) for _ in range(3))
    vid, txt = m.pos_embed([(1, H, W)], max_txt_seq_len=L, device=torch.device("cpu"))
    rot = rot_from_freqs(torch.cat([txt, vid]))
    qr, kr = rope(q, rot), rope(k, rot)
    mask = torch.zeros(H * W, L + H * W, dtype=torch.bool)
    mask[:, :L] = True
    for r in range(H):
        for c in range(W):
            r0, c0 = min(max((r // 2) * 2 + 1 - 2, 0), H - 4), min(max((c // 3) * 3 + 1 - 3, 0), W - 6)
            for a in range(r0, r0 + 4):
                mask[r * W + c, L + a * W + c0:L + a * W + c0 + 6] = True
    ref = torch.nn.functional.scaled_dot_product_attention(
        qr[:, L:].transpose(1, 2), kr.transpose(1, 2), v.transpose(1, 2), attn_mask=mask).transpose(1, 2)
    assert torch.allclose(local_branch(qr, kr, v, L, (H, W), (2, 3), (4, 6)), ref, atol=1e-5)


def test_gate_and_mix():
    m, H, W = model(), 8, 8
    mk = lambda **kw: Ctx(mode="sca", n0=4, tile=2, window=4, local_impl="gather", **kw)
    glob = attn_outputs(m, mk(pi=PiTable("const", value=0.0)), H, W)
    gated = attn_outputs(m, mk(pi=PiTable("const", value=0.7), gate_sigma=0.4), H, W)      # sigma 0.5 > 0.4
    open_gate = attn_outputs(m, mk(pi=PiTable("const", value=0.7), gate_sigma=0.6), H, W)  # sigma 0.5 < 0.6
    plain = attn_outputs(m, mk(pi=PiTable("const", value=0.7)), H, W)
    loc = attn_outputs(m, mk(pi=PiTable("const", value=1.0)), H, W)
    assert torch.allclose(flat(glob), flat(gated), atol=1e-5)
    assert torch.allclose(flat(plain), flat(open_gate), atol=1e-5)
    assert not torch.allclose(flat(plain), flat(glob), atol=1e-4)
    assert torch.equal(glob[1], loc[1])              # text queries read O_glob only: identical for every pi
    # the mix is affine in pi before the affine output projection (to_out has a bias: weights sum to one)
    assert torch.allclose(plain[0], 0.3 * glob[0] + 0.7 * loc[0], atol=1e-5)
    t = PiTable("head", np.random.rand(2, HEADS, 3), np.array([0.2, 0.5, 0.8]))
    assert torch.isfinite(run(m, inputs(H, W), mk(pi=t))).all()
    bad = PiTable("head", np.random.rand(3, HEADS, 3), np.array([0.2, 0.5, 0.8]))
    try:
        run(m, inputs(H, W), mk(pi=bad))
        raise AssertionError("pi table with 3 layers accepted for a 2-layer model")
    except ValueError:
        pass


def test_flex_fused_gather_and_bridge_paths_agree():
    m, x = model(), inputs(8, 8)
    mk = lambda **kw: Ctx(mode="sca", n0=4, tile=2, window=4, pi=PiTable("const", value=0.6), **kw)
    ref = run(m, x, mk(local_impl="gather"))
    for fused in (False, True):
        out = run(m, x, mk(local_impl="flex", fused=fused))
        assert torch.allclose(out, ref, atol=1e-6), (fused, (out - ref).abs().max())
    # sca_v2 operator (bridge + window schedule + matched weights) runs and differs from the plain mix
    t = PiTable("head", np.random.rand(2, HEADS, 3), np.array([0.2, 0.5, 0.8]))
    v2 = mk(bridge_cells=2, bridge_k=2, win_sched=((0.6, 6),))
    v2.pi = t
    out = run(m, x, v2)
    assert torch.isfinite(out).all()


def test_text_length_change_between_calls():
    """CFG calls the transformer twice per step with different text lengths; caches are keyed per call."""
    m = model()
    ctx = Ctx(mode="sca", n0=4, tile=2, window=4, pi=PiTable("const", value=0.6))
    undo = install(m, ctx)
    try:
        with torch.no_grad():
            a1 = m(**inputs(8, 8, L=5))[0]
            b1 = m(**inputs(8, 8, L=2, seed=2))[0]
            a2 = m(**inputs(8, 8, L=5))[0]
    finally:
        undo()
    assert torch.equal(a1, a2) and a1.shape == b1.shape


def test_calib_matches_stock_and_pi_one_when_local_exact():
    m = model()
    x = inputs(8, 8)
    ref = run(m, x)
    ctx = Ctx(mode="calib", n0=4, tile=2, calib_scales=(2,), calib_window=8)
    ctx.calib = CalibStats(2, HEADS, (2,), bins=4, mass_queries=16)
    out = run(m, x, ctx)
    assert torch.allclose(out, ref, atol=1e-5)
    st = ctx.calib.state()
    S, b = st["stats"][0], ctx.calib.bin(0.7)
    A, B = S[0, ..., b], S[1, ..., b]
    assert S.shape[1:3] == (2, HEADS) and (A > 0).all() and np.allclose(B / A, 1.0, atol=1e-4)
    assert np.allclose(S[3, ..., b], 0, atol=1e-8)
    assert st["sigma"][1, b] == 1


def test_calib_geometry_at_1344():
    """84 x 84 tokens: ratio 2 and ratio 4 tile the grid (83 x 83 at 1328^2 would not)."""
    ctx = Ctx(mode="calib", n0=84, tile=8, calib_scales=(2, 4))
    ctx.grid = (84, 84)
    assert ctx.geometry(2) == ((2, 2), (4, 4), (42, 42))
    assert ctx.geometry(4) == ((4, 4), (2, 2), (21, 21))
    ctx.grid = (83, 83)
    assert ctx.geometry(2)[0] == (1, 1)        # prime grid: no pooling possible


def test_inference_geometry():
    from sca.attn import pool_factors
    assert pool_factors((332, 332), 83) == (4, 4)       # 5312^2: coarse grid = native 83 x 83
    assert pool_factors((256, 256), 83) == (4, 4)       # 4096^2: 64 x 64 coarse grid


def test_schedule_and_gate_rule():
    mu = mu_for(83 * 83)
    assert abs(mu - (0.5 + 0.4 * (6889 - 256) / (8192 - 256))) < 1e-9
    assert mu_for(256 * 256) == mu and mu_for(256 * 256, "default") > 3.5
    s = sigma_schedule(50, mu)
    assert len(s) == 50 and abs(s[0] - 1.0) < 1e-6 and s[-1] > 0
    g = gate_from_steps(s, 14)
    assert s[13] > g > s[14] and int((s > g).sum()) == 14
    ws = win_sched_from_steps(s, 14, (166, 145, 124, 104), per_step=2)
    assert [w for _, w in ws] == [166, 145, 124, 104]
    for i, (thr, _) in enumerate(ws):   # window i on steps 14 + 2i and 15 + 2i
        assert s[15 + 2 * i] > thr > s[16 + 2 * i]


# ----------------------------------------------------------------------------- script plumbing (tiny pipeline)

def tiny_pipeline():
    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImagePipeline
    from sca.qwen import scheduler_config
    torch.manual_seed(0)
    vae = AutoencoderKLQwenImage(base_dim=8, z_dim=4, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                                 temperal_downsample=[False, True, True], latents_mean=[0.0] * 4,
                                 latents_std=[1.0] * 4).eval()
    tr = model(layers=1)
    sch = FlowMatchEulerDiscreteScheduler(**scheduler_config())
    return QwenImagePipeline(scheduler=sch, vae=vae, text_encoder=None, tokenizer=None, transformer=tr)


def patch_loaders(monkeypatch, tmp_path):
    import sca.qwen as Q
    pipe = tiny_pipeline()
    monkeypatch.setattr(Q, "load_pipeline", lambda device="cpu": pipe)
    monkeypatch.setattr(Q, "snapshot", lambda: tmp_path / "snap0")
    calls = []

    def enc(prompts, device="cpu", **kw):
        calls.append(list(prompts))
        return [torch.randn(1, 2 + len(p) % 4, 24, generator=torch.Generator().manual_seed(len(p))) for p in prompts]
    monkeypatch.setattr(Q, "encode_prompts", enc)
    return pipe, calls


def write_prompts(path, n):
    import json
    path.write_text("".join(json.dumps({"id": f"p{i}", "prompt": f"prompt number {i} " + "x" * i}) + "\n"
                            for i in range(n)))
    return path


def test_sample_script_cpu(tmp_path, monkeypatch):
    import json
    from scripts import sample_qwen
    pipe, calls = patch_loaders(monkeypatch, tmp_path)
    prompts = write_prompts(tmp_path / "p.jsonl", 3)
    out = tmp_path / "out"
    base = ["--prompts", str(prompts), "--res", "128x128", "--steps", "3", "--device", "cpu", "--out", str(out),
            "--start", "0", "--end", "2"]
    sca_args = ["--method", "sca", "--pi-const", "0.5", "--n0", "4", "--tile", "2", "--gate-sigma", "0.9",
                "--label", "qwen_sca"]
    sample_qwen.main(base + sca_args)
    metas = sorted(out.glob("*.json"))
    assert len(metas) == 2 and len(list(out.glob("*.png"))) == 2
    m = json.loads(metas[0].read_text())
    assert m["complete"] and m["config"]["method"] == "qwen_sca" and m["config"]["res"] == "128x128"
    assert m["config"]["sca"]["pool_factors"] == [2, 2] and len(m["sigmas"]) == 3
    assert m["config"]["true_cfg_scale"] == 4.0 and m["config"]["negative_prompt"] == " "
    assert calls[0][-1] == " "                                  # negative prompt encoded with the chunk
    assert m["pi_trace"] and len(m["pi_trace"]) % 2 == 0       # two transformer calls per step (true CFG)
    n = len(calls)
    sample_qwen.main(base + sca_args)                           # resume: nothing regenerated
    assert calls[n] == [" "]
    sample_qwen.main(base + ["--method", "full", "--label", "qwen_full", "--out", str(tmp_path / "full")])
    assert len(list((tmp_path / "full").glob("*.png"))) == 2
    assert type(pipe.transformer.transformer_blocks[0].attn.processor).__name__ == "QwenDoubleStreamAttnProcessor2_0"


def test_calibrate_script_cpu(tmp_path, monkeypatch):
    import json
    from scripts import calibrate_qwen
    patch_loaders(monkeypatch, tmp_path)
    prompts = write_prompts(tmp_path / "c.jsonl", 4)
    out = tmp_path / "calib" / "t.npz"
    base = ["--prompts", str(prompts), "--count", "4", "--res", "128x128", "--steps", "3", "--n0", "8",
            "--tile", "2", "--mass-queries", "8", "--device", "cpu", "--out", str(out)]
    calibrate_qwen.main(base + ["--start", "0", "--end", "2", "--no-merge"])
    assert not out.exists() and len(list(out.parent.glob("t_parts/*.npz"))) == 2
    calibrate_qwen.main(base + ["--start", "2", "--end", "4", "--no-merge"])
    calibrate_qwen.main(base + ["--report-only"])
    rep = json.loads(out.with_suffix(".json").read_text())
    assert rep["n_prompts"] == 4 and set(rep["scales"]) == {"2", "4"}
    t = PiTable.load(out, "head", target_scale=4)
    assert t.scale == 4 and t(0.5).shape == (1, HEADS)
