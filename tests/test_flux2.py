"""CPU tests for the FLUX.2 port on a tiny random Flux2Transformer2DModel (float32)."""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusers import Flux2Transformer2DModel  # noqa: E402

from sca.attn import PiTable, global_branch, local_branch, rope  # noqa: E402
from sca.calib import CalibStats  # noqa: E402
from sca.flux2 import (Flux2SCAContext as Ctx, empirical_mu, flops, gate_from_steps, install,  # noqa: E402
                       sigma_schedule)

L_TXT, HEADS, DH = 6, 2, 16


def model():
    torch.manual_seed(0)
    return Flux2Transformer2DModel(patch_size=1, in_channels=4, num_layers=2, num_single_layers=2,
                                   attention_head_dim=DH, num_attention_heads=HEADS, joint_attention_dim=24,
                                   timestep_guidance_channels=32, mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4),
                                   guidance_embeds=True).eval()


def img_ids(H, W):
    ids = torch.zeros(H, W, 4, dtype=torch.long)  # (t, h, w, l) as Flux2Pipeline._prepare_latent_ids
    ids[..., 1] = torch.arange(H)[:, None]
    ids[..., 2] = torch.arange(W)[None, :]
    return ids.reshape(-1, 4)


def txt_ids(L=L_TXT):
    ids = torch.zeros(L, 4, dtype=torch.long)  # (0, 0, 0, j) as Flux2Pipeline._prepare_text_ids
    ids[:, 3] = torch.arange(L)
    return ids


def inputs(H, W, seed=1):
    g = torch.Generator().manual_seed(seed)
    return dict(hidden_states=torch.randn(1, H * W, 4, generator=g),
                encoder_hidden_states=torch.randn(1, L_TXT, 24, generator=g),
                timestep=torch.tensor([0.7]), guidance=torch.tensor([4.0]),
                img_ids=img_ids(H, W)[None], txt_ids=txt_ids()[None], return_dict=False)


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
def attn_outputs(m, ctx, H, W, sigma=0.5):
    """Per-module outputs (double: (img, txt); single: [txt|img]) for a fixed random input."""
    undo = install(m, ctx) if ctx is not None else None
    if ctx is not None:
        ctx.set_inputs(img_ids(H, W), sigma=sigma)
    rot = m.pos_embed(torch.cat([txt_ids(), img_ids(H, W)]))
    g = torch.Generator().manual_seed(5)
    img, txt = torch.randn(1, H * W, m.inner_dim, generator=g), torch.randn(1, L_TXT, m.inner_dim, generator=g)
    outs = [m.transformer_blocks[0].attn(img, txt, image_rotary_emb=rot),
            m.single_transformer_blocks[0].attn(torch.cat([txt, img], 1), image_rotary_emb=rot)]
    if undo:
        undo()
    return outs


def flat(o):
    return torch.cat([o[0][0].flatten(), o[0][1].flatten(), o[1].flatten()])


def test_full_matches_stock():
    m = model()
    for H, W in ((8, 8), (6, 10)):
        x = inputs(H, W)
        ref = run(m, x)
        out = run(m, x, Ctx(mode="full"))
        assert torch.allclose(out, ref, atol=1e-5, rtol=0), (out - ref).abs().max()
    # stock processors are restored by uninstall
    assert type(m.transformer_blocks[0].attn.processor).__name__ == "Flux2AttnProcessor"
    assert type(m.single_transformer_blocks[0].attn.processor).__name__ == "Flux2ParallelSelfAttnProcessor"


def test_module_outputs_full_mode():
    m = model()
    ref = attn_outputs(m, None, 8, 8)
    out = attn_outputs(m, Ctx(mode="full"), 8, 8)
    assert torch.allclose(flat(out), flat(ref), atol=1e-5)


def test_local_full_window_equals_full_attention():
    m = model()
    H = W = 8
    ref = attn_outputs(m, None, H, W)
    for impl in ("flex", "gather"):
        for tile in (2, 8):
            ctx = Ctx(mode="sca", n0=4, tile=tile, window=8, local_impl=impl, pi=PiTable("const", value=1.0))
            out = attn_outputs(m, ctx, H, W)
            assert ctx.geometry()[0] == (2, 2)
            assert torch.allclose(out[0][0], ref[0][0], atol=1e-5)                  # double, image stream
            assert torch.allclose(out[1][:, L_TXT:], ref[1][:, L_TXT:], atol=1e-5)  # single, image part


def test_global_f1_equals_full():
    m = model()
    x = inputs(8, 8)
    ref = run(m, x)
    for pi in (0.0, 0.37):  # f = 1 and a window covering the grid make both branches exact
        ctx = Ctx(mode="sca", n0=8, tile=4, window=8, pi=PiTable("const", value=pi))
        assert torch.allclose(run(m, x, ctx), ref, atol=1e-5)
    ctx = Ctx(mode="sca", n0=8, tile=4, window=2, pi=PiTable("const", value=0.0))
    assert torch.allclose(run(m, x, ctx), ref, atol=1e-5)  # pi = 0: local window size irrelevant


def test_coarse_rope_4d_bruteforce():
    """Global branch with 4-D ids: pooled keys at block centres with block-index (row, col),
    queries at floor(id / f), t and l axes untouched, text keys always present."""
    m = model()
    H, W, L, h, D = 8, 12, L_TXT, HEADS, DH
    g = torch.Generator().manual_seed(7)
    q, k, v = (torch.randn(1, L + H * W, h, D, generator=g) for _ in range(3))
    ids = img_ids(H, W)
    f = (2, 3)
    q_ids = ids.clone()
    q_ids[:, 1], q_ids[:, 2] = ids[:, 1] // 2, ids[:, 2] // 3
    cen = [(a * 2 + 1) * W + b * 3 + 1 for a in range(H // 2) for b in range(W // 3)]
    k_ids = torch.tensor([[0, a, b, 0] for a in range(H // 2) for b in range(W // 3)])
    txt = m.pos_embed(txt_ids())
    rq, rk = m.pos_embed(q_ids), m.pos_embed(k_ids)
    qg = torch.cat([rope(q[:, :L], txt), rope(q[:, L:], rq)], 1)
    kg = torch.cat([rope(k[:, :L], txt), rope(k[:, [L + i for i in cen]], rk)], 1)
    vg = torch.cat([v[:, :L], v[:, [L + i for i in cen]]], 1)
    ref = torch.nn.functional.scaled_dot_product_attention(qg.transpose(1, 2), kg.transpose(1, 2),
                                                           vg.transpose(1, 2)).transpose(1, 2)
    ctx = Ctx(mode="sca")
    ctx.pos_embed = m.pos_embed
    ctx.set_inputs(ids)
    assert ctx.grid == (H, W)
    out = global_branch(q, k, v, L, (H, W), f, "center", txt, *ctx.coarse_rope(f, q.device))
    assert torch.allclose(out, ref, atol=1e-5)
    # local branch (original 4-D RoPE) against an explicit window mask
    rot = m.pos_embed(torch.cat([txt_ids(), ids]))
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
    # text queries take O_glob only: identical for every pi
    assert torch.equal(glob[0][1], loc[0][1]) and torch.equal(glob[1][:, :L_TXT], loc[1][:, :L_TXT])
    # the mix is affine in pi before the (linear) output projections: 0.7 = 0.3 * glob + 0.7 * loc
    # holds for the double-block image stream (to_out is linear; no bias in FLUX.2)
    mix = 0.3 * glob[0][0] + 0.7 * loc[0][0]
    assert torch.allclose(plain[0][0], mix, atol=1e-5)
    # per-head tables of the right shape (2 double + 2 single layers) run, wrong shapes fail
    t = PiTable("head", np.random.rand(4, HEADS, 3), np.array([0.2, 0.5, 0.8]))
    out = run(m, inputs(H, W), mk(pi=t))
    assert torch.isfinite(out).all()
    bad = PiTable("head", np.random.rand(3, HEADS, 3), np.array([0.2, 0.5, 0.8]))
    try:
        run(m, inputs(H, W), mk(pi=bad))
        raise AssertionError("pi table with 3 layers accepted for a 4-layer model")
    except ValueError:
        pass


def test_flex_fused_and_gather_agree():
    m, x = model(), inputs(8, 8)
    mk = lambda **kw: Ctx(mode="sca", n0=4, tile=2, window=4, pi=PiTable("const", value=0.6), **kw)
    ref = run(m, x, mk(local_impl="gather"))
    for fused in (False, True):
        out = run(m, x, mk(local_impl="flex", fused=fused))
        assert torch.allclose(out, ref, atol=1e-6), (fused, (out - ref).abs().max())


def test_calib_matches_stock_and_pi_one_when_local_exact():
    m = model()
    x = inputs(8, 8)
    ref = run(m, x)
    ctx = Ctx(mode="calib", n0=4, tile=2, calib_scales=(2,), calib_window=8)
    ctx.calib = CalibStats(4, HEADS, (2,), bins=4, mass_queries=16)
    out = run(m, x, ctx)
    assert torch.allclose(out, ref, atol=1e-5)  # sampling unchanged in calib mode
    st = ctx.calib.state()
    S, b = st["stats"][0], ctx.calib.bin(0.7)
    A, B = S[0, ..., b], S[1, ..., b]
    assert S.shape[1:3] == (4, HEADS) and (A > 0).all() and np.allclose(B / A, 1.0, atol=1e-4)
    assert np.allclose(S[3, ..., b], 0, atol=1e-8)
    assert st["sigma"][1, b] == 1  # one transformer call recorded in the sigma bin of t = 0.7


def test_schedule_and_gate_rule():
    mu = empirical_mu(4096, 28)
    assert abs(mu - 2.1514) < 1e-3
    assert empirical_mu(256 * 256, 28) == mu                  # native: capped at the 64 x 64 grid
    assert empirical_mu(256 * 256, 28, "default") > 11       # pipeline extrapolation at 4096^2
    s = sigma_schedule(28, mu)
    assert len(s) == 28 and s[0] == 1.0
    g = gate_from_steps(s, 8)
    assert s[7] > g > s[8] and abs(g - 0.959) < 1e-9
    assert int((s > g).sum()) == 8


def test_flops_flux2():
    r = flops(4096, 4096)
    assert r["pool"] == [4, 4] and r["attn_flop_ratio"] > 5


# ----------------------------------------------------------------------------- script plumbing (tiny pipeline)

def tiny_pipeline():
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2Pipeline
    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(block_out_channels=(4, 4, 4, 4), layers_per_block=1, latent_channels=2,
                             norm_num_groups=2, mid_block_add_attention=False).eval()
    tr = Flux2Transformer2DModel(patch_size=1, in_channels=8, num_layers=1, num_single_layers=1,
                                 attention_head_dim=DH, num_attention_heads=HEADS, joint_attention_dim=24,
                                 timestep_guidance_channels=32, axes_dims_rope=(4, 4, 4, 4)).eval()
    sch = FlowMatchEulerDiscreteScheduler(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=4096,
                                          max_shift=1.15, shift=3.0, time_shift_type="exponential",
                                          use_dynamic_shifting=True)
    return Flux2Pipeline(scheduler=sch, vae=vae, text_encoder=None, tokenizer=None, transformer=tr)


def patch_loaders(monkeypatch, tmp_path):
    import sca.flux2 as F2
    pipe = tiny_pipeline()
    monkeypatch.setattr(F2, "load_pipeline", lambda device="cpu": pipe)
    monkeypatch.setattr(F2, "snapshot", lambda: tmp_path / "snap0")
    calls = []

    def enc(prompts, device="cpu", **kw):
        calls.append(list(prompts))
        return [torch.randn(1, L_TXT, 24, generator=torch.Generator().manual_seed(len(p))) for p in prompts]
    monkeypatch.setattr(F2, "encode_prompts", enc)
    return pipe, calls


def write_prompts(path, n):
    import json
    path.write_text("".join(json.dumps({"id": f"p{i}", "prompt": f"prompt number {i} " + "x" * i}) + "\n"
                            for i in range(n)))
    return path


def test_sample_script_cpu(tmp_path, monkeypatch):
    import json
    from scripts import sample_flux2
    pipe, calls = patch_loaders(monkeypatch, tmp_path)
    prompts = write_prompts(tmp_path / "p.jsonl", 3)
    out = tmp_path / "out"
    base = ["--prompts", str(prompts), "--res", "128x128", "--steps", "3", "--device", "cpu", "--out", str(out),
            "--start", "0", "--end", "2"]
    sample_flux2.main(base + ["--method", "sca", "--pi-const", "0.5", "--n0", "4", "--tile", "2",
                              "--gate-sigma", "0.9", "--label", "flux2_sca"])
    metas = sorted(out.glob("*.json"))
    assert len(metas) == 2 and len(list(out.glob("*.png"))) == 2
    m = json.loads(metas[0].read_text())
    assert m["complete"] and m["config"]["method"] == "flux2_sca" and m["config"]["res"] == "128x128"
    assert m["config"]["sca"]["pool_factors"] == [2, 2] and len(m["sigmas"]) == 3
    assert m["pi_trace"] and m["pi_trace"][0]["step"] >= 1  # step 0 has sigma 1 > 0.9: gated, no pi lookup
    assert abs(m["config"]["mu"] - empirical_mu(64, 3)) < 1e-12
    # resume: nothing re-encoded or regenerated
    n = len(calls)
    sample_flux2.main(base + ["--method", "sca", "--pi-const", "0.5", "--n0", "4", "--tile", "2",
                              "--gate-sigma", "0.9", "--label", "flux2_sca"])
    assert calls[n] == []
    # full attention through the same script
    sample_flux2.main(base + ["--method", "full", "--label", "flux2_full", "--out", str(tmp_path / "full")])
    assert len(list((tmp_path / "full").glob("*.png"))) == 2
    # processors restored after the run
    assert type(pipe.transformer.transformer_blocks[0].attn.processor).__name__ == "Flux2AttnProcessor"


def test_calibrate_script_cpu(tmp_path, monkeypatch):
    import json
    from scripts import calibrate_flux2
    from sca.attn import PiTable
    patch_loaders(monkeypatch, tmp_path)
    prompts = write_prompts(tmp_path / "c.jsonl", 4)
    out = tmp_path / "calib" / "t.npz"
    base = ["--prompts", str(prompts), "--count", "4", "--res", "128x128", "--steps", "3", "--n0", "4",
            "--tile", "2", "--mass-queries", "8", "--device", "cpu", "--out", str(out)]
    calibrate_flux2.main(base + ["--start", "0", "--end", "2", "--no-merge"])
    assert not out.exists() and len(list(out.parent.glob("t_parts/*.npz"))) == 2
    try:
        calibrate_flux2.main(base + ["--report-only"])
        raise AssertionError("merged with missing parts")
    except SystemExit:
        pass
    calibrate_flux2.main(base + ["--start", "2", "--end", "4", "--no-merge"])
    calibrate_flux2.main(base + ["--report-only"])
    rep = json.loads(out.with_suffix(".json").read_text())
    assert rep["n_prompts"] == 4 and set(rep["scales"]) == {"2", "4"}
    t = PiTable.load(out, "head", target_scale=4)
    assert t.scale == 4 and t(0.5).shape == (2, HEADS)  # 1 double + 1 single layer
