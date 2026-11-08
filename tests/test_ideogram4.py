"""CPU tests for the Ideogram 4 port on a tiny random Ideogram4Transformer2DModel (pinned SEGA copy, float32).

head_dim stays 256 (128 frequencies) because the interleaved M-RoPE combs (mrope_section 24, 20, 20) index frequencies
up to 59; the model is made small through 2 heads, 2 blocks and narrow MLP / text widths."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sca.attn import PiTable, global_branch, rope  # noqa: E402
from sca.calib import CalibStats  # noqa: E402
from sca.ideogram4 import (IMAGE_POSITION_OFFSET as OFF, Ideogram4SCAContext as Ctx, coarse_positions,  # noqa: E402
                           gate_from_steps, image_positions, install, packed_layout, pair_perm, rope_rotate_half,
                           sega_modules, sigma_schedule, text_positions, to_interleaved, valid_caption,
                           win_sched_from_steps)

T, P = sega_modules()
HEADS, DH, CIN, LLM = 2, 256, 8, 16


def model(layers=2, seed=0):
    torch.manual_seed(seed)
    return T.Ideogram4Transformer2DModel(in_channels=CIN, num_layers=layers, attention_head_dim=DH,
                                         num_attention_heads=HEADS, intermediate_size=64, adaln_dim=32,
                                         llm_features_dim=LLM).eval()


def inputs(H, W, L=5, seed=1, t=0.3):
    """Keyword inputs of one packed call [text (L)][image (H x W)] as the pipeline builds them (L = 0: the
    unconditional pass, no text features)."""
    g = torch.Generator().manual_seed(seed)
    N = H * W
    pos = torch.cat([text_positions(L), image_positions(H, W)])[None]
    ind = torch.tensor([T.LLM_TOKEN_INDICATOR] * L + [T.OUTPUT_IMAGE_INDICATOR] * N)[None]
    hs = torch.cat([torch.zeros(1, L, CIN), torch.randn(1, N, CIN, generator=g)], 1)
    enc = torch.randn(1, L, LLM, generator=g) if L else None
    return dict(hidden_states=hs, timestep=torch.tensor([t]), encoder_hidden_states=enc, position_ids=pos,
                segment_ids=torch.ones(1, L + N, dtype=torch.long), indicator=ind, return_dict=False)


@torch.no_grad()
def run(m, x, ctx=None):
    if ctx is None:
        return m(**x)[0]
    undo = install(m, ctx)
    try:
        return m(**x)[0]
    finally:
        undo()


def logits(q, k):
    return torch.einsum("bshd,bthd->bhst", q, k)


def test_pair_perm_turns_rotate_half_into_interleaved_rope():
    m = model()
    pos = torch.cat([text_positions(7), image_positions(5, 6)])[None]
    cos, sin = m.rotary_emb(pos)
    g = torch.Generator().manual_seed(3)
    q, k = (torch.randn(1, pos.shape[1], HEADS, DH, generator=g) for _ in range(2))
    ref = logits(rope_rotate_half(q, cos, sin), rope_rotate_half(k, cos, sin))
    p, rot = pair_perm(DH), to_interleaved(cos[0], sin[0])
    out = logits(rope(q[..., p], rot), rope(k[..., p], rot))
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-5), (out - ref).abs().max()
    # the rotated vectors themselves agree up to the permutation
    assert torch.allclose(rope(q[..., p], rot), rope_rotate_half(q, cos, sin)[..., p], atol=1e-5)
    assert sorted(p.tolist()) == list(range(DH)) and p[:4].tolist() == [0, 128, 1, 129]


def test_positions_match_pipeline_layout():
    for lengths, (H, W) in (([5], (4, 6)), ([9], (8, 8))):
        pos, seg, ind = P.Ideogram4Pipeline._prepare_ids(lengths, H, W, 12, torch.device("cpu"))
        first = int((seg != T.SEQUENCE_PADDING_INDICATOR).any(0).long().argmax())   # the pipeline's trim
        pos, seg, ind = pos[:, first:], seg[:, first:], ind[:, first:]
        L = lengths[0]
        assert torch.equal(pos[0], torch.cat([text_positions(L), image_positions(H, W)]))
        assert packed_layout(pos, ind, seg) == (L, H, W)
        n = H * W   # unconditional branch: image-only slices
        assert packed_layout(pos[:, -n:], ind[:, -n:], seg[:, -n:]) == (0, H, W)


def test_packed_layout_rejects_padding_and_mixed_segments():
    pos, seg, ind = P.Ideogram4Pipeline._prepare_ids([3, 5], 4, 4, 8, torch.device("cpu"))
    with pytest.raises(NotImplementedError):
        packed_layout(pos, ind, seg)          # two prompts of different length: padding, two segments
    pos, seg, ind = P.Ideogram4Pipeline._prepare_ids([3], 4, 4, 8, torch.device("cpu"))
    with pytest.raises(ValueError):
        packed_layout(pos, ind, torch.ones_like(seg))   # untrimmed left padding inside one segment
    bad = pos[:, 5:].clone()
    bad[0, 4, 1] += 1                                # a non-grid image position
    with pytest.raises(ValueError):
        packed_layout(bad, ind[:, 5:], seg[:, 5:])


def test_full_mode_is_bit_identical_to_stock():
    m = model()
    for H, W, L in ((6, 6, 5), (4, 8, 3), (5, 7, 0)):
        x = inputs(H, W, L)
        ref = run(m, x)
        out = run(m, x, Ctx(mode="full"))
        assert torch.equal(out, ref)
    assert type(m.layers[0].attention.processor).__name__ == "Ideogram4AttnProcessor"   # uninstalled


@pytest.mark.parametrize("L", [5, 0])
def test_sca_at_native_extent_equals_full_attention(L):
    """Mix disabled: with f = 1 and a window covering the grid both branches are exact attention, so SCA reproduces
    the stock model for any pi (global only, local only, a blend), gather and flex local branches."""
    m = model()
    for H, W in ((6, 6), (5, 7)):
        x = inputs(H, W, L)
        ref = run(m, x)
        n = max(H, W)
        for impl in ("gather", "flex"):
            for pi in (0.0, 1.0, 0.37):
                ctx = Ctx(mode="sca", n0=n, tile=1, window=n, local_impl=impl, pi=PiTable("const", value=pi))
                out = run(m, x, ctx)
                assert ctx.geometry()[0] == (1, 1)
                assert torch.allclose(out, ref, atol=2e-5, rtol=1e-4), (impl, pi, (out - ref).abs().max())


def test_coarse_positions_change_only_the_hw_combs():
    m = model()
    H, W, f = 8, 12, (2, 3)
    q_ids, k_ids = coarse_positions(H, W, f)
    true = image_positions(H, W)
    assert torch.equal(q_ids[:, 0], true[:, 0]) and int(q_ids[:, 0].unique()) == OFF   # t keeps the 65536 offset
    assert torch.equal(q_ids[:, 1:] - OFF, torch.stack([true[:, 1] - OFF, true[:, 2] - OFF], 1) // torch.tensor(f))
    Hc, Wc = H // f[0], W // f[1]
    assert torch.equal(k_ids, image_positions(Hc, Wc))                                  # a native Hc x Wc image
    cq, sq = m.rotary_emb(q_ids[None])
    ct, st = m.rotary_emb(true[None])
    axis = m.rotary_emb.axis_of_freq                                                    # 0 = t, 1 = h, 2 = w
    t_comb = torch.cat([axis == 0, axis == 0])                                          # rotate-half layout
    assert torch.equal(cq[0][:, t_comb], ct[0][:, t_comb]) and torch.equal(sq[0][:, t_comb], st[0][:, t_comb])
    moved = (true[:, 1] - OFF) % f[0] + (true[:, 2] - OFF) % f[1] > 0
    assert not torch.allclose(cq[0][moved][:, ~t_comb], ct[0][moved][:, ~t_comb])     # h / w combs do move
    assert int((axis == 1).sum()) == 20 and int((axis == 2).sum()) == 20 and int((axis == 0).sum()) == 88


def test_global_branch_is_native_geometry():
    """The global branch (image queries at their block's coarse id, pooled keys at block centres, text at (j, j, j))
    equals attention over a native Hc x Wc image built from the model's own RoPE at the native positions: every
    text-image and image-image phase is that of training."""
    m = model()
    p = pair_perm(DH)
    for (H, W), f, L in (((8, 12), (2, 3), 5), ((12, 8), (4, 2), 3), ((16, 16), (4, 4), 0)):
        Hc, Wc = H // f[0], W // f[1]
        g = torch.Generator().manual_seed(7)
        q, k, v = (torch.randn(1, L + H * W, HEADS, DH, generator=g) for _ in range(3))
        nat = image_positions(Hc, Wc)
        qn = nat.view(Hc, Wc, 3).repeat_interleave(f[0], 0).repeat_interleave(f[1], 1).reshape(-1, 3)
        cen = [L + (a * f[0] + f[0] // 2) * W + b * f[1] + f[1] // 2 for a in range(Hc) for b in range(Wc)]
        tp = text_positions(L)
        cq, sq = m.rotary_emb(torch.cat([tp, qn])[None])                                 # model layout, fp32
        ck, sk = m.rotary_emb(torch.cat([tp, nat])[None])
        qr = rope_rotate_half(q, cq, sq)
        kr = rope_rotate_half(torch.cat([k[:, :L], k[:, cen]], 1), ck, sk)
        vr = torch.cat([v[:, :L], v[:, cen]], 1)
        ref = torch.nn.functional.scaled_dot_product_attention(qr.transpose(1, 2), kr.transpose(1, 2),
                                                               vr.transpose(1, 2)).transpose(1, 2)
        ctx = Ctx(mode="sca")
        ctx.pos_embed = m.rotary_emb
        ctx.rope_dtype = torch.float32
        ctx.known_grid = (H, W)
        ctx.set_inputs(ctx.grid_ids(H, W))
        ct, st = m.rotary_emb(torch.cat([tp, image_positions(H, W)])[None])             # what the model passes
        txt = to_interleaved(ct[0, :L], st[0, :L])
        out = global_branch(q[..., p], k[..., p], v, L, (H, W), f, "center", txt, *ctx.coarse_rope(f, q.device))
        assert torch.allclose(out, ref, atol=2e-5), (out - ref).abs().max()


def test_gate_mix_and_unconditional_pass():
    m = model()
    H = W = 8
    mk = lambda **kw: Ctx(mode="sca", n0=4, tile=2, window=4, local_impl="gather", **kw)
    for L in (5, 0):
        x = inputs(H, W, L, t=0.3)                     # sigma = 1 - t = 0.7
        glob = run(m, x, mk(pi=PiTable("const", value=0.0)))
        gated = run(m, x, mk(pi=PiTable("const", value=1.0), gate_sigma=0.6))
        plain = run(m, x, mk(pi=PiTable("const", value=0.5)))
        assert torch.allclose(glob, gated, atol=1e-5)
        assert not torch.allclose(plain, glob, atol=1e-4)
        assert torch.isfinite(plain).all()


def test_flex_gather_and_bridge_paths_agree():
    m = model()
    x = inputs(8, 8, 5)
    mk = lambda **kw: Ctx(mode="sca", n0=4, tile=2, window=4, pi=PiTable("const", value=0.6), **kw)
    ref = run(m, x, mk(local_impl="gather"))
    out = run(m, x, mk(local_impl="flex"))
    assert torch.allclose(out, ref, atol=2e-5), (out - ref).abs().max()
    for L in (5, 0):
        y = run(m, inputs(8, 8, L), mk(local_impl="flex", bridge_cells=2, bridge_k=2))
        assert torch.isfinite(y).all()


def test_calib_matches_stock_and_fits_pi_one_when_local_exact():
    m = model()
    H = W = 8
    x = inputs(H, W, 5)
    ref = run(m, x)
    ctx = Ctx(mode="calib", n0=8, tile=2, calib_scales=(2,), calib_window=8, local_impl="gather")
    ctx.calib = CalibStats(2, HEADS, (2,), 14, mass_queries=4)
    out = run(m, x, ctx)
    assert torch.allclose(out, ref, atol=2e-5)           # calib returns exact attention
    S = ctx.calib.stats.numpy()[0]                         # window 8 covers the grid: O_loc = O_full
    b = ctx.calib.bin(0.7)
    A, B = S[0, ..., b], S[1, ..., b]
    assert S.shape[1:3] == (2, HEADS) and (A > 0).all() and np.allclose(B / A, 1.0, atol=1e-4)
    assert np.allclose(S[3, ..., b], 0, atol=1e-6)


def test_schedule_gate_and_window_rule():
    s = sigma_schedule(48, 8192, 8192)
    assert s.shape == (48,) and np.all(np.diff(s) < 0) and 0.99 < s[0] <= 1.0
    gate = gate_from_steps(s, 14)
    assert (s > gate).sum() == 14
    sched = win_sched_from_steps(s, 14, (512, 448, 384, 320), per_step=2)
    win = [next((w for t, w in sched if sig > t), 256) for sig in s[14:24]]
    assert win == [512, 512, 448, 448, 384, 384, 320, 320, 256, 256]
    # the caption sees only the aspect ratio, so 4096^2 and 8192^2 share one schedule shape but not one mu
    assert not np.allclose(sigma_schedule(48, 4096, 4096), s)


def test_valid_caption():
    good = ('{"high_level_description":"A watch.","compositional_deconstruction":{"background":"Dark wood.",'
            '"elements":[{"type":"obj","desc":"A steel watch."}]}}')
    assert valid_caption(good)
    assert not valid_caption(good[:-1]) and not valid_caption('{"high_level_description":"x"}')


# ----------------------------------------------------------------------------- scripts end to end on a tiny pipeline

def tiny_pipeline(device="cpu", guidance_mode="dual"):
    """The pinned SEGA Ideogram4Pipeline with tiny random components: 2-block transformers, an 8-channel
    AutoencoderKLFlux2 (16 px per token, as the real one) and an encode_prompt that packs random text features."""
    from diffusers.models.autoencoders import AutoencoderKLFlux2
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), latent_channels=32, norm_num_groups=4,
                             layers_per_block=1).eval()
    tf = lambda seed: (torch.manual_seed(seed), T.Ideogram4Transformer2DModel(
        in_channels=128, num_layers=2, attention_head_dim=DH, num_attention_heads=HEADS, intermediate_size=64,
        adaln_dim=32, llm_features_dim=LLM).eval())[1]
    pipe = P.Ideogram4Pipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae, text_encoder=None,
                               tokenizer=None, transformer=tf(1), unconditional_transformer=tf(2))

    def encode_prompt(prompt, grid_h, grid_w, max_sequence_length, device):
        n = 3 + len(prompt) % 5
        g = torch.Generator().manual_seed(len(prompt))
        feats = torch.zeros(1, max_sequence_length, LLM)
        feats[:, -n:] = torch.randn(1, n, LLM, generator=g)
        return (feats, *P.Ideogram4Pipeline._prepare_ids([n], grid_h, grid_w, max_sequence_length, device))
    pipe.encode_prompt = encode_prompt
    return pipe


def test_scripts_cpu(tmp_path, monkeypatch):
    import json
    import sca.ideogram4 as I
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import calibrate_ideogram4
    import sample_ideogram4
    monkeypatch.setattr(I, "load_pipeline", tiny_pipeline)
    monkeypatch.setattr(I, "snapshot", lambda repo=I.REPO: tmp_path / "snapshot0")
    cap = ('{"high_level_description":"A thing.","compositional_deconstruction":{"background":"Plain.",'
           '"elements":[{"type":"obj","desc":"A thing."}]}}')
    prompts, captions = tmp_path / "p.jsonl", tmp_path / "c.jsonl"
    rows = [{"id": f"x{i}", "prompt": f"prompt number {i}" + "!" * i} for i in range(3)]
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    captions.write_text("".join(json.dumps({"id": r["id"], "prompt": r["prompt"], "valid": True, "caption": cap,
                                            "caption_sha256": "0" * 64}) + "\n" for r in rows))
    stem = tmp_path / "calib" / "tiny"
    calibrate_ideogram4.main(["--prompts", str(prompts), "--captions", str(captions), "--count", "2",
                              "--res", "256x256", "--steps", "4", "--out", str(stem), "--device", "cpu",
                              "--local-impl", "gather", "--mass-queries", "4", "--tile", "2"])
    for name in ("cond", "uncond"):
        z = np.load(f"{stem}_{name}.npz")
        assert z["s2_pi_head"].shape[:2] == (2, HEADS) and int(z["n_prompts"]) == 2
        assert json.loads(Path(f"{stem}_{name}.json").read_text())["transformer"] == name
    out = tmp_path / "out"
    base = ["--prompts", str(prompts), "--captions", str(captions), "--res", "256x256", "--steps", "4",
            "--device", "cpu", "--end", "2"]
    sca = ["--method", "sca", "--pi-table", f"{stem}_cond.npz", "--pi-table-uncond", f"{stem}_uncond.npz",
           "--n0", "8", "--tile", "2", "--gate-sigma", "0.9", "--bridge-cells", "2", "--bridge-k", "2",
           "--win-sched", "0.6:16,0.5:12", "--win-pi-match", "--win-pi-interp"]
    sample_ideogram4.main(base + sca + ["--out", str(out / "sca")])
    sample_ideogram4.main(base + ["--method", "sega", "--train-grid", "8", "--out", str(out / "sega")])
    for d in ("sca", "sega"):
        metas = sorted((out / d).glob("*.json"))
        assert len(metas) == 2
        m = json.loads(metas[0].read_text())
        assert m["complete"] and m["config"]["res"] == "256x256" and m["caption"] == cap
        assert Image.open(m["image"]).size == (256, 256)
    m = json.loads(sorted((out / "sca").glob("*.json"))[0].read_text())
    assert set(m["pi_trace"]) == {"cond", "uncond"} and m["config"]["sca"]["cond"]["pool_factors"] == [2, 2]
    assert json.loads(sorted((out / "sega").glob("*.json"))[0].read_text())["config"]["rope_scaling"]["method"] == "sega"
    sample_ideogram4.main(base + sca + ["--out", str(out / "sca")])   # resume: all present


def test_eval_driver_plumbing_cpu(tmp_path):
    """scripts/ideogram4_eval.py without the GPU metrics: links complete sets, pairs SCA with each baseline on common
    prompts only, writes summaries, timing (shared flag from the GPU logs) and SHA256SUMS."""
    import csv
    import json
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import ideogram4_eval as EV
    from PIL import Image
    ids = [f"sc_x{i}" for i in range(4)]
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("".join(json.dumps({"id": i, "prompt": f"prompt {i}"}) + "\n" for i in ids))
    out, E, R = tmp_path / "out", tmp_path / "eval", tmp_path / "results"
    have = {"sca": ids, "full": ids[:3], "sega": ids, "ntk": ids[:2]}
    g = np.random.default_rng(0)
    rows = {"extra": [], "base": [], "musiq": []}
    for m, keep in have.items():
        d = out / f"ideogram4_{m}" / "8192x8192"
        d.mkdir(parents=True)
        for i in keep:
            Image.new("RGB", (8, 8)).save(d / f"{i}.png")
            (d / f"{i}.json").write_text(json.dumps({"complete": True, "seconds": 100.0 + len(m),
                                                     "peak_memory_bytes": 5e10, "gpu": "fake",
                                                     "config": {"method": m, "prompt_id": i, "res": "8192x8192",
                                                                "seed": 0}}))
            base = {"method": m, "res": "8192x8192", "prompt_id": i, "seed": 0, "path": "x"}
            rows["extra"].append({**base, "ir": g.normal() + (m == "sca"), "hpsv21": g.normal()})
            rows["base"].append({**base, "clip": g.normal(), "pickscore": g.normal()})
            rows["musiq"].append({**base, "musiq_4096": g.normal(), "musiq_crop1024": g.normal()})
    (out / "_gpu").mkdir()
    (out / "_gpu" / "00a1_30_sca8k_sc_x0.apps").write_text("start t h gpu0\n123, 2000 MiB\ncmd x\nend t h gpu0\n")
    (out / "_gpu" / "00a1_31_sca8k_sc_x1.apps").write_text("start t h gpu0\ncmd x\nend t h gpu0\n")
    for name, f in (("extra", E / "extra8k" / "extra_per_image.csv"), ("base", E / "base8k" / "per_image.csv"),
                    ("musiq", E / "musiq8k.csv")):
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[name][0]))
            w.writeheader()
            w.writerows(rows[name])
    EV.main(["--prompts", str(prompts), "--eval", str(E), "--results", str(R), "--outputs", str(out),
             "--skip-metrics", "--min-pairs", "2", "--device", "cpu"])
    pairs = {p.name: list(csv.DictReader(open(p))) for p in R.glob("sca_vs_*.csv")}
    assert set(pairs) == {"sca_vs_full.csv", "sca_vs_sega.csv", "sca_vs_ntk.csv"}
    assert {r["n_prompts"] for r in pairs["sca_vs_full.csv"]} == {"3"}
    assert {r["n_prompts"] for r in pairs["sca_vs_ntk.csv"]} == {"2"}
    assert {r["metric"] for r in pairs["sca_vs_sega.csv"]} == {"ir", "hpsv21", "clip", "pickscore", "musiq_4096",
                                                               "musiq_crop1024"}
    t = {(r["method"], r["prompt_id"]): r for r in csv.DictReader(open(R / "timing.csv"))}
    assert t[("sca", "sc_x0")]["gpu_shared"] == "True" and t[("sca", "sc_x1")]["gpu_shared"] == "False"
    summ = {r["method"]: r for r in csv.DictReader(open(R / "summary_8192.csv"))}
    assert summ["full"]["n"] == "3" and "musiq_crop1024" in summ["sca"]
    sums = (R / "SHA256SUMS").read_text()
    assert "./per_image_8192.csv" in sums and "./timing_summary.csv" in sums and "./prompts.jsonl" in sums
