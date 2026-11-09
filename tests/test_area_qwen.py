"""CPU tests for scripts/run_area_qwen.py: AReA's released Qwen-Image code (third_party/AReA_3efb7e5) on a tiny random
pipeline, the settings read from its generate.py, and the schedule its README states."""

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "AReA_3efb7e5"))
from test_qwen import tiny_pipeline, write_prompts  # noqa: E402

from scripts import run_area_qwen as R  # noqa: E402


def test_upstream_defaults_from_generate_py():
    d = R.upstream_defaults()
    assert d["steps"] == 40 and d["guidance"] == 4.0 and d["guidance_arg"] == "true_cfg_scale"
    assert d["negative_prompt"] == "" and d["vae_tiling"] == "always" and d["checkpoint"] == "Qwen/Qwen-Image"
    assert d["suffix"].startswith(", Ultra HD, 4K, cinematic composition, faithful to the described content")
    assert d["suffix"].endswith("smooth tonal gradations.")


def test_schedule_matches_readme():
    s4 = R.area_schedule((4096, 4096), 40, None, txt_len=0)
    assert s4["phase1_steps"] == 20 and s4["mu"] == 2.0          # README: Phase II from step 20 of 40 at 4096^2
    assert s4["factors"][0] == 4 and s4["factors"][-1] == 2 and s4["windows"][-1] == 64
    s5 = R.area_schedule((5312, 5312), 40, None, txt_len=0)
    assert s5["phase1_steps"] == 22 and abs(s5["mu"] - 2.26) < 1e-3
    assert len(s5["factors"]) + len(s5["windows"]) == 40 and s5["windows"][-1] == 64
    assert R.area_schedule((1328, 1328), 40, None, txt_len=0)["phase1_steps"] == 0     # native: plain attention


def test_area_hash_is_stable():
    assert R.area_hash() == R.area_hash() and len(R.area_hash()) == 64


def test_run_area_qwen_cpu(tmp_path, monkeypatch):
    import area.attention as A
    import sca.qwen as Q
    pipe = tiny_pipeline()
    stock_scheduler, stock_proc = pipe.scheduler, type(pipe.transformer.transformer_blocks[0].attn.processor)
    monkeypatch.setattr(Q, "load_pipeline", lambda device="cpu": pipe)
    monkeypatch.setattr(Q, "snapshot", lambda: tmp_path / "snap0")
    calls, used = [], {"coarse": 0, "fine": 0}

    def enc(prompts, device="cpu", **kw):
        calls.append(list(prompts))
        return [torch.randn(1, 3 + len(p) % 5, 24, generator=torch.Generator().manual_seed(len(p))) for p in prompts]
    monkeypatch.setattr(Q, "encode_prompts", enc)
    pool, win = A.pool_keys_values, A.windowed_attention

    def pool_count(*a, **k):
        used["coarse"] += 1
        return pool(*a, **k)

    def win_count(*a, **k):
        used["fine"] += 1
        return win(*a, **k)
    monkeypatch.setattr(A, "pool_keys_values", pool_count)
    monkeypatch.setattr(A, "windowed_attention", win_count)
    prompts = write_prompts(tmp_path / "p.jsonl", 3)
    out = tmp_path / "out"
    base = ["--prompts", str(prompts), "--res", "1344x1344", "--steps", "6", "--device", "cpu", "--out", str(out),
            "--start", "0", "--end", "2", "--exclusive-wait", "0"]
    R.main(base)
    suffix = R.upstream_defaults()["suffix"]
    metas = sorted(out.glob("*.json"))
    assert len(metas) == 2 and len(list(out.glob("*.png"))) == 2
    m = json.loads(metas[0].read_text())
    c = m["config"]
    assert m["complete"] and c["method"] == "qwen_area" and c["res"] == "1344x1344" and c["steps"] == 6
    assert c["prompt"] == "prompt number 0 " and c["conditioning_prompt"] == "prompt number 0 " + suffix
    assert c["negative_prompt"] == "" and c["true_cfg_scale"] == 4.0 and c["area"]["switch_sigma"] == 0.87
    assert calls[0][:2] == ["prompt number 0 " + suffix, "prompt number 1 x" + suffix] and calls[0][-1] == ""
    plan = m["area_schedule"]
    k = plan["phase1_steps"]
    assert 1 <= k < 6 and len(plan["factors"]) == k and len(plan["windows"]) == 6 - k
    # both phases ran, on both true-CFG passes of every step, for each of the 2 images (1 block in the tiny model)
    assert used["coarse"] == 2 * k * 2 and used["fine"] == 2 * (6 - k) * 2
    assert m["first_call_in_process"] is True and m["seconds"] > 0 and len(m["sigmas"]) == 6
    assert m["mu_area"] == 2.0 and m["gpu_watch"]["available"] in (True, False)
    # detached afterwards: stock processors and scheduler
    assert type(pipe.transformer.transformer_blocks[0].attn.processor) is stock_proc and pipe.scheduler is stock_scheduler
    n = len(calls)
    R.main(base)                                     # resume: nothing regenerated, nothing encoded
    assert len(calls) == n
    R.main(base + ["--no-prompt-suffix", "--out", str(tmp_path / "plain"), "--end", "1"])
    p = json.loads(next((tmp_path / "plain").glob("*.json")).read_text())
    assert p["config"]["conditioning_prompt"] == p["config"]["prompt"]


def test_same_initial_noise_as_sample_qwen():
    """Both scripts seed a CPU generator with image_seed(prompt id, seed)."""
    src_a = (ROOT / "scripts" / "run_area_qwen.py").read_text()
    src_s = (ROOT / "scripts" / "sample_qwen.py").read_text()
    needle = 'torch.Generator(device="cpu").manual_seed(image_seed(row["id"], args.seed))'
    assert needle in src_a and needle in src_s
