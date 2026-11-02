"""Review round 1 additions (slurm/review_tasks.py): the window-tied bridge size (--bridge-ratio), AReA's Phase II
window override (scripts/run_area.py --phase2-windows), the late-duplication options in scripts/sample_flux2.py, and the
schedules and task scripts the generator writes. CPU only."""

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "slurm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sca.attn import PiTable, SCAContext  # noqa: E402
import review_tasks as rt  # noqa: E402
from sca.dedup import win_sched  # noqa: E402
from test_dup_round3 import BRIDGE, GRID, _close, _ctx  # noqa: E402,F401
from test_sca import attn_outputs, model  # noqa: E402


def per_cell(w, rho=0.5, cells=64):
    return max(1, round(max(1, round(rho * w)) ** 2 / cells))


def test_area_schedule_maps_one_window_per_step():
    s = rt.flux1_sigmas()
    assert abs(s[8] - 0.88758) < 1e-5 and abs(s[27] - 0.10472) < 1e-5   # values recorded in the sca_v2 sidecars
    c = SCAContext(mode="sca", gate_sigma=0.89, win_sched=win_sched(rt.AREA_SCHED), bridge_cells=8, bridge_ratio=0.5)
    c.grid = (256, 256)
    assert s[7] > 0.89 > s[8]                                                      # step 8 is the first local step
    for k, w in enumerate(rt.AREA_WINDOWS):
        c.sigma = s[8 + k]
        assert c.window_now() == w, (8 + k, c.window_now(), w)
        assert c.bridge_k_now(64) == per_cell(w) and c.bridge_now() == ((8, 8), per_cell(w))
    assert [per_cell(w) for w in (180, 128, 64)] == [127, 64, 16]                  # 8128, 4096 and 1024 bridge tokens
    v2 = SCAContext(mode="sca", gate_sigma=0.89, win_sched=win_sched(rt.V2_SCHED))
    v2.grid = (256, 256)
    for k, w in enumerate((128, 112, 96, 80, 64, 64)):                           # the adopted schedule, same rule
        v2.sigma = s[8 + k]
        assert v2.window_now() == w


def test_flux2_v2_schedule_on_flux2_sigmas():
    from sca.flux2 import empirical_mu, sigma_schedule
    sig = [float(x) for x in sigma_schedule(28, empirical_mu(256 * 256, 28, "native"))]
    for k, v in rt.FLUX2_SIGMAS.items():
        assert abs(sig[k] - v) < 6e-5, (k, sig[k], v)
    c = SCAContext(mode="sca", gate_sigma=0.959, win_sched=win_sched(rt.FLUX2_V2_SCHED))
    c.grid = (256, 256)
    assert sig[7] > 0.959 > sig[8]
    for k, w in zip(range(8, 14), (128, 112, 96, 80, 64, 64)):
        c.sigma = sig[k]
        assert c.window_now() == w


def test_bridge_ratio_validation_and_default_off():
    assert SCAContext(mode="sca").bridge_ratio == 0.0
    with pytest.raises(ValueError):
        SCAContext(mode="sca", bridge_ratio=0.5)                                   # needs bridge cells
    with pytest.raises(ValueError):
        SCAContext(mode="sca", bridge_cells=8, bridge_ratio=0.5, bridge_k_hi=32)
    c = SCAContext(mode="sca", bridge_cells=8, bridge_k=16)
    c.grid, c.sigma = (256, 256), 0.5
    assert c.bridge_k_now() == 16 and c.bridge_k_now(64) == 16                    # published bridge unchanged


def test_bridge_ratio_equals_the_fixed_count_it_implies():
    m = model()  # attn_outputs runs at sigma 0.5; GRID 16 x 16, base window 4, cells 2 x 2
    wide = attn_outputs(m, _ctx(win_sched=((0.4, 8),), bridge_cells=2, bridge_ratio=0.5), *GRID)   # (0.5 * 8)^2 / 4 = 4
    assert _close(wide, attn_outputs(m, _ctx(win_sched=((0.4, 8),), bridge_cells=2, bridge_k=4), *GRID))
    base = attn_outputs(m, _ctx(win_sched=((0.6, 8),), bridge_cells=2, bridge_ratio=0.5), *GRID)   # (0.5 * 4)^2 / 4 = 1
    assert _close(base, attn_outputs(m, _ctx(bridge_cells=2, bridge_k=1), *GRID))
    assert not _close(wide, attn_outputs(m, _ctx(win_sched=((0.4, 8),), bridge_cells=2, bridge_k=1), *GRID), 1e-4)


def test_ratios_below_two_read_the_scale_two_table():
    path = next((p for p in (ROOT / "calib/k1_1024.npz", Path(rt.K1)) if p.exists() and p.stat().st_size > 1000), None)
    if path is None:
        pytest.skip("calibrated table not on this machine")
    pi = PiTable.load(path, "head", None, None, 4)
    c = SCAContext(mode="sca", pi=pi, gate_sigma=0.89, win_pi_match=True, win_pi_interp=True,
                   win_sched=win_sched(rt.AREA_SCHED))
    c.grid = (256, 256)
    s = rt.flux1_sigmas()
    p2, p4 = PiTable.load(path, "head", 2), PiTable.load(path, "head", 4)
    for k in range(7):                       # windows 180 .. 130: ratio 256 / w < 2, clamped to the scale-2 table
        c.sigma = s[8 + k]
        assert 256 / c.window_now() < 2 and np.allclose(c.pi_table_now()(c.sigma), p2(c.sigma))
    c.sigma = s[15]                          # window 123: ratio 2.08, log-linear between scale 2 and 4
    t = math.log2((256 / 123) / 2)
    assert np.allclose(c.pi_table_now()(c.sigma), (1 - t) * p2(c.sigma) + t * p4(c.sigma))


def _run_area():
    try:
        import scripts.run_area as ra
        import area  # noqa: F401
    except Exception as exc:  # AReA checkout or the evaluator snapshot missing on this machine
        pytest.skip(f"AReA not importable: {exc}")
    return ra


def test_area_phase2_windows_override():
    ra = _run_area()
    from area.schedule import build_schedule
    args = ra.parser().parse_args(["--res", "4096x4096", "--prompts", "x.jsonl", "--out", "o"])
    cfg = ra.area_config(args)
    plain = ra.area_schedule((4096, 4096), 28, cfg)
    assert tuple(plain["windows"]) == rt.AREA_WINDOWS and "phase2_windows" not in plain     # default unchanged
    new = ra.area_schedule((4096, 4096), 28, cfg, (128, 112, 96, 80))
    assert new["windows"] == [128, 112, 96, 80] + [64] * 16 and new["factors"] == plain["factors"]
    assert new["area_windows"] == plain["windows"] and new["phase2_windows"] == [128, 112, 96, 80]
    for bad in ((63,), (257,), (100,) * 21):
        with pytest.raises(ValueError):
            ra.override_windows(plain["windows"], bad, 256)

    class Stub:
        def schedule(self, h, w, steps, txt_len=0):
            return None if max(h, w) <= 64 else build_schedule(h, w, steps, 8, 64, txt_len=txt_len)
    b = Stub()
    ra.patch_schedule(b, (128, 112, 96, 80))
    s = b.schedule(256, 256, 28, 512)
    assert s.windows == (128, 112, 96, 80) + (64,) * 16 and s.at(8) == ("fine", 128) and s.at(12) == ("fine", 64)
    assert s.factors == build_schedule(256, 256, 28, 8, 64, txt_len=512).factors and b.schedule(64, 64, 28) is None
    a2 = ra.parser().parse_args(["--res", "4096x4096", "--prompts", "x", "--out", "o", "--phase2-windows",
                                 "128,112,96,80", "--label", "area_sca4"])
    assert a2.phase2_windows == (128, 112, 96, 80) and a2.label == "area_sca4"


def test_area_dry_run_records_label_and_windows(tmp_path, capsys):
    ra = _run_area()
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"id": "p0", "prompt": "a cat"}) + "\n")
    ra.main(["--res", "4096x4096", "--prompts", str(prompts), "--out", str(tmp_path / "o"), "--dry-run",
             "--phase2-windows", "128,112,96,80", "--label", "area_sca4"])
    text = capsys.readouterr().out
    cfg = json.loads(text[:text.index("\ntodo")])
    assert cfg["method"] == "area_sca4" and cfg["area"]["schedule"]["windows"][:5] == [128, 112, 96, 80, 64]


def test_flux2_sca_v2_options(tmp_path, monkeypatch):
    from test_flux2 import Ctx, attn_outputs as f2_outputs, flat, model as f2_model, patch_loaders, write_prompts
    m, H, W = f2_model(), 8, 8
    mk = lambda **kw: Ctx(**{**dict(mode="sca", n0=4, tile=2, window=4, gate_sigma=0.9, pi=PiTable("const", value=0.7)), **kw})
    sched = f2_outputs(m, mk(win_sched=((0.4, 8),), bridge_cells=2, bridge_k=2), H, W)
    assert torch.allclose(flat(sched), flat(f2_outputs(m, mk(window=8, bridge_cells=2, bridge_k=2), H, W)), atol=1e-5)
    ratio = f2_outputs(m, mk(win_sched=((0.4, 8),), bridge_cells=2, bridge_ratio=0.5), H, W)     # (0.5 * 8)^2 / 4 = 4
    assert torch.allclose(flat(ratio), flat(f2_outputs(m, mk(window=8, bridge_cells=2, bridge_k=4), H, W)), atol=1e-5)
    assert not torch.allclose(flat(sched), flat(f2_outputs(m, mk(), H, W)), atol=1e-4)
    from scripts import sample_flux2
    patch_loaders(monkeypatch, tmp_path)
    out = tmp_path / "out"
    sample_flux2.main(["--prompts", str(write_prompts(tmp_path / "p.jsonl", 1)), "--res", "128x128", "--steps", "3",
                       "--device", "cpu", "--out", str(out), "--method", "sca", "--pi-const", "0.5", "--n0", "4",
                       "--tile", "2", "--gate-sigma", "0.9", "--label", "flux2_sca_v2", "--bridge-cells", "2",
                       "--bridge-k", "2", "--win-sched", "0.6:8", "--win-pi-match", "--win-pi-interp"])
    meta = json.loads(next(out.glob("*.json")).read_text())
    sca = meta["config"]["sca"]
    assert meta["complete"] and sca["bridge_cells"] == 2 and sca["win_sched"] == [[0.6, 8]] and sca["win_pi_interp"]
    with pytest.raises(SystemExit):
        sample_flux2.build_context(sample_flux2.parser().parse_args(["--prompts", "p", "--method", "sca", "--pi-const",
                                                                     "0.5", "--anchor-lo", "0.3"]), (8, 8))


def test_generated_tasks(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "FREEZE_COMMIT").write_text("0123456789abcdef\n")
    out = tmp_path / "hold"
    subprocess.run([sys.executable, str(ROOT / "slurm/review_tasks.py"), "--code", str(code), "--out", str(out)],
                   check=True, capture_output=True)
    names = sorted(p.name for p in out.glob("*.sh"))
    assert len(names) == 148 and (out / "README.md").exists()
    assert sum(n.startswith("r1_g_") for n in names) == 78 and sum(n.startswith("r2_g_") for n in names) == 18
    for p in out.glob("*.sh"):
        text = p.read_text()
        assert subprocess.run(["bash", "-n", str(p)]).returncode == 0, p.name
        assert "rm -" not in text and "—" not in text and f"C={code}" in text or f"cd {code}" in text, p.name
        assert text.splitlines()[1].startswith("# SCA_MIN_SECONDS="), p.name
    gen = (out / "r2_g_area_sca4_0044_0050.sh").read_text()
    assert "--phase2-windows 128,112,96,80 --label area_sca4" in gen and "--start 44 --end 50" in gen
    assert f"mv {out}/r2_z_eval_abl46_pair.sh" in gen
    ev = (out / "r2_z_eval_abl46_pair.sh").read_text()
    pat = ev.split("# RELEASE_AFTER: ^(")[1].split(")")[0].split("|")
    assert sorted(x + ".sh" for x in pat) == [n for n in names if n.startswith("r2_g_")]
    assert "--bridge-ratio 0.5" in (out / "r2_g_sca_area20_0004_0009.sh").read_text()
    assert f"{rt.QC}/pending" in (out / "r3_g_8192_sca_v2_0010_0012.sh").read_text()
    assert "--win-sched " + rt.FLUX2_V2_SCHED in (out / "r3_g_flux2_sca_v2_0045_0050.sh").read_text()
    assert "—" not in (out / "README.md").read_text() and "§" not in (out / "README.md").read_text()
