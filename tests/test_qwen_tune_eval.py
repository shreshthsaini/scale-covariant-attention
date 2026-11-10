"""CPU test for scripts/qwen_tune_eval.py with stand-in metric CSVs (--skip-metrics): loading, linking with the
label as method, merge, paired bootstrap, timing, the d_z screen ranking and the archive."""

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen_tune_eval as T  # noqa: E402


def fake_outputs(tmp, labels, ids, res="64x64"):
    from PIL import Image
    dirs = {}
    for k, label in enumerate(labels):
        d = tmp / "out" / label / res
        d.mkdir(parents=True)
        for i, pid in enumerate(ids):
            png = d / f"{pid}_abc_seed0.png"
            Image.new("RGB", (64, 64), (10 * k, 20, 30)).save(png)
            meta = {"complete": True, "seconds": 100.0 + 10 * k + i, "peak_memory_bytes": 1e9,
                    "config": {"method": f"orig_{label}", "res": res, "prompt_id": pid, "prompt": f"p {pid}",
                               "steps": 40, "seed": 0}}
            (d / f"{pid}_abc_seed0.json").write_text(json.dumps(meta))
        dirs[label] = d
    return dirs


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_eval_skip_metrics(tmp_path, monkeypatch):
    import qwen_area_eval as qae
    monkeypatch.setattr(qae, "OUT", tmp_path / "showcase")          # no task logs: every image clean
    ids = [f"sc_x{i:02d}" for i in range(6)]
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("".join(json.dumps({"id": i, "prompt": f"p {i}"}) + "\n" for i in ids))
    dirs = fake_outputs(tmp_path, ["ref", "good", "bad"], ids)
    E, R = tmp_path / "eval", tmp_path / "results"
    off = {"ref": 0.0, "good": 1.0, "bad": -1.0}
    extra, base, musiq = [], [], []
    for m, o in off.items():
        for i, pid in enumerate(ids):
            j = 0.1 * ((i * 7) % 5)   # prompt-dependent noise shared by all methods, plus a method offset
            e = 0.01 * ((i * 3 + list(off).index(m)) % 4)   # method-specific noise: every paired difference varies
            extra.append({"method": m, "res": "64x64", "prompt_id": pid, "seed": 0, "path": "x",
                          "ir": 0.5 + j + o * 0.2 + e,
                          "aesthetic": 6, "hpsv2": 0.3, "hpsv21": 0.3 + j * 0.01 + o * 0.01 + e * 0.1,
                          "clipiqa": 0.6, "arniqa": 0.7})
            base.append({"clip": 30, "method": m, "path": "x", "pickscore": 22 + j + o * 0.3 + e, "prompt_id": pid,
                         "res": "64x64", "seconds": 1, "seed": 0})
            musiq.append({"method": m, "res": "64x64", "prompt_id": pid, "seed": 0, "path": "x", "musiq_4096": 50,
                          "musiq_crop1024": 50 + o})
    write(E / "extra/extra_per_image.csv", extra)
    write(E / "base/per_image.csv", base)
    write(E / "musiq.csv", musiq)
    T.main(["--prompts", str(prompts), "--res", "64x64", "--reference", "ref", "--eval", str(E), "--results", str(R),
            "--skip-metrics"] + [f"--method={m}={d}" for m, d in dirs.items()])
    sc = list(csv.DictReader(open(R / "screen.csv")))
    assert [r["method"] for r in sc] == ["good", "bad"]
    assert float(sc[0]["mean_dz"]) > 0 > float(sc[1]["mean_dz"]) and int(sc[0]["n_prompts"]) == 6
    assert abs(float(sc[0]["pickscore_delta"]) - 0.3) < 0.03 and float(sc[0]["musiq_crop1024_delta"]) == 1.0
    linked = json.loads(next((E / "root" / "good" / "64x64").glob("*.json")).read_text())
    assert linked["config"]["method"] == "good"                       # label written for the metric tools
    assert (R / "good_minus_ref.csv").exists() and (R / "bad_minus_ref.csv").exists()
    t = list(csv.DictReader(open(R / "timing_summary.csv")))
    assert {r["method"] for r in t} == {"ref", "good", "bad"}
    about = json.loads((R / "about.json").read_text())
    assert about["reference"] == "ref" and len(about["images"]["good"]) == 6
    sums = (R / "SHA256SUMS").read_text()
    assert "screen.csv" in sums and "per_image.csv" in sums
