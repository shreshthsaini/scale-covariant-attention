"""Read-only byte and provenance audit of the repaired inherited E9 baseline view.

No scoring, model imports, image writes, or mutation of original outputs.
Archive output is written only after all exact-coverage assertions pass.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sca.runtime import image_seed, image_stem


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=Path("/scratch/09032/saini_2/sca"))
    parser.add_argument("--archive", type=Path, default=ROOT / "results/evidence/krea_overlay_complete")
    args = parser.parse_args()
    prior = ROOT / "results/evidence/krea_baseline_provenance"
    preparation = json.loads((prior / "repair_preparation.json").read_text())
    inherited = json.loads((prior / "receipt.json").read_text())
    overlay = Path(preparation["overlay"])
    manifest = args.scratch / "eval100.jsonl"
    selected = jsonl(manifest)
    assert sha(manifest) == sha(ROOT / "results/evidence/prompts/eval100.jsonl")
    expected = {str(r["id"]): r["prompt"] for r in selected}
    assert len(selected) == len(expected) == 100
    dev = {str(r["id"]) for r in jsonl(ROOT / "results/evidence/prompts/dev4.jsonl")}
    assert not dev.intersection(expected), "Development prompt in E9 manifest"
    repair_ids = {str(r["id"]) for r in jsonl(prior / "unmatched_sega_prompts.jsonl")}
    assert len(repair_ids) == 7
    links = {(r["method"], r["prompt_id"]): r for r in preparation["links"]}
    assert len(links) == 200
    known_hashes = set(inherited["committed_implementation_fingerprints"].values())
    records, repaired_sidecars, sigma_hashes = [], [], set()
    seen, repairs, checked_images = Counter(), set(), {}

    def check_image(path, declared):
        resolved = path.resolve(strict=True)
        if resolved not in checked_images:
            actual = sha(resolved)
            with Image.open(resolved) as image:
                assert image.format == "PNG" and image.size == (4096, 4096), str(path)
                image.verify()  # PNG chunk CRC/structure validation, not metric decoding.
            checked_images[resolved] = actual
        assert checked_images[resolved] == declared, f"PNG hash mismatch: {path}"
        return checked_images[resolved]

    for method in ("dype", "sega_official"):
        directory = overlay / method / "4096x4096"
        assert len(list(directory.glob("*.json"))) == len(list(directory.glob("*.png"))) == 100
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text())
            cfg = data["config"]
            pid = cfg["prompt_id"]
            assert pid in expected and cfg["prompt"] == expected[pid]
            assert data["complete"] is True and cfg["method"] == method
            assert cfg["res"] == "4096x4096" and cfg["seed"] == 0
            assert cfg["derived_seed"] == image_seed(pid, 0)
            assert cfg["steps"] == 28 and cfg["guidance"] == 4.5
            assert cfg["dtype"] == "bfloat16" and cfg["batch_size"] == 1
            assert cfg["rope"]["method"] == method
            assert cfg["model"] == "black-forest-labs/FLUX.1-dev"  # Base component label, not transformer.
            assert cfg["model_revision"] == "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
            assert cfg["implementation_sha256"] in known_hashes
            fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            assert fingerprint == data["config_fingerprint"]
            assert path.stem == image_stem(pid, 0)
            seen[method, pid] += 1
            link = links[method, pid]
            assert path.is_symlink() and path.with_suffix(".png").is_symlink()
            assert path.resolve() == Path(link["target_json"]).resolve()
            assert path.with_suffix(".png").resolve() == Path(link["target_json"]).with_suffix(".png").resolve()
            original = Path(link["original_json"])
            original_hash = sha(original)
            assert original_hash == link["original_sidecar_sha256"], f"Original sidecar changed: {original}"
            original_data = json.loads(original.read_text())
            original_png_hash = check_image(original.with_suffix(".png"), original_data["png_sha256"])
            png_hash = check_image(path.with_suffix(".png"), data["png_sha256"])
            replacement = bool(link["replacement"])
            if replacement:
                assert method == "sega_official" and pid in repair_ids
                assert path.resolve() != original.resolve()
                assert cfg["implementation_sha256"] == preparation["implementation_sha256"]
                repairs.add(pid)
                repaired_sidecars.append(dict(path=str(path.resolve()), sha256=sha(path), record=data))
            else:
                assert path.resolve() == original.resolve()
            sigma_hashes.add(hashlib.sha256(json.dumps(data["sigmas"]).encode()).hexdigest())
            records.append(dict(method=method, prompt_id=pid, overlay_path=str(path),
                resolved_sidecar=str(path.resolve()), sidecar_sha256=sha(path), png_sha256=png_hash,
                png_bytes=path.with_suffix(".png").stat().st_size, dimensions=[4096,4096],
                config_fingerprint=fingerprint, implementation_sha256=cfg["implementation_sha256"],
                declared_code_revision=cfg["code_revision"], replacement=replacement,
                original_sidecar=str(original), original_sidecar_sha256=original_hash,
                original_png_sha256=original_png_hash, original_preserved=True))
        assert {p for m,p in seen if m == method} == set(expected)
        print(f"Verified {method}: 100 exact prompt/image pairs", flush=True)
    assert all(n == 1 for n in seen.values()) and repairs == repair_ids and len(sigma_hashes) == 1
    launches = sorted(Path(preparation["runtime_receipts_directory"]).glob("*.json"))
    assert len(launches) == 1
    launch = json.loads(launches[0].read_text())
    assert set(launch["prompt_ids"]) == repair_ids
    assert launch["implementation_sha256"] == preparation["implementation_sha256"]
    assert launch["transformer_override"] == preparation["transformer_snapshot"]
    assert launch["transformer_config_sha256"] == sha(Path(preparation["transformer_snapshot"]) / "transformer/config.json")
    assert launch["manifest_sha256"] == sha(prior / "unmatched_sega_prompts.jsonl")
    task = args.scratch / "taskq/done/a_krea_sega_source_repair_7.sh"
    assert sha(task) == launch["executed_task_sha256"]
    files = ["crope/" + n for n in ("__init__.py","runtime.py","gain.py","rope.py","sega_official.py","spectral.py","utils.py")]
    files += ["scripts/" + n for n in ("sample.py","calibrate.py","probe_mtf.py")]
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode()); digest.update((Path(preparation["source_root"]) / name).read_bytes())
    assert digest.hexdigest() == preparation["implementation_sha256"]
    args.archive.mkdir(parents=True, exist_ok=True)
    pairs = args.archive / "verified_pairs.jsonl"
    pairs.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    with gzip.GzipFile(filename=str(args.archive / "repair_sidecars.jsonl.gz"), mode="wb", mtime=0) as handle:
        handle.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in repaired_sidecars).encode())
    shutil.copyfile(launches[0], args.archive / "repair_launch_receipt.json")
    shutil.copyfile(task, args.archive / task.name)
    receipt = dict(status="complete_overlay_byte_and_provenance_audit_passed", created_utc=datetime.now(timezone.utc).isoformat(),
        overlay=str(overlay), selected_manifest=str(manifest), manifest_sha256=sha(manifest),
        exact_pairs=200, methods={"dype":100,"sega_official":100}, unchanged_pairs=193, repaired_pairs=7,
        regenerated_pngs_identical_to_originals=sum(r["replacement"] and r["png_sha256"] == r["original_png_sha256"] for r in records),
        original_sidecars_unchanged=200, distinct_png_files_hashed_and_crc_verified=len(checked_images),
        repaired_prompt_ids=sorted(repairs), unique_sigma_schedules=1,
        known_implementation_hash_counts=dict(Counter(r["implementation_sha256"] for r in records)),
        pairs_sha256=sha(pairs), repair_task_sha256=sha(task), frozen_repair_source_fingerprint=digest.hexdigest(),
        prior_receipt_sha256=sha(prior / "receipt.json"), preparation_sha256=sha(prior / "repair_preparation.json"),
        audit_script_sha256=sha(__file__),
        limitations=["Krea transformer attribution uses archived launch/loader evidence; sidecars label base FLUX.1-dev components.",
                     "Inherited 193 outputs retain multiple recovered source revisions; no claim of one frozen historical revision.",
                     "No rescoring, model loads, weights rehash, full-model equivalence run, or new generation.",
                     "This certifies the two baseline methods only, not completion of SCA/AReA E9 generation or evaluation."])
    (args.archive / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k:v for k,v in receipt.items() if k not in ["limitations","repaired_prompt_ids"]}), flush=True)


if __name__ == "__main__":
    main()
