"""Record cached model revisions without loading models or hashing large weights."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-Krea-dev", "black-forest-labs/FLUX.2-dev")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", type=Path, default=Path("/scratch/09032/saini_2/tmp/hf"))
    parser.add_argument("--out", type=Path, default=ROOT / "repro/model_snapshots.json")
    args = parser.parse_args()
    sidecars = ROOT / "results/evidence/generation_sidecars.jsonl.gz"
    observed = {}
    with gzip.open(sidecars, "rt") as handle:
        for line in handle:
            config = json.loads(line)["sidecar"]["config"]
            key = (config["model"], config["method"])
            observed.setdefault(key, Counter())[config.get("model_revision", "not_recorded")] += 1
    models = []
    for repo in MODELS:
        cached = args.hf_home / "hub" / ("models--" + repo.replace("/", "--"))
        ref = cached / "refs/main"
        revision = ref.read_text().strip()
        assert re.fullmatch(r"[0-9a-f]{40}", revision), (repo, revision)
        snapshot = cached / "snapshots" / revision
        files = []
        for path in sorted(snapshot.rglob("*")):
            if path.is_dir():
                continue
            target = path.resolve()
            row = {"path": str(path.relative_to(snapshot)), "exists": path.is_file(),
                   "resolved_path": str(target), "bytes": path.stat().st_size if path.is_file() else None}
            if target.parent.name == "blobs" and re.fullmatch(r"[0-9a-f]{64}", target.name):
                row["cache_blob_sha256_identifier"] = target.name
            if path.is_file() and path.suffix == ".json" and path.stat().st_size <= 1024 * 1024:
                data = path.read_bytes()
                copy = args.out.parent / "model_metadata" / repo.split("/")[1] / row["path"]
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_bytes(data)
                row.update(metadata_sha256=hashlib.sha256(data).hexdigest(),
                           archived_metadata=str(copy.relative_to(args.out.parent)))
            files.append(row)
        models.append({"repo_id": repo, "observed_cached_revision": revision,
                       "ref_source": str(ref), "ref_sha256": hashlib.sha256(ref.read_bytes()).hexdigest(),
                       "snapshot": str(snapshot), "files": files,
                       "archived_generation_records": [{"method": method, "recorded_revision_counts": dict(counts)}
                                                       for (model, method), counts in sorted(observed.items()) if model == repo]})
    report = {"observed_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Current local cache inventory, recorded retrospectively. SCA and FLUX.2 historical sidecars do not record model snapshot revisions. These cache observations do not retroactively add missing run-time evidence.",
              "weight_verification": "Weight bytes were not rehashed or copied. SHA256-named HF blob identifiers and current path sizes are recorded; small JSON metadata bytes are hashed and archived.",
              "generation_sidecars_sha256": hashlib.sha256(sidecars.read_bytes()).hexdigest(), "models": models}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({row["repo_id"]: {"revision": row["observed_cached_revision"], "files": len(row["files"]),
                                     "missing_files": sum(not f["exists"] for f in row["files"])} for row in models}))


if __name__ == "__main__":
    main()
