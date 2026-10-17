"""Print the metric-asset plan; download only with explicit --download."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent


def verify(path, expected):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    if h.hexdigest() != expected:
        raise RuntimeError(f"Checksum mismatch: {path}; preserve this file and investigate")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-home", type=Path, required=True)
    p.add_argument("--torch-home", type=Path, required=True)
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    record = json.loads((HERE / "checkpoints.json").read_text())
    urls = json.loads((HERE / "torch_asset_urls.json").read_text())["urls"]
    total = sum(row["bytes"] for entry in record["hub"].values() for row in entry["files"] if row["name"] in entry["restore_files"])
    total += sum(row["bytes"] for row in record["torch_files"].values())
    print(json.dumps({"download_enabled": args.download, "asset_bytes": total,
                      "hf_home": str(args.hf_home), "torch_home": str(args.torch_home),
                      "scope": "Six pinned HF snapshots (two tokenizer/processor-only), seven hash-verified Torch assets"}, indent=2))
    if not args.download:
        return
    from huggingface_hub import snapshot_download
    cache = args.hf_home / "hub"
    for model, entry in record["hub"].items():
        base = cache / ("models--" + model.replace("/", "--"))
        ref = base / "refs/main"
        if ref.exists() and ref.read_text().strip() != entry["revision"]:
            raise RuntimeError(f"Refusing to change existing cache revision: {ref}")
        snapshot = Path(snapshot_download(model, revision=entry["revision"],
                        allow_patterns=entry["restore_files"], cache_dir=cache, max_workers=2))
        for row in entry["files"]:
            if row["name"] in entry["restore_files"]:
                expected = row.get("verified_sha256") or row.get("blob_sha256")
                if expected:
                    verify(snapshot / row["name"], expected)
        # Evaluators address model IDs without a revision argument. In this isolated
        # cache, make their offline main lookup point to the verified pinned snapshot.
        ref.parent.mkdir(parents=True, exist_ok=True)
        if not ref.exists():
            with ref.open("x") as f:
                f.write(entry["revision"] + "\n")
    for relative, entry in record["torch_files"].items():
        target = args.torch_home / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            with partial.open("xb") as output, urllib.request.urlopen(urls[relative], timeout=60) as response:
                for block in iter(lambda: response.read(8 << 20), b""):
                    output.write(block)
            verify(partial, entry["verified_sha256"])
            partial.rename(target)
        verify(target, entry["verified_sha256"])
    print("All selected assets restored and verified. No scoring was performed.")


if __name__ == "__main__":
    main()
