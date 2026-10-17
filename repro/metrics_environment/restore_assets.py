"""Print the metric-asset plan; download only with explicit --download."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

HERE = Path(__file__).resolve().parent


def verify(path, expected):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    if h.hexdigest() != expected:
        raise RuntimeError(f"Checksum mismatch: {path}; preserve this file and investigate")
    return h.hexdigest()


def copy_verified(source, target, expected, receipt):
    """Copy only verified bytes, retaining every pre-existing target on failure."""
    source = Path(source)
    if not source.is_file():
        raise RuntimeError(f"Missing source asset: {source}")
    digest = verify(source, expected)
    if target.exists():
        verify(target, digest)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        with source.open("rb") as inp, partial.open("xb") as out:
            shutil.copyfileobj(inp, out, 8 << 20)
        verify(partial, digest)
        partial.rename(target)
    receipt.append({"source": str(source), "target": str(target),
                    "bytes": target.stat().st_size, "sha256": digest})


def restore_from_cache(record, hf_home, torch_home):
    """Populate isolated caches without network access or modifying source caches."""
    receipt = []
    for model, entry in record["hub"].items():
        base = hf_home / "hub" / ("models--" + model.replace("/", "--"))
        ref = base / "refs/main"
        if ref.exists() and ref.read_text() != entry["revision"]:
            raise RuntimeError(f"Refusing to change existing cache revision: {ref}")
        snapshot = base / "snapshots" / entry["revision"]
        by_name = {row["name"]: row for row in entry["files"]}
        for name in entry["restore_files"]:
            row = by_name[name]
            expected = row.get("verified_sha256") or row.get("blob_sha256")
            if not expected:
                # Small HF files can be addressed by Git blob SHA1 rather than
                # LFS SHA256. Authenticate that archived identity before copying.
                source = Path(row["resolved_path"])
                data = source.read_bytes()
                git_digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                if source.name != git_digest or len(data) != row["bytes"]:
                    raise RuntimeError(f"Git blob identity mismatch: {source}")
                expected = hashlib.sha256(data).hexdigest()
            copy_verified(row["resolved_path"], snapshot / name, expected, receipt)
        ref.parent.mkdir(parents=True, exist_ok=True)
        if not ref.exists():
            with ref.open("x") as f:
                f.write(entry["revision"])
    for relative, entry in record["torch_files"].items():
        copy_verified(entry["path"], torch_home / relative,
                      entry["verified_sha256"], receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-home", type=Path, required=True)
    p.add_argument("--torch-home", type=Path, required=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--from-cache", action="store_true",
                   help="Copy assets from archived source paths; verify source and destination SHA256")
    p.add_argument("--receipt", type=Path, help="Write cache-copy identities to this new JSON file")
    args = p.parse_args()
    if args.download and args.from_cache:
        p.error("Choose --download or --from-cache")
    if args.receipt and not args.from_cache:
        p.error("--receipt requires --from-cache")
    if args.receipt and args.receipt.exists():
        p.error("Refusing to overwrite an existing receipt")
    record = json.loads((HERE / "checkpoints.json").read_text())
    urls = json.loads((HERE / "torch_asset_urls.json").read_text())["urls"]
    total = sum(row["bytes"] for entry in record["hub"].values() for row in entry["files"] if row["name"] in entry["restore_files"])
    total += sum(row["bytes"] for row in record["torch_files"].values())
    print(json.dumps({"download_enabled": args.download, "asset_bytes": total,
                      "hf_home": str(args.hf_home), "torch_home": str(args.torch_home),
                      "scope": "Six pinned HF snapshots (two tokenizer/processor-only), seven hash-verified Torch assets"}, indent=2))
    if args.from_cache:
        receipt = restore_from_cache(record, args.hf_home, args.torch_home)
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            with args.receipt.open("x") as f:
                json.dump({"source": "archived cache paths", "network_requests": 0,
                           "assets": receipt}, f, indent=2)
                f.write("\n")
        print(f"Copied and verified {len(receipt)} assets. No scoring was performed.")
        return
    if not args.download:
        return
    from huggingface_hub import snapshot_download
    cache = args.hf_home / "hub"
    for model, entry in record["hub"].items():
        base = cache / ("models--" + model.replace("/", "--"))
        ref = base / "refs/main"
        if ref.exists() and ref.read_text() != entry["revision"]:
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
                f.write(entry["revision"])
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
