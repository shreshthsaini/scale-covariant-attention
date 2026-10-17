"""Inventory known metric assets; hash small/local Torch files, never download weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
HUB_MODELS = {
    "openai/clip-vit-large-patch14": "CLIP and aesthetic backbone",
    "yuvalkirstain/PickScore_v1": "PickScore model",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": "PickScore processor only; large model weights are not required",
    "THUDM/ImageReward": "ImageReward.pt and med_config.json",
    "bert-base-uncased": "ImageReward BLIP tokenizer only; BERT model weights are not required",
    "xswu/HPSv2": "HPS_v2_compressed.pt and HPS_v2.1_compressed.pt",
}
TORCH_FILES = {
    "hub/pyiqa/musiq_koniq_ckpt-e95806b9.pth": "MUSIQ",
    "hub/clip/RN50.pt": "CLIP-IQA",
    "hub/checkpoints/ARNIQA.pth": "ARNIQA encoder",
    "hub/checkpoints/regressor_koniq10k.pth": "ARNIQA KonIQ regressor",
    "hub/checkpoints/resnet50-0676ba61.pth": "ARNIQA constructor's initial ResNet50 weights",
    "hub/checkpoints/weights-inception-2015-12-05-6726825d.pth": "FID/KID, patch FID/KID, Inception Score",
    "hub/checkpoints/laion_aesthetic/sac+logos+ava1-l14-linearMSE.pth": "Aesthetic MLP",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-home", type=Path, default=Path("/scratch/09032/saini_2/tmp/hf"))
    p.add_argument("--torch-home", type=Path, default=Path("/scratch/09032/saini_2/tmp/torch"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    record = {"checked_utc": datetime.now(timezone.utc).isoformat(), "hf_home": str(args.hf_home),
              "torch_home": str(args.torch_home), "hub": {}, "torch_files": {},
              "observed_cache_environment": {k: os.environ.get(k) for k in ("HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME")},
              "scope": "Retrospective cache inventory. HF blob identities are recorded, not rehashed for large weights. All seven listed Torch asset files are newly SHA256-hashed. No weights copied or downloaded."}
    for model, role in HUB_MODELS.items():
        base = args.hf_home / "hub" / ("models--" + model.replace("/", "--"))
        ref = base / "refs/main"
        entry = {"role": role, "main_ref_exists": ref.exists(), "files": []}
        if ref.exists():
            revision = ref.read_text().strip()
            entry["revision"] = revision
            snap = base / "snapshots" / revision
            for file in sorted(snap.rglob("*")):
                if file.is_dir():
                    continue
                relative = str(file.relative_to(snap))
                if model in ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "bert-base-uncased") and file.suffix not in (".json", ".txt", ".model"):
                    continue
                row = {"name": relative, "exists": file.exists(), "resolved_path": str(file.resolve())}
                if file.exists():
                    row["bytes"] = file.stat().st_size
                    blob = file.resolve().name
                    row["blob_sha256"] = blob if re.fullmatch(r"[a-f0-9]{64}", blob) else None
                    if file.stat().st_size < 2 << 20:
                        row["verified_sha256"] = sha256(file)
                entry["files"].append(row)
            # Avoid downloading duplicate formats and unrelated model/export artifacts.
            available = {row["name"] for row in entry["files"]}
            entry["restore_files"] = [name for name in sorted(available)
                if "/" not in name and name not in ("README.md", ".gitattributes")
                and not (name == "pytorch_model.bin" and "model.safetensors" in available)]
        record["hub"][model] = entry
    for relative, role in TORCH_FILES.items():
        file = args.torch_home / relative
        entry = {"role": role, "exists": file.is_file(), "path": str(file)}
        if file.is_file():
            entry.update(bytes=file.stat().st_size, verified_sha256=sha256(file))
        record["torch_files"][relative] = entry
    alternative = os.environ.get("TRANSFORMERS_CACHE")
    record["legacy_transformers_cache"] = {}
    if alternative:
        for model, entry in record["hub"].items():
            base = Path(alternative) / ("models--" + model.replace("/", "--"))
            ref = base / "refs/main"
            if not ref.exists():
                continue
            revision = ref.read_text().strip()
            snapshot = base / "snapshots" / revision
            matches = {}
            for row in entry["files"]:
                file = snapshot / row["name"]
                if file.exists():
                    matches[row["name"]] = {
                        "bytes": file.stat().st_size,
                        "resolved_blob_name": file.resolve().name,
                        "same_blob_identity_as_hub": file.resolve().name == Path(row["resolved_path"]).name,
                    }
            record["legacy_transformers_cache"][model] = {
                "revision": revision, "matches_hub_revision": revision == entry.get("revision"), "files": matches}
    # Compare the revisions explicitly recorded in a historical completed main report.
    base = json.loads((ROOT / "results/evidence/eval/final4k_nb_base/metrics.json").read_text())
    extra = json.loads((ROOT / "results/evidence/eval/final4k_nb_extra/extra_metrics.json").read_text())
    historical = dict(base["model_revisions"])
    historical["xswu/HPSv2"] = extra["provenance"]["hpsv2"]["checkpoint_sha"]
    record["historical_revision_comparisons"] = {
        model: {"recorded": historical.get(model), "current": entry.get("revision"),
                "match": historical.get(model) == entry.get("revision") if historical.get(model) else None}
        for model, entry in record["hub"].items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"hub_revisions": {m: e.get("revision") for m, e in record["hub"].items()},
                      "missing_torch_assets": [p for p, e in record["torch_files"].items() if not e["exists"]]}, indent=2))


if __name__ == "__main__":
    main()
