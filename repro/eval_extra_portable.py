#!/usr/bin/env python3
"""Use archived helpers and the supplied Torch cache for extra-metric reproduction."""
import importlib.util
import json
from pathlib import Path

from extra_report_identity import (IDENTITY_FILE, POLICY, REPORT_FILES, environment,
                                   identities, sha256, source_hashes, write_json)

ROOT = Path(__file__).resolve().parents[1]


def main():
    spec = importlib.util.spec_from_file_location("sca_eval_extra_portable", ROOT / "scripts/eval_extra.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DEFAULT_CROPE = str(ROOT / "repro/evaluator_snapshot")
    args = module.parser().parse_args()
    module.AESTHETIC_WEIGHTS = str(Path(args.torch_home) / "hub/checkpoints/laion_aesthetic/sac+logos+ava1-l14-linearMSE.pth")
    module.setup_environment(args)
    from crope.metrics.io import discover_samples, read_jsonl

    def inputs():
        records = discover_samples(args.root, read_jsonl(args.prompts), args.res)
        if args.methods:
            keep = {m.strip() for m in args.methods.split(",")}
            records = [r for r in records if r["method"] in keep]
        if not records:
            raise ValueError("No inputs remain after method filtering")
        return identities(records)

    out = Path(args.out)
    identity_path = out / IDENTITY_FILE
    before = inputs()
    manifest = {"policy": POLICY, "phase": "running", "identities": before,
                "environment": environment(), "source_hashes": source_hashes()}
    if identity_path.exists():
        old = json.loads(identity_path.read_text())
        for field in ("policy", "identities", "environment", "source_hashes"):
            if old.get(field) != manifest[field]:
                raise ValueError(f"Refusing cache reuse after {field} changed; use a new output directory")
    elif any((out / name).exists() for name in ("extra_cache.jsonl", *REPORT_FILES)):
        raise ValueError("Existing report/cache lacks input identities; use a new output directory")
    write_json(identity_path, manifest)
    result = module.main()
    if inputs() != before:
        manifest.update(phase="failed", error="Inputs changed during scoring")
        write_json(identity_path, manifest)
        raise ValueError(manifest["error"])
    report = json.loads((out / "extra_metrics.json").read_text())
    manifest["phase"] = "complete" if result == 0 and report.get("complete") is True and not report.get("errors") else "failed"
    manifest["report_files"] = {name: sha256(out / name) for name in REPORT_FILES}
    write_json(identity_path, manifest)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
