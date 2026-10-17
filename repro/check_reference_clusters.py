#!/usr/bin/env python3
"""Check evaluator image clusters against the archived-feature bootstrap implementation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repro.evaluate_corrected import reference_clusters, reference_identity
from scripts.check_archived_distributions import load_bank
from crope.metrics.statistics import distribution_bootstrap, fid, kid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "repro/reference_cluster_validation.json")
    args = parser.parse_args()
    root = ROOT / "results/evidence/distributions"
    index = json.loads((root / "final4k_nb_base_banks.json").read_text())
    checks = []
    for metric in ("fid", "kid", "patch_fid", "patch_kid"):
        patch = metric.startswith("patch_")
        entry = index["banks"][f"reference_images:patch={patch}"]
        records = [{"prompt_id": r["prompt_id"], "path": r["image_signature"]["path"]}
                   for r in entry["records"]]
        real = reference_clusters(records, load_bank(root, entry, "rows", "rows"), "images")
        archived = load_bank(root, entry, "rows", "images")
        generated = load_bank(root, index["banks"][f"sca_nb:patch={patch}"])
        baseline = load_bank(root, index["banks"][f"area:patch={patch}"])
        statistic = kid if metric.endswith("kid") else fid
        common = dict(draws=args.draws, seed=0, baseline=baseline)
        current = distribution_bootstrap(generated, real, statistic, **common)
        previous = distribution_bootstrap(generated, archived, statistic, **common)
        error = max(abs(current[k] - previous[k]) for k in ("value", "ci_low", "ci_high"))
        assert error < 1e-9, (metric, current, previous)
        assert len(real) == 168 and len(records) == 195
        checks.append({"metric": metric, "max_absolute_difference": error, "result": current,
                       "reference_rows": len(records), "reference_clusters": len(real),
                       "identity_sources": sorted({reference_identity(r)[1] for r in records})})
    result = {"experiment": "final4k_nb_base", "draws": args.draws, "seed": 0,
              "scope": "Cached-feature numerical equivalence; no image encoder inference", "checks": checks}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
