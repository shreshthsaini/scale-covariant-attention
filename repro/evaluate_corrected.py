#!/usr/bin/env python3
"""Evaluate with reference features indexed by prompt row, preserving the reference pool.

The historical evaluator_snapshot is unchanged. This subclass changes only
distribution feature assembly and records reference_prompt_row_v2 in reports.
Feature cache keys remain compatible: they already include prompt identity.
"""

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

SNAPSHOT = Path(__file__).resolve().parent / "evaluator_snapshot"
sys.path.insert(0, str(SNAPSHOT))


def configure_cli():
    spec = importlib.util.spec_from_file_location("sca_historical_eval_cli", SNAPSHOT / "scripts/eval.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    parser = cli.parser()
    parser.add_argument("--musiq-query-chunk", type=int, default=0,
                        help="Opt-in inference attention query chunking; preserves native MUSIQ patches (0 disables)")
    args = parser.parse_args()
    if args.musiq_query_chunk < 0:
        parser.error("--musiq-query-chunk must be nonnegative")
    if args.bootstrap < 0 or args.crops < 1 or not 1 <= args.threads <= 8:
        raise SystemExit("--bootstrap must be nonnegative, --crops positive, and --threads between 1 and 8")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ.setdefault("HF_HOME", "/scratch/09032/saini_2/tmp/hf")
    os.environ.setdefault("TORCH_HOME", "/scratch/09032/saini_2/tmp/torch")
    os.environ.setdefault("XDG_CACHE_HOME", "/scratch/09032/saini_2/tmp/cache")
    os.environ.setdefault("WANDB_DIR", "/scratch/09032/saini_2/sca/wandb")
    os.environ.setdefault("WANDB_PROJECT", "scale-covariant-attention")
    return args


# Set the requested BLAS thread limits before importing NumPy for CLI execution.
CLI_ARGS = configure_cli() if __name__ == "__main__" else None

import numpy as np
from crope.metrics.evaluate import (
    DISTRIBUTIONS, Evaluation as HistoricalEvaluation, cluster_features,
    release_model, synchronize,
)
from crope.metrics.models import InceptionFeatures
from crope.metrics.statistics import distribution_bootstrap, fid, kid

ASSEMBLY_POLICY = "reference_prompt_row_v2"


class Evaluation(HistoricalEvaluation):
    def key(self, record, metric, extra=None):
        chunk = getattr(self.args, "musiq_query_chunk", 0)
        if metric == "musiq" and chunk:
            extra = {"original_extra": extra, "execution_adapter": "musiq_all_keys_query_chunks_v1",
                     "query_chunk": chunk}
        return super().key(record, metric, extra)

    def learned(self, metric):
        chunk = getattr(self.args, "musiq_query_chunk", 0)
        if metric != "musiq" or not chunk:
            return super().learned(metric)
        from crope.metrics import models
        from unittest.mock import patch
        # This evaluator is sequential. Temporarily replace its scorer factory,
        # preserving the historical learned-score loop and restoring on exit.
        try:
            from repro.musiq_chunked import install
        except ModuleNotFoundError:
            from musiq_chunked import install
        original = models.load_scorer

        def adapted_scorer(name, device):
            scorer = original(name, device)
            scorer.provenance.update(install(scorer.model, chunk))
            return scorer

        with patch.object(models, "load_scorer", adapted_scorer):
            return super().learned(metric)

    def distributions(self):
        selected = [metric for metric in DISTRIBUTIONS if metric in self.metrics]
        if not selected:
            return
        if not self.references:
            for metric in selected:
                self.failed(metric, ValueError("--references is required for FID/KID"))
            return
        if len({r["prompt_id"] for r in self.references}) != len(self.references):
            for metric in selected:
                self.failed(metric, ValueError("Reference prompt IDs must be unique; repeated image paths are allowed"))
            return
        extractor, provenance = None, None
        banks = {}
        try:
            for patch in sorted({metric.startswith("patch_") for metric in selected}):
                generated_bank, reference_bank = {}, {}
                records = [(False, r) for r in self.records] + [(True, r) for r in self.references]
                for is_reference, record in records:
                    key = self.key(record, "inception_features", {"patch": patch})
                    path = self.out / "features" / f"{key}.npz"
                    if path.is_file():
                        with np.load(path, allow_pickle=False) as saved:
                            values = saved["features"]
                            current_provenance = json.loads(str(saved["provenance"]))
                            elapsed = float(saved["seconds"])
                    else:
                        if extractor is None:
                            extractor = InceptionFeatures(self.args.device)
                        current_provenance = extractor.provenance
                        synchronize(self.args.device)
                        start = time.perf_counter()
                        values = extractor.extract(record, patch=patch, crop_seed=self.args.crop_seed,
                                                   crops=self.args.crops)
                        synchronize(self.args.device)
                        elapsed = time.perf_counter() - start
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temp = path.with_name(path.name + ".tmp")
                        with temp.open("wb") as handle:
                            np.savez_compressed(handle, features=values,
                                                provenance=json.dumps(current_provenance), seconds=elapsed)
                        temp.replace(path)
                    if provenance is not None and provenance != current_provenance:
                        raise ValueError("Inception cache mixes backends; use a new --out directory")
                    provenance = current_provenance
                    if is_reference:
                        # Distinct prompt IDs may resolve to the same HF image blob.
                        # Crop coordinates depend on prompt ID, so these rows must not alias.
                        reference_bank[record["prompt_id"]] = values
                    else:
                        generated_bank[record["path"]] = values
                        for row in self.rows:
                            if row["path"] == record["path"]:
                                row["eval_seconds"]["inception_patch1024" if patch else "inception299"] = elapsed
                banks[patch] = generated_bank, reference_bank
        except Exception as exc:
            for metric in selected:
                self.failed(metric, exc)
            return
        finally:
            del extractor
            release_model()
        groups = sorted({(r["method"], r["res"]) for r in self.records})
        for metric in selected:
            self.status[metric] = {
                "status": "complete", "provenance": provenance,
                "estimator": "unbiased all-pairs polynomial degree 3" if metric.endswith("kid") else "empirical Gaussian FID",
                "bootstrap": "generated and real prompt clusters independently; shared real draw for paired deltas",
                "feature_assembly": ASSEMBLY_POLICY,
                "reference_rows": len(self.references),
                "reference_distinct_paths": len({r["path"] for r in self.references}),
                "reference_weighting": "one prompt cluster per manifest row; repeated image content retained",
            }
            bank, real = banks[metric.startswith("patch_")]
            statistic = kid if metric.endswith("kid") else fid
            for method, res in groups:
                try:
                    generated = cluster_features(self.records, bank, method=method, res=res)
                    summary = distribution_bootstrap(generated, real, statistic,
                        draws=self.args.bootstrap, seed=self.args.bootstrap_seed)
                    self.summaries.append({"method": method, "res": res, "metric": metric, **summary})
                    reference = self.args.reference_method
                    if method != reference and (reference, res) in groups:
                        aligned = cluster_features(self.records, bank, method=method, res=res, align_with=reference)
                        baseline = cluster_features(self.records, bank, method=reference, res=res, align_with=method)
                        delta = distribution_bootstrap(aligned, real, statistic, baseline=baseline,
                            draws=self.args.bootstrap, seed=self.args.bootstrap_seed)
                        self.paired.append({"method": method, "res": res, "metric": metric,
                                            "reference_method": reference, "delta_definition": "method_minus_reference",
                                            **delta})
                except Exception as exc:
                    self.failed(metric, ValueError(f"{method}/{res}: {exc}"))


if __name__ == "__main__":
    raise SystemExit(Evaluation(CLI_ARGS).run())
