"""Adversarial acceptance checks for restoration parity receipts."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import types
import unittest

PATH = Path(__file__).resolve().parents[1] / "check_metric_extra_parity.py"
SPEC = importlib.util.spec_from_file_location("extra_parity", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class RestorationParityAcceptance(unittest.TestCase):
    def test_declared_tolerance_accepts_roundoff(self):
        self.assertTrue(MOD.compare_values([[1.0]], [[1.0 + 1e-6]], "ir")["pass"])

    def test_drift_is_failure(self):
        self.assertFalse(MOD.compare_values([[0.3]], [[0.3001]], "hpsv2")["pass"])

    def test_inception_compares_every_logit(self):
        candidate = [0.0] * 1008
        candidate[-1] = 0.001
        result = MOD.compare_values([[0.0] * 1008], [candidate], "is")
        self.assertFalse(result["pass"])
        self.assertEqual(result["values_compared"], 1008)

    def test_nonfinite_and_partial_vectors_are_rejected(self):
        for reference, candidate in (([[1.0]], [[math.nan]]), ([[1.0, 2.0]], [[1.0]]),
                                     ([[1.0]], []), ([], [])):
            with self.assertRaises(ValueError):
                MOD.compare_values(reference, candidate, "ir")

    def test_altered_input_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image"
            path.write_bytes(b"original image")
            record = {"path": str(path), "image_sha256": MOD.sha256(path)}
            path.write_bytes(b"changed image")
            with self.assertRaisesRegex(ValueError, "Input identity changed"):
                MOD.check_inputs([record])

    def test_missing_metric_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = {"policy": MOD.POLICY, "manifest_sha256": "fixture", "inputs": [],
                     "tolerances": MOD.TOLERANCES, "device": "cpu", "source_hashes": {},
                     "asset_receipt": {}, "cache_configuration": {},
                     "results": {}, "limitations": []}
            for name in ("reference", "candidate"):
                (root / name).write_text(json.dumps(value))
            args = types.SimpleNamespace(reference=root / "reference", candidate=root / "candidate",
                                         metrics="ir", out=root / "report")
            self.assertEqual(MOD.compare(args), 1)
            self.assertFalse(json.loads(args.out.read_text())["passed"])
            changed = copy.deepcopy(value)
            changed["manifest_sha256"] = "changed"
            args.candidate.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                MOD.compare(args)


if __name__ == "__main__":
    unittest.main()
