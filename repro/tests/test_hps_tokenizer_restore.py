import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "metrics_environment/restore_hps_tokenizer.py"
SPEC = importlib.util.spec_from_file_location("hps_restore", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class TokenizerRestoreTests(unittest.TestCase):
    def test_bad_source_does_not_create_target(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.write_bytes(b"wrong tokenizer")
            with self.assertRaises(ValueError):
                MOD.restore(source, target)
            self.assertFalse(target.exists())

    def test_different_existing_target_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.write_bytes(b"fixture")
            target.write_bytes(b"different existing artifact")
            with patch.object(MOD, "EXPECTED", hashlib.sha256(b"fixture").hexdigest()):
                with self.assertRaises(ValueError):
                    MOD.restore(source, target)
            self.assertEqual(target.read_bytes(), b"different existing artifact")

    def test_copy_and_idempotence_verify_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.write_bytes(b"fixture")
            with patch.object(MOD, "EXPECTED", hashlib.sha256(b"fixture").hexdigest()):
                result = MOD.restore(source, target)
                self.assertEqual(result["action"], "copied_missing_package_data")
                self.assertEqual(target.read_bytes(), source.read_bytes())
                self.assertEqual(MOD.restore(source, target)["action"], "already_present_verified")


if __name__ == "__main__":
    unittest.main()
