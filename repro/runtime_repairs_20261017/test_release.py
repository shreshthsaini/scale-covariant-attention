"""No scheduler calls or real queue mutations: every queue is temporary."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("repair_release", Path(__file__).with_name("release.py"))
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class RepairReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / "taskq"
        for state in ("failed", "hold_review", "pending", "running", "done", "quarantine", "recovered"):
            (self.queue / state).mkdir(parents=True)
        evidence = self.root / "terminal_failure.json"
        evidence.write_text('{"terminal_failures":2}')
        self.manifest = {"schema_version": 1, "owning_job": MOD.OWNING_JOB,
                         "terminal_failure_evidence": {"path": evidence.name, "sha256": MOD.sha256(evidence)},
                         "repairs": []}
        for original, replacement in MOD.EXPECTED.items():
            old, new = self.queue / "failed" / original, self.queue / "hold_review" / replacement
            old.write_text("#!/bin/bash\necho original " + original + "\n")
            new.write_text("#!/bin/bash\necho replacement " + replacement + "\n")
            self.manifest["repairs"].append({"original": original, "replacement": replacement,
                                              "original_sha256": MOD.sha256(old), "replacement_sha256": MOD.sha256(new)})
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.outputs = ["", "1075814|FAILED|1:0\n1075814.0|FAILED|1:0\n", ""]
        self.calls = []

    def runner(self, command):
        self.calls.append(command)
        return subprocess.CompletedProcess(command, 0, self.outputs[len(self.calls) - 1], "")

    def run_release(self, apply=False):
        self.calls = []
        receipt = {}
        MOD.release(self.manifest, self.queue, receipt, apply, self.runner)
        return receipt

    def snapshot(self):
        return {str(p.relative_to(self.queue)): MOD.sha256(p) for p in self.queue.rglob("*")
                if p.is_file() and p.name != ".runtime_repairs_1076947.lock"}

    def test_manifest_evidence_and_names(self):
        self.assertEqual(MOD.load_manifest(self.manifest_path, self.root), self.manifest)
        bad = copy.deepcopy(self.manifest)
        bad["repairs"][0]["replacement"] = "../escape.sh"
        self.manifest_path.write_text(json.dumps(bad))
        with self.assertRaises(MOD.Refusal): MOD.load_manifest(self.manifest_path, self.root)
        self.manifest_path.write_text(json.dumps(self.manifest))
        (self.root / "terminal_failure.json").write_text("changed")
        with self.assertRaises(MOD.Refusal): MOD.load_manifest(self.manifest_path, self.root)

    def test_default_dry_run_preserves_every_task(self):
        before = self.snapshot()
        receipt = self.run_release()
        self.assertEqual(receipt["status"], "dry_run")
        self.assertEqual(before, self.snapshot())
        self.assertEqual(len(self.calls), 3)
        self.assertFalse((self.queue / ".runtime_repairs_1076947.lock").exists())

    def test_apply_and_reentry_in_pending_running_done(self):
        receipt = self.run_release(True)
        self.assertEqual(receipt["status"], "applied")
        for row in self.manifest["repairs"]:
            self.assertFalse((self.queue / "failed" / row["original"]).exists())
            self.assertEqual(MOD.sha256(self.queue / MOD.RECOVERY / row["original"]), row["original_sha256"])
        for state in ("pending", "running", "done"):
            for row in self.manifest["repairs"]:
                current = self.queue / "pending" / row["replacement"]
                if not current.exists():
                    candidates = list((self.queue / "running").glob(row["replacement"] + "*"))
                    current = candidates[0]
                target = self.queue / state / (row["replacement"] + (".worker" if state == "running" else ""))
                if current != target: current.rename(target)
            self.assertEqual(self.run_release(True)["status"], "already_released")

    def test_running_pending_completing_and_unclear_states_refuse(self):
        before = self.snapshot()
        for state in ("RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "FAILED+"):
            self.outputs[0] = "1075814|" + state + "\n"
            with self.assertRaises(MOD.Refusal): self.run_release(True)
            self.assertEqual(before, self.snapshot())

    def test_live_or_unknown_sacct_step_refuses(self):
        for raw in ("1075814|FAILED|1:0\n1075814.0|RUNNING|0:0\n", "", "1075814|FAILED+|1:0\n"):
            self.outputs[1] = raw
            with self.assertRaises(MOD.Refusal): self.run_release(True)

    def test_any_squeue_step_refuses_even_if_terminal(self):
        self.outputs[2] = "1075814.0\n"
        with self.assertRaises(MOD.Refusal): self.run_release(True)

    def test_account_wide_queries_allow_unrelated_jobs_and_steps(self):
        self.outputs[0] = "1076947|RUNNING\n1076923|PENDING\n10758140|RUNNING\n"
        self.outputs[2] = "1076947.batch\n10758140.0\n"
        self.assertEqual(self.run_release()["status"], "dry_run")
        self.assertIn("--user", self.calls[0])
        self.assertNotIn("--jobs", self.calls[0])
        self.assertEqual(self.calls[2][-1], "%i")

    def test_scheduler_error_warning_timeout_refuses(self):
        for result in (subprocess.CompletedProcess([], 1, "", "error"),
                       subprocess.CompletedProcess([], 0, "", "warning")):
            with self.assertRaises(MOD.Refusal): MOD.release(self.manifest, self.queue, {}, True, lambda command: result)
        with self.assertRaises(MOD.Refusal):
            MOD.release(self.manifest, self.queue, {}, True,
                        lambda command: (_ for _ in ()).throw(subprocess.TimeoutExpired(command, 15)))

    def test_every_repair_is_preflighted_before_any_task_mutation(self):
        row = self.manifest["repairs"][1]
        (self.queue / "hold_review" / row["replacement"]).write_text("wrong bytes")
        before = self.snapshot()
        with self.assertRaises(MOD.Refusal): self.run_release(True)
        self.assertEqual(before, self.snapshot())

    def test_quarantine_unexpected_state_and_duplicate_refuse(self):
        row = self.manifest["repairs"][0]
        held = self.queue / "hold_review" / row["replacement"]
        for state in ("quarantine", "failed", "hold_unknown"):
            target = self.queue / state / row["replacement"]
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(held.read_bytes())
            with self.assertRaises(MOD.Refusal): self.run_release(True)
            target.unlink()

    def test_resume_after_archive_and_interrupted_archive_link(self):
        archived_dir = self.queue / MOD.RECOVERY
        archived_dir.mkdir()
        for i, row in enumerate(self.manifest["repairs"]):
            source, target = self.queue / "failed" / row["original"], archived_dir / row["original"]
            if i == 0: source.rename(target)
            else: os.link(source, target)
        self.assertEqual(self.run_release(True)["status"], "applied")
        self.assertFalse(any((self.queue / "failed").iterdir()))

    def test_resume_after_atomic_publication_before_held_unlink(self):
        self.run_release(True)
        row = self.manifest["repairs"][0]
        pending = self.queue / "pending" / row["replacement"]
        held = self.queue / "hold_review" / row["replacement"]
        os.link(pending, held)
        self.assertEqual(self.run_release(True)["status"], "applied")
        self.assertFalse(held.exists())

    def interrupted_link(self, boundary):
        original_link = os.link
        interrupted = False
        def linked_then_interrupted(source, target, **kwargs):
            nonlocal interrupted
            original_link(source, target, **kwargs)
            at_boundary = (Path(target).parent.name == "runtime_repairs_1076947" if boundary == "archive"
                           else Path(target).parent.name == "pending")
            if at_boundary and not interrupted:
                interrupted = True
                raise OSError("Simulated interruption after atomic link")
        with patch.object(MOD.os, "link", side_effect=linked_then_interrupted):
            with self.assertRaises(OSError): self.run_release(True)
        self.assertTrue(interrupted)
        self.assertEqual(self.run_release(True)["status"], "applied")
        self.assertEqual(self.run_release(True)["status"], "already_released")

    def test_real_interrupted_archive_link_resumes(self):
        self.interrupted_link("archive")

    def test_real_interrupted_publication_link_resumes(self):
        self.interrupted_link("publication")

    def test_dispatcher_can_claim_immediately_after_publication(self):
        original_link = os.link
        def linked_then_claimed(source, target, **kwargs):
            original_link(source, target, **kwargs)
            if Path(target).parent.name == "pending":
                Path(target).rename(self.queue / "running" / (Path(target).name + ".worker"))
        with patch.object(MOD.os, "link", side_effect=linked_then_claimed):
            self.assertEqual(self.run_release(True)["status"], "applied")
        self.assertEqual(self.run_release(True)["status"], "already_released")

    def test_exclusive_move_cannot_overwrite(self):
        source, target = self.root / "source", self.root / "target"
        source.write_text("source"); target.write_text("target")
        with self.assertRaises(FileExistsError): MOD.exclusive_move(source, target)
        self.assertEqual(source.read_text(), "source")
        self.assertEqual(target.read_text(), "target")

    def test_lock_contention_refuses(self):
        with MOD.apply_lock(self.queue):
            with self.assertRaises(MOD.Refusal): self.run_release(True)

    def test_receipts_never_overwrite(self):
        receipt = self.root / "receipt.json"
        MOD.save_receipt(receipt, {"status": "dry_run"})
        with self.assertRaises(FileExistsError): MOD.save_receipt(receipt, {"status": "changed"})
        self.assertEqual(json.loads(receipt.read_text())["status"], "dry_run")


if __name__ == "__main__": unittest.main()
