"""CPU regression cases for the native launch boundary."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


GUARD = Path(__file__).with_name("launch_gated.py")


class GatedLaunchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        launcher = self.root / "native-wrapper"
        launcher.write_text('#!/bin/sh\nmkdir "$BIRD_RUN_ROOT"\nprintf "%s" "$1" > "$BIRD_RUN_ROOT/started"\n')
        launcher.chmod(0o700)
        actor = self.root / "actor.py"
        actor.write_text("candidate actor\n")
        kernel = self.root / "kernel.py"
        kernel.write_text("candidate kernel\n")
        endpoint = {"bind_domain": "eval-host", "port": 15024}
        self.manifest = {
            "job_root": str(self.root),
            "launcher": {**self.pin(launcher), "args": ["pinned-argument"]},
            "actor": self.pin(actor),
            "kernel": self.pin(kernel),
            "devices": [f"gpu-{rank}" for rank in range(4)],
            "evaluation_endpoints": {"validation": [endpoint], "milestones": []},
            "validation_watcher_host": "watcher-host",
            "gate_window_seconds": 120,
        }
        self.workers = {"workers": [
            {
                "rank": rank,
                "device_uuid": device,
                "actor_sha256": self.manifest["actor"]["sha256"],
                "kernel_sha256": self.manifest["kernel"]["sha256"],
                "seal_installed_by_production_path": True,
                "supported_case": {"fixed_delta": 2, "q_positive_finite": True, "k_positive_finite": True},
                "negative_case": {"fixed_delta": 0},
            }
            for rank, device in enumerate(self.manifest["devices"])
        ]}
        self.validation = {
            "evaluation_endpoints": deepcopy(self.manifest["evaluation_endpoints"]),
            "watcher_ready": True,
            "watcher_host": "watcher-host",
            "published_via_watcher": True,
            "fresh_gate_decision": "CONTINUE",
            "late_gate_decision": "STOP",
            "checkpoint_to_receipt_seconds": 90,
        }

    def pin(self, path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def run_guard(self, *options):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(self.manifest))
        manifest_digest = self.pin(manifest)["sha256"]
        workers = self.root / "workers.json"
        workers.write_text(json.dumps({"manifest_sha256": manifest_digest, **self.workers}))
        validation = self.root / "validation.json"
        validation.write_text(json.dumps({"manifest_sha256": manifest_digest, **self.validation}))
        return subprocess.run(
            [sys.executable, str(GUARD), "--manifest", str(manifest), "--worker-proof", str(workers),
             "--validation-proof", str(validation), *options],
            capture_output=True, text=True, timeout=10,
        )

    def assert_blocked(self, reason):
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0, result)
        self.assertIn(reason, result.stderr)
        self.assertFalse((self.root / "native").exists())
        self.assertFalse((self.root / "gated-launch.claim").exists())

    def test_check_is_read_only_and_valid_launch_runs_once(self):
        checked = self.run_guard("--check")
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertFalse((self.root / "gated-launch.claim").exists())
        result = self.run_guard()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "native/started").read_text(), "pinned-argument")
        self.assertIn("already exists", self.run_guard().stderr)

    def test_stock_actor_installed_after_worker_proof_is_blocked(self):
        Path(self.manifest["actor"]["path"]).write_text("restored stock actor\n")
        self.assert_blocked("actor hash differs")

    def test_matching_files_without_fixed_worker_branch_are_blocked(self):
        self.workers["workers"][3]["supported_case"]["fixed_delta"] = 0
        self.assert_blocked("rank 3 did not prove the fixed backward branch")

    def test_validation_and_milestone_port_collision_is_blocked(self):
        endpoints = self.manifest["evaluation_endpoints"]
        endpoints["milestones"] = deepcopy(endpoints["validation"])
        self.validation["evaluation_endpoints"] = deepcopy(endpoints)
        self.assert_blocked("evaluation port collision")

    def test_watcher_on_wrong_host_is_blocked(self):
        self.validation["watcher_host"] = "trainer"
        self.assert_blocked("wrong host")

    def test_receipt_at_gate_deadline_is_blocked(self):
        self.validation["checkpoint_to_receipt_seconds"] = 120
        self.assert_blocked("did not fit the configured gate window")

    def test_proof_from_a_prior_launch_is_blocked(self):
        self.workers["manifest_sha256"] = "0" * 64
        self.assert_blocked("different launch manifest")


if __name__ == "__main__":
    unittest.main()
