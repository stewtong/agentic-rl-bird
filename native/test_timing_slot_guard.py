"""CPU regression cases for the timing-slot guard (isolation failures)."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import timing_slot_guard as guard

NOW = datetime(2026, 11, 23, 10, 43, 30, tzinfo=timezone.utc)
SLUG = "q512-frozen-strict-r1"


def row(disposition, pid="1"):
    return {"pid": pid, "name": "python3", "cores": 1.0, "disposition": disposition}


def sample(utc, *rows, gpu=()):
    return {"utc": utc, "rows": list(rows), "gpu_processes": list(gpu)}


class TimingSlotGuardTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.job = self.make_job("job")
        self.registry = self.root / "registry"
        self.registry.mkdir()
        self.lock = self.root / "node-a.lock"
        self.slot = {"slug": SLUG, "job_root": str(self.job), "attempt": 1, "node": "A",
                     "node_lock": str(self.lock), "attempt_registry": str(self.registry),
                     "max_sample_age_seconds": 30, "max_gap_seconds": 60}
        self.sampler = {"node": "A", "running": True, "first_sample_utc": "2026-11-23T10:40:00Z",
                        "last_sample_utc": "2026-11-23T10:43:20Z", "gpu_processes": [],
                        "latest_rows": [row("baseline_daemon")]}
        self.native = {"started_utc": "2026-11-23T10:43:38Z", "first_bar_utc": "2026-11-23T10:46:31Z",
                       "finished_utc": "2026-11-23T10:48:00Z", "exit_code": 0}

    def make_job(self, name):
        path = self.root / name
        path.mkdir()
        return path

    def write(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data))
        return path

    def run_preflight(self):
        return guard.preflight(self.write("slot.json", self.slot), self.write("sampler.json", self.sampler), NOW)

    def samples(self, first="2026-11-23T10:43:30Z", last="2026-11-23T10:48:10Z", extra=()):
        middle = [sample(f"2026-11-23T10:{m}Z", row("native_job")) for m in ("44:20", "45:10", "46:00", "46:50", "47:40")]
        middle = [s for s in middle if first < s["utc"] < last]  # sampling starts at `first`, ends at `last`
        return [sample(first, row("sampler")), *middle, sample(last, row("sampler")), *extra]

    def close(self, samples, job=None):
        return guard.close(job or self.job, self.write("samples.json", samples), self.write("native.json", self.native))

    def failed_original(self):
        self.run_preflight()
        late = self.samples(first="2026-11-23T10:47:00Z")
        self.assertEqual(self.close(late)["status"], "needs_rerun")
        rerun = self.make_job("job-r1b")
        self.slot.update(job_root=str(rerun), attempt=2, rerun_of_job_root=str(self.job))
        return rerun

    def test_clean_slot_is_accepted_and_releases_lock(self):
        self.run_preflight()
        self.assertTrue(self.lock.exists())
        self.assertEqual(self.close(self.samples())["status"], "accepted")
        self.assertFalse(self.lock.exists())

    def test_late_sampling_needs_rerun(self):
        self.run_preflight()
        result = self.close(self.samples(first="2026-11-23T10:47:00Z"))
        self.assertFalse(result["covers_first_bar"])
        self.assertEqual(result["status"], "needs_rerun")

    def test_sampler_gap_or_early_stop_needs_rerun(self):
        self.run_preflight()
        gappy = [s for s in self.samples() if s["utc"][14:16] not in ("45", "46")]
        result = self.close(gappy)
        self.assertGreater(result["largest_gap_seconds"], 60)
        self.assertEqual(result["status"], "needs_rerun")

    def test_sampler_stopping_before_finish_needs_rerun(self):
        self.run_preflight()
        result = self.close(self.samples(last="2026-11-23T10:47:50Z"))
        self.assertFalse(result["covers_finish"])
        self.assertEqual(result["status"], "needs_rerun")

    def test_operator_work_in_rerun_is_unresolved(self):
        rerun = self.failed_original()
        self.run_preflight()
        busy = sample("2026-11-23T10:47:00Z", row("operator_test_work", "3599565"))
        result = self.close(self.samples(extra=[busy]), job=rerun)
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["work_rows"][0]["pid"], "3599565")

    def test_unknown_process_or_gpu_process_counts_as_overlap(self):
        self.run_preflight()
        unknown = sample("2026-11-23T10:47:10Z", row("unknown"))
        self.assertEqual(self.close(self.samples(extra=[unknown]))["status"], "needs_rerun")

    def test_unattributed_gpu_process_counts_as_overlap(self):
        self.run_preflight()
        gpu = sample("2026-11-23T10:47:10Z", gpu=[row("unknown", "77")])
        self.assertEqual(self.close(self.samples(extra=[gpu]))["status"], "needs_rerun")

    def test_non_integer_exit_code_is_rejected(self):
        self.run_preflight()
        self.native["exit_code"] = False
        with self.assertRaisesRegex(ValueError, "integer"):
            self.close(self.samples())

    def test_repeated_original_is_refused(self):
        self.run_preflight()
        self.close(self.samples(first="2026-11-23T10:47:00Z"))
        self.slot["job_root"] = str(self.make_job("job-again"))
        with self.assertRaises(FileExistsError):
            self.run_preflight()
        self.assertFalse(self.lock.exists())
        self.assertFalse((self.root / "job-again" / guard.MARKER).exists())

    def test_third_attempt_is_refused(self):
        self.slot.update(attempt=3, rerun_of_job_root=str(self.job))
        with self.assertRaisesRegex(ValueError, "attempt"):
            self.run_preflight()

    def test_rerun_needs_a_failed_original(self):
        self.run_preflight()
        self.close(self.samples())
        self.slot.update(job_root=str(self.make_job("job-r1b")), attempt=2, rerun_of_job_root=str(self.job))
        with self.assertRaisesRegex(ValueError, "did not need a rerun"):
            self.run_preflight()

    def test_stale_or_stopped_sampler_blocks_launch(self):
        self.sampler["last_sample_utc"] = "2026-11-23T10:40:00Z"
        with self.assertRaisesRegex(ValueError, "old"):
            self.run_preflight()
        self.sampler.update(last_sample_utc="2026-11-23T10:43:20Z", running=False)
        with self.assertRaisesRegex(ValueError, "not running"):
            self.run_preflight()
        self.assertFalse(self.lock.exists())

    def test_foreign_work_or_gpu_process_blocks_launch(self):
        self.sampler["latest_rows"] = [row("operator_test_work")]
        with self.assertRaisesRegex(ValueError, "non-exempt"):
            self.run_preflight()
        self.sampler.update(latest_rows=[], gpu_processes=[{"pid": 9, "gpu": 6}])
        with self.assertRaisesRegex(ValueError, "GPU processes"):
            self.run_preflight()

    def test_held_node_lock_blocks_launch_and_leaves_no_marker(self):
        self.lock.write_text(json.dumps({"slot": "other", "job_root": "/elsewhere"}))
        with self.assertRaises(FileExistsError):
            self.run_preflight()
        self.assertFalse((self.job / guard.MARKER).exists())
        self.assertFalse(any(self.registry.iterdir()))

    def test_close_refuses_a_lock_held_by_another_slot(self):
        self.run_preflight()
        self.lock.write_text(json.dumps({"slot": "other", "job_root": "/elsewhere"}))
        with self.assertRaisesRegex(ValueError, "another slot"):
            self.close(self.samples())
        self.assertFalse((self.job / guard.DISPOSITION).exists())

    def test_marker_blocks_reuse_of_job_root(self):
        self.run_preflight()
        self.lock.unlink()
        with self.assertRaisesRegex(ValueError, "fresh root"):
            self.run_preflight()


if __name__ == "__main__":
    unittest.main()
