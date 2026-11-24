import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
CLI = REPOSITORY / "scripts" / "audit_bird_overlap.py"


class OverlapCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="overlap-", dir=Path(__file__).resolve().parent
        )
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_json(self, name, payload):
        path = self.directory / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def run_cli(self, left, right):
        return subprocess.run(
            [
                sys.executable,
                str(CLI),
                str(left),
                str(right),
                "--left-revision",
                "dev-test-revision",
                "--right-revision",
                "mini-dev-test-revision",
            ],
            cwd=REPOSITORY,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

    def test_exact_overlap(self):
        rows = [
            {
                "question_id": 7,
                "db_id": "birds",
                "question": "How many birds?",
            }
        ]
        result = self.run_cli(
            self.write_json("left.json", rows),
            self.write_json("right.json", rows),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["question_id_overlap"]["matched_identifiers"], ["7"])
        self.assertEqual(report["normalized_overlap"]["matched_pair_count"], 1)
        self.assertEqual(
            report["revisions"],
            {"left": "dev-test-revision", "right": "mini-dev-test-revision"},
        )

    def test_renamed_ids_with_identical_normalized_text(self):
        left = self.write_json(
            "left.json",
            [
                {
                    "question_id": "dev-9",
                    "db_id": "birds",
                    "question": "  How   many\nBIRDS? ",
                }
            ],
        )
        right = self.write_json(
            "right.json",
            [
                {
                    "question_id": "mini-42",
                    "database_id": "birds",
                    "question": "how many birds?",
                }
            ],
        )
        result = self.run_cli(left, right)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["question_id_overlap"]["count"], 0)
        match = report["normalized_overlap"]["matches"][0]
        self.assertEqual(match["left_question_ids"], ["dev-9"])
        self.assertEqual(match["right_question_ids"], ["mini-42"])

    def test_same_question_with_different_database_ids_does_not_match(self):
        left = self.write_json(
            "left.json",
            [{"question_id": 1, "db_id": "birds_dev", "question": "Count birds"}],
        )
        right = self.write_json(
            "right.json",
            [{"question_id": 2, "db_id": "birds_mini", "question": "count birds"}],
        )
        result = self.run_cli(left, right)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["question_id_overlap"]["count"], 0)
        self.assertEqual(report["normalized_overlap"]["matched_pair_count"], 0)

    def test_malformed_inputs_are_rejected(self):
        valid = self.write_json("valid.json", [])
        malformed_json = self.directory / "malformed.json"
        malformed_json.write_text("[not-json", encoding="utf-8")
        wrong_schema = self.write_json("wrong-schema.json", {"question_id": 1})
        for path in (malformed_json, wrong_schema):
            with self.subTest(path=path.name):
                result = self.run_cli(path, valid)
                self.assertEqual(result.returncode, 2)
                self.assertIn("left:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_duplicate_ids_are_rejected(self):
        duplicate = self.write_json(
            "duplicate.json",
            [
                {"question_id": "same", "db_id": "birds", "question": "One"},
                {"question_id": "same", "db_id": "birds", "question": "Two"},
            ],
        )
        result = self.run_cli(duplicate, self.write_json("right.json", []))
        self.assertEqual(result.returncode, 2)
        self.assertIn("duplicate question_id 'same'", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
