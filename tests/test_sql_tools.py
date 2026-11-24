"""Pin read-only SQLite execution and EX scoring behaviour in native/bird_evaluator/sql_tools.py."""

from contextlib import closing
import hashlib
import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[1]
NATIVE_DIR = REPOSITORY / "native"
if str(NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(NATIVE_DIR))

from bird_evaluator import sql_tools


MISSING_SCORER_REASON = (
    "BIRD Mini-Dev scorer not fetched; run: python3 scripts/fetch_bird_scorer.py"
)
MISSING_FUNC_TIMEOUT_REASON = (
    "func_timeout not installed; run: pip install func_timeout"
)
HAS_FUNC_TIMEOUT = importlib.util.find_spec("func_timeout") is not None


class SqlToolsDependencyTests(unittest.TestCase):
    def test_missing_scorer_raises_with_fetch_command(self):
        with tempfile.TemporaryDirectory(prefix="sql-tools-noscorer-") as directory:
            missing_scorer = Path(directory) / "absent_evaluation_ex.py"
            db_path = Path(directory) / "birds.sqlite"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("CREATE TABLE bird (id INTEGER PRIMARY KEY, name TEXT)")
                conn.commit()
            with patch.object(sql_tools, "SCORER_PATH", missing_scorer):
                with self.assertRaises(FileNotFoundError) as ctx:
                    sql_tools.ex_score("SELECT 1", "SELECT 1", str(db_path))
            self.assertIn("scripts/fetch_bird_scorer.py", str(ctx.exception))

    def test_missing_func_timeout_raises_with_install_command(self):
        with patch.object(sql_tools, "_load_calculate_ex", return_value=lambda a, b: 1):
            with patch.dict(sys.modules, {"func_timeout": None}):
                with self.assertRaises(ModuleNotFoundError) as ctx:
                    sql_tools.ex_score("SELECT 1", "SELECT 1", "ignored.sqlite")
        self.assertIn("pip install func_timeout", str(ctx.exception))


@unittest.skipUnless(sql_tools.SCORER_PATH.is_file(), MISSING_SCORER_REASON)
class SqlToolsExecuteReadOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix="sql-tools-ro-")
        cls.database = Path(cls.temporary_directory.name) / "birds.sqlite"
        with closing(sqlite3.connect(cls.database)) as conn:
            conn.execute("CREATE TABLE bird (id INTEGER PRIMARY KEY, name TEXT)")
            conn.executemany(
                "INSERT INTO bird VALUES (?, ?)", [(1, "Ada"), (2, "Grace")]
            )
            conn.commit()

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_execute_read_only_result_comparison(self):
        gold = "SELECT name FROM bird ORDER BY id"
        cases = (
            ("exact", "SELECT name FROM bird ORDER BY id", gold, 1),
            ("reordered", "SELECT name FROM bird ORDER BY id DESC", gold, 1),
            (
                "duplicates",
                "SELECT name FROM bird CROSS JOIN (SELECT 1 AS n UNION ALL SELECT 2)",
                gold,
                1,
            ),
            ("mismatch", "SELECT name FROM bird WHERE id = 1", gold, 0),
            (
                "empty_equal",
                "SELECT name FROM bird WHERE id = 99",
                "SELECT name FROM bird WHERE id = 98",
                1,
            ),
            ("column_order", "SELECT name, id FROM bird", "SELECT id, name FROM bird", 0),
        )
        for label, predicted, reference, expected in cases:
            with self.subTest(case=label):
                self.assertEqual(
                    sql_tools.execute_read_only(predicted, reference, str(self.database)),
                    expected,
                )


@unittest.skipUnless(sql_tools.SCORER_PATH.is_file(), MISSING_SCORER_REASON)
@unittest.skipUnless(HAS_FUNC_TIMEOUT, MISSING_FUNC_TIMEOUT_REASON)
class SqlToolsExScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix="sql-tools-ex-")
        cls.root = Path(cls.temporary_directory.name)
        cls.database = cls.root / "birds.sqlite"
        cls.missing_database = cls.root / "absent.sqlite"
        with closing(sqlite3.connect(cls.database)) as conn:
            conn.execute("CREATE TABLE bird (id INTEGER PRIMARY KEY, name TEXT)")
            conn.executemany(
                "INSERT INTO bird VALUES (?, ?)", [(1, "Ada"), (2, "Grace")]
            )
            conn.commit()
        cls.before_digest = hashlib.sha256(cls.database.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_ex_score_success_and_sql_error(self):
        ok = sql_tools.ex_score(
            "SELECT name FROM bird ORDER BY id DESC",
            "SELECT name FROM bird ORDER BY id",
            str(self.database),
        )
        self.assertEqual(ok, {"correct": True, "raw_result": {"sql_idx": 0, "res": 1}})

        bad = sql_tools.ex_score(
            "SELECT missing_column FROM bird",
            "SELECT 1",
            str(self.database),
        )
        self.assertEqual(bad, {"correct": False, "raw_result": {"sql_idx": 0, "res": 0}})
        self.assertEqual(
            hashlib.sha256(self.database.read_bytes()).hexdigest(),
            self.before_digest,
        )

    def test_ex_score_timeout_returns_zero_via_func_timeout(self):
        from func_timeout import FunctionTimedOut, func_timeout as real_func_timeout

        slow_query = (
            "WITH RECURSIVE counter(value) AS ("
            "SELECT 1 UNION ALL SELECT value + 1 FROM counter WHERE value < 3000000"
            ") SELECT sum(value) FROM counter"
        )
        timed_out_seen = False

        def spy_func_timeout(timeout, func, args=()):
            nonlocal timed_out_seen
            try:
                return real_func_timeout(timeout, func, args=args)
            except FunctionTimedOut:
                timed_out_seen = True
                raise

        with patch.object(
            sql_tools,
            "_require_func_timeout",
            return_value=(spy_func_timeout, FunctionTimedOut),
        ):
            result = sql_tools.ex_score(
                slow_query,
                "SELECT 1",
                str(self.database),
                timeout_seconds=0.05,
            )
        self.assertTrue(timed_out_seen)
        self.assertEqual(result, {"correct": False, "raw_result": {"sql_idx": 0, "res": 0}})

    def test_ex_score_missing_database_raises(self):
        with self.assertRaises(sql_tools.DatabaseUnavailableError):
            sql_tools.ex_score("SELECT 1", "SELECT 1", str(self.missing_database))
        self.assertFalse(self.missing_database.exists())


if __name__ == "__main__":
    unittest.main()
