import ast
from contextlib import closing
import hashlib
import signal
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
OFFICIAL_SOURCE = REPOSITORY / "vendor" / "bird_mini_dev" / "evaluation" / "evaluation_ex.py"
OFFICIAL_SHA256 = "da1bbcd4530be83692d7c650c814ea9704bb710d0c953eb75d02ccb38233cf89"
MISSING_SCORER_REASON = (
    "BIRD Mini-Dev scorer not fetched; run: python3 scripts/fetch_bird_scorer.py"
)


class FunctionTimedOut(Exception):
    pass


def immediate_func_timeout(_timeout, function, args=()):
    return function(*args)


def alarm_func_timeout(timeout, function, args=()):
    timer_fired = False

    def raise_timeout(_signum, _frame):
        nonlocal timer_fired
        timer_fired = True
        raise FunctionTimedOut()

    previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return function(*args)
    except sqlite3.OperationalError as error:
        if timer_fired:
            raise FunctionTimedOut() from error
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def fixture_execute_sql(predicted_sql, ground_truth, db_path, sql_dialect, calculate_ex):
    if sql_dialect != "SQLite":
        raise ValueError("these CPU fixtures support SQLite only")
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.set_progress_handler(lambda: None, 1000)
        predicted_rows = connection.execute(predicted_sql).fetchall()
        ground_truth_rows = connection.execute(ground_truth).fetchall()
    return calculate_ex(predicted_rows, ground_truth_rows)


def load_official_functions(**namespace):
    source_bytes = OFFICIAL_SOURCE.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != OFFICIAL_SHA256:
        raise AssertionError("pinned official evaluation_ex.py hash changed")
    source_path = str(OFFICIAL_SOURCE)
    tree = ast.parse(source_bytes, filename=source_path)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"calculate_ex", "execute_model"}
    ]
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, source_path, "exec"), namespace)
    return namespace


@unittest.skipUnless(OFFICIAL_SOURCE.is_file(), MISSING_SCORER_REASON)
class OfficialExecutionAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory(
            prefix="official-ex-", dir=Path(__file__).resolve().parent
        )
        cls.database = Path(cls.temporary_directory.name) / "birds.sqlite"
        with closing(sqlite3.connect(cls.database)) as connection:
            connection.execute("CREATE TABLE bird (id INTEGER PRIMARY KEY, name TEXT)")
            connection.executemany(
                "INSERT INTO bird VALUES (?, ?)", [(1, "Ada"), (2, "Grace")]
            )
            connection.commit()

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def rows(self, sql):
        with closing(sqlite3.connect(self.database)) as connection:
            return connection.execute(sql).fetchall()

    def test_direct_official_result_set_cases(self):
        official = load_official_functions(sys=sys)["calculate_ex"]
        ground_truth = self.rows("SELECT name FROM bird ORDER BY id")
        cases = {
            "correct": ("SELECT name FROM bird ORDER BY id", 1),
            "incorrect": ("SELECT name FROM bird WHERE id = 1", 0),
            "reordered": ("SELECT name FROM bird ORDER BY id DESC", 1),
            "duplicate_multiplicity": (
                "SELECT name FROM bird CROSS JOIN "
                "(SELECT 1 AS n UNION ALL SELECT 2)",
                1,
            ),
        }
        for name, (predicted_sql, expected) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(official(self.rows(predicted_sql), ground_truth), expected)

    def test_direct_official_empty_and_column_order_edge_cases(self):
        official = load_official_functions(sys=sys)["calculate_ex"]
        self.assertEqual(official([], []), 1)
        self.assertEqual(official([("Ada", 1)], [(1, "Ada")]), 0)

    def execute_model(self, predicted, ground_truth, index=3, timeout=1.0, timer=None):
        official = load_official_functions(
            sys=sys,
            func_timeout=timer or immediate_func_timeout,
            FunctionTimedOut=FunctionTimedOut,
            execute_sql=fixture_execute_sql,
        )
        return official["execute_model"](
            predicted, ground_truth, str(self.database), index, timeout, "SQLite"
        )

    def test_official_execute_model_success(self):
        result = self.execute_model(
            "SELECT name FROM bird ORDER BY id DESC",
            "SELECT name FROM bird ORDER BY id",
        )
        self.assertEqual(result, {"sql_idx": 3, "res": 1})

    def test_official_execute_model_sql_error(self):
        official = load_official_functions(sys=sys)["calculate_ex"]
        with self.assertRaises(sqlite3.OperationalError):
            fixture_execute_sql(
                "SELECT missing_column FROM bird",
                "SELECT 1",
                str(self.database),
                "SQLite",
                official,
            )
        result = self.execute_model("SELECT missing_column FROM bird", "SELECT 1")
        self.assertEqual(result, {"sql_idx": 3, "res": 0})

    @unittest.skipUnless(hasattr(signal, "setitimer"), "requires POSIX interval timers")
    def test_official_execute_model_real_sqlite_timeout(self):
        endless_query = (
            "WITH RECURSIVE counter(value) AS ("
            "SELECT 1 UNION ALL SELECT value + 1 FROM counter"
            ") SELECT sum(value) FROM counter"
        )
        official = load_official_functions(sys=sys)["calculate_ex"]
        with self.assertRaises(FunctionTimedOut):
            alarm_func_timeout(
                0.02,
                fixture_execute_sql,
                (
                    endless_query,
                    "SELECT 1",
                    str(self.database),
                    "SQLite",
                    official,
                ),
            )
        result = self.execute_model(
            endless_query, "SELECT 1", index=9, timeout=0.02, timer=alarm_func_timeout
        )
        self.assertEqual(result, {"sql_idx": 9, "res": 0})


if __name__ == "__main__":
    unittest.main()
