"""Read-only SQLite tools and pinned official Mini-Dev EX execution comparison."""

from __future__ import annotations

import ast
from contextlib import closing
from functools import lru_cache
import hashlib
import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import quote


MAX_DISPLAY_ROWS = 50
READ_START = re.compile(r"^(?:select|with|values)\b", re.IGNORECASE | re.DOTALL)
SCORER_PATH = (
    Path(__file__).resolve().parents[2]
    / "vendor"
    / "bird_mini_dev"
    / "evaluation"
    / "evaluation_ex.py"
)
SCORER_SHA256 = "da1bbcd4530be83692d7c650c814ea9704bb710d0c953eb75d02ccb38233cf89"
FETCH_COMMAND = "python3 scripts/fetch_bird_scorer.py"
FUNC_TIMEOUT_INSTALL = "pip install func_timeout"


class DatabaseUnavailableError(RuntimeError):
    """The SQLite database could not be opened for read-only use."""


def display_cell(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": type(value).__name__, "byte_length": len(value)}
    if isinstance(value, str) and len(value) > 256 and value.startswith(("0x", "0X")):
        try:
            decoded = bytes.fromhex(value[2:])
        except ValueError:
            pass
        else:
            return {"type": "hex", "byte_length": len(decoded)}
    return value


def display_row(row):
    return tuple(display_cell(value) for value in row)


def connection_uri(path):
    return f"file:{quote(str(Path(path).resolve()), safe='/')}?mode=ro&immutable=1"


def open_read_only(path):
    connection = None
    try:
        connection = sqlite3.connect(connection_uri(path), uri=True, timeout=0.05)
        connection.execute("PRAGMA query_only=ON")
        return connection
    except sqlite3.OperationalError as exc:
        if connection is not None:
            connection.close()
        raise DatabaseUnavailableError(f"unable to open read-only database: {path}") from exc


def read_only_statement(sql):
    return isinstance(sql, str) and bool(READ_START.match(sql.strip()))


def execute_query(path, sql, timeout_seconds=2.0):
    start = time.monotonic()
    connection = open_read_only(path)
    try:
        if not read_only_statement(sql):
            raise ValueError("read-only SQL statements are required")
        connection.set_progress_handler(lambda: 1 if time.monotonic() - start >= timeout_seconds else 0, 10_000)
        cursor = connection.execute(sql)
        columns = [item[0] for item in cursor.description or []]
        rows = cursor.fetchmany(MAX_DISPLAY_ROWS + 1)
        timed_out = time.monotonic() - start >= timeout_seconds
        return {
            "columns": columns,
            "rows": [display_row(row) for row in rows[:MAX_DISPLAY_ROWS]],
            "truncated": len(rows) > MAX_DISPLAY_ROWS,
            "timed_out": timed_out,
        }
    finally:
        connection.close()


def describe_database(path, table_name=None):
    connection = open_read_only(path)
    try:
        tables = table_name or connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        if isinstance(table_name, str):
            tables = [(table_name,)]
        output = {}
        for (table,) in tables:
            identifier = table.replace('"', '""')
            columns = connection.execute(f'PRAGMA table_info("{identifier}")').fetchall()
            if not columns:
                raise ValueError(f"unknown table: {table}")
            sample = connection.execute(f'SELECT * FROM "{identifier}" LIMIT 3').fetchall()
            output[table] = {
                "columns": [{"name": col[1], "dtype": col[2], "required": bool(col[3]), "pk": bool(col[5])} for col in columns],
                "sample_rows": [display_row(row) for row in sample],
                "create_sql": connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone()[0],
            }
        return output
    finally:
        connection.close()


def final_sql(text):
    blocks = re.findall(r"```(?:sql|sqlite)?\s*\n?(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if len(blocks) != 1:
        raise ValueError("final response must contain exactly one SQL code block")
    query = blocks[0].strip().rstrip(";").strip()
    if not read_only_statement(query):
        raise ValueError("one read-only SQL statement is required")
    return query


@lru_cache(maxsize=4)
def _load_calculate_ex(source_path_text: str):
    source_path = Path(source_path_text)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"BIRD Mini-Dev scorer not found at {source_path}; run: {FETCH_COMMAND}"
        )
    source_bytes = source_path.read_bytes()
    actual_digest = hashlib.sha256(source_bytes).hexdigest()
    if actual_digest != SCORER_SHA256:
        raise RuntimeError(
            f"BIRD Mini-Dev scorer hash mismatch at {source_path}: expected {SCORER_SHA256}, got {actual_digest}"
        )
    parsed = ast.parse(source_bytes, filename=str(source_path))
    functions = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "calculate_ex"
    ]
    if len(functions) != 1:
        raise RuntimeError(f"calculate_ex definition not found in {source_path}")
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {"sys": sys}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["calculate_ex"]


def _require_func_timeout():
    try:
        from func_timeout import FunctionTimedOut, func_timeout
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"func_timeout is required for ex_score; install with: {FUNC_TIMEOUT_INSTALL}"
        ) from exc
    return func_timeout, FunctionTimedOut


def _run_pair_read_only(db_path, candidate_sql, reference_sql):
    with closing(open_read_only(db_path)) as conn:
        candidate_rows = conn.execute(candidate_sql).fetchall()
        reference_rows = conn.execute(reference_sql).fetchall()
    return candidate_rows, reference_rows


def execute_read_only(predicted, gold, db_place):
    comparator = _load_calculate_ex(str(SCORER_PATH))
    candidate_rows, reference_rows = _run_pair_read_only(db_place, predicted, gold)
    return comparator(candidate_rows, reference_rows)


def ex_score(predicted, gold, db_place, timeout_seconds=30.0):
    _load_calculate_ex(str(SCORER_PATH))
    run_with_timeout, timeout_error = _require_func_timeout()
    try:
        score = run_with_timeout(
            timeout_seconds,
            execute_read_only,
            args=(predicted, gold, db_place),
        )
    except DatabaseUnavailableError:
        raise
    except timeout_error:
        score = 0
    except Exception:
        score = 0
    return {"correct": bool(score), "raw_result": {"sql_idx": 0, "res": score}}
