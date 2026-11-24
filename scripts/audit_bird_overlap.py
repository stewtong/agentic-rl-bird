#!/usr/bin/env python3
"""Report explicit ID and normalized database/question overlap between BIRD JSON files."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Sequence


class AuditInputError(ValueError):
    pass


class Row(NamedTuple):
    question_id: str
    database_id: str
    question: str

    @property
    def normalized_pair(self) -> tuple[str, str]:
        return self.database_id, normalize_question(self.question)


def normalize_question(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(normalized.split())


def _question_id(value: object, location: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise AuditInputError(f"{location}.question_id must be a string or integer")
    canonical = str(value).strip()
    if not canonical:
        raise AuditInputError(f"{location}.question_id must not be empty")
    return canonical


def _database_id(item: dict[str, object], location: str) -> str:
    has_database_id = "database_id" in item
    has_db_id = "db_id" in item
    if has_database_id == has_db_id:
        raise AuditInputError(
            f"{location} must contain exactly one of database_id or db_id"
        )
    value = item["database_id" if has_database_id else "db_id"]
    if not isinstance(value, str):
        raise AuditInputError(f"{location} database_id/db_id must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized:
        raise AuditInputError(f"{location} database_id/db_id must not be empty")
    return normalized


def load_dataset(path: Path, label: str) -> list[Row]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditInputError(f"{label}: cannot read {path}: {error}") from error
    if not isinstance(payload, list):
        raise AuditInputError(f"{label}: top-level JSON value must be a list")

    rows: list[Row] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload):
        location = f"{label}[{index}]"
        if not isinstance(item, dict):
            raise AuditInputError(f"{location} must be an object")
        question_id = _question_id(item.get("question_id"), location)
        if question_id in seen_ids:
            raise AuditInputError(
                f"{label}: duplicate question_id {question_id!r} at index {index}"
            )
        seen_ids.add(question_id)
        question = item.get("question")
        if not isinstance(question, str):
            raise AuditInputError(f"{location}.question must be a string")
        if not question.strip():
            raise AuditInputError(f"{location}.question must not be empty")
        rows.append(Row(question_id, _database_id(item, location), question))
    return rows


def _paired_ids(rows: Sequence[Row]) -> dict[tuple[str, str], list[str]]:
    paired: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        paired[row.normalized_pair].append(row.question_id)
    return dict(paired)


def build_report(
    left_rows: Sequence[Row],
    right_rows: Sequence[Row],
    left_revision: str,
    right_revision: str,
) -> dict[str, object]:
    left_ids = {row.question_id for row in left_rows}
    right_ids = {row.question_id for row in right_rows}
    matched_question_ids = sorted(left_ids & right_ids)
    left_pairs = _paired_ids(left_rows)
    right_pairs = _paired_ids(right_rows)
    matched_pairs = sorted(left_pairs.keys() & right_pairs.keys())

    return {
        "schema_version": "bird-overlap-audit/v1",
        "revisions": {"left": left_revision, "right": right_revision},
        "datasets": {
            label: {
                "row_count": len(rows),
                "question_id_count": len(rows),
                "normalized_pair_count": len(_paired_ids(rows)),
            }
            for label, rows in (("left", left_rows), ("right", right_rows))
        },
        "question_id_overlap": {
            "count": len(matched_question_ids),
            "matched_identifiers": matched_question_ids,
        },
        "normalized_overlap": {
            "matched_pair_count": len(matched_pairs),
            "left_matched_row_count": sum(len(left_pairs[pair]) for pair in matched_pairs),
            "right_matched_row_count": sum(len(right_pairs[pair]) for pair in matched_pairs),
            "matches": [
                {
                    "database_id": database_id,
                    "normalized_question": normalized_question,
                    "left_question_ids": sorted(left_pairs[(database_id, normalized_question)]),
                    "right_question_ids": sorted(right_pairs[(database_id, normalized_question)]),
                }
                for database_id, normalized_question in matched_pairs
            ],
        },
        "normalization": {
            "question_id": "Convert integers to strings and trim surrounding whitespace",
            "database_id": "Unicode NFKC and trim surrounding whitespace; preserve case",
            "question": "Unicode NFKC, case-fold, and collapse whitespace runs",
        },
        "limitations": [
            "Overlap is detected only within the two supplied JSON inputs.",
            "Unmatched rows do not prove absence of contamination.",
            "Normalized text matching is not semantic equivalence; punctuation differences remain significant.",
            "A question-ID match does not imply matching database IDs or question text.",
        ],
    }


def _revision(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("revision must not be empty")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="left dataset JSON list")
    parser.add_argument("right", type=Path, help="right dataset JSON list")
    parser.add_argument("--left-revision", required=True, type=_revision)
    parser.add_argument("--right-revision", required=True, type=_revision)
    args = parser.parse_args(argv)
    try:
        left_rows = load_dataset(args.left, "left")
        right_rows = load_dataset(args.right, "right")
    except AuditInputError as error:
        parser.error(str(error))
    report = build_report(
        left_rows, right_rows, args.left_revision, args.right_revision
    )
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
