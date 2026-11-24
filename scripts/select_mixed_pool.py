#!/usr/bin/env python3
"""Select a reproducible mixed-outcome pool from exported frozen attempts."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


EXPECTED_ATTEMPTS = 8
DEFAULT_MAX_SELECTED = 512
SELECTION_ALGORITHM = (
    "Sort qualifying question IDs lexicographically. If their count exceeds "
    "max_selected, take random.Random(seed).sample(sorted_ids, max_selected); "
    "otherwise retain every sorted ID. Emit selected candidate objects in "
    "original candidate-file order."
)


class InputError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_json_constant(value: str) -> None:
    raise InputError(f"non-finite JSON number {value!r} is not allowed")


def read_jsonl(path: Path, label: str) -> list[dict]:
    rows = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read {label} {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=reject_json_constant)
            except json.JSONDecodeError as exc:
                raise InputError(
                    f"{label} line {line_number} is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise InputError(f"{label} line {line_number} must be a JSON object")
            rows.append(row)
    return rows


def required_string(row: dict, key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise InputError(f"{context} requires nonempty string {key}")
    return value


def load_candidates(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = read_jsonl(path, "candidates")
    if not rows:
        raise InputError("candidates must contain at least one row")
    by_id = {}
    for index, row in enumerate(rows, start=1):
        context = f"candidate row {index}"
        question_id = required_string(row, "question_id", context)
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise InputError(f"{context} requires object metadata")
        metadata_id = required_string(metadata, "question_id", context)
        if metadata_id != question_id:
            raise InputError(f"{context} has mismatched top-level and metadata question_id")
        database_id = required_string(metadata, "db_id", context)
        for key in ("db_path", "gold_sql"):
            required_string(metadata, key, context)
        for key in ("prompt", "label"):
            if key not in row:
                raise InputError(f"{context} is missing {key}")
        if question_id in by_id:
            raise InputError(f"duplicate candidate question_id {question_id!r}")
        by_id[question_id] = {
            "question_id": question_id,
            "database_id": database_id,
            "candidate_ordinal": index - 1,
        }
    return rows, by_id


def load_attempts(
    path: Path, candidates: dict[str, dict], condition: str
) -> dict[str, list[dict]]:
    rows = read_jsonl(path, "attempts")
    grouped = {question_id: [] for question_id in candidates}
    seen_attempt_ids = set()
    for index, row in enumerate(rows, start=1):
        context = f"attempt row {index}"
        question_id = required_string(row, "question_id", context)
        if question_id not in candidates:
            raise InputError(f"{context} has unknown question_id {question_id!r}")
        attempt_id = required_string(row, "attempt_id", context)
        attempt_key = (question_id, attempt_id)
        if attempt_key in seen_attempt_ids:
            raise InputError(
                f"duplicate attempt_id {attempt_id!r} for question {question_id!r}"
            )
        seen_attempt_ids.add(attempt_key)
        if row.get("arm") != "frozen":
            raise InputError(f"{context} must have arm='frozen' (mixed or non-frozen arm)")
        if row.get("condition") != condition:
            raise InputError(
                f"{context} condition {row.get('condition')!r} does not match "
                f"requested condition {condition!r}"
            )
        expected_database = candidates[question_id]["database_id"]
        if row.get("database_id") != expected_database:
            raise InputError(
                f"{context} database_id {row.get('database_id')!r} does not match "
                f"candidate database_id {expected_database!r}"
            )
        if row.get("grade_status") != "graded" or type(row.get("correct")) is not bool:
            raise InputError(f"{context} lacks a literal Boolean domain grade")
        infrastructure_errors = row.get("infrastructure_errors")
        if not isinstance(infrastructure_errors, list):
            raise InputError(f"{context} requires list infrastructure_errors")
        if infrastructure_errors:
            raise InputError(f"{context} has infrastructure errors")
        generated_tokens = row.get("generated_token_count")
        if type(generated_tokens) is not int or generated_tokens < 0:
            raise InputError(
                f"{context} generated_token_count must be a nonnegative integer"
            )
        required_string(row, "generated_token_count_source", context)
        if "truncated" in row and type(row["truncated"]) is not bool:
            raise InputError(f"{context} truncated must be a literal Boolean when supplied")
        grouped[question_id].append(row)

    for question_id, attempts in grouped.items():
        if len(attempts) != EXPECTED_ATTEMPTS:
            raise InputError(
                f"question {question_id!r} has {len(attempts)} attempts; "
                f"expected exactly {EXPECTED_ATTEMPTS}"
            )
    return grouped


def question_summaries(
    candidates: dict[str, dict], grouped: dict[str, list[dict]]
) -> list[dict]:
    summaries = []
    for question_id in sorted(candidates):
        attempts = sorted(grouped[question_id], key=lambda row: row["attempt_id"])
        correct_count = sum(row["correct"] for row in attempts)
        if correct_count == EXPECTED_ATTEMPTS:
            outcome = "all_pass"
        elif correct_count == 0:
            outcome = "all_fail"
        else:
            outcome = "mixed"
        summaries.append(
            {
                "question_id": question_id,
                "database_id": candidates[question_id]["database_id"],
                "outcome": outcome,
                "correct_count": correct_count,
                "attempts": [
                    {"attempt_id": row["attempt_id"], "correct": row["correct"]}
                    for row in attempts
                ],
            }
        )
    return summaries


def token_reference(selected_ids: set[str], grouped: dict[str, list[dict]]) -> dict:
    attempts = [
        row for question_id in sorted(selected_ids) for row in grouped[question_id]
    ]
    generated_total = sum(row["generated_token_count"] for row in attempts)
    population_count = len(attempts)
    truncation_supplied = bool(attempts) and all("truncated" in row for row in attempts)
    truncated_count = (
        sum(row["truncated"] for row in attempts) if truncation_supplied else None
    )
    return {
        "population_count": population_count,
        "generated_token_total": generated_total,
        "mean_generated_tokens_per_attempt": (
            generated_total / population_count if population_count else None
        ),
        "generated_token_count_source": dict(
            sorted(Counter(row["generated_token_count_source"] for row in attempts).items())
        ),
        "truncated_population_count": population_count if truncation_supplied else None,
        "truncated_attempt_count": truncated_count,
        "selected_truncated_attempt_fraction": (
            truncated_count / population_count if truncation_supplied else None
        ),
    }


def select(args: argparse.Namespace) -> dict:
    candidate_rows, candidates = load_candidates(args.candidates)
    grouped = load_attempts(args.attempts, candidates, args.condition)
    summaries = question_summaries(candidates, grouped)
    outcome_counts = Counter(summary["outcome"] for summary in summaries)
    mixed_ids = sorted(
        summary["question_id"] for summary in summaries if summary["outcome"] == "mixed"
    )
    if len(mixed_ids) > args.max_selected:
        sampled_ids = random.Random(args.seed).sample(mixed_ids, args.max_selected)
    else:
        sampled_ids = mixed_ids.copy()
    selected_ids = set(sampled_ids)
    selected_rows = [
        row for row in candidate_rows if row["question_id"] in selected_ids
    ]
    selected_question_ids = [row["question_id"] for row in selected_rows]
    status = "selected" if selected_rows else "no_mixed_questions"
    report = {
        "schema_version": "mixed-pool-selection/v1",
        "status": status,
        "arm": "frozen",
        "condition": args.condition,
        "expected_attempts_per_question": EXPECTED_ATTEMPTS,
        "seed": args.seed,
        "max_selected": args.max_selected,
        "selection_algorithm": SELECTION_ALGORITHM,
        "sampling_cap_applied": len(mixed_ids) > args.max_selected,
        "counts": {
            "candidates": len(candidate_rows),
            "attempts": sum(len(rows) for rows in grouped.values()),
            "all_pass": outcome_counts["all_pass"],
            "all_fail": outcome_counts["all_fail"],
            "mixed": outcome_counts["mixed"],
            "selected": len(selected_rows),
        },
        "selected_question_ids": selected_question_ids,
        "sampled_question_ids_before_candidate_order_restore": sampled_ids,
        "questions": summaries,
        "generated_token_reference": token_reference(selected_ids, grouped),
        "inputs": {
            "candidates": {
                "path": str(args.candidates),
                "sha256": sha256(args.candidates),
            },
            "attempts": {"path": str(args.attempts), "sha256": sha256(args.attempts)},
        },
    }

    if args.output_dir.exists():
        raise InputError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "selected-pool.jsonl").open("x", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (args.output_dir / "selection-report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return report


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-selected", type=positive_int, default=DEFAULT_MAX_SELECTED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = select(args)
    except (InputError, OSError) as exc:
        print(f"select_mixed_pool: error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": report["counts"],
                "mean_generated_tokens_per_attempt": report[
                    "generated_token_reference"
                ]["mean_generated_tokens_per_attempt"],
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
