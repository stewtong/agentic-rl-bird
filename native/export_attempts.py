#!/usr/bin/env python3
"""Export native retained Samples without losing attempts or inventing SQL EX grades."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from miles.ray.rollout.debug_data import load_debug_rollout_data
from miles.utils.types import Sample

from native_eval import load_request


ANALYSIS_FIELDS = ("question_id", "database_id", "condition", "training_seed", "attempt_id", "correct")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def dataset_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            question_id = metadata.get("question_id", row.get("question_id"))
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"dataset line {line_number} lacks metadata.question_id")
            rows.append(row)
    if not rows:
        raise ValueError("dataset has no questions")
    return rows


def sql_errors(metadata: dict) -> list[dict]:
    errors = []
    for call in metadata.get("calls") or []:
        if not isinstance(call, dict) or call.get("tool") != "run_sql":
            continue
        output = call.get("tool_output")
        if isinstance(output, dict) and output.get("error") is not None:
            errors.append({"args": call.get("args"), "tool_output": output})
    return errors


def model_errors(metadata: dict) -> dict:
    return {
        key: metadata[key]
        for key in ("format_error", "ungradeable")
        if metadata.get(key) is not None
    }


def infrastructure_errors(sample: Sample, metadata: dict) -> list:
    errors = list(metadata.get("infrastructure_errors") or [])
    if sample.status in {Sample.Status.ABORTED, Sample.Status.PENDING}:
        errors.append(f"native_sample_status:{sample.status.value}")
    return errors


def native_token_accounting(sample: Sample) -> dict:
    response_span_token_count = sample.response_length
    if type(response_span_token_count) is not int or response_span_token_count < 0:
        raise ValueError("native response_length must be a nonnegative integer")
    loss_mask = sample.loss_mask
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if loss_mask is None:
        if metadata.get("num_turns") != 1:
            raise ValueError("native loss_mask is required unless metadata.num_turns == 1")
        generated_token_count = response_span_token_count
        generated_token_count_source = "single_turn_response_length_fallback_no_loss_mask"
    else:
        if len(loss_mask) != response_span_token_count:
            raise ValueError("native loss_mask length must equal response_length")
        if any(type(value) is not int or value not in (0, 1) for value in loss_mask):
            raise ValueError("native loss_mask must contain only integer zero or one")
        generated_token_count = sum(loss_mask)
        generated_token_count_source = "native_loss_mask"
    return {
        "generated_token_count": generated_token_count,
        "response_span_token_count": response_span_token_count,
        "observation_token_count": response_span_token_count - generated_token_count,
        "generated_token_count_source": generated_token_count_source,
    }


def export_record(
    sample: Sample,
    question_id: str,
    database_id: str | None,
    request: dict,
    selected: dict,
    arm: str,
    prompt_ordinal: int,
    attempt_ordinal: int,
) -> dict:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    grade = metadata.get("execution_correct")
    graded = type(grade) is bool
    finish_reasons = metadata.get("finish_reasons") or []
    truncation_sources = []
    if sample.status == Sample.Status.TRUNCATED:
        truncation_sources.append("native_sample_status")
    if "length" in finish_reasons:
        truncation_sources.append("agent_finish_reason_length")
    return {
        "question_id": question_id,
        "database_id": database_id,
        "condition": request["condition"],
        "training_seed": selected.get("training_seed"),
        "attempt_id": f"attempt-{attempt_ordinal:02d}",
        "correct": grade if graded else None,
        "arm": arm,
        "run_id": request["run_id"],
        "evidence_class": request["evidence_class"],
        "sample_index": sample.index,
        "prompt_ordinal": prompt_ordinal,
        "attempt_ordinal": attempt_ordinal,
        "grade_status": "graded" if graded else "missing_domain_grade",
        "domain_ex_source": "sample.metadata.execution_correct",
        "native_reward_retained_not_used_for_ex": sample.reward,
        "native_sample_status": sample.status.value,
        "native_failure_requires_review": sample.status == Sample.Status.FAILED,
        "sql_errors": sql_errors(metadata),
        "model_errors": model_errors(metadata),
        "infrastructure_errors": infrastructure_errors(sample, metadata),
        "tool_call_count": len(metadata.get("calls") or []),
        "turn_count": metadata.get("num_turns"),
        "truncated": bool(truncation_sources),
        "truncation_sources": truncation_sources,
        **native_token_accounting(sample),
        "finish_reasons": finish_reasons,
        "rollout_seed": request["rollout_seed"],
        "attempt_sampling_seed": None,
        "sampling_seed_note": "Launcher does not enable deterministic per-attempt sampling_seed assignment.",
        "weight_checkpoint_provenance": {
            "model_name": request["model_name"],
            "model_root": request["model_root"],
            "model_revision": request["model_revision"],
            "adapter_path": selected.get("adapter_path"),
            "sample_weight_versions": list(sample.weight_versions),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=("frozen", "checkpoint"), required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format-fixture", action="store_true")
    args = parser.parse_args()
    request, selected = load_request(args.manifest, args.arm)
    if args.format_fixture != (request["evidence_class"] == "training-format-fixture"):
        raise ValueError("--format-fixture and manifest evidence_class must agree")
    loader_args = SimpleNamespace(
        load_debug_rollout_data=str(args.raw),
        load_debug_rollout_data_subsample=None,
    )
    samples = load_debug_rollout_data(loader_args, rollout_id=0)
    rows = dataset_rows(Path(request["dataset_path"]))
    expected_question_ids = [row["metadata"]["question_id"] for row in rows]
    if len(set(expected_question_ids)) != len(expected_question_ids):
        raise ValueError("dataset question IDs must be unique")
    attempt_count = request["attempt_count"]
    expected_indexes = set(range(len(rows) * attempt_count))
    actual_indexes = [sample.index for sample in samples]
    duplicate_indexes = sorted({index for index in actual_indexes if actual_indexes.count(index) > 1})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    accounting = {
        "schema_version": "bird-attempt-accounting/v1",
        "run_id": request["run_id"],
        "arm": args.arm,
        "evidence_class": request["evidence_class"],
        "raw_path": str(args.raw),
        "expected_question_ids": expected_question_ids,
        "expected_attempt_count": len(expected_indexes),
        "actual_sample_count": len(samples),
        "expected_attempts_per_question": attempt_count,
        "expected_sample_indexes": sorted(expected_indexes),
        "actual_sample_indexes": actual_indexes,
        "missing_sample_indexes": sorted(expected_indexes - set(actual_indexes)),
        "unexpected_sample_indexes": sorted(set(actual_indexes) - expected_indexes),
        "duplicate_sample_indexes": duplicate_indexes,
        "attempt_id_rule": "attempt-{zero-padded attempt ordinal}; native sample.index = prompt ordinal * attempt_count + attempt ordinal",
        "domain_ex_rule": "only literal Boolean sample.metadata.execution_correct; never native/group reward or missing-as-zero",
    }
    metadata_mismatches = []
    by_index = {}
    for sample in samples:
        if sample.index in by_index:
            continue
        by_index[sample.index] = sample
        if sample.index not in expected_indexes:
            continue
        prompt_ordinal = sample.index // attempt_count
        expected_id = expected_question_ids[prompt_ordinal]
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        if metadata.get("question_id") != expected_id:
            metadata_mismatches.append({
                "sample_index": sample.index,
                "expected_question_id": expected_id,
                "actual_question_id": metadata.get("question_id"),
            })
        if not args.format_fixture:
            expected_overrides = {
                "eval_run_id": request["run_id"],
                "eval_arm": args.arm,
                "eval_condition": request["condition"],
                "eval_training_seed": selected.get("training_seed"),
                "eval_mode": request["mode"],
                "eval_model_revision": request["model_revision"],
                "eval_adapter_path": selected.get("adapter_path"),
                "eval_attempt_count": attempt_count,
            }
            for key, expected in expected_overrides.items():
                if metadata.get(key) != expected:
                    metadata_mismatches.append({
                        "sample_index": sample.index,
                        "field": key,
                        "expected": expected,
                        "actual": metadata.get(key),
                    })
    accounting["metadata_mismatches"] = metadata_mismatches
    structural_errors = (
        accounting["missing_sample_indexes"]
        or accounting["unexpected_sample_indexes"]
        or duplicate_indexes
        or metadata_mismatches
    )
    if structural_errors:
        accounting["status"] = "failed_exact_accounting"
        write_json(args.output_dir / "accounting.json", accounting)
        return 4
    records = []
    for index in sorted(expected_indexes):
        sample = by_index[index]
        prompt_ordinal = index // attempt_count
        row = rows[prompt_ordinal]
        record = export_record(
            sample=sample,
            question_id=expected_question_ids[prompt_ordinal],
            database_id=(row["metadata"] or {}).get("db_id"),
            request=request,
            selected=selected,
            arm=args.arm,
            prompt_ordinal=prompt_ordinal,
            attempt_ordinal=index % attempt_count,
        )
        records.append(record)
    with (args.output_dir / "detailed-results.jsonl").open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    missing_grade_ids = [
        {"question_id": record["question_id"], "attempt_id": record["attempt_id"]}
        for record in records
        if record["grade_status"] != "graded"
    ]
    accounting["graded_attempt_count"] = len(records) - len(missing_grade_ids)
    accounting["missing_domain_grade_attempts"] = missing_grade_ids
    accounting["analysis_projection_written"] = not missing_grade_ids
    accounting["status"] = "complete" if not missing_grade_ids else "withheld_analysis_missing_domain_grade"
    write_json(args.output_dir / "accounting.json", accounting)
    if missing_grade_ids:
        return 3
    with (args.output_dir / "analysis-results.jsonl").open("x", encoding="utf-8") as handle:
        for record in records:
            projected = {key: record[key] for key in ANALYSIS_FIELDS}
            if projected["database_id"] is None:
                del projected["database_id"]
            handle.write(json.dumps(projected, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
