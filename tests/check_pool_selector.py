#!/usr/bin/env python3
"""Exercise the mixed-pool selector through actual CLI subprocesses."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")

import atexit
import copy
import hashlib
import json
import random
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = Path(tempfile.mkdtemp(prefix="bird-pool-proof-"))
atexit.register(shutil.rmtree, ROOT, True)
(ROOT / "output").mkdir()
SELECTOR = REPOSITORY / "scripts/select_mixed_pool.py"
WORK_ROOT = ROOT / "work/cpu-proof"
WORK_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ROOT = Path(tempfile.mkdtemp(prefix="run-", dir=WORK_ROOT))
CONDITION = "frozen-thinking-off"
CHECKS = []
LOG_LINES = [
    "CPU-only stdlib selector proof: synthetic fixtures; actual CLI subprocesses.",
    f"Python: {sys.version.split()[0]}",
    f"Fixture run root: {RUN_ROOT.relative_to(ROOT)}",
    "",
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def candidate(question_id: str, database_id: str | None = None) -> dict:
    database_id = database_id or f"db-{question_id}"
    return {
        "question_id": question_id,
        "prompt": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Synthetic SQL prompt",
                    }
                ],
            }
        ],
        "label": "",
        "metadata": {
            "question_id": question_id,
            "db_id": database_id,
            "db_path": f"/synthetic/{database_id}.sqlite",
            "gold_sql": "SELECT 1",
            "split": "synthetic-train",
        },
        "original_extra": {"preserve": [question_id, {"nested": True}]},
    }


def attempts_for(
    candidate_row: dict,
    correct_values: list[bool],
    generated_counts: list[int] | None = None,
) -> list[dict]:
    generated_counts = generated_counts or list(range(1, 9))
    rows = []
    for index, (correct, generated_count) in enumerate(
        zip(correct_values, generated_counts, strict=True)
    ):
        rows.append(
            {
                "question_id": candidate_row["question_id"],
                "database_id": candidate_row["metadata"]["db_id"],
                "condition": CONDITION,
                "training_seed": 11,
                "attempt_id": f"attempt-{index:02d}",
                "correct": correct,
                "arm": "frozen",
                "run_id": "synthetic-run",
                "evidence_class": "synthetic-fixture",
                "sample_index": index,
                "prompt_ordinal": 0,
                "attempt_ordinal": index,
                "grade_status": "graded",
                "domain_ex_source": "sample.metadata.execution_correct",
                "native_reward_retained_not_used_for_ex": float(not correct),
                "native_sample_status": "COMPLETED",
                "native_failure_requires_review": False,
                "sql_errors": [],
                "model_errors": {},
                "infrastructure_errors": [],
                "tool_call_count": 1,
                "turn_count": 2,
                "truncated": False,
                "truncation_sources": [],
                "generated_token_count": generated_count,
                "response_span_token_count": 1000 + index * 31,
                "observation_token_count": 1000 + index * 31 - generated_count,
                "generated_token_count_source": (
                    "native_loss_mask"
                    if index < 6
                    else "single_turn_response_length_fallback_no_loss_mask"
                ),
                "finish_reasons": ["stop"],
                "rollout_seed": 17,
                "weight_checkpoint_provenance": {
                    "model_name": "synthetic-model-name",
                    "model_revision": "synthetic-revision",
                    "adapter_path": None,
                    "sample_weight_versions": [],
                },
            }
        )
    return rows


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def invoke(
    name: str,
    candidate_rows: list[dict],
    attempt_rows: list[dict] | None,
    *,
    seed: int = 23,
    raw_attempts: str | None = None,
) -> dict:
    case_root = RUN_ROOT / name
    case_root.mkdir()
    candidates_path = case_root / "candidates.jsonl"
    attempts_path = case_root / "attempts.jsonl"
    output_dir = case_root / "output"
    write_jsonl(candidates_path, candidate_rows)
    if raw_attempts is not None:
        attempts_path.write_text(raw_attempts, encoding="utf-8")
    else:
        write_jsonl(attempts_path, attempt_rows)
    command = [
        sys.executable,
        str(SELECTOR),
        "--candidates",
        str(candidates_path),
        "--attempts",
        str(attempts_path),
        "--condition",
        CONDITION,
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    process = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, timeout=60, check=False
    )
    display_command = ["python3" if part == sys.executable else part for part in command]
    display_command = [
        display_path(Path(part))
        if part.startswith(str(ROOT))
        else display_path(Path(part))
        if part == str(SELECTOR)
        else part
        for part in display_command
    ]
    return {
        "name": name,
        "case_root": case_root,
        "candidates_path": candidates_path,
        "attempts_path": attempts_path,
        "candidate_rows": candidate_rows,
        "output_dir": output_dir,
        "seed": seed,
        "command": shlex.join(display_command),
        "process": process,
    }


def invoke_repeat(context: dict, name: str) -> dict:
    output_dir = RUN_ROOT / name / "output"
    output_dir.parent.mkdir()
    command = [
        sys.executable,
        str(SELECTOR),
        "--candidates",
        str(context["candidates_path"]),
        "--attempts",
        str(context["attempts_path"]),
        "--condition",
        CONDITION,
        "--seed",
        str(context["seed"]),
        "--output-dir",
        str(output_dir),
    ]
    process = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, timeout=60, check=False
    )
    display_command = ["python3" if part == sys.executable else part for part in command]
    display_command = [
        display_path(Path(part)) if part.startswith(str(ROOT)) else part
        for part in display_command
    ]
    return {
        "command": shlex.join(display_command),
        "process": process,
        "output_dir": output_dir,
    }


def record(check_id: str, description: str, contexts: list[dict], assertions: list[str]) -> None:
    invocations = []
    for context in contexts:
        process = context["process"]
        invocations.append(
            {
                "command": context["command"],
                "exit_code": process.returncode,
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip(),
            }
        )
        LOG_LINES.extend(
            [
                f"[{check_id}] {description}",
                f"$ {context['command']}",
                f"exit={process.returncode}",
                f"stdout={process.stdout.strip()}",
                f"stderr={process.stderr.strip()}",
                "",
            ]
        )
    CHECKS.append(
        {
            "id": check_id,
            "description": description,
            "status": "passed",
            "invocations": invocations,
            "assertions": assertions,
        }
    )


def load_success(context: dict) -> tuple[dict, list[dict], bytes]:
    assert context["process"].returncode == 0, context["process"].stderr
    report = json.loads(
        (context["output_dir"] / "selection-report.json").read_text(encoding="utf-8")
    )
    pool_bytes = (context["output_dir"] / "selected-pool.jsonl").read_bytes()
    pool = [json.loads(line) for line in pool_bytes.decode("utf-8").splitlines()]
    return report, pool, pool_bytes


mixed_candidate = candidate("q-mixed")
pass_candidate = candidate("q-pass")
fail_candidate = candidate("q-fail")
basic_candidates = [pass_candidate, mixed_candidate, fail_candidate]
mixed_correct = [True, False, True, False, True, False, True, False]
token_counts = [3, 5, 7, 11, 13, 17, 19, 23]
mixed_attempts = attempts_for(mixed_candidate, mixed_correct, token_counts)
mixed_attempts[1]["native_sample_status"] = "FAILED"
mixed_attempts[1]["native_failure_requires_review"] = True
mixed_attempts[1]["model_errors"] = {"format_error": "synthetic malformed SQL"}
mixed_attempts[1]["truncated"] = True
basic_attempts = (
    attempts_for(pass_candidate, [True] * 8)
    + mixed_attempts
    + attempts_for(fail_candidate, [False] * 8)
)
basic = invoke("eligibility-and-arithmetic", basic_candidates, basic_attempts)
report, pool, _ = load_success(basic)
assert report["status"] == "selected"
assert report["counts"] == {
    "candidates": 3,
    "attempts": 24,
    "all_pass": 1,
    "all_fail": 1,
    "mixed": 1,
    "selected": 1,
}
assert pool == [mixed_candidate]
assert report["selected_question_ids"] == ["q-mixed"]
assert {
    summary["question_id"]: summary["outcome"] for summary in report["questions"]
} == {"q-fail": "all_fail", "q-mixed": "mixed", "q-pass": "all_pass"}
reference = report["generated_token_reference"]
assert reference["population_count"] == 8
assert reference["generated_token_total"] == 98
assert reference["mean_generated_tokens_per_attempt"] == 12.25
assert reference["mean_generated_tokens_per_attempt"] != sum(
    row["response_span_token_count"] for row in mixed_attempts
) / 8
assert reference["generated_token_count_source"] == {
    "native_loss_mask": 6,
    "single_turn_response_length_fallback_no_loss_mask": 2,
}
assert reference["selected_truncated_attempt_fraction"] == 0.125
assert mixed_attempts[1]["correct"] is False
record(
    "P1",
    "Mixed eligibility, object preservation, native-false eligibility, and token arithmetic",
    [basic],
    [
        "One mixed question selected; all-pass and all-fail questions excluded.",
        "Selected candidate object exactly equals its original JSON object.",
        "Explicit false grade on native FAILED attempt remains valid.",
        "Eight-attempt generated-token total 98 and mean 12.25 use generated_token_count.",
        "Deliberately different response_span_token_count values do not enter the mean.",
        "Source frequencies are 6 loss-mask and 2 fallback; truncation fraction is 1/8.",
    ],
)

empty = invoke(
    "honest-empty",
    [pass_candidate, fail_candidate],
    attempts_for(pass_candidate, [True] * 8)
    + attempts_for(fail_candidate, [False] * 8),
)
empty_report, empty_pool, empty_bytes = load_success(empty)
assert empty_report["status"] == "no_mixed_questions"
assert empty_pool == [] and empty_bytes == b""
assert empty_report["counts"]["selected"] == 0
assert empty_report["generated_token_reference"]["population_count"] == 0
assert empty_report["generated_token_reference"]["mean_generated_tokens_per_attempt"] is None
record(
    "P2",
    "Honest empty result without substitution",
    [empty],
    [
        "Empty JSONL pool and no_mixed_questions status emitted.",
        "Selected population is zero and token reference mean is null.",
    ],
)

cap_candidates = [candidate(f"q-{index:04d}", "shared-db") for index in range(512, -1, -1)]
cap_attempts = []
for candidate_index, candidate_row in enumerate(reversed(cap_candidates)):
    counts = [((candidate_index * 8 + attempt_index) % 17) + 1 for attempt_index in range(8)]
    cap_attempts.extend(
        attempts_for(candidate_row, [True] + [False] * 7, counts)
    )
cap = invoke("cap-first", cap_candidates, cap_attempts, seed=8675309)
cap_report, cap_pool, cap_pool_bytes = load_success(cap)
cap_repeat = invoke_repeat(cap, "cap-repeat")
cap_repeat_report, _, cap_repeat_pool_bytes = load_success(cap_repeat)
expected_sample = set(
    random.Random(8675309).sample(sorted(row["question_id"] for row in cap_candidates), 512)
)
candidate_order = [row["question_id"] for row in cap_candidates]
expected_order = [question_id for question_id in candidate_order if question_id in expected_sample]
assert cap_report["counts"]["mixed"] == 513
assert cap_report["counts"]["selected"] == 512
assert cap_report["sampling_cap_applied"] is True
assert cap_report["selected_question_ids"] == expected_order
assert set(cap_report["selected_question_ids"]) == expected_sample
assert len(cap_report["selected_question_ids"]) == 512
assert cap_report["generated_token_reference"]["population_count"] == 4096
assert cap_repeat_report == cap_report and cap_repeat_pool_bytes == cap_pool_bytes
record(
    "P3",
    "Deterministic >512 cap, candidate-order restoration, and byte-exact repeatability",
    [cap, cap_repeat],
    [
        "Exactly 512 of 513 mixed questions selected using the default cap.",
        "Selected set matches sorted-ID random.Random(8675309).sample exactly.",
        "Output pool follows descending candidate-file order, not sample order.",
        "Reference population is 512 * 8 = 4096 attempts.",
        "Repeated CLI run produces byte-identical pool and identical report.",
    ],
)

negative_candidate = candidate("q-negative")
valid_negative = attempts_for(negative_candidate, [True, False, False, False, False, False, False, False])
negative_cases = {}
negative_cases["incomplete-eight-required"] = valid_negative[:7]
duplicate_attempts = copy.deepcopy(valid_negative)
duplicate_attempts[-1]["attempt_id"] = duplicate_attempts[0]["attempt_id"]
negative_cases["duplicate-attempt-id"] = duplicate_attempts
missing_grade = copy.deepcopy(valid_negative)
missing_grade[0]["correct"] = None
missing_grade[0]["grade_status"] = "missing_domain_grade"
negative_cases["missing-grade"] = missing_grade
infrastructure = copy.deepcopy(valid_negative)
infrastructure[0]["infrastructure_errors"] = ["synthetic process failure"]
negative_cases["infrastructure-error"] = infrastructure
mixed_arm = copy.deepcopy(valid_negative)
mixed_arm[0]["arm"] = "checkpoint"
negative_cases["mixed-arm"] = mixed_arm
mixed_condition = copy.deepcopy(valid_negative)
mixed_condition[0]["condition"] = "frozen-thinking-on"
negative_cases["mixed-condition"] = mixed_condition
unknown_question = copy.deepcopy(valid_negative)
unknown_question[0]["question_id"] = "q-unknown"
negative_cases["unknown-question"] = unknown_question
mismatched_database = copy.deepcopy(valid_negative)
mismatched_database[0]["database_id"] = "different-db"
negative_cases["mismatched-database"] = mismatched_database
bad_token_count = copy.deepcopy(valid_negative)
bad_token_count[0]["generated_token_count"] = -1
negative_cases["negative-generated-token-count"] = bad_token_count

for name, rows in negative_cases.items():
    context = invoke(name, [negative_candidate], rows)
    assert context["process"].returncode != 0
    assert not context["output_dir"].exists()
    record(
        f"FAIL:{name}",
        "Malformed or ineligible input rejected before any output directory is created",
        [context],
        ["Nonzero CLI exit.", "No output directory or selected pool left behind."],
    )

malformed_json = invoke(
    "malformed-json", [negative_candidate], None, raw_attempts="{not-json\n"
)
assert malformed_json["process"].returncode != 0
assert not malformed_json["output_dir"].exists()
record(
    "FAIL:malformed-json",
    "Malformed JSONL rejected through actual CLI",
    [malformed_json],
    ["Nonzero CLI exit.", "No output directory or selected pool left behind."],
)

selector_digest = hashlib.sha256(SELECTOR.read_bytes()).hexdigest()
proof = {
    "schema_version": "mixed-pool-selector-cpu-proof/v1",
    "status": "passed",
    "fixture_class": "synthetic-only",
    "actual_cli_subprocesses": True,
    "python_version": sys.version.split()[0],
    "selector_sha256": selector_digest,
    "check_count": len(CHECKS),
    "success_case_count": 3,
    "failure_case_count": len(CHECKS) - 3,
    "gpu_or_model_access_performed": False,
    "real_pool_selection_performed": False,
    "checks": CHECKS,
}
(ROOT / "output/cpu-proof.json").write_text(
    json.dumps(proof, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
(ROOT / "output/cpu-proof.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "status": "passed",
            "check_count": len(CHECKS),
            "cleaned_up": True,
        }
    )
)
