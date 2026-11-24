#!/usr/bin/env python3
"""Analyze frozen/trained outcomes and multi-attempt oracle pass@k."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any


K_VALUES = (1, 2, 4, 8)
ATTEMPTS_PER_QUESTION = 8
DEFAULT_REPLICATES = 10_000
DEFAULT_RNG_SEED = 20_261_117
CONFIDENCE_LEVEL = 0.95
CLAIM_THRESHOLD = 0.03


class InputError(ValueError):
    pass


def require_id(value: Any, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"line {line_number}: {field} must be a nonempty string")
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    required = {"question_id", "condition", "training_seed", "attempt_id", "correct"}
    allowed = required | {"database_id"}
    rows = []
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise InputError(f"line {line_number}: expected a JSON object")
            missing = required - record.keys()
            unknown = record.keys() - allowed
            if missing:
                raise InputError(f"line {line_number}: missing fields {sorted(missing)}")
            if unknown:
                raise InputError(f"line {line_number}: unknown fields {sorted(unknown)}")
            question_id = require_id(record["question_id"], "question_id", line_number)
            condition = require_id(record["condition"], "condition", line_number)
            attempt_id = require_id(record["attempt_id"], "attempt_id", line_number)
            seed = record["training_seed"]
            if seed is not None:
                seed = require_id(seed, "training_seed", line_number)
            if type(record["correct"]) is not bool:
                raise InputError(f"line {line_number}: correct must be JSON true or false")
            database_id = record.get("database_id")
            if database_id is not None:
                database_id = require_id(database_id, "database_id", line_number)
            key = (database_id, question_id, condition, seed, attempt_id)
            if key in keys:
                raise InputError(
                    f"line {line_number}: duplicate question/condition/seed/attempt key"
                )
            keys.add(key)
            rows.append(
                {
                    "question_id": question_id,
                    "database_id": database_id,
                    "question_key": (database_id, question_id),
                    "condition": condition,
                    "training_seed": seed,
                    "attempt_id": attempt_id,
                    "correct": int(record["correct"]),
                }
            )
    if not rows:
        raise InputError("input contains no outcomes")
    return rows


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(differences: list[float], replicates: int, rng_seed: int) -> dict[str, Any]:
    rng = random.Random(rng_seed)
    question_count = len(differences)
    bootstrap_means = []
    for _ in range(replicates):
        total = sum(differences[rng.randrange(question_count)] for _ in range(question_count))
        bootstrap_means.append(total / question_count)
    alpha = 1.0 - CONFIDENCE_LEVEL
    low = percentile(bootstrap_means, alpha / 2.0)
    high = percentile(bootstrap_means, 1.0 - alpha / 2.0)
    return {
        "point_estimate": statistics.fmean(differences),
        "interval_95_percentile": {"low": low, "high": high},
        "bootstrap_replicates": replicates,
        "analysis_rng_seed": rng_seed,
        "resampling_unit": "question_id (including database_id when present)",
        "interval_method": "percentile bootstrap with linearly interpolated 2.5/97.5 percentiles",
    }


def exact_mcnemar(gains: int, losses: int) -> dict[str, Any]:
    discordant = gains + losses
    if discordant == 0:
        p_value = 1.0
    else:
        tail_terms = sum(math.comb(discordant, index) for index in range(min(gains, losses) + 1))
        p_value = float(min(Fraction(1), Fraction(2 * tail_terms, 2**discordant)))
    return {
        "trained_correct_frozen_incorrect": gains,
        "frozen_correct_trained_incorrect": losses,
        "discordant_questions": discordant,
        "exact_two_sided_p_value": p_value,
        "test": "exact conditional binomial McNemar; doubled smaller tail capped at 1",
    }


def analyze_primary(rows: list[dict[str, Any]], seeds: list[str], replicates: int, rng_seed: int) -> dict[str, Any]:
    conditions = {row["condition"] for row in rows}
    if len(conditions) != 1:
        raise InputError("primary input must contain exactly one condition")
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(not seed.strip() for seed in seeds):
        raise InputError("primary requires exactly three distinct nonempty training seeds")
    observed_seeds = {row["training_seed"] for row in rows if row["training_seed"] is not None}
    if observed_seeds != set(seeds):
        raise InputError(
            f"trained seed set mismatch: input has {sorted(observed_seeds)}, requested {sorted(seeds)}"
        )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["question_key"], row["training_seed"])].append(row)
    for key, group in grouped.items():
        if len(group) != 1:
            raise InputError(
                f"primary requires one attempt per question/seed; {key!r} has {len(group)}"
            )
    frozen_questions = {
        key[0] for key in grouped if key[1] is None
    }
    if not frozen_questions:
        raise InputError("primary requires frozen outcomes with training_seed null")
    samples: dict[str | None, dict[Any, int]] = {}
    for seed in [None, *seeds]:
        question_keys = {key[0] for key in grouped if key[1] == seed}
        if question_keys != frozen_questions:
            missing = sorted(frozen_questions - question_keys, key=str)
            extra = sorted(question_keys - frozen_questions, key=str)
            raise InputError(
                f"inconsistent question sample for seed {seed!r}; missing={missing}, extra={extra}"
            )
        samples[seed] = {key[0]: group[0]["correct"] for key, group in grouped.items() if key[1] == seed}
    ordered_questions = sorted(frozen_questions, key=str)
    seed_results = {}
    seed_gains = []
    for seed_index, seed in enumerate(seeds):
        differences = [samples[seed][question] - samples[None][question] for question in ordered_questions]
        interval = paired_bootstrap(differences, replicates, rng_seed + seed_index)
        gains = sum(value == 1 for value in differences)
        losses = sum(value == -1 for value in differences)
        seed_results[seed] = {
            "trained_accuracy": statistics.fmean(samples[seed][question] for question in ordered_questions),
            "paired_gain": interval,
            "mcnemar": exact_mcnemar(gains, losses),
        }
        seed_gains.append(interval["point_estimate"])
    pooled_differences = [
        statistics.fmean(samples[seed][question] for seed in seeds) - samples[None][question]
        for question in ordered_questions
    ]
    pooled = paired_bootstrap(pooled_differences, replicates, rng_seed + len(seeds))
    lower_bound = pooled["interval_95_percentile"]["low"]
    return {
        "schema_version": "bird-paired-analysis/v1",
        "mode": "primary",
        "condition": next(iter(conditions)),
        "question_count": len(ordered_questions),
        "training_seeds": seeds,
        "frozen_accuracy": statistics.fmean(samples[None][question] for question in ordered_questions),
        "seed_specific": seed_results,
        "pooled_question_analysis": {
            **pooled,
            "trained_accuracy_mean_within_question": statistics.fmean(
                statistics.fmean(samples[seed][question] for seed in seeds)
                for question in ordered_questions
            ),
            "interpretation": (
                "Conditional on these three observed training seeds; trained binary outcomes are "
                "averaged within each question before subtracting the frozen binary outcome. "
                "This is not training-seed population uncertainty."
            ),
            "claim_rule": {
                "rule": "interval_95_percentile.low > 0.03",
                "threshold": CLAIM_THRESHOLD,
                "strictly_greater": True,
                "passes": lower_bound > CLAIM_THRESHOLD,
            },
        },
        "training_seed_spread": {
            "minimum_gain": min(seed_gains),
            "maximum_gain": max(seed_gains),
            "range": max(seed_gains) - min(seed_gains),
            "population_standard_deviation": statistics.pstdev(seed_gains),
        },
    }


def unbiased_pass_at_k(correct_count: int, k: int, attempts: int = ATTEMPTS_PER_QUESTION) -> float:
    return 1.0 - math.comb(attempts - correct_count, k) / math.comb(attempts, k)


def analyze_sampling(
    rows: list[dict[str, Any]], attempts_per_question: int = ATTEMPTS_PER_QUESTION
) -> dict[str, Any]:
    k_values = K_VALUES if attempts_per_question == ATTEMPTS_PER_QUESTION else (1, 2, 4, 16)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["question_key"], row["training_seed"])].append(row)
    for key, group in grouped.items():
        if len(group) != attempts_per_question:
            raise InputError(
                f"sampling requires exactly {attempts_per_question} distinct attempts for each question/condition/seed; "
                f"{key!r} has {len(group)}"
            )
    condition_results = []
    for condition in sorted({row["condition"] for row in rows}):
        condition_groups = {key: value for key, value in grouped.items() if key[0] == condition}
        question_keys = {key[1] for key in condition_groups}
        seed_groups: dict[str | None, dict[Any, list[dict[str, Any]]]] = defaultdict(dict)
        for (_, question_key, seed), attempts in condition_groups.items():
            seed_groups[seed][question_key] = attempts
        for seed, questions in seed_groups.items():
            if set(questions) != question_keys:
                raise InputError(f"inconsistent question sample for condition {condition!r}, seed {seed!r}")
        seed_results = []
        for seed in sorted(seed_groups, key=lambda value: (value is not None, value or "")):
            questions = seed_groups[seed]
            per_question = []
            for question_key in sorted(questions, key=str):
                correct_count = sum(attempt["correct"] for attempt in questions[question_key])
                per_question.append(
                    {
                        str(k): unbiased_pass_at_k(correct_count, k, attempts_per_question)
                        for k in k_values
                    }
                )
            seed_results.append(
                {
                    "training_seed": seed,
                    "question_count": len(questions),
                    "attempts_per_question": attempts_per_question,
                    "oracle_pass_at_k": {
                        str(k): statistics.fmean(question[str(k)] for question in per_question)
                        for k in k_values
                    },
                }
            )
        condition_results.append({"condition": condition, "seeds": seed_results})
    return {
        "schema_version": "bird-sampling-analysis/v1",
        "mode": "sampling",
        "conditions": condition_results,
        "estimator": (
            f"per question: 1 - C({attempts_per_question}-c,k)/C({attempts_per_question},k); "
            "report the arithmetic mean across questions"
        ),
        "k_values": list(k_values),
        "interpretation": "Gold-scored oracle pass@k upper bound; not deployable selected-answer accuracy.",
        "gold_free_selected_answer_accuracy": {
            "status": "not_implemented",
            "reason": "Execution-result inputs and a specified selection/tie-break rule are required.",
            "future_selector": "Execution-result consensus with a deterministic tie-break, scored only on its selected answer.",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    primary = subparsers.add_parser("primary", help="Analyze one frozen outcome and three trained seeds")
    primary.add_argument("input", type=Path, help="Outcome JSONL path")
    primary.add_argument("--seeds", nargs=3, required=True, metavar=("SEED_A", "SEED_B", "SEED_C"))
    primary.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    primary.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    sampling = subparsers.add_parser("sampling", help="Compute unbiased oracle pass@k")
    sampling.add_argument("input", type=Path, help="Outcome JSONL path")
    sampling.add_argument("--attempts", type=int, choices=(8, 16), default=ATTEMPTS_PER_QUESTION)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rows = load_rows(args.input)
        if args.mode == "primary":
            if args.replicates <= 0:
                raise InputError("replicates must be a positive integer")
            result = analyze_primary(rows, args.seeds, args.replicates, args.rng_seed)
        else:
            result = analyze_sampling(rows, args.attempts)
    except (InputError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
