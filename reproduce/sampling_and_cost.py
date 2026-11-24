#!/usr/bin/env python3
"""Regenerate BIRD 16-draw sampling, best-of-16, and matched-cost results.

What this script does:
    Computes first-draw accuracy, any-of-16 oracle coverage, unbiased pass@k,
    reference-free best-of-16 accuracy, paired contrasts with 95% confidence
    intervals, GPU timing summaries, and matched-cost outcomes across all eight
    evaluated arms on the 1,534 BIRD dev questions.

What it reads:
    Standard library only (Python 3.11+). Reads the eight gzip-compressed grade
    files (*.grades.jsonl.gz), eight gzip-compressed selector pick files
    (*.picks.jsonl.gz), and timing.json from evidence/bird_16draw/. No SQL is
    executed and no answers are re-graded.

Bootstrap settings:
    Paired 95% intervals use random.Random(20261120) with 10,000 resamples of
    the 1,534 per-question paired differences in dev-0 through dev-1533 order
    via rng.choices(delta, k=1534), taking linearly interpolated 2.5th and
    97.5th percentiles of the resampled percentage-point differences.

Matched-cost rule:
    For each trained operating point and timing repeat, take the largest frozen
    draw budget k in {1, 2, 4, 16} whose saturated GPU-seconds per question are
    at most 1.15 times the trained point's cost (or k=1 if even frozen k=1
    exceeds 1.15 times the trained cost) and compare paired accuracy on the
    1,534 questions. Outcomes are "training wins at matched cost" (paired lower
    bound above +3.00 points at matched k, or above 0.00 when frozen k=1
    exceeds 1.15x trained cost), "favors frozen" (upper bound below +3.00
    points at matched k, or below 0.00 when frozen k=1 exceeds 1.15x trained
    cost), "training adds on top of sampling" (same k in {2, 4}, frozen-to-
    trained cost ratio in [0.85, 1.15], and lower bound above +3.00 points), or
    "open" (otherwise, when repeats disagree on matched k or rule branch, or
    when any repeat lacks valid isolation timing).
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
import random
from statistics import mean
import sys

N_QUESTIONS = 1534
IDS = tuple(f"dev-{i}" for i in range(N_QUESTIONS))
DRAWS = tuple(range(16))
KS = (1, 2, 4, 16)
BOOTSTRAP_SEED = 20261120
BOOTSTRAP_REPLICATES = 10_000

ARMS = (
    "frozen-12turn",
    "frozen-6turn",
    "seed44-update8-12turn",
    "seed44-update8-6turn",
    "seed42-update50-12turn",
    "seed42-update50-6turn",
    "seed43-update50-12turn",
    "seed43-update50-6turn",
)

ARM_LABELS = {
    "frozen-12turn": "frozen (12-turn limits)",
    "frozen-6turn": "frozen (6-turn limits)",
    "seed44-update8-12turn": "seed 44, update 8 (12-turn limits)",
    "seed44-update8-6turn": "seed 44, update 8 (6-turn limits)",
    "seed42-update50-12turn": "seed 42, update 50 (12-turn limits)",
    "seed42-update50-6turn": "seed 42, update 50 (6-turn limits)",
    "seed43-update50-12turn": "seed 43, update 50 (12-turn limits)",
    "seed43-update50-6turn": "seed 43, update 50 (6-turn limits)",
}

SESSION1_ARMS = (
    "frozen-6turn",
    "frozen-12turn",
    "seed44-update8-12turn",
    "seed42-update50-12turn",
    "seed44-update8-6turn",
    "seed42-update50-6turn",
)

SESSION2_ARMS = (
    "frozen-12turn",
    "frozen-6turn",
    "seed42-update50-12turn",
    "seed43-update50-12turn",
    "seed43-update50-6turn",
)

OUTCOME_WINS = "training wins at matched cost"
OUTCOME_FROZEN = "favors frozen"
OUTCOME_ADDS = "training adds on top of sampling"
OUTCOME_OPEN = "open"


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_population(evidence_dir: Path, arm: str) -> dict:
    grade_rows = read_jsonl_gz(evidence_dir / f"{arm}.grades.jsonl.gz")
    grades: dict[tuple[str, int], dict] = {}
    for row in grade_rows:
        key = (row["question_id"], row["draw"])
        if key in grades or type(row["correct"]) is not bool:
            raise ValueError(f"invalid grade row in {arm}: {row}")
        grades[key] = row
    expected_grades = {(q, d) for q in IDS for d in DRAWS}
    if grades.keys() != expected_grades:
        raise ValueError(f"incomplete grade grid for {arm}: {len(grades)}")

    pick_rows = read_jsonl_gz(evidence_dir / f"{arm}.picks.jsonl.gz")
    picks: dict[tuple[str, int], int | None] = {}
    for row in pick_rows:
        q, k, val = row["question_id"], row["k"], row["pick"]
        if (q, k) in picks or k not in KS:
            raise ValueError(f"invalid pick row in {arm}: {row}")
        if val is not None and not (type(val) is int and 0 <= val < k):
            raise ValueError(f"out-of-range pick in {arm}: {row}")
        picks[(q, k)] = val
    expected_picks = {(q, k) for q in IDS for k in KS}
    if picks.keys() != expected_picks:
        raise ValueError(f"incomplete pick grid for {arm}: {len(picks)}")

    draw0 = tuple(int(grades[(q, 0)]["correct"]) for q in IDS)
    correct_counts = tuple(sum(int(grades[(q, d)]["correct"]) for d in DRAWS) for q in IDS)
    any_of_16 = tuple(int(c > 0) for c in correct_counts)
    selected = {
        k: tuple(
            int(picks[(q, k)] is not None and grades[(q, picks[(q, k)])]["correct"])
            for q in IDS
        )
        for k in KS
    }
    unbiased_pass = {
        k: 100.0 * math.fsum(1.0 - math.comb(16 - c, k) / math.comb(16, k) for c in correct_counts) / N_QUESTIONS
        for k in (1, 2, 4, 16)
    }

    draw0_rows = [grades[(q, 0)] for q in IDS]
    all_rows = list(grades.values())
    return {
        "arm": arm,
        "label": ARM_LABELS[arm],
        "draw0": draw0,
        "any_of_16": any_of_16,
        "correct_counts": correct_counts,
        "selected": selected,
        "summary": {
            "arm": arm,
            "label": ARM_LABELS[arm],
            "questions": N_QUESTIONS,
            "draws_per_question": 16,
            "first_draw_correct": sum(draw0),
            "first_draw_accuracy_pct": 100.0 * sum(draw0) / N_QUESTIONS,
            "any_of_16_correct": sum(any_of_16),
            "any_of_16_accuracy_pct": 100.0 * sum(any_of_16) / N_QUESTIONS,
            "selected_correct": {str(k): sum(selected[k]) for k in KS},
            "selected_accuracy_pct": {str(k): 100.0 * sum(selected[k]) / N_QUESTIONS for k in KS},
            "unbiased_pass_at_pct": {str(k): unbiased_pass[k] for k in (1, 2, 4, 16)},
            "first_draw_generated_tokens": sum(int(r["generated_tokens"]) for r in draw0_rows),
            "all_draws_generated_tokens": sum(int(r["generated_tokens"]) for r in all_rows),
            "first_draw_truncated": sum(int(r["truncated"]) for r in draw0_rows),
            "all_draws_truncated": sum(int(r["truncated"]) for r in all_rows),
            "first_draw_turn_limit_exits": sum(int(r["turn_limit_exit"]) for r in draw0_rows),
            "all_draws_turn_limit_exits": sum(int(r["turn_limit_exit"]) for r in all_rows),
            "null_picks_total": sum(1 for v in picks.values() if v is None),
        },
    }


@lru_cache(maxsize=None)
def bootstrap_ci(delta: tuple[int, ...]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(delta)
    samples = sorted(sum(rng.choices(delta, k=n)) for _ in range(BOOTSTRAP_REPLICATES))
    bounds = []
    for quantile in (0.025, 0.975):
        pos = (len(samples) - 1) * quantile
        idx = int(pos)
        val = samples[idx] + (samples[idx + 1] - samples[idx]) * (pos - idx)
        bounds.append(100.0 * val / n)
    return (bounds[0], bounds[1])


def paired_contrast(vec_a: tuple[int, ...], vec_b: tuple[int, ...]) -> dict:
    if len(vec_a) != N_QUESTIONS or len(vec_b) != N_QUESTIONS:
        raise ValueError("paired vectors must have length 1534")
    delta = tuple(a - b for a, b in zip(vec_a, vec_b))
    low, high = bootstrap_ci(delta)
    return {
        "a_correct": sum(vec_a),
        "b_correct": sum(vec_b),
        "a_only": delta.count(1),
        "b_only": delta.count(-1),
        "difference_pp": 100.0 * sum(delta) / N_QUESTIONS,
        "ci95_low_pp": low,
        "ci95_high_pp": high,
    }


def check_saturation(stages: list[list[dict]]) -> list[dict]:
    results = []
    for stage in stages:
        if len(stage) != 3 or {r["draws"] for r in stage} != {2, 4, 8}:
            raise ValueError(f"unexpected saturation stage: {stage}")
        q_counts = {r["attempts"] // r["draws"] for r in stage}
        if len(q_counts) != 1:
            raise ValueError(f"inconsistent question counts: {q_counts}")
        questions = q_counts.pop()
        sec_per_attempt = {}
        for row in stage:
            k = row["draws"]
            sec_per_attempt[k] = row["native_seconds"] / row["attempts"]
        rel_to_4 = {k: sec_per_attempt[k] / sec_per_attempt[4] - 1.0 for k in (2, 8)}
        passed = all(0.90 <= sec_per_attempt[k] / sec_per_attempt[4] <= 1.10 for k in (2, 8))
        results.append({
            "questions": questions,
            "seconds_per_attempt": {str(k): sec_per_attempt[k] for k in (2, 4, 8)},
            "relative_to_4": {str(k): rel_to_4[k] for k in (2, 8)},
            "passed": passed,
        })
    return results


def build_session1_series(timing: dict) -> tuple[list[dict], list[dict[str, dict[int, float]]], dict]:
    rows = timing["session1_single_seed"]["rows"]
    by_key = {}
    for row in rows:
        key = (row["arm"], row["repeat"], bool(row["isolated_rerun"]))
        if key in by_key:
            raise ValueError(f"duplicate session 1 slot: {key}")
        by_key[key] = row

    used_rows = []
    series: list[dict[str, dict[int, float]]] = []
    for rep in (1, 2, 3):
        rep_costs: dict[str, dict[int, float]] = {}
        for arm in SESSION1_ARMS:
            if rep == 2 and (arm, 2, True) in by_key:
                chosen = by_key[(arm, 2, True)]
            else:
                chosen = by_key[(arm, rep, False)]
            used_rows.append(chosen)
            sec_per_draw = chosen["native_seconds"] / chosen["attempts"]
            rep_costs[arm] = {k: k * sec_per_draw for k in KS}
        series.append(rep_costs)

    # Repeat-2 isolated rerun comparison and cutoff margins for seed42-update50-6turn
    frozen_6_rep2 = by_key[("frozen-6turn", 2, False)]["native_seconds"]
    cutoff_k2_sec = 2.0 * frozen_6_rep2 / 1.15
    rerun_summary = {
        "frozen_12turn_rep2_initial_s": by_key[("frozen-12turn", 2, False)]["native_seconds"],
        "frozen_12turn_rep2_isolated_s": by_key[("frozen-12turn", 2, True)]["native_seconds"],
        "seed42_update50_6turn_rep2_initial_s": by_key[("seed42-update50-6turn", 2, False)]["native_seconds"],
        "seed42_update50_6turn_rep2_isolated_s": by_key[("seed42-update50-6turn", 2, True)]["native_seconds"],
        "frozen_6turn_rep2_s": frozen_6_rep2,
        "k2_cutoff_threshold_s": cutoff_k2_sec,
        "seed42_update50_6turn_rep2_isolated_margin_s": by_key[("seed42-update50-6turn", 2, True)]["native_seconds"] - cutoff_k2_sec,
        "seed42_update50_6turn_rep2_initial_margin_s": by_key[("seed42-update50-6turn", 2, False)]["native_seconds"] - cutoff_k2_sec,
    }
    return used_rows, series, rerun_summary


def evaluate_matched_cost(
    pops: dict[str, dict],
    series: list[dict[str, dict[int, float]]],
    trained_arm: str,
    trained_k: int,
    frozen_arm: str,
    unresolved_repeats: set[int] | None = None,
) -> dict:
    unresolved = unresolved_repeats or set()
    repeat_rows = []
    for idx, cost in enumerate(series, start=1):
        if idx in unresolved:
            repeat_rows.append({
                "repeat": idx,
                "isolation_valid": False,
                "matched_frozen_k": None,
                "within_cutoff": None,
                "outcome": OUTCOME_OPEN,
                "note": "missing valid isolation timing",
            })
            continue
        budget = cost[trained_arm][trained_k]
        frozen_cost = cost[frozen_arm]
        eligible = [j for j in KS if frozen_cost[j] <= 1.15 * budget]
        match_k = max(eligible, default=1)
        within_cutoff = bool(eligible)
        trained_vec = pops[trained_arm]["draw0"] if trained_k == 1 else pops[trained_arm]["selected"][trained_k]
        frozen_vec = pops[frozen_arm]["draw0"] if match_k == 1 else pops[frozen_arm]["selected"][match_k]
        pair = paired_contrast(trained_vec, frozen_vec)
        low, high = pair["ci95_low_pp"], pair["ci95_high_pp"]
        if within_cutoff:
            if low > 3.0:
                outcome = OUTCOME_WINS
            elif high < 3.0:
                outcome = OUTCOME_FROZEN
            else:
                outcome = OUTCOME_OPEN
            branch = "ordinary_matched"
        else:
            if low > 0.0:
                outcome = OUTCOME_WINS
            elif high < 0.0:
                outcome = OUTCOME_FROZEN
            else:
                outcome = OUTCOME_OPEN
            branch = "cheaper_than_frozen_k1"
        repeat_rows.append({
            "repeat": idx,
            "isolation_valid": True,
            "matched_frozen_k": match_k,
            "within_cutoff": within_cutoff,
            "branch": branch,
            "trained_seconds_per_question": budget,
            "frozen_seconds_per_question": frozen_cost[match_k],
            "frozen_to_trained_cost_ratio": frozen_cost[match_k] / budget,
            "outcome": outcome,
            **pair,
        })

    valid_rows = [r for r in repeat_rows if r["isolation_valid"]]
    identities = {(r["matched_frozen_k"], r["within_cutoff"], r["outcome"]) for r in valid_rows}
    if unresolved:
        overall = OUTCOME_OPEN
        valid_only_outcome = valid_rows[0]["outcome"] if len(identities) == 1 and valid_rows else OUTCOME_OPEN
        reason = (
            f"repeat {', '.join(str(r) for r in sorted(unresolved))} lacked valid isolation timing; "
            f"valid repeats ({', '.join(str(r['repeat']) for r in valid_rows)}) give {valid_only_outcome} "
            f"at frozen k={valid_rows[0]['matched_frozen_k']}"
            if valid_rows and len(identities) == 1
            else "lacked valid isolation timing"
        )
    elif len(identities) == 1:
        overall = valid_rows[0]["outcome"]
        reason = ""
    else:
        overall = OUTCOME_OPEN
        ks = [r["matched_frozen_k"] for r in valid_rows]
        if len(set(ks)) == 1:
            reason = (
                "repeats disagree on the 1.15x cost-cutoff branch "
                "(repeat 1 ordinary matched-cost, repeats 2 and 3 cheaper than frozen k=1); "
                "all three repeats favor training"
            )
        else:
            reason = f"repeats disagree on matched frozen k ({ks})"

    return {
        "trained_arm": trained_arm,
        "trained_label": ARM_LABELS[trained_arm],
        "trained_k": trained_k,
        "frozen_arm": frozen_arm,
        "frozen_label": ARM_LABELS[frozen_arm],
        "overall_outcome": overall,
        "note": reason,
        "repeats": repeat_rows,
    }


def evaluate_same_k(
    pops: dict[str, dict],
    series: list[dict[str, dict[int, float]]],
    trained_arm: str,
    k: int,
    frozen_arm: str,
) -> dict:
    pair = paired_contrast(pops[trained_arm]["selected"][k], pops[frozen_arm]["selected"][k])
    low = pair["ci95_low_pp"]
    ratios = [cost[frozen_arm][k] / cost[trained_arm][k] for cost in series]
    repeat_outcomes = []
    for ratio in ratios:
        if ratio > 1.15:
            band = "adapter_cheaper"
        elif ratio < 0.85:
            band = "frozen_cheaper"
        else:
            band = "comparable"
        outcome = OUTCOME_OPEN
        if band == "adapter_cheaper" and low > 0.0:
            outcome = OUTCOME_WINS
        elif band == "comparable" and k in (2, 4) and low > 3.0:
            outcome = OUTCOME_ADDS
        repeat_outcomes.append((band, outcome))
    overall = repeat_outcomes[0][1] if len(set(repeat_outcomes)) == 1 else OUTCOME_OPEN
    return {
        "trained_arm": trained_arm,
        "trained_label": ARM_LABELS[trained_arm],
        "k": k,
        "frozen_arm": frozen_arm,
        "frozen_label": ARM_LABELS[frozen_arm],
        "frozen_to_trained_cost_ratios": ratios,
        "mean_cost_ratio": mean(ratios),
        "overall_outcome": overall,
        **pair,
    }


def build_session2_series(timing: dict) -> tuple[list[dict], list[dict[str, dict[int, float]]], dict[str, set[int]], dict]:
    rows = timing["session2_two_seed"]["rows"]
    by_key = {(r["arm"], r["repeat"]): r for r in rows}
    series: list[dict[str, dict[int, float]]] = []
    unresolved_by_arm: dict[str, set[int]] = {arm: set() for arm in SESSION2_ARMS}
    for rep in (1, 2, 3):
        rep_costs: dict[str, dict[int, float]] = {}
        for arm in SESSION2_ARMS:
            row = by_key[(arm, rep)]
            if not row["isolation_valid"]:
                unresolved_by_arm[arm].add(rep)
            sec_per_draw = row["native_seconds"] / row["attempts"]
            rep_costs[arm] = {k: k * sec_per_draw for k in KS}
        series.append(rep_costs)

    arm_timing_summary = {}
    for arm in SESSION2_ARMS:
        rep_sec = [by_key[(arm, r)]["native_seconds"] / by_key[(arm, r)]["attempts"] for r in (1, 2, 3)]
        valid_reps = [r for r in (1, 2, 3) if r not in unresolved_by_arm[arm]]
        valid_sec = [by_key[(arm, r)]["native_seconds"] / by_key[(arm, r)]["attempts"] for r in valid_reps]
        arm_timing_summary[arm] = {
            "arm": arm,
            "label": ARM_LABELS[arm],
            "seconds_per_draw_by_repeat": rep_sec,
            "native_seconds_by_repeat": [by_key[(arm, r)]["native_seconds"] for r in (1, 2, 3)],
            "valid_repeats": valid_reps,
            "unresolved_repeats": sorted(unresolved_by_arm[arm]),
            "all_valid_mean_seconds_per_draw": mean(rep_sec) if len(valid_reps) == 3 else None,
            "valid_only_mean_seconds_per_draw": mean(valid_sec) if valid_sec else None,
        }
    return rows, series, unresolved_by_arm, arm_timing_summary


def compute_all(evidence_dir: Path) -> dict:
    pops = {arm: load_population(evidence_dir, arm) for arm in ARMS}
    timing = json.loads((evidence_dir / "timing.json").read_text(encoding="utf-8"))

    frozen_12_draw0_tokens = pops["frozen-12turn"]["summary"]["first_draw_generated_tokens"]
    for arm in ARMS:
        t0 = pops[arm]["summary"]["first_draw_generated_tokens"]
        pops[arm]["summary"]["first_draw_tokens_vs_frozen_12turn_pct"] = (
            100.0 * (t0 / frozen_12_draw0_tokens - 1.0)
        )

    paired_first_draw = [
        ("seed42-update50-12turn", "frozen-12turn"),
        ("seed43-update50-12turn", "frozen-12turn"),
        ("seed43-update50-12turn", "seed42-update50-12turn"),
        ("seed44-update8-12turn", "frozen-12turn"),
        ("seed44-update8-6turn", "frozen-6turn"),
        ("seed42-update50-6turn", "frozen-6turn"),
        ("seed43-update50-6turn", "frozen-6turn"),
        ("seed43-update50-6turn", "seed42-update50-6turn"),
        ("frozen-12turn", "frozen-6turn"),
    ]
    first_draw_contrasts = [
        {
            "a_arm": a,
            "b_arm": b,
            "a_label": ARM_LABELS[a],
            "b_label": ARM_LABELS[b],
            **paired_contrast(pops[a]["draw0"], pops[b]["draw0"]),
        }
        for a, b in paired_first_draw
    ]

    paired_any16 = [
        ("seed42-update50-12turn", "frozen-12turn"),
        ("seed42-update50-12turn", "frozen-6turn"),
        ("seed43-update50-12turn", "frozen-12turn"),
        ("seed43-update50-12turn", "frozen-6turn"),
        ("seed43-update50-12turn", "seed42-update50-12turn"),
        ("seed43-update50-6turn", "seed42-update50-6turn"),
    ]
    any_of_16_contrasts = [
        {
            "a_arm": a,
            "b_arm": b,
            "a_label": ARM_LABELS[a],
            "b_label": ARM_LABELS[b],
            **paired_contrast(pops[a]["any_of_16"], pops[b]["any_of_16"]),
        }
        for a, b in paired_any16
    ]

    best_of_16_pairs = [
        ("seed42-update50-12turn", "frozen-12turn"),
        ("seed43-update50-12turn", "frozen-12turn"),
        ("seed44-update8-12turn", "frozen-12turn"),
        ("seed42-update50-6turn", "frozen-6turn"),
        ("seed43-update50-6turn", "frozen-6turn"),
        ("seed44-update8-6turn", "frozen-6turn"),
    ]
    best_of_16_contrasts = [
        {
            "adapter_arm": a,
            "frozen_arm": f,
            "adapter_label": ARM_LABELS[a],
            "frozen_label": ARM_LABELS[f],
            **paired_contrast(pops[a]["draw0"], pops[f]["selected"][16]),
        }
        for a, f in best_of_16_pairs
    ]

    sat = check_saturation(timing["saturation"])
    s1_used_rows, s1_series, s1_rerun = build_session1_series(timing)

    s1_matched_cases = [
        ("seed44-update8-12turn", 1, "frozen-6turn"),
        ("seed44-update8-12turn", 2, "frozen-6turn"),
        ("seed42-update50-12turn", 1, "frozen-12turn"),
        ("seed42-update50-12turn", 1, "frozen-6turn"),
        ("seed42-update50-6turn", 1, "frozen-6turn"),
        ("seed44-update8-6turn", 1, "frozen-6turn"),
        ("seed44-update8-6turn", 2, "frozen-6turn"),
    ]
    s1_matched = [
        evaluate_matched_cost(pops, s1_series, tr, tk, fr)
        for tr, tk, fr in s1_matched_cases
    ]
    s1_same_k = [
        evaluate_same_k(pops, s1_series, "seed44-update8-12turn", 2, "frozen-6turn"),
    ]
    s1_arm_timing = {}
    for arm in SESSION1_ARMS:
        vals = [rep[arm][1] for rep in s1_series]
        s1_arm_timing[arm] = {
            "arm": arm,
            "label": ARM_LABELS[arm],
            "seconds_per_draw_by_repeat": vals,
            "mean_seconds_per_draw": mean(vals),
        }

    s2_rows, s2_series, s2_unresolved, s2_arm_timing = build_session2_series(timing)
    s2_matched_cases = [
        ("seed43-update50-12turn", 1, "frozen-12turn"),
        ("seed43-update50-12turn", 1, "frozen-6turn"),
        ("seed43-update50-6turn", 1, "frozen-6turn"),
    ]
    s2_matched = [
        evaluate_matched_cost(
            pops,
            s2_series,
            tr,
            tk,
            fr,
            unresolved_repeats=s2_unresolved[tr] | s2_unresolved[fr],
        )
        for tr, tk, fr in s2_matched_cases
    ]

    # Five decisive two-seed checks for seed 43 update 50 (12-turn limits)
    s43_first = paired_contrast(pops["seed43-update50-12turn"]["draw0"], pops["frozen-12turn"]["draw0"])
    s43_cov_12 = paired_contrast(pops["seed43-update50-12turn"]["any_of_16"], pops["frozen-12turn"]["any_of_16"])
    s43_cov_6 = paired_contrast(pops["seed43-update50-12turn"]["any_of_16"], pops["frozen-6turn"]["any_of_16"])
    s43_mc_12 = s2_matched[0]
    s43_mc_6 = s2_matched[1]

    decisive_checks = [
        {
            "comparison": "seed 43, update 50 (12-turn limits) first draw vs frozen (12-turn limits)",
            "criterion": "95% lower bound > +3.00 points",
            "replicated": s43_first["ci95_low_pp"] > 3.0,
            "detail": f"{s43_first['a_correct']:,} vs {s43_first['b_correct']:,} ({s43_first['difference_pp']:+.2f}, {s43_first['ci95_low_pp']:+.2f} to {s43_first['ci95_high_pp']:+.2f})",
        },
        {
            "comparison": "seed 43, update 50 (12-turn limits) matched cost vs frozen (12-turn limits)",
            "criterion": "training wins at matched cost across all 3 repeats",
            "replicated": s43_mc_12["overall_outcome"] == OUTCOME_WINS,
            "detail": f"{s43_mc_12['overall_outcome']} (matched frozen k=1 in repeats 1, 2, 3)",
        },
        {
            "comparison": "seed 43, update 50 (12-turn limits) matched cost vs frozen (6-turn limits)",
            "criterion": "training wins at matched cost across all 3 repeats",
            "replicated": s43_mc_6["overall_outcome"] == OUTCOME_WINS,
            "detail": f"{s43_mc_6['overall_outcome']} ({s43_mc_6['note']})",
        },
        {
            "comparison": "seed 43, update 50 (12-turn limits) any-of-16 vs frozen (12-turn limits)",
            "criterion": "95% lower bound > 0.00 points",
            "replicated": s43_cov_12["ci95_low_pp"] > 0.0,
            "detail": f"{s43_cov_12['a_correct']:,} vs {s43_cov_12['b_correct']:,} ({s43_cov_12['difference_pp']:+.2f}, {s43_cov_12['ci95_low_pp']:+.2f} to {s43_cov_12['ci95_high_pp']:+.2f})",
        },
        {
            "comparison": "seed 43, update 50 (12-turn limits) any-of-16 vs frozen (6-turn limits)",
            "criterion": "95% lower bound > 0.00 points",
            "replicated": s43_cov_6["ci95_low_pp"] > 0.0,
            "detail": f"{s43_cov_6['a_correct']:,} vs {s43_cov_6['b_correct']:,} ({s43_cov_6['difference_pp']:+.2f}, {s43_cov_6['ci95_low_pp']:+.2f} to {s43_cov_6['ci95_high_pp']:+.2f})",
        },
    ]
    replicated_count = sum(1 for c in decisive_checks if c["replicated"])
    two_seed_label = (
        "the update-50 result replicates across two seeds"
        if replicated_count == len(decisive_checks)
        else "the update-50 result is seed-dependent"
    )

    return {
        "questions": N_QUESTIONS,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "method": "paired question resamples with replacement in dev-ID order, linear percentiles",
        },
        "arms": {arm: pops[arm]["summary"] for arm in ARMS},
        "first_draw_contrasts": first_draw_contrasts,
        "any_of_16_contrasts": any_of_16_contrasts,
        "frozen_best_of_16": {
            "frozen-12turn": {
                "correct": pops["frozen-12turn"]["summary"]["selected_correct"]["16"],
                "accuracy_pct": pops["frozen-12turn"]["summary"]["selected_accuracy_pct"]["16"],
            },
            "frozen-6turn": {
                "correct": pops["frozen-6turn"]["summary"]["selected_correct"]["16"],
                "accuracy_pct": pops["frozen-6turn"]["summary"]["selected_accuracy_pct"]["16"],
            },
        },
        "best_of_16_contrasts": best_of_16_contrasts,
        "saturation": sat,
        "session1": {
            "arm_timing": s1_arm_timing,
            "used_slots": s1_used_rows,
            "repeat2_isolated_reruns": s1_rerun,
            "matched_cost": s1_matched,
            "same_k": s1_same_k,
        },
        "session2": {
            "arm_timing": s2_arm_timing,
            "slots": s2_rows,
            "matched_cost": s2_matched,
            "same_session_seed43_over_seed42_ratios": [
                s2_series[i]["seed43-update50-12turn"][1] / s2_series[i]["seed42-update50-12turn"][1]
                for i in range(3)
            ],
            "two_seed_replication": {
                "replicated_count": replicated_count,
                "total_decisive": len(decisive_checks),
                "label": two_seed_label,
                "checks": decisive_checks,
            },
        },
    }


def fmt_ci(low: float, high: float) -> str:
    return f"{low:+.2f} to {high:+.2f}"


def render_markdown(res: dict) -> str:
    lines: list[str] = []
    lines.append("# BIRD 16-draw sampling, best-of-16, and matched-cost results")
    lines.append("")
    lines.append(
        "Matched-cost rule: for each trained operating point and timing repeat, take the largest "
        "frozen draw budget `k` in `{1, 2, 4, 16}` whose saturated GPU-seconds per question are at "
        "most `1.15` times the trained point's cost (or `k=1` if even frozen `k=1` exceeds `1.15` "
        "times the trained cost), and compare paired accuracy on the 1,534 BIRD dev questions. "
        "Outcomes are `training wins at matched cost` (paired 95% lower bound above `+3.00` points "
        "at matched `k`, or above `0.00` when frozen `k=1` exceeds `1.15`x trained cost), "
        "`favors frozen` (upper bound below `+3.00` points at matched `k`), "
        "`training adds on top of sampling` (same `k` in `{2, 4}`, frozen-to-trained cost ratio "
        "in `[0.85, 1.15]`, and lower bound above `+3.00` points), or `open` (otherwise, when "
        "repeats disagree on matched `k` or rule branch, or when a repeat lacks valid isolation timing)."
    )
    lines.append("")
    lines.append(
        f"Questions: {res['questions']:,}. Paired 95% intervals: {res['bootstrap']['replicates']:,} "
        f"question bootstrap resamples with seed `{res['bootstrap']['seed']}` "
        "(linearly interpolated 2.5th and 97.5th percentiles)."
    )
    lines.append("")

    # 1. Per-arm first draw and any-of-16
    lines.append("## First-draw and any-of-16 counts per arm")
    lines.append("")
    lines.append(
        "`first draw` is draw 0 of 16; `any-of-16` counts a question correct if any of its 16 draws "
        "is correct (an oracle upper bound); `best-of-16` picks one answer per question using a "
        "fixed selector that never sees grades."
    )
    lines.append("")
    lines.append(
        "| Arm | First draw | First draw % | Any-of-16 | Any-of-16 % | "
        "Selected k=2 | Selected k=4 | Best-of-16 | Unbiased pass@1 % | Unbiased pass@16 % | "
        "First-draw tokens | vs frozen 12-turn % |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for arm in ARMS:
        a = res["arms"][arm]
        lines.append(
            f"| {a['label']} | {a['first_draw_correct']:,} | {a['first_draw_accuracy_pct']:.2f}% | "
            f"{a['any_of_16_correct']:,} | {a['any_of_16_accuracy_pct']:.2f}% | "
            f"{a['selected_correct']['2']:,} | {a['selected_correct']['4']:,} | "
            f"{a['selected_correct']['16']:,} | {a['unbiased_pass_at_pct']['1']:.2f}% | "
            f"{a['unbiased_pass_at_pct']['16']:.2f}% | {a['first_draw_generated_tokens']:,} | "
            f"{a['first_draw_tokens_vs_frozen_12turn_pct']:+.2f}% |"
        )
    lines.append("")

    # 2. Paired first-draw and any-of-16 contrasts
    lines.append("## Paired first-draw and any-of-16 contrasts")
    lines.append("")
    lines.append("| Contrast (A minus B) | Mode | A correct | B correct | A only | B only | Difference (pp) | 95% interval (pp) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in res["first_draw_contrasts"]:
        lines.append(
            f"| {row['a_label']} vs {row['b_label']} | first draw | "
            f"{row['a_correct']:,} | {row['b_correct']:,} | {row['a_only']:,} | {row['b_only']:,} | "
            f"{row['difference_pp']:+.2f} | {fmt_ci(row['ci95_low_pp'], row['ci95_high_pp'])} |"
        )
    for row in res["any_of_16_contrasts"]:
        lines.append(
            f"| {row['a_label']} vs {row['b_label']} | any-of-16 | "
            f"{row['a_correct']:,} | {row['b_correct']:,} | {row['a_only']:,} | {row['b_only']:,} | "
            f"{row['difference_pp']:+.2f} | {fmt_ci(row['ci95_low_pp'], row['ci95_high_pp'])} |"
        )
    lines.append("")

    # 3. Best-of-16 for frozen and adapter first-draw contrasts against it
    lines.append("## Frozen best-of-16 and adapter first-draw contrasts")
    lines.append("")
    f12 = res["frozen_best_of_16"]["frozen-12turn"]
    f6 = res["frozen_best_of_16"]["frozen-6turn"]
    lines.append(
        f"- Frozen (`12-turn limits`) best-of-16: **{f12['correct']:,}** / 1,534 ({f12['accuracy_pct']:.2f}%)"
    )
    lines.append(
        f"- Frozen (`6-turn limits`) best-of-16: **{f6['correct']:,}** / 1,534 ({f6['accuracy_pct']:.2f}%)"
    )
    lines.append("")
    lines.append("| Adapter first draw vs frozen best-of-16 | Adapter correct | Frozen best-of-16 | Adapter only | Frozen only | Difference (pp) | 95% interval (pp) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in res["best_of_16_contrasts"]:
        lines.append(
            f"| {row['adapter_label']} vs {row['frozen_label']} | "
            f"{row['a_correct']:,} | {row['b_correct']:,} | {row['a_only']:,} | {row['b_only']:,} | "
            f"{row['difference_pp']:+.2f} | {fmt_ci(row['ci95_low_pp'], row['ci95_high_pp'])} |"
        )
    lines.append("")

    # 4. Saturation checks
    lines.append("## GPU timing saturation check")
    lines.append("")
    lines.append("| Cohort questions | 2 draws (s/draw) | 4 draws (s/draw) | 8 draws (s/draw) | Passed +/-10% vs 4 draws |")
    lines.append("| ---: | ---: | ---: | ---: | --- |")
    for st in res["saturation"]:
        s = st["seconds_per_attempt"]
        lines.append(
            f"| {st['questions']:,} | {s['2']:.4f} | {s['4']:.4f} | {s['8']:.4f} | "
            f"{'yes' if st['passed'] else 'no'} |"
        )
    lines.append("")

    # 5. Session 1 timings and matched-cost results
    lines.append("## Matched-cost results: single-seed session (seeds 44 and 42)")
    lines.append("")
    lines.append("Slot timings (512 questions x 4 draws = 2,048 attempts per slot; isolated reruns used for repeat 2 where present):")
    lines.append("")
    lines.append("| Arm | Repeat 1 (s) | Repeat 2 (s) | Repeat 3 (s) | Mean s/draw |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    s1_slots_by_arm_rep = {
        (r["arm"], r["repeat"]): r["native_seconds"] for r in res["session1"]["used_slots"]
    }
    for arm in SESSION1_ARMS:
        t = res["session1"]["arm_timing"][arm]
        lines.append(
            f"| {t['label']} | {s1_slots_by_arm_rep[(arm, 1)]:,} | "
            f"{s1_slots_by_arm_rep[(arm, 2)]:,} | {s1_slots_by_arm_rep[(arm, 3)]:,} | "
            f"{t['mean_seconds_per_draw']:.4f} |"
        )
    lines.append("")
    rr = res["session1"]["repeat2_isolated_reruns"]
    lines.append(
        f"Repeat-2 isolated reruns vs initial repeat 2: `frozen (12-turn limits)` "
        f"{rr['frozen_12turn_rep2_isolated_s']:,} s (initial {rr['frozen_12turn_rep2_initial_s']:,} s); "
        f"`seed 42, update 50 (6-turn limits)` {rr['seed42_update50_6turn_rep2_isolated_s']:,} s "
        f"(initial {rr['seed42_update50_6turn_rep2_initial_s']:,} s), clearing the frozen k=2 cutoff "
        f"(`2 * {rr['frozen_6turn_rep2_s']} / 1.15 = {rr['k2_cutoff_threshold_s']:.4f} s`) by "
        f"{rr['seed42_update50_6turn_rep2_isolated_margin_s']:.4f} s "
        f"(initial {rr['seed42_update50_6turn_rep2_initial_margin_s']:.4f} s)."
    )
    lines.append("")
    lines.append("| Trained point | Comparator | Matched frozen k (rep 1, 2, 3) | Cost ratio frozen/trained (rep 1, 2, 3) | Difference (pp) | 95% interval (pp) | Outcome | Notes |")
    lines.append("| --- | --- | --- | --- | ---: | --- | --- | --- |")
    for m in res["session1"]["matched_cost"]:
        ks = ", ".join(str(r["matched_frozen_k"]) for r in m["repeats"])
        ratios = ", ".join(f"{r['frozen_to_trained_cost_ratio']:.3f}" for r in m["repeats"])
        r0 = m["repeats"][0]
        if len({r["matched_frozen_k"] for r in m["repeats"]}) == 1:
            diff_s = f"{r0['difference_pp']:+.2f}"
            ci_s = fmt_ci(r0["ci95_low_pp"], r0["ci95_high_pp"])
        else:
            diff_s = "varies by repeat"
            ci_s = "varies by repeat"
        lines.append(
            f"| {m['trained_label']}, k={m['trained_k']} | {m['frozen_label']} | "
            f"{ks} | {ratios} | {diff_s} | {ci_s} | {m['overall_outcome']} | {m['note']} |"
        )
    for sk in res["session1"]["same_k"]:
        ratios = ", ".join(f"{x:.3f}" for x in sk["frozen_to_trained_cost_ratios"])
        lines.append(
            f"| {sk['trained_label']}, same k={sk['k']} | {sk['frozen_label']}, k={sk['k']} | "
            f"{sk['k']}, {sk['k']}, {sk['k']} | {ratios} (mean {sk['mean_cost_ratio']:.3f}) | "
            f"{sk['difference_pp']:+.2f} | {fmt_ci(sk['ci95_low_pp'], sk['ci95_high_pp'])} | "
            f"{sk['overall_outcome']} | comparable cost band [0.85, 1.15] |"
        )
    lines.append("")

    # 6. Session 2 timings and matched-cost results (two-seed comparison)
    lines.append("## Matched-cost results: two-seed same-session comparison (seeds 42 and 43)")
    lines.append("")
    lines.append("Same-session slot timings (512 questions x 4 draws = 2,048 attempts per slot):")
    lines.append("")
    lines.append("| Arm | Repeat 1 (s; s/draw) | Repeat 2 (s; s/draw) | Repeat 3 (s; s/draw) | Valid mean s/draw | Status |")
    lines.append("| --- | --- | --- | --- | ---: | --- |")
    for arm in SESSION2_ARMS:
        t = res["session2"]["arm_timing"][arm]
        cells = []
        for idx in range(3):
            rep_num = idx + 1
            sec = t["native_seconds_by_repeat"][idx]
            spd = t["seconds_per_draw_by_repeat"][idx]
            tag = " (unresolved)" if rep_num in t["unresolved_repeats"] else ""
            cells.append(f"{sec:,} ({spd:.4f}){tag}")
        if t["all_valid_mean_seconds_per_draw"] is not None:
            mean_s = f"{t['all_valid_mean_seconds_per_draw']:.4f}"
            status_s = "all 3 repeats valid"
        else:
            mean_s = f"{t['valid_only_mean_seconds_per_draw']:.4f}"
            status_s = f"repeat {', '.join(str(r) for r in t['unresolved_repeats'])} unresolved; mean over repeats {', '.join(str(r) for r in t['valid_repeats'])}"
        lines.append(f"| {t['label']} | {cells[0]} | {cells[1]} | {cells[2]} | {mean_s} | {status_s} |")
    lines.append("")
    s43_s42_ratios = ", ".join(f"{x:.3f}" for x in res["session2"]["same_session_seed43_over_seed42_ratios"])
    lines.append(
        f"Same-session `seed 43, update 50 (12-turn limits)` / `seed 42, update 50 (12-turn limits)` "
        f"cost ratio by repeat: {s43_s42_ratios} "
        f"(mean {mean(res['session2']['same_session_seed43_over_seed42_ratios']):.3f})."
    )
    lines.append("")
    lines.append("| Trained point | Comparator | Matched frozen k (rep 1, 2, 3) | Cost ratio frozen/trained (valid repeats) | Difference on valid repeats (pp) | 95% interval (pp) | Outcome | Notes |")
    lines.append("| --- | --- | --- | --- | ---: | --- | --- | --- |")
    for m in res["session2"]["matched_cost"]:
        ks = ", ".join("unresolved" if not r["isolation_valid"] else str(r["matched_frozen_k"]) for r in m["repeats"])
        valid = [r for r in m["repeats"] if r["isolation_valid"]]
        ratios = ", ".join(f"{r['frozen_to_trained_cost_ratio']:.3f}" for r in valid)
        r0 = valid[0]
        lines.append(
            f"| {m['trained_label']}, k={m['trained_k']} | {m['frozen_label']} | "
            f"{ks} | {ratios} | {r0['difference_pp']:+.2f} | "
            f"{fmt_ci(r0['ci95_low_pp'], r0['ci95_high_pp'])} | {m['overall_outcome']} | {m['note']} |"
        )
    lines.append("")

    # 7. Two-seed replication summary
    rep = res["session2"]["two_seed_replication"]
    lines.append("## Two-seed replication summary")
    lines.append("")
    lines.append(
        f"Decisive checks replicated on valid evidence: **{rep['replicated_count']} of {rep['total_decisive']}**. "
        f"Assigned two-seed label: **{rep['label']}**."
    )
    lines.append("")
    lines.append("| Decisive comparison | Criterion | Replicated | Result |")
    lines.append("| --- | --- | --- | --- |")
    for chk in rep["checks"]:
        lines.append(
            f"| {chk['comparison']} | {chk['criterion']} | "
            f"{'yes' if chk['replicated'] else 'no'} | {chk['detail']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "evidence" / "bird_16draw",
        help="Path to the bird_16draw evidence folder (default: evidence/bird_16draw).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write the full computed results as JSON.",
    )
    args = parser.parse_args()
    results = compute_all(args.evidence_dir)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    sys.stdout.write(render_markdown(results))


if __name__ == "__main__":
    main()
