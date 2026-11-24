"""Regenerate the BIRD and Spider findings tables and supporting counts from compressed row files.

Stdlib only (Python 3.11+). Run from the repository root with no arguments:
    python3 reproduce/findings_tables.py
Optional JSON output:
    python3 reproduce/findings_tables.py --json PATH

Inputs read (relative to the repository root):
  - evidence/bird_rescore/{frozen-off,frozen-on,seed42,seed43,seed42-on,seed43-on}.rows.jsonl.gz
  - evidence/spider/{frozen,seed42,seed43}.rows.jsonl.gz
  - evidence/spider/{frozen,seed42,seed43}.original-scorer.jsonl.gz

This script grades no SQL: every grade is a stored boolean in the evidence files.
Scoring rules:
  - As run: a draw parsed by the parser as run keeps strict_correct; unparsed draws score wrong.
  - With recovered answers (R2): unparsed draws take r2_correct (null scores wrong).
  - R1 and R3 work the same way using r1_correct and r3_correct.

Bootstrap settings for 95% paired intervals:
  - Paired differences over questions sorted by numeric question ID.
  - 10,000 resamples with replacement using random.Random(20261120).
  - Linear 2.5th and 97.5th percentiles, printed in percentage points to two decimals.
"""
import argparse
import gzip
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BIRD_DIR = REPO_ROOT / "evidence/bird_rescore"
SPIDER_DIR = REPO_ROOT / "evidence/spider"

BOOTSTRAP_SEED = 20261120
BOOTSTRAP_RESAMPLES = 10000

SEED43_FOOTNOTE = (
    "* Seed 43's recovered score comes from a rerun that allowed four grader"
    " disagreements; the check was first set at three."
)

BIRD_ARMS = [
    ("frozen-off", "BIRD, frozen, thinking off"),
    ("frozen-on", "BIRD, frozen, thinking on"),
    ("seed42", "BIRD, seed 42, update 50"),
    ("seed43", "BIRD, seed 43, update 50"),
    ("seed42-on", "BIRD, seed 42, update 50, thinking on"),
    ("seed43-on", "BIRD, seed 43, update 50, thinking on"),
]

SPIDER_ARMS = [
    ("frozen", "Spider, frozen (result-set grade)"),
    ("seed42", "Spider, seed 42, update 50 (result-set grade)"),
    ("seed43", "Spider, seed 43, update 50 (result-set grade)"),
]

SPIDER_ORIG_LABELS = {
    "frozen": "Spider, frozen (original scorer)",
    "seed42": "Spider, seed 42, update 50 (original scorer)",
    "seed43": "Spider, seed 43, update 50 (original scorer)",
}

TABLE_CONTRASTS = [
    (
        "bird_thinking_on_minus_off",
        "BIRD, frozen, thinking on minus frozen, thinking off",
        "bird",
        "frozen-on",
        "frozen-off",
        False,
    ),
    (
        "bird_seed42_minus_thinking_on",
        "BIRD, seed 42, update 50 minus frozen, thinking on",
        "bird",
        "seed42",
        "frozen-on",
        False,
    ),
    (
        "bird_seed42_minus_thinking_off",
        "BIRD, seed 42, update 50 minus frozen, thinking off",
        "bird",
        "seed42",
        "frozen-off",
        False,
    ),
    (
        "bird_seed43_minus_thinking_on",
        "BIRD, seed 43, update 50 minus frozen, thinking on",
        "bird",
        "seed43",
        "frozen-on",
        True,
    ),
    (
        "bird_seed43_minus_thinking_off",
        "BIRD, seed 43, update 50 minus frozen, thinking off",
        "bird",
        "seed43",
        "frozen-off",
        True,
    ),
    (
        "bird_seed42_on_minus_seed42",
        "BIRD, seed 42, update 50, thinking on minus seed 42, update 50",
        "bird",
        "seed42-on",
        "seed42",
        False,
    ),
    (
        "bird_seed43_on_minus_seed43",
        "BIRD, seed 43, update 50, thinking on minus seed 43, update 50",
        "bird",
        "seed43-on",
        "seed43",
        True,
    ),
    (
        "bird_seed42_on_minus_frozen_on",
        "BIRD, seed 42, update 50, thinking on minus frozen, thinking on",
        "bird",
        "seed42-on",
        "frozen-on",
        False,
    ),
    (
        "bird_seed43_on_minus_frozen_on",
        "BIRD, seed 43, update 50, thinking on minus frozen, thinking on",
        "bird",
        "seed43-on",
        "frozen-on",
        False,
    ),
    (
        "spider_seed42_minus_frozen",
        "Spider, seed 42, update 50 minus frozen (result-set grade)",
        "spider",
        "seed42",
        "frozen",
        False,
    ),
    (
        "spider_seed43_minus_frozen",
        "Spider, seed 43, update 50 minus frozen (result-set grade)",
        "spider",
        "seed43",
        "frozen",
        False,
    ),
]


def load_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="ascii") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def qid_num(qid: str) -> int:
    return int(qid.rsplit("-", 1)[1])


def score_row(row: dict, rule: str, exclude_inside_reasoning: bool = False) -> bool:
    if row["strict_parse_ok"]:
        if exclude_inside_reasoning and row["strict_only_block_inside_reasoning"]:
            return False
        return bool(row["strict_correct"])
    if rule == "as_run":
        return False
    val = row[f"{rule.lower()}_correct"]
    return bool(val) if val is not None else False


def bootstrap_interval(diffs: tuple[int, ...]) -> tuple[float, float]:
    n = len(diffs)
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(sum(rng.choices(diffs, k=n)) for _ in range(BOOTSTRAP_RESAMPLES))
    bounds = []
    for q in (0.025, 0.975):
        pos = (len(samples) - 1) * q
        lo_idx = int(pos)
        interp = samples[lo_idx] + (samples[lo_idx + 1] - samples[lo_idx]) * (pos - lo_idx)
        bounds.append(100.0 * interp / n)
    return bounds[0], bounds[1]


def paired_contrast(left: dict[str, bool], right: dict[str, bool], ids: list[str]) -> dict:
    diffs = tuple(int(left[q]) - int(right[q]) for q in ids)
    diff_pp = 100.0 * sum(diffs) / len(diffs)
    lo_pp, hi_pp = bootstrap_interval(diffs)
    return {
        "n": len(ids),
        "difference_pp": round(diff_pp, 2),
        "interval_95_pp": [round(lo_pp, 2), round(hi_pp, 2)],
        "formatted": f"{diff_pp:+.2f} [{lo_pp:+.2f}, {hi_pp:+.2f}]",
    }


def fmt_pct(k: int, n: int, decimals: int = 2) -> str:
    return f"{100.0 * k / n:.{decimals}f}%" if n else "0.00%"


def compute_all() -> tuple[str, dict]:
    bird_all = {
        arm: {(r["question_id"], r["draw"]): r for r in load_jsonl_gz(BIRD_DIR / f"{arm}.rows.jsonl.gz")}
        for arm, _ in BIRD_ARMS
    }
    spider_all = {
        arm: {(r["question_id"], r["draw"]): r for r in load_jsonl_gz(SPIDER_DIR / f"{arm}.rows.jsonl.gz")}
        for arm, _ in SPIDER_ARMS
    }
    spider_orig_all = {
        arm: {(r["question_id"], r["draw"]): bool(r["correct"]) for r in load_jsonl_gz(SPIDER_DIR / f"{arm}.original-scorer.jsonl.gz")}
        for arm, _ in SPIDER_ARMS
    }

    bird_ids = sorted({q for q, _ in bird_all["frozen-off"]}, key=qid_num)
    spider_ids = sorted({q for q, _ in spider_all["frozen"]}, key=qid_num)

    bird_d0 = {arm: {q: bird_all[arm][(q, 0)] for q in bird_ids} for arm, _ in BIRD_ARMS}
    spider_d0 = {arm: {q: spider_all[arm][(q, 0)] for q in spider_ids} for arm, _ in SPIDER_ARMS}
    spider_orig_d0 = {arm: {q: spider_orig_all[arm][(q, 0)] for q in spider_ids} for arm, _ in SPIDER_ARMS}

    # Block 1: First-draw accuracy by arm
    accuracy_rows = []
    md_lines = [
        "## Accuracy by arm (first draw)",
        "",
        "| Arm | Questions | As run | With recovered answers |",
        "| --- | --- | --- | --- |",
    ]

    for suite, arms_list, d0_map, ids in (
        ("bird", BIRD_ARMS, bird_d0, bird_ids),
        ("spider", SPIDER_ARMS, spider_d0, spider_ids),
    ):
        n = len(ids)
        for arm, label in arms_list:
            k_run = sum(score_row(d0_map[arm][q], "as_run") for q in ids)
            k_r2 = sum(score_row(d0_map[arm][q], "R2") for q in ids)
            pct_run = round(100.0 * k_run / n, 2)
            pct_r2 = round(100.0 * k_r2 / n, 2)
            mark = "*" if (suite == "bird" and arm == "seed43") else ""
            accuracy_rows.append({
                "suite": suite,
                "arm": arm,
                "label": label,
                "questions": n,
                "as_run_correct": k_run,
                "as_run_pct": pct_run,
                "recovered_correct": k_r2,
                "recovered_pct": pct_r2,
                "seed43_rerun_note": bool(mark),
            })
            md_lines.append(
                f"| {label} | {n:,} | {k_run} ({100.0 * k_run / n:.2f}%) | {k_r2} ({100.0 * k_r2 / n:.2f}%){mark} |"
            )

    md_lines.extend(["", SEED43_FOOTNOTE, ""])

    # Block 2: Paired contrasts (first draw)
    contrast_rows = []
    md_lines.extend([
        "## Paired contrasts (first draw, points with 95% interval)",
        "",
        "| Contrast | As run | With recovered answers |",
        "| --- | --- | --- |",
    ])

    contrast_lookup = {}
    for key, label, suite, left_arm, right_arm, has_seed43_r2 in TABLE_CONTRASTS:
        d0_map = bird_d0 if suite == "bird" else spider_d0
        ids = bird_ids if suite == "bird" else spider_ids
        res_run = paired_contrast(
            {q: score_row(d0_map[left_arm][q], "as_run") for q in ids},
            {q: score_row(d0_map[right_arm][q], "as_run") for q in ids},
            ids,
        )
        res_r2 = paired_contrast(
            {q: score_row(d0_map[left_arm][q], "R2") for q in ids},
            {q: score_row(d0_map[right_arm][q], "R2") for q in ids},
            ids,
        )
        mark = "*" if has_seed43_r2 else ""
        rec = {
            "key": key,
            "contrast": label,
            "suite": suite,
            "left": left_arm,
            "right": right_arm,
            "n": len(ids),
            "as_run": {
                "difference_pp": res_run["difference_pp"],
                "interval_95_pp": res_run["interval_95_pp"],
            },
            "with_recovered_answers": {
                "difference_pp": res_r2["difference_pp"],
                "interval_95_pp": res_r2["interval_95_pp"],
            },
            "seed43_rerun_note": has_seed43_r2,
        }
        contrast_rows.append(rec)
        contrast_lookup[key] = rec
        md_lines.append(f"| {label} | {res_run['formatted']} | {res_r2['formatted']}{mark} |")

    md_lines.extend(["", SEED43_FOOTNOTE, ""])

    # Block 3: Counts behind the findings
    counts_data: dict = {}
    md_lines.extend(["## Counts behind the findings", ""])

    # 3a. First-draw gain in correct answers (Finding 1) and recovery rules R1/R2/R3 (Finding 2)
    s42_run = sum(score_row(bird_d0["seed42"][q], "as_run") for q in bird_ids)
    s43_run = sum(score_row(bird_d0["seed43"][q], "as_run") for q in bird_ids)
    off_run = sum(score_row(bird_d0["frozen-off"][q], "as_run") for q in bird_ids)
    s42_gain_as_run = s42_run - off_run

    def diff_pp_rule(left_arm: str, right_arm: str, rule: str) -> float:
        d = sum(
            int(score_row(bird_d0[left_arm][q], rule)) - int(score_row(bird_d0[right_arm][q], rule))
            for q in bird_ids
        )
        return 100.0 * d / len(bird_ids)

    lead_s42_run = diff_pp_rule("seed42", "frozen-off", "as_run")
    lead_s43_run = diff_pp_rule("seed43", "frozen-off", "as_run")
    lead_s42_r1 = diff_pp_rule("seed42", "frozen-off", "R1")
    lead_s43_r1 = diff_pp_rule("seed43", "frozen-off", "R1")
    lead_s42_r2 = diff_pp_rule("seed42", "frozen-off", "R2")
    lead_s43_r2 = diff_pp_rule("seed43", "frozen-off", "R2")
    lead_s42_r3 = diff_pp_rule("seed42", "frozen-off", "R3")
    lead_s43_r3 = diff_pp_rule("seed43", "frozen-off", "R3")

    share_s42_r2 = 100.0 * lead_s42_r2 / lead_s42_run
    share_s43_r2 = 100.0 * lead_s43_r2 / lead_s43_run
    share_s42_r3 = 100.0 * lead_s42_r3 / lead_s42_run
    share_s43_r3 = 100.0 * lead_s43_r3 / lead_s43_run

    counts_data["bird_recovery_leads"] = {
        "seed42_minus_frozen_off_as_run_correct_gain": s42_gain_as_run,
        "seed42_minus_frozen_off_r1_pp": round(lead_s42_r1, 2),
        "seed43_minus_frozen_off_r1_pp": round(lead_s43_r1, 2),
        "seed42_minus_frozen_off_r2_share_of_as_run_pct": round(share_s42_r2, 2),
        "seed43_minus_frozen_off_r2_share_of_as_run_pct": round(share_s43_r2, 2),
        "seed42_minus_frozen_off_r3_pp": round(lead_s42_r3, 2),
        "seed43_minus_frozen_off_r3_pp": round(lead_s43_r3, 2),
        "seed42_minus_frozen_off_r3_share_of_as_run_pct": round(share_s42_r3, 2),
        "seed43_minus_frozen_off_r3_share_of_as_run_pct": round(share_s43_r3, 2),
    }

    md_lines.extend([
        f"- BIRD, seed 42, update 50 minus frozen, thinking off (as run, correct answers): {s42_gain_as_run}",
        f"- BIRD, seed 42, update 50 minus frozen, thinking off, share of as-run lead kept under R2: {share_s42_r2:.2f}%",
        f"- BIRD, seed 43, update 50 minus frozen, thinking off, share of as-run lead kept under R2: {share_s43_r2:.2f}%*",
        f"- BIRD, seed 42, update 50 minus frozen, thinking off (R1): {lead_s42_r1:+.2f} points",
        f"- BIRD, seed 43, update 50 minus frozen, thinking off (R1): {lead_s43_r1:+.2f} points*",
        f"- BIRD, seed 42, update 50 minus frozen, thinking off (R3): {lead_s42_r3:+.2f} points ({share_s42_r3:.2f}% of as-run lead)",
        f"- BIRD, seed 43, update 50 minus frozen, thinking off (R3): {lead_s43_r3:+.2f} points ({share_s43_r3:.2f}% of as-run lead)*",
    ])

    # 3b. Answered, recovered, and recovered-correct counts per arm (first draw)
    per_arm_recovery = {}
    for suite, arms_list, d0_map, ids in (
        ("bird", BIRD_ARMS, bird_d0, bird_ids),
        ("spider", SPIDER_ARMS, spider_d0, spider_ids),
    ):
        for arm, label in arms_list:
            rows0 = [d0_map[arm][q] for q in ids]
            as_run_ans = sum(bool(r["strict_parse_ok"]) for r in rows0)
            as_run_ok = sum(score_row(r, "as_run") for r in rows0)
            r1_rec = sum((not r["strict_parse_ok"]) and r["r1_correct"] is not None for r in rows0)
            r1_rec_ok = sum((not r["strict_parse_ok"]) and bool(r["r1_correct"]) for r in rows0)
            r2_rec = sum((not r["strict_parse_ok"]) and r["r2_correct"] is not None for r in rows0)
            r2_rec_ok = sum((not r["strict_parse_ok"]) and bool(r["r2_correct"]) for r in rows0)
            r2_ans = as_run_ans + r2_rec
            key = f"{suite}:{arm}"
            per_arm_recovery[key] = {
                "label": label,
                "as_run_correct": as_run_ok,
                "as_run_answered": as_run_ans,
                "as_run_correct_of_answered_pct": round(100.0 * as_run_ok / as_run_ans, 2) if as_run_ans else 0.0,
                "r1_recovered": r1_rec,
                "r1_recovered_correct": r1_rec_ok,
                "r2_recovered": r2_rec,
                "r2_recovered_correct": r2_rec_ok,
                "r2_recovered_correct_pct": round(100.0 * r2_rec_ok / r2_rec, 2) if r2_rec else 0.0,
                "r2_answered": r2_ans,
            }
            as_run_share = f"{100.0 * as_run_ok / as_run_ans:.1f}%" if as_run_ans else "0.0%"
            r2_share = f"{100.0 * r2_rec_ok / r2_rec:.1f}%" if r2_rec else "0.0%"
            md_lines.append(
                f"- {label} (first draw, as run): {as_run_ok} of {as_run_ans:,} answered ({as_run_share})"
            )
            if suite == "bird" and arm == "frozen-on":
                r1_share = f"{100.0 * r1_rec_ok / r1_rec:.1f}%" if r1_rec else "0.0%"
                md_lines.append(
                    f"- {label} (first draw, R1 recovered): {r1_rec_ok} of {r1_rec} recovered ({r1_share})"
                )
            md_lines.append(
                f"- {label} (first draw, R2 recovered): {r2_rec_ok} of {r2_rec} recovered ({r2_share}), {r2_ans:,} answered total"
            )
    counts_data["per_arm_recovery"] = per_arm_recovery

    # 3c. Answers taken from inside the reasoning and sensitivity contrasts (first draw, R2)
    inside_counts = {}
    for arm, label in BIRD_ARMS:
        rows0 = [bird_d0[arm][q] for q in bird_ids]
        inside_rows = [r for r in rows0 if r["strict_parse_ok"] and r["strict_only_block_inside_reasoning"]]
        n_in = len(inside_rows)
        k_in = sum(bool(r["strict_correct"]) for r in inside_rows)
        r2_excl = sum(score_row(r, "R2", exclude_inside_reasoning=True) for r in rows0)
        inside_counts[arm] = {
            "label": label,
            "inside_reasoning": n_in,
            "inside_reasoning_correct": k_in,
            "r2_correct_excluding_inside_reasoning": r2_excl,
        }
        md_lines.append(
            f"- {label} (first draw, parsed as run from inside reasoning): {n_in} ({k_in} correct)"
        )

    on_excl = inside_counts["frozen-on"]["r2_correct_excluding_inside_reasoning"]
    md_lines.append(
        f"- BIRD, frozen, thinking on, with recovered answers, excluding inside-reasoning answers: {on_excl} ({100.0 * on_excl / len(bird_ids):.2f}%)"
    )

    excl_specs = [
        (
            "bird_thinking_on_minus_off",
            "BIRD, frozen, thinking on minus frozen, thinking off (R2, excluding inside-reasoning answers)",
            "frozen-on",
            "frozen-off",
            False,
        ),
        (
            "bird_seed42_minus_thinking_on",
            "BIRD, seed 42, update 50 minus frozen, thinking on (R2, excluding inside-reasoning answers)",
            "seed42",
            "frozen-on",
            False,
        ),
        (
            "bird_seed43_minus_thinking_on",
            "BIRD, seed 43, update 50 minus frozen, thinking on (R2, excluding inside-reasoning answers)",
            "seed43",
            "frozen-on",
            True,
        ),
        (
            "bird_seed42_on_minus_frozen_on",
            "BIRD, seed 42, update 50, thinking on minus frozen, thinking on (R2, excluding inside-reasoning answers)",
            "seed42-on",
            "frozen-on",
            False,
        ),
        (
            "bird_seed43_on_minus_frozen_on",
            "BIRD, seed 43, update 50, thinking on minus frozen, thinking on (R2, excluding inside-reasoning answers)",
            "seed43-on",
            "frozen-on",
            False,
        ),
    ]
    excl_contrasts = {}
    for key, label, left_arm, right_arm, has_mark in excl_specs:
        res = paired_contrast(
            {q: score_row(bird_d0[left_arm][q], "R2", exclude_inside_reasoning=True) for q in bird_ids},
            {q: score_row(bird_d0[right_arm][q], "R2", exclude_inside_reasoning=True) for q in bird_ids},
            bird_ids,
        )
        excl_contrasts[key] = {
            "label": label,
            "n": res["n"],
            "difference_pp": res["difference_pp"],
            "interval_95_pp": res["interval_95_pp"],
        }
        mark = "*" if has_mark else ""
        md_lines.append(f"- {label}: {res['formatted']}{mark}")

    s42_r2_cnt = sum(score_row(bird_d0["seed42"][q], "R2") for q in bird_ids)
    s43_r2_cnt = sum(score_row(bird_d0["seed43"][q], "R2") for q in bird_ids)
    on_r2_cnt = sum(score_row(bird_d0["frozen-on"][q], "R2") for q in bird_ids)
    gain_over_thinking_lo = s42_r2_cnt - on_r2_cnt
    gain_over_thinking_hi = s43_r2_cnt - on_excl
    md_lines.append(
        f"- BIRD, training gain over frozen, thinking on (R2, correct answers): {gain_over_thinking_lo} to {gain_over_thinking_hi}"
    )

    counts_data["inside_reasoning"] = {
        "by_arm": inside_counts,
        "contrasts_excluding_inside_reasoning": excl_contrasts,
        "training_correct_gain_over_frozen_thinking_on_range": [gain_over_thinking_lo, gain_over_thinking_hi],
    }

    # 3d. Reasoning close-tag counts (draws 0 to 3, rows with final message)
    close_tags = {}
    for arm, label in BIRD_ARMS:
        all_rows = list(bird_all[arm].values())
        final_rows = [r for r in all_rows if r["final_key_present"]]
        with_tag = sum(bool(r["had_close_tag"]) for r in final_rows)
        close_tags[arm] = {
            "label": label,
            "with_close_tag": with_tag,
            "final_answers": len(final_rows),
        }
        md_lines.append(
            f"- {label} (draws 0 to 3, final answers with reasoning close tag): {with_tag:,} of {len(final_rows):,}"
        )
    counts_data["close_tags_all_draws"] = close_tags

    # 3e. No-answer and turn-limit counts after R2 (first draw)
    no_answer = {}
    for arm, label in BIRD_ARMS:
        rows0 = [bird_d0[arm][q] for q in bird_ids]
        unans = [r for r in rows0 if (not r["strict_parse_ok"]) and r["r2_correct"] is None]
        tl = sum(bool(r["turn_limit"]) for r in unans)
        no_answer[arm] = {
            "label": label,
            "no_answer_after_r2": len(unans),
            "turn_limit_exits": tl,
        }
        md_lines.append(
            f"- {label} (first draw, no answer after R2): {len(unans)} ({tl} turn-limit exits)"
        )
    counts_data["no_answer_after_r2"] = no_answer

    # 3f. Spider no-final-SQL counts and original-scorer counts and contrasts (first draw)
    spider_no_final = {}
    for arm, _ in SPIDER_ARMS:
        spider_no_final[arm] = sum(not spider_d0[arm][q]["strict_parse_ok"] for q in spider_ids)
    drop_s42 = spider_no_final["frozen"] - spider_no_final["seed42"]
    drop_s43 = spider_no_final["frozen"] - spider_no_final["seed43"]

    orig_counts = {arm: sum(spider_orig_d0[arm][q] for q in spider_ids) for arm, _ in SPIDER_ARMS}
    orig_gain_s42 = orig_counts["seed42"] - orig_counts["frozen"]
    orig_gain_s43 = orig_counts["seed43"] - orig_counts["frozen"]
    orig_c42 = paired_contrast(spider_orig_d0["seed42"], spider_orig_d0["frozen"], spider_ids)
    orig_c43 = paired_contrast(spider_orig_d0["seed43"], spider_orig_d0["frozen"], spider_ids)
    orig_seed_diff_pp = 100.0 * (orig_counts["seed42"] - orig_counts["seed43"]) / len(spider_ids)

    counts_data["spider_no_final_sql"] = {
        "frozen": spider_no_final["frozen"],
        "seed42": spider_no_final["seed42"],
        "seed43": spider_no_final["seed43"],
        "drop_seed42_from_frozen": drop_s42,
        "drop_seed43_from_frozen": drop_s43,
    }
    counts_data["spider_original_scorer"] = {
        "questions": len(spider_ids),
        "counts": {
            arm: {
                "label": SPIDER_ORIG_LABELS[arm],
                "correct": orig_counts[arm],
                "pct": round(100.0 * orig_counts[arm] / len(spider_ids), 2),
            }
            for arm, _ in SPIDER_ARMS
        },
        "correct_gain_seed42_minus_frozen": orig_gain_s42,
        "correct_gain_seed43_minus_frozen": orig_gain_s43,
        "seed42_minus_frozen": {
            "difference_pp": orig_c42["difference_pp"],
            "interval_95_pp": orig_c42["interval_95_pp"],
        },
        "seed43_minus_frozen": {
            "difference_pp": orig_c43["difference_pp"],
            "interval_95_pp": orig_c43["interval_95_pp"],
        },
        "seed42_minus_seed43_pp": round(orig_seed_diff_pp, 2),
    }

    md_lines.extend([
        f"- Spider, frozen (first draw, no final SQL as run): {spider_no_final['frozen']}",
        f"- Spider, seed 42, update 50 (first draw, no final SQL as run): {spider_no_final['seed42']} (drop of {drop_s42} from frozen)",
        f"- Spider, seed 43, update 50 (first draw, no final SQL as run): {spider_no_final['seed43']} (drop of {drop_s43} from frozen)",
        f"- Spider, frozen (first draw, original scorer): {orig_counts['frozen']} of {len(spider_ids):,} ({100.0 * orig_counts['frozen'] / len(spider_ids):.2f}%)",
        f"- Spider, seed 42, update 50 (first draw, original scorer): {orig_counts['seed42']} of {len(spider_ids):,} ({100.0 * orig_counts['seed42'] / len(spider_ids):.2f}%, gain of {orig_gain_s42} over frozen)",
        f"- Spider, seed 43, update 50 (first draw, original scorer): {orig_counts['seed43']} of {len(spider_ids):,} ({100.0 * orig_counts['seed43'] / len(spider_ids):.2f}%, gain of {orig_gain_s43} over frozen)",
        f"- Spider, seed 42, update 50 minus frozen (first draw, original scorer): {orig_c42['formatted']}",
        f"- Spider, seed 43, update 50 minus frozen (first draw, original scorer): {orig_c43['formatted']}",
        f"- Spider, seed 42, update 50 minus seed 43, update 50 (first draw, original scorer): {orig_seed_diff_pp:+.2f} points ({orig_seed_diff_pp:.1f} points to one decimal)",
    ])

    # 3g. Grader disagreements on BIRD regrades and contrasts with the six questions dropped
    off_arms = ("frozen-off", "seed42", "seed43")
    on_arms = ("frozen-on", "seed42-on", "seed43-on")
    regraded = {}
    disagreements = {}
    for arm, _ in BIRD_ARMS:
        rr = [r for r in bird_all[arm].values() if r["regrade_strict_correct"] is not None]
        regraded[arm] = len(rr)
        disagreements[arm] = [r for r in rr if bool(r["regrade_strict_correct"]) != bool(r["strict_correct"])]

    dis_off_rows = [r for arm in off_arms for r in disagreements[arm]]
    dis_on_rows = [r for arm in on_arms for r in disagreements[arm]]
    n_regraded_off = sum(regraded[arm] for arm in off_arms)
    n_regraded_on = sum(regraded[arm] for arm in on_arms)
    n_regraded_adapters_on = regraded["seed42-on"] + regraded["seed43-on"]
    dis_qids = sorted({r["question_id"] for r in dis_off_rows}, key=qid_num)
    all_orig_false_regrade_true = all(
        (not bool(r["strict_correct"])) and bool(r["regrade_strict_correct"]) for r in dis_off_rows
    )
    kept_ids = [q for q in bird_ids if q not in set(dis_qids)]

    def drop_diff_pp(left_arm: str, right_arm: str) -> float:
        d = sum(
            int(score_row(bird_d0[left_arm][q], "R2")) - int(score_row(bird_d0[right_arm][q], "R2"))
            for q in kept_ids
        )
        return 100.0 * d / len(kept_ids)

    drop_on_off = drop_diff_pp("frozen-on", "frozen-off")
    drop_s42_on = drop_diff_pp("seed42", "frozen-on")
    drop_s42_off = drop_diff_pp("seed42", "frozen-off")

    counts_data["grader_disagreements"] = {
        "thinking_off_disagreements": len(dis_off_rows),
        "thinking_off_regrades": n_regraded_off,
        "thinking_off_by_arm": {
            arm: {"disagreements": len(disagreements[arm]), "regrades": regraded[arm]} for arm in off_arms
        },
        "distinct_questions": dis_qids,
        "all_originally_incorrect_and_regrade_correct": all_orig_false_regrade_true,
        "thinking_on_disagreements": len(dis_on_rows),
        "thinking_on_regrades": n_regraded_on,
        "thinking_on_by_arm": {
            arm: {"disagreements": len(disagreements[arm]), "regrades": regraded[arm]} for arm in on_arms
        },
        "thinking_on_adapters_regrades": n_regraded_adapters_on,
        "contrasts_dropping_disagreement_questions": {
            "n": len(kept_ids),
            "bird_thinking_on_minus_off_r2_pp": round(drop_on_off, 2),
            "bird_seed42_minus_thinking_on_r2_pp": round(drop_s42_on, 2),
            "bird_seed42_minus_thinking_off_r2_pp": round(drop_s42_off, 2),
        },
    }

    md_lines.extend([
        (
            f"- BIRD thinking-off regrade disagreements: {len(dis_off_rows)} of {n_regraded_off:,}"
            f" (frozen, thinking off: {len(disagreements['frozen-off'])} of {regraded['frozen-off']:,};"
            f" seed 42, update 50: {len(disagreements['seed42'])} of {regraded['seed42']:,};"
            f" seed 43, update 50: {len(disagreements['seed43'])} of {regraded['seed43']:,})"
        ),
        (
            f"- BIRD thinking-off disagreement questions: {len(dis_qids)} distinct"
            f" ({', '.join(dis_qids)});"
            f" all originally incorrect and correct on regrade: {'yes' if all_orig_false_regrade_true else 'no'}"
        ),
        (
            f"- BIRD thinking-on regrade disagreements: {len(dis_on_rows)} of {n_regraded_on:,}"
            f" (frozen, thinking on: {len(disagreements['frozen-on'])} of {regraded['frozen-on']:,};"
            f" two adapters with thinking on: {len(disagreements['seed42-on']) + len(disagreements['seed43-on'])} of {n_regraded_adapters_on:,})"
        ),
        f"- BIRD, frozen, thinking on minus frozen, thinking off (R2, {len(dis_qids)} disagreement questions dropped, n={len(kept_ids):,}): {drop_on_off:+.2f} points",
        f"- BIRD, seed 42, update 50 minus frozen, thinking on (R2, {len(dis_qids)} disagreement questions dropped, n={len(kept_ids):,}): {drop_s42_on:+.2f} points",
        f"- BIRD, seed 42, update 50 minus frozen, thinking off (R2, {len(dis_qids)} disagreement questions dropped, n={len(kept_ids):,}): {drop_s42_off:+.2f} points",
    ])

    payload = {
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "method": "percentile_linear_2.5_97.5",
        },
        "accuracy_by_arm": accuracy_rows,
        "paired_contrasts": contrast_rows,
        "counts_behind_findings": counts_data,
    }
    return "\n".join(md_lines) + "\n", payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate BIRD and Spider findings tables and supporting counts from compressed row files."
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        help="Optional path to write the computed tables and counts as JSON.",
    )
    args = parser.parse_args()
    markdown, payload = compute_all()
    print(markdown, end="")
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
