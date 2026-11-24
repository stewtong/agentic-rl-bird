# Mixed-pool selector

Run from the repository root with Python 3. The standard-library CLI takes the
original training-candidate JSONL and the frozen arm's `detailed-results.jsonl`
from [`export_attempts.py`](../native/export_attempts.py).

```bash
python3 scripts/select_mixed_pool.py \
  --candidates /path/to/training-rows.jsonl \
  --attempts /path/to/detailed-results.jsonl \
  --condition frozen-thinking-off \
  --seed 42 \
  --output-dir /path/to/new-selection-output
```

`--max-selected` defaults to 512 and must be positive. The output directory must
not already exist. Every candidate must have exactly eight distinct attempts with
literal Boolean `correct`, `grade_status: graded`, no infrastructure errors, and a
nonnegative integer `generated_token_count`. All attempts must match the requested
condition, frozen arm, candidate ID, and candidate database ID. Explicitly graded
false native/model failures remain eligible.

The CLI validates all candidates and attempts before creating output. Mixed means
one through seven correct attempts. If more than the cap qualify, it sorts their IDs,
uses `random.Random(seed).sample`, then restores original candidate-file order.
Selected candidate JSON objects retain their original fields and values.

Outputs:

- `selected-pool.jsonl`: selected original candidate objects, or an empty file.
- `selection-report.json`: counts, all per-question Boolean outcomes, selection
  algorithm and seed, selected IDs, input SHA-256 digests, and the generated-token
  reference over exactly eight attempts per selected question.

If nothing qualifies, report status is `no_mixed_questions`; the token mean is null,
not zero. Generated tokens use exported `generated_token_count` (loss-mask sum),
never `response_span_token_count`. A new output directory is required so malformed
input failures cannot leave a misleading pool from this invocation.

Synthetic proof:

```bash
python3 -B tests/check_pool_selector.py
```

The proof invokes the real CLI through subprocesses in a temporary directory
that is removed on exit. No real pool selection or GPU/model access is performed.

## Whole-question eligibility precedes the strict selector

The derivation for Qwen3.5-9B with thinking off excludes entire infrastructure-affected groups from the original retained screen before reward-based selection. It preserves all eight attempts for every eligible question and records all excluded IDs and reasons. It yielded 903 complete groups and retained all 340 mixed groups. The strict selector itself is unchanged. This filter does not apply to held-out scoring.
