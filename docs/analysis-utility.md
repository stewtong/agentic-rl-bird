# BIRD analysis utility

[`scripts/analyze_bird.py`](../scripts/analyze_bird.py) analyzes scored JSONL
records with Python 3.11 or newer. It uses the standard library. Run the
examples from the repository root.

## JSONL contract

Each nonblank line is one JSON object with:

- `question_id`, `condition`, `attempt_id`: nonempty strings.
- `training_seed`: `null` for frozen/base outcomes, otherwise a nonempty string.
- `correct`: JSON `true` or `false` only (numeric `0`/`1` is rejected).
- `database_id`: optional nonempty string; when present it is part of the question key.

No other fields are accepted. The full question/condition/seed/attempt key must
be unique. Missing or malformed records fail nonzero; records are never dropped.

## Primary paired analysis

```bash
python3 scripts/analyze_bird.py primary outcomes.jsonl \
  --seeds seed-a seed-b seed-c
```

The file must contain exactly one condition, one frozen attempt per question,
and one attempt per question for each of the exactly three named seeds. All seed
question sets must match. `--replicates` defaults to 10,000 and `--rng-seed` to
20261117; both values are reported. Use the flags to choose other explicit
settings. For each seed and for the pooled analysis, the percentile bootstrap
resamples question IDs with replacement and computes 95% intervals from the
2.5th and 97.5th percentiles using linear interpolation. Independent RNG
streams are derived by adding 0, 1, 2, and 3 to the analysis RNG seed.

For each seed the gain is `trained - frozen`. The pooled difference first
averages the three trained binary outcomes within question, then subtracts the
frozen outcome. Its interval is explicitly conditional on those observed seeds,
not a training-seed population interval. The output also reports exact two-sided
McNemar tests (conditional exact binomial, doubled smaller tail capped at one)
and the min/max/range/population standard deviation of seed-specific gains.
The claim rule passes only when the pooled interval lower bound is strictly
greater than 0.03 (3 percentage points).

## Multi-attempt oracle sampling

```bash
python3 scripts/analyze_bird.py sampling attempts.jsonl
```

By default, each question/condition/training-seed group needs eight distinct
attempts, with the same question set for every seed within a condition. For each
question, the unbiased estimator at k = 1, 2, 4, 8 is
`1 - C(8-c,k)/C(8,k)`, where `c` is its correct attempt count. Group results are
the arithmetic mean over questions. `--attempts 16` reads sixteen attempts per group and
reports k = 1, 2, 4, 16, the k values used for best-of-k selection;
pass@8 is not reported on that path. This is gold-scored **oracle pass@k**, an
upper bound for a selector choosing from the same candidate draws. Report
selected-answer accuracy separately.

To evaluate a selector, save its chosen answer using the candidate SQL and
execution results before joining the reference grades.

All output is JSON on stdout. Validation and I/O errors go to stderr and exit 2.
