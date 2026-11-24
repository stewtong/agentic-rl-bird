# BIRD execution scoring fixtures and overlap audit

The overlap CLI compares two local dataset files. The CPU fixtures exercise the
pinned SQLite execution-accuracy comparator. Both use the Python standard library.

## Overlap CLI

Each input must be a JSON list with unique `question_id` values (strings or
integers), a nonempty `question` string, and exactly one of `db_id` or
`database_id`. Revisions are caller-supplied labels; the tool does not resolve or
download them.

```bash
python3 scripts/audit_bird_overlap.py dev.json mini_dev.json \
  --left-revision bird-dev-2023 \
  --right-revision mini-dev-2026-10-28
```

JSON stdout reports exact question-ID matches separately from normalized
`(database_id, question)` matches. Question normalization applies Unicode NFKC,
case-folding, and whitespace collapsing. Database IDs receive NFKC and trimming
but retain case. Duplicate IDs and malformed inputs exit with status 2.

Matches describe only the two supplied inputs. Unmatched rows do not establish
absence of contamination, and normalized matching does not establish semantic
equivalence.

## CPU fixtures

Run from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

The fixtures AST-load `calculate_ex` and `execute_model` from the pinned
`vendor/bird_mini_dev/evaluation/evaluation_ex.py` (fetched with
`python3 scripts/fetch_bird_scorer.py`). Their SQLite execution and
POSIX interval timer adapters are described separately from the upstream functions. No
third-party package is installed or required. The timeout test requires POSIX
`signal.setitimer`; other environments skip that test. The SQLite adapter polls
its progress handler so the pending timer exception is delivered even for an
unbounded recursive query, then translates SQLite's resulting `interrupted`
error back to the test timeout signal only when the timer actually fired.
