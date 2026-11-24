# BIRD 16-draw evaluation evidence

This folder holds the per-question execution grades, reference-free selector picks, and GPU timing records used to regenerate the BIRD dev 16-draw sampling, best-of-16, and matched-cost results for Qwen3.5-9B with database schema in the prompt and thinking disabled.

None of the files in this folder hold question text, database rows, reference SQL, or model-generated SQL. Questions appear only as positional identifiers `dev-0` through `dev-1533` matching BIRD dev release `bird_sql_dev_20251106`.

## Agent limits and arms

- **12-turn limits**: 12 turns and 1,536 generated tokens per model call.
- **6-turn limits**: 6 turns and 768 generated tokens per model call.
- **Frozen**: untrained Qwen3.5-9B.
- **Trained adapters**: LoRA weights trained under 12-turn limits, evaluated under both 12-turn limits and 6-turn limits:
  - `seed44-update8-*`: training seed 44, optimizer update 8.
  - `seed42-update50-*`: training seed 42, optimizer update 50.
  - `seed43-update50-*`: training seed 43, optimizer update 50.

## Files

All `.jsonl.gz` files are deterministically gzip-compressed (`mtime=0`, no embedded filename). `SHA256SUMS` records the SHA-256 checksum of all 16 `.jsonl.gz` files and `timing.json`.

### Grade files (`*.grades.jsonl.gz`)

Each grade file has 24,544 JSONL rows (1,534 BIRD dev questions times 16 ordered draws, `draw` 0 to 15).

| File | Model / Seed / Update | Agent limits | Rows | Gzip bytes |
| --- | --- | --- | ---: | ---: |
| `frozen-12turn.grades.jsonl.gz` | Frozen (untrained) | 12-turn limits | 24,544 | 135,315 |
| `frozen-6turn.grades.jsonl.gz` | Frozen (untrained) | 6-turn limits | 24,544 | 134,385 |
| `seed44-update8-12turn.grades.jsonl.gz` | Seed 44, update 8 | 12-turn limits | 24,544 | 133,370 |
| `seed44-update8-6turn.grades.jsonl.gz` | Seed 44, update 8 | 6-turn limits | 24,544 | 133,171 |
| `seed42-update50-12turn.grades.jsonl.gz` | Seed 42, update 50 | 12-turn limits | 24,544 | 140,629 |
| `seed42-update50-6turn.grades.jsonl.gz` | Seed 42, update 50 | 6-turn limits | 24,544 | 140,286 |
| `seed43-update50-12turn.grades.jsonl.gz` | Seed 43, update 50 | 12-turn limits | 24,544 | 224,367 |
| `seed43-update50-6turn.grades.jsonl.gz` | Seed 43, update 50 | 6-turn limits | 24,544 | 218,283 |

Field list for grade rows:
- `question_id` (string): BIRD dev question index from `dev-0` to `dev-1533`.
- `draw` (integer): zero-based draw index from `0` to `15`; draw `0` is the first draw.
- `correct` (boolean): whether the draw's SQL matched the reference result rows.
- `generated_tokens` (integer): generated token count for that draw.
- `prompt_tokens` (integer, present in `seed43-update50-*` grade files): prompt token count for that draw.
- `truncated` (boolean): whether a model call hit the per-call token cap.
- `turn_limit_exit` (boolean): whether the agent reached the maximum turn limit without finishing.

### Pick files (`*.picks.jsonl.gz`)

Each pick file has 6,136 JSONL rows (1,534 questions times 4 sample budgets `k` in `{1, 2, 4, 16}`). Picks come from a fixed reference-free selector that chooses one draw among the first `k` draws without seeing grades.

| File | Model / Seed / Update | Agent limits | Rows | Null picks | Gzip bytes |
| --- | --- | --- | ---: | ---: | ---: |
| `frozen-12turn.picks.jsonl.gz` | Frozen (untrained) | 12-turn limits | 6,136 | 824 | 18,104 |
| `frozen-6turn.picks.jsonl.gz` | Frozen (untrained) | 6-turn limits | 6,136 | 1,026 | 18,245 |
| `seed44-update8-12turn.picks.jsonl.gz` | Seed 44, update 8 | 12-turn limits | 6,136 | 417 | 17,803 |
| `seed44-update8-6turn.picks.jsonl.gz` | Seed 44, update 8 | 6-turn limits | 6,136 | 578 | 17,037 |
| `seed42-update50-12turn.picks.jsonl.gz` | Seed 42, update 50 | 12-turn limits | 6,136 | 227 | 17,558 |
| `seed42-update50-6turn.picks.jsonl.gz` | Seed 42, update 50 | 6-turn limits | 6,136 | 723 | 17,117 |
| `seed43-update50-12turn.picks.jsonl.gz` | Seed 43, update 50 | 12-turn limits | 6,136 | 201 | 17,515 |
| `seed43-update50-6turn.picks.jsonl.gz` | Seed 43, update 50 | 6-turn limits | 6,136 | 596 | 17,662 |

Field list for pick rows:
- `question_id` (string): BIRD dev question index from `dev-0` to `dev-1533`.
- `k` (integer): number of draws available to the selector (`1`, `2`, `4`, or `16`).
- `pick` (integer or `null`): chosen draw index in `0..k-1`, or `null` when no valid candidate was selected (counted as incorrect).

### Timing file (`timing.json`, 15,513 bytes)

Uncompressed JSON holding path-free saturated GPU-timing measurements (512 questions times 4 draws = 2,048 attempts per slot on one RTX PRO 6000 card):
- `saturation`: two saturation-check stages (256-question and 512-question cohorts at `draws` 2, 4, and 8), each row with `draws`, `attempts`, and `native_seconds`.
- `session1_single_seed.rows` (20 rows): three repeats across six arms (`frozen`, `seed44-update8`, and `seed42-update50` under both 12-turn and 6-turn limits), plus two isolated repeat-2 reruns (`isolated_rerun: true`) replacing initial repeat 2 for `frozen-12turn` and `seed42-update50-6turn`.
- `session2_two_seed.rows` (15 rows): three repeats across five arms (`frozen-12turn`, `frozen-6turn`, `seed42-update50-12turn`, `seed43-update50-12turn`, `seed43-update50-6turn`) timed in the same session; repeat 1 has `isolation_valid: false` for `frozen-6turn` and `seed43-update50-6turn` with `initial_native_seconds`, `rerun_native_seconds`, and `unresolved_reason`.
- `prefix_prompt_tokens`: summed prompt tokens across all 1,534 questions for prefixes `k` in `{1, 2, 4, 16}` per arm.

Field list for timing slot rows:
- `slot` (string): slot label encoding cohort size, arm, and repeat.
- `arm` (string): public arm identifier matching the grade and pick filenames.
- `repeat` (integer): repeat number (`1`, `2`, or `3`).
- `isolated_rerun` (boolean, session 1): whether the slot is an isolated repeat-2 replacement.
- `attempts` (integer): total attempts timed in the slot (`2048`).
- `native_seconds` (integer): evaluation-phase elapsed seconds excluding model startup.
- `native_exit_code` (integer, session 1): process exit status (`0`).
- `generated_tokens` (integer): generated tokens summed over the 2,048 timed attempts.
- `prompt_tokens` (integer): prompt tokens summed over the 2,048 timed attempts.
- `cached_prompt_tokens` (integer): KV-cache hit prompt tokens summed over the 2,048 timed attempts.
- `isolation_valid` (boolean): whether isolation monitoring validated the slot.
- `isolation_note` (string, optional): note on isolation verification scope.
- `initial_native_seconds` / `rerun_native_seconds` (integer, unresolved session-2 slots): elapsed seconds for the initial attempt and its rerun.
- `unresolved_reason` (string, unresolved session-2 slots): reason isolation validation failed.

## Verification

From this folder, verify file integrity and run the reproduction script from the repository root:

```sh
shasum -a 256 -c SHA256SUMS
```

From the repository root:

```sh
python3 reproduce/sampling_and_cost.py
```
