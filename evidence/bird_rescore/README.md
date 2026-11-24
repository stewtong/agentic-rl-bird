# BIRD per-question grade rows

Per-question evaluation rows for six Qwen3.5-9B arms on the 1,534 BIRD dev questions across draws 0 to 3. Each file is gzip-compressed JSON Lines (`mtime=0`, no embedded file name) and listed in `SHA256SUMS`.

None of these files hold question text, reference SQL, or model SQL.

## Files

| File | Arm | `arm` field value | Rows | Questions | Draws |
| --- | --- | --- | ---: | ---: | --- |
| `frozen-off.rows.jsonl.gz` | frozen, thinking off | `bird-off` | 6,136 | 1,534 | 0-3 |
| `frozen-on.rows.jsonl.gz` | frozen, thinking on | `bird-on` | 6,136 | 1,534 | 0-3 |
| `seed42.rows.jsonl.gz` | seed 42, update 50 | `bird-seed42` | 6,136 | 1,534 | 0-3 |
| `seed43.rows.jsonl.gz` | seed 43, update 50* | `bird-seed43` | 6,136 | 1,534 | 0-3 |
| `seed42-on.rows.jsonl.gz` | seed 42, update 50, thinking on | `bird-seed42-on` | 6,136 | 1,534 | 0-3 |
| `seed43-on.rows.jsonl.gz` | seed 43, update 50, thinking on | `bird-seed43-on` | 6,136 | 1,534 | 0-3 |

* Seed 43's recovered score comes from a rerun that allowed four grader disagreements; the check was first set at three.

## Fields

Each line is one JSON object with 16 fields:

- `arm` (`str`): identifier for the evaluated arm.
- `question_id` (`str`): BIRD dev question identifier (`dev-0` through `dev-1533`).
- `draw` (`int`): sampled attempt index (`0`, `1`, `2`, or `3`); draw `0` is the first draw.
- `final_key_present` (`bool`): whether the rollout ended with a final assistant message.
- `strict_parse_ok` (`bool`): whether the parser as run extracted a single SQL block from the final message.
- `strict_exception_type` (`str | null`): parser exception type when the as-run parser rejected a final message (`ValueError`), otherwise `null`.
- `turn_limit` (`bool`): whether the rollout stopped at the turn limit without a final answer.
- `capped` (`bool`): whether the rollout hit the generation length cap.
- `strict_correct` (`bool`): execution correctness recorded in-run for an as-run parsed answer (`false` when unparsed).
- `strict_only_block_inside_reasoning` (`bool | null`): for a row parsed as run, whether the single SQL block sat inside the reasoning span before the close tag (`null` when unparsed).
- `had_close_tag` (`bool | null`): whether the final message contained a reasoning close tag (`null` when `final_key_present` is `false`).
- `blocks_in_answer_span` (`int | null`): count of fenced SQL blocks after the reasoning close tag, or in the full final message if no close tag appeared (`null` when `final_key_present` is `false`).
- `r1_correct` (`bool | null`): execution correctness under recovery rule R1 (strip reasoning before parsing) for rows rejected as run; `null` when R1 recovered no SQL or the row parsed as run.
- `r2_correct` (`bool | null`): execution correctness under recovery rule R2 (R1 plus identical duplicate SQL blocks); `null` when R2 recovered no SQL or the row parsed as run.
- `r3_correct` (`bool | null`): execution correctness under recovery rule R3 (R2 plus taking the last block when blocks differ); `null` when R3 recovered no SQL or the row parsed as run.
- `regrade_strict_correct` (`bool | null`): idle-machine regrade result for first-draw rows parsed as run; `null` on other rows.
