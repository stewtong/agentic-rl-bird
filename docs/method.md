<!-- register: public method note | reader: an engineer checking how a number in the README was produced | consumed: looked up by section -->
# Method and recorded settings

Every setting below was recorded during the study. Settings that were not recorded are listed at the end.

## Agent

The agent is a chat loop around one model endpoint. The system prompt describes a read-only SQLite task and two tools:

- `describe_schema` lists tables, columns and sample rows, for one table or all of them.
- `run_sql` executes a `SELECT`, `WITH` or `VALUES` statement and returns rows.

Each turn the model either calls tools or stops. Tool results return to the model as JSON in a tool message. The run ends when the model replies without a tool call, when a reply hits the per-call token cap, or when the turn limit is reached. A final reply counts as an answer only if it holds exactly one fenced SQL block containing one read-only statement. A run that reaches the turn limit, or whose final reply fails that rule, scores 0.

The per-question prompt carries the database ID, the question and the BIRD evidence text. In every evaluation and training run reported in the README it also carries the database schema, rendered from the database files without reference SQL or result rows. The one exception is the schema comparison, which scores the frozen model with and without it.

| Name used in this repository | Turns | Generated tokens per model call |
| --- | ---: | ---: |
| 12-turn limits | 12 | 1,536 |
| 6-turn limits | 6 | 768 |

[`native/bird_evaluator/bird_agent.py`](../native/bird_evaluator/bird_agent.py) is the agent for the 6-turn limits, and the included data preparation builds prompts without the schema. The 12-turn agent and the schema prompt builder are not included.

## Training

| Setting | Value | Recorded for |
| --- | --- | --- |
| Base model | `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, thinking off | Update-50 runs |
| Algorithm | GRPO with LoRA in Miles; rollouts served by SGLang, training in Megatron | Update-50 runs |
| Loss | Token-level aggregation, KL coefficient 0, symmetric clip range 0.20, group standard-deviation normalization | Update-50 runs |
| Learning rate | 1e-5 | Update-50 runs |
| Batch | 16 questions per update, 8 attempts per question, 128 episodes | Update-50 runs |
| Reward | Binary execution accuracy of the final SQL | Update-50 runs |
| Agent limits | 12-turn limits | Update-50 runs |
| Parallelism | Four RTX PRO 6000 cards, tensor parallel 1, data parallel 4, rollout engines on the same cards. The largest sampled device memory in the seed-42 run was 97,159 MiB | Update-50 runs |
| LoRA | Rank 32, alpha 64, on the attention query-key-value and output projections and both MLP projections of every decoder layer | Included launcher |
| Optimizer | Adam, betas 0.9 and 0.98, constant learning rate | Included launcher |
| Sampling | Temperature 1.0 | Included launcher and evaluation |

Rows marked "Included launcher" come from [`native/run_campaign_9b.sh`](../native/run_campaign_9b.sh), which is the launcher for an earlier setup. The update-50 runs used a later version of it that was not retained in this repository, so those rows are the closest recorded values. The included launcher requests eight GPUs; the update-50 runs used four.

Both seeds used the settings above. Seed 42 stopped with a CUDA out-of-memory error at update 53, while Triton was autotuning the L2-normalization backward kernel of the flash-linear-attention library. Seed 43 ran with a fixed configuration for that kernel and a modified trainer file, and was stopped at update 50 by plan. Neither the kernel configuration nor the trainer change is included here. Two earlier seed-43 launches stopped at updates 6 and 12 and are excluded from its reported GPU-hours.

A third run, seed 44, used the same settings and has a scored adapter at update 8. It appears in the 16-draw and timing evidence as an early-training reference.

Checkpoints at updates 2, 4, 8, 16, 32, 50 and 100 were named for scoring before training started. Update 50 is reported because it is the last of them that seed 42 reached before it stopped. All checkpoints were scored on BIRD dev.

## Training questions

1. The BIRD train release has 69 databases. Seven were held out for validation and 62 kept for training.
2. Up to 1,024 questions were sampled from the 62 training databases with a fixed seed.
3. The frozen model attempted each candidate eight times.
4. Questions with any infrastructure failure among their eight attempts were dropped, leaving 903.
5. The 340 questions with mixed outcomes, some attempts correct and some incorrect, became the training pool.

[`native/prepare_bird_data.py`](../native/prepare_bird_data.py) performs steps 1 and 2 and [`scripts/select_mixed_pool.py`](../scripts/select_mixed_pool.py) performs step 5. BIRD dev questions were not used to screen or select training questions. The 12-turn limits and the schema prompt were first used in evaluations on BIRD dev and then adopted for training, so the agent setup was chosen with dev results in view.

## Evaluation

- **BIRD.** Dev release `bird_sql_dev_20251106`, 1,534 questions, identified in the published rows as `dev-0` to `dev-1533` in file order.
- **Spider.** The 1,034 Spider 1.0 dev questions from the `xlangai/spider` release, identified as `spider-dev-0000` to `spider-dev-1033`.
- **Draws.** Each question was sampled 4 or 16 times at temperature 1. Draw 0 is the first draw.
- **Grade.** Predicted and reference SQL run on the same read-only database. The answer is correct when the comparison function from the BIRD Mini-Dev scorer accepts the two result sets. A statement that fails or exceeds 30 seconds scores 0.
- **Spider grades.** The original Spider execution scorer and the result-set grade above were both applied. The original scorer parses each statement with its own SQL grammar before comparing results and marks a statement it cannot parse as wrong. It could not parse 484, 482 and 570 first-draw statements from the frozen model, seed 42 and seed 43; examples include a table alias written without `AS` and an `IN` list of numbers. Among statements it parsed, 287 of 332, 449 of 516 and 383 of 427 were correct. Counts under one scorer are never compared with counts under the other.
- **Intervals.** Question IDs are resampled with replacement 10,000 times with seed 20261120, and the 2.5th and 97.5th percentiles of the paired difference are reported.
- **Decision margin.** Before the comparisons were run, a lead was defined as established only when the interval's lower bound is above +3 points.

## Recovery rules

The rules apply to final replies the as-run parser rejected. An answer the parser accepted keeps its in-run grade.

| Rule | What it accepts |
| --- | --- |
| R1 | The reply after the model's reasoning is removed |
| R2 | R1, plus a reply whose SQL blocks are identical copies |
| R3 | R2, plus the last block when the blocks differ |

"With recovered answers" in the README means R2. Failed tool calls and turn-limit exits are never recovered. Every accepted first-draw answer was also graded again on an idle machine; an arm with more than three disagreements was to stop. Seed 43 had four and was rerun with the limit at four.

## Sampling and cost

- **Any of 16** counts a question correct when any of its 16 draws is correct. It is an upper bound that needs the reference answer.
- **Best of k** applies a fixed selector to the first k draws. The selector executes each candidate answer, groups candidates by the set of rows they return, and picks the earliest draw in the largest group; when groups tie in size it takes the group containing the earliest draw. It saw no grades and no reference SQL, and its picks were written before grades were joined. Its code is not included; its picks are in `evidence/bird_16draw/*.picks.jsonl.gz`. The selector's CPU time is not counted in any cost figure.
- **GPU timing.** One timing slot runs 512 questions with 4 draws each, 2,048 attempts, on one card with enough concurrency to keep the engine busy. Seconds per draw are the slot's evaluation seconds divided by 2,048, with model startup excluded. Each arm has three repeats. A slot counts only when a monitoring record covers its whole interval and shows nothing else running on the machine. A slot that failed that test was rerun once; if the rerun also failed, the slot is reported as unresolved.
- **Timing sessions.** Timing was done twice. The first session timed the frozen model, seed 44 and seed 42. The second, after seed 43 finished, timed the frozen model, seed 42 and seed 43 together.
- **Matched cost.** For each trained arm, take the largest frozen draw count k in {1, 2, 4, 16} whose GPU-seconds per question are at most 1.15 times the trained arm's, then compare the adapter's first draw with the selector's pick among those k frozen draws. `sampling_and_cost.py` prints the full rule and its outcome names.
- **Payback.** One-time training GPU-seconds divided by the GPU-seconds saved per query, where frozen best of 16 costs 16 draws and the adapter one. For seed 43, with second-session timings and its whole run: 16 x 0.4266 - 0.6755 = 6.15 seconds saved per query, and 23.16 GPU-hours is 83,376 seconds, giving about 13,600 queries. For seed 42, with first-session timings and its GPU-hours through update 50: 16 x 0.4398 - 0.8011 = 6.24 seconds, and 19.95 GPU-hours is 71,820 seconds, giving about 11,500. With second-session timings seed 42 comes to about 11,900.

## Early short runs

Before the runs reported in the README, three short training runs (seeds 42, 43 and 44, stopped after two, two and one optimizer updates) were scored on BIRD dev with the earlier setup that the code in `native/` implements: the 6-turn limits and no schema in the prompt. Averaged over the three, they answered 17.30% of questions correctly against 12.26% for the frozen model, a paired difference of +5.04 points [+3.41, +6.69]. Their grade rows are in `evidence/short_runs/` as the worked example for the analysis utility:

```sh
python3 scripts/analyze_bird.py primary evidence/short_runs/dev-derived-input.jsonl --seeds 42 43 44 --replicates 10000 --rng-seed 20261117
```

The output matches `reproduce/expected/short_runs_paired.json` byte for byte on Python 3.12 and newer. On Python 3.11 two interval bounds differ in their last digit.

## Figures that differ between sources

- Seed 42 at update 50 scores 921 in the single-draw checkpoint sweep and 926 as the first draw of the 16-draw evaluation. These are separate sampling runs.
- The frozen model at the 6-turn limits scores 601 in the schema comparison and 600 as the first draw of the 16-draw evaluation, for the same reason.

## Settings that were not recorded

LoRA dropout, top-p and top-k, the maximum context length, the NVIDIA driver and CUDA runtime versions, the host operating system, CPU and memory, the dataset revision and download source of the BIRD dev questions and databases, the rule that ordered draws at sampling time, and the installed versions of `func_timeout`, `transformers`, `safetensors`, Megatron Bridge, `flash-linear-attention` and Triton. Software versions that were recorded are in [`native/README.md`](../native/README.md).
