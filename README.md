# Reinforcement learning for text-to-SQL on BIRD

We tested whether reinforcement learning (RL) could make [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) more likely to write correct SQL on its first attempt. On 1,534 [BIRD](https://bird-bench.github.io/) dev questions, **50 optimizer updates raised first-attempt execution accuracy from 39.3% to 60.4% and 61.3%** in two training runs. That is 323 and 338 more correct answers. A frozen model given 16 attempts per question reached 58.3% with result-set voting. [Accuracy by model](reproduce/expected/findings_tables.md) · [Sampling comparison](reproduce/expected/sampling_and_cost.md#frozen-best-of-16-and-adapter-first-draw-contrasts)

## How much did RL improve SQL accuracy?

The frozen model and both trained adapters used the same agent. It received the question, database schema and BIRD evidence text, and could inspect the schema and run read-only SQLite queries. One attempt allowed up to 12 turns and 1,536 generated tokens per model call. The grader counted an answer as correct when the final SQL returned the reference result. The first-attempt comparison used temperature 1 with thinking mode off. [Agent and grading method](docs/method.md)

| Model | Correct on first attempt | Accuracy | Gain over frozen, percentage points (paired 95% interval) |
| --- | ---: | ---: | ---: |
| Frozen Qwen3.5-9B | 603 / 1,534 | 39.3% | Baseline |
| RL, seed 42, update 50 | 926 / 1,534 | 60.4% | +21.06 (+18.64 to +23.53) |
| RL, seed 43, update 50 | 941 / 1,534 | 61.3% | +22.03 (+19.62 to +24.58) |

Training used GRPO to update a LoRA adapter in [Miles](https://github.com/radixark/miles). Each update sampled eight attempts on each of 16 questions and rewarded correct execution. The 340 training questions came from BIRD train and were selected because the frozen model sometimes succeeded and sometimes failed on them. Both seeds improved through the checkpoint sweep. The intervals above measure variation across questions. Only two training seeds reached update 50. [Training settings and pool selection](docs/method.md#training) · [Paired results](reproduce/expected/findings_tables.md)

## What improved: finishing the task or writing correct SQL?

**Both measures improved.** The frozen agent finished with a parseable answer on 1,088 questions, and 55.4% of those answers were correct. The trained agents finished with an answer on 1,395 and 1,417 questions, respectively, and both answered 66.4% of their submitted questions correctly. The sets of questions answered differed across models. [Per-question grades](evidence/bird_rescore/README.md) · [Completion and correctness counts](reproduce/expected/findings_tables.md#counts-behind-the-findings)

Some failures came from the parser. After inspecting the results, we re-scored replies whose SQL could be recovered by removing reasoning text or collapsing identical SQL blocks. With thinking off, this recovered 36 additional correct answers for the frozen agent and three for each adapter. About 90% of the first-attempt RL gain survived that change. [Recovery rules](docs/method.md#recovery-rules)

## When is training worth the compute?

Giving the frozen agent the database schema in its prompt raised its first-attempt score from 378 to 603 correct under the 12-turn limit. Thinking mode also helped once the parser accounted for SQL quoted inside reasoning: the frozen agent scored 743 to 777 correct, depending on whether answers extracted from reasoning counted. Thinking roughly doubled the frozen agent’s generated tokens. On the trained adapters it used about 40% more tokens with no measurable accuracy gain after parser recovery. Few-shot prompting with worked SQL examples remains an open comparison. [Schema comparison](evidence/aggregates/results-2026-11-20.json) · [Thinking and recovery results](reproduce/expected/findings_tables.md)

| Strategy, thinking off | Draws per question | BIRD accuracy | GPU-seconds per question¹ | Upfront training GPU-hours² |
| --- | ---: | ---: | ---: | ---: |
| Frozen, first attempt | 1 | 39.3% | 0.43 | 0 |
| Frozen, result-set voting | 16 | 58.3% | 6.83 | 0 |
| RL, seed 42 | 1 | 60.4% | 0.78 | 19.95 |
| RL, seed 43 | 1 | 61.3% | 0.68 | 23.16 |

¹ Batched throughput on one RTX PRO 6000 in the common timing session. The 16-draw figure is 16 × 0.4266 GPU-seconds. Voting executes the candidates and picks an answer from the largest group returning the same rows, without seeing reference SQL. The frozen model generated a correct answer somewhere among its 16 draws on 1,150 questions. The voter selected one on 894. The measurements exclude model startup and the selector's CPU time. ² Seed 42 through update 50 and seed 43 for its completed run. These totals exclude failed launches, setup and evaluation. [Timing records and selector picks](evidence/bird_16draw/README.md) · [Timing and payback formula](docs/method.md#sampling-and-cost)

**Against frozen 16-draw voting, one draw from the trained adapter saves enough GPU time per query to recover training costs after about 11,900 queries for seed 42 and 13,600 for seed 43.** These estimates use the common timing session above. The earlier seed-42 timings give 11,500 queries. A single frozen draw used less GPU time per query than either adapter. Dollar costs and single-query latency were unmeasured. [GPU-time calculation](docs/method.md#sampling-and-cost)

On accuracy, one adapter draw beat frozen 16-draw voting by 2.09 percentage points for seed 42 and 3.06 for seed 43. Neither met the study's predefined margin for an established lead. The lower bounds of the paired 95% intervals were +0.13 and +1.17 points, and the rule required a lower bound above +3. With a per-query GPU budget capped at 1.15 times one adapter attempt, only two frozen draws fit for seed 42 and one for seed 43. Against those alternatives, the adapters led by 13.69 and 22.03 points. [Paired comparisons and matched-cost results](reproduce/expected/sampling_and_cost.md) · [Accuracy decision rule](docs/method.md#evaluation)

## Does the training run on SM120, and what can I reproduce?

Yes. Both runs reached update 50 on **four PCIe RTX PRO 6000 GPUs**, with training and rollout generation sharing the cards. Seed 42 used 19.95 GPU-hours through update 50 and 21.20 before it stopped. Seed 43 used 23.16 GPU-hours for the completed run. [Hardware and training record](docs/method.md#training)

**Verify the scores on a laptop.** From the repository root, run these commands with Python 3.11 or newer. They use the standard library and the published grades, selector picks and timing records. Each prints nothing when its output matches the committed tables. The scripts recompute the analysis. Model inference, SQL regrading and training require additional inputs. [BIRD grade records](evidence/bird_rescore/README.md) · [Expected outputs](reproduce/expected/)

```sh
python3 reproduce/findings_tables.py | diff - reproduce/expected/findings_tables.md
python3 reproduce/sampling_and_cost.py | diff - reproduce/expected/sampling_and_cost.md
```

**Adapt the GPU code.** The included launcher requests eight GPUs and implements the earlier six-turn setup without a schema prompt. Both four-card runs used a later 12-turn agent with the schema in the prompt. Seed 42 reached update 50 before kernel autotuning ran out of memory at update 53. The successful seed-43 run used a fixed backward-kernel configuration and a modified trainer. This checkout omits the later agent, schema prompt builder, seed-43 kernel and trainer changes, and trained adapters. [GPU code and launch instructions](native/README.md) · [Per-seed training record](docs/method.md#training)

## What failed, and how can future runs avoid it?

| Observed failure | What to check on the next run |
| --- | --- |
| Seed 42 ran out of GPU memory at update 53 while Triton autotuned the L2-normalization backward kernel. Seed 43 reached update 50 with a fixed kernel configuration and a modified trainer. | Validate the fixed kernel on the training workers and verify the loaded trainer and kernel against the launch manifest. [Training record](docs/method.md#training) · [Launch guard](docs/gated-training-launch.md) |
| The parser rejected answers with one final SQL block when it also found SQL quoted inside reasoning. Frozen thinking mode scored 533 correct as run. | Test the parser on reasoning followed by final SQL before comparing thinking modes. Keep original and recovered grades. Nine idle-machine regrades disagreed with original grades across six questions for an unresolved reason. Seed 43 exceeded the preset disagreement limit by one, so its recovered figures use a rerun with the limit raised from three to four. [Grading and recovery](docs/method.md#recovery-rules) |
| One timing repeat lacked full isolation monitoring, and its rerun overlapped other CPU work. The matched-cost comparison against the six-turn frozen agent remained unresolved for seed 43. | Start monitoring before generation and verify isolation through the end of every timing slot. Four of five predefined replication checks passed, including the first-attempt accuracy gain. [Timing guard](docs/timing-slot-guard.md) · [Five checks and verdict](reproduce/expected/sampling_and_cost.md#two-seed-replication-summary) |

## Did the gain transfer beyond BIRD?

Both BIRD-trained adapters improved first-attempt accuracy on **Spider 1.0 dev**, a second text-to-SQL benchmark with 1,034 questions. The size of the gain depended on the scorer. The original Spider scorer parses SQL with its own grammar and can reject a statement that SQLite executes. Both rows use the same attempts with thinking off. [Spider grade records](evidence/spider/README.md) · [Scorer methods](docs/method.md#evaluation)

| Correct answers on Spider | Frozen | RL, seed 42 | RL, seed 43 | Gain over frozen, percentage points |
| --- | ---: | ---: | ---: | --- |
| Original Spider scorer | 287 | 449 | 383 | +15.67 / +9.28 |
| Graded by returned rows | 594 | 770 | 745 | +17.02 / +14.60 |

The returned-row comparison was added after the original results. These experiments cover one model and one task family. The schema prompt and turn limits were chosen with BIRD dev results in view. A duplicate-question audit between the 340 training questions and BIRD dev remains open. [Paired Spider results](reproduce/expected/findings_tables.md) · [BIRD overlap audit utility](docs/scoring-utility.md)

Code, documentation and study records use the [Apache License 2.0](LICENSE). BIRD and Spider retain their own terms. The published evidence omits question text, database files, reference SQL and model-generated SQL. [Third-party notices and citations](THIRD_PARTY_NOTICES.md)
