<!-- register: public README | reader: applied AI engineers weighing RL fine-tuning for an agent task | consumed: findings skimmed first, then the two reproduction commands -->
# Reinforcement learning for a text-to-SQL agent on BIRD: Qwen3.5-9B with LoRA on four RTX PRO 6000 cards

This repository holds the results, per-question grades and analysis code from a study of reinforcement learning (RL) on a multi-turn text-to-SQL agent. The agent is [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) with a LoRA adapter. It inspects a SQLite database through two read-only tools and finishes with one SQL statement. Training used the open-source [Miles](https://github.com/radixark/miles) framework on four RTX PRO 6000 cards. Accuracy is execution accuracy on the 1,534 questions of the [BIRD](https://bird-bench.github.io/) dev set: an answer is correct when its SQL returns the same rows as the reference SQL.

Fifty optimizer updates raised first-draw accuracy from 603 correct (39.3%) to 926 (60.4%) for training seed 42 and 941 (61.3%) for seed 43. Each run used 20 to 23 GPU-hours, which is four cards for five to six hours. Two steps that need no training closed part of that gap, and sampling the untrained model 16 times closed most of it at 16 times the draws.

| Configuration (BIRD dev, 1,534 questions) | Correct as run | Correct with recovered answers | Generated tokens per question |
| --- | ---: | ---: | ---: |
| Frozen, thinking off | 603 (39.3%) | 639 (41.7%) | 443 |
| Frozen, thinking on | 533 (34.7%) | 777 (50.7%) | 881 [1] |
| Frozen, best of 16 draws | 894 (58.3%) | | 16 draws |
| Seed 42, update 50 | 926 (60.4%) | 929 (60.6%) | 744 |
| Seed 43, update 50 | 941 (61.3%) | 944 (61.5%) [2] | 725 |
| Seed 42, update 50, thinking on | 864 (56.3%) | 939 (61.2%) | 1,045 [1] |
| Seed 43, update 50, thinking on | 888 (57.9%) | 919 (59.9%) | 1,039 [1] |

Terms used in the table and below:

- **Frozen** is the untrained Qwen3.5-9B. **Seed 42** and **seed 43** are two training runs with the same training settings; **update 50** is the adapter after 50 optimizer steps.
- A **draw** is one sampled attempt at a question, at temperature 1. Every row except best of 16 scores the first draw.
- **As run** is the evaluation harness as it ran. Its parser accepted an answer only if the final message held exactly one SQL block.
- **With recovered answers** re-scores the same outputs after pulling SQL out of messages the parser rejected for format. The recovery rules were written after the as-run results were known, apply to every row, and send recovered SQL to the same grader.
- **Best of 16** samples the frozen model 16 times per question. A fixed selector runs the 16 answers, groups them by the rows they return and picks from the largest group. It never sees grades or reference SQL.
- **Thinking** is the model's reasoning mode. Both adapters were trained with thinking off.
- **12-turn limits** allow the agent 12 turns and 1,536 generated tokens per model call. Training and every row above used them. **6-turn limits** allow 6 turns and 768 tokens.
- An **arm** is one model configuration evaluated on the full question set.
- [1] Token counts reported from the original runs; they are not in the published files.
- [2] From a rerun of the re-scoring that allowed four grading disagreements where the first run allowed three. The Limits section describes the check.

## Findings

Differences are in percentage points with a paired 95% bootstrap interval over questions.

**1. Fifty updates added 323 correct answers for seed 42 and 338 for seed 43.** The adapters lead the frozen model by +21.06 points [+18.64, +23.53] and +22.03 [+19.62, +24.58]. The two seeds differ from each other by 0.98 points [-0.91, +2.93]. Accuracy rose through training. Seed 42 scored 670, 743, 754, 789, 848 and 921 at updates 2, 4, 8, 16, 32 and 50 in a separate single-draw evaluation, which is why its update-50 count there is 921 and not 926. Seed 43's checkpoints scored 650, 709, 732, 798, 897 and 941.

**2. A tolerant answer parser plus thinking mode took the frozen model from 603 to between 743 and 777 correct, with no training.** As run, thinking on scored 4.56 points below thinking off [-7.37, -1.69]. The parser counted SQL blocks quoted inside the model's reasoning, so an answer with one block after the reasoning was rejected for holding two. Recovery restores 401 first-draw answers, 244 of them correct, and thinking on then leads thinking off by +9.00 points [+6.52, +11.54]. The 777 includes 91 answers the as-run parser took from inside the reasoning (34 correct). Without them the score is 743 and the lead is +6.78 [+4.30, +9.26]. Thinking on generated about twice the tokens.

**3. About 90% of the training gain survives that correction.** With recovered answers the adapters lead frozen thinking off by +18.90 [+16.49, +21.25] and +19.88 [+17.41, +22.36], which is 89.8% and 90.2% of their as-run leads. They lead frozen thinking on by +9.91 [+7.69, +12.13] and +10.89 [+8.67, +13.10], or by +12.13 [+9.84, +14.41] and +13.10 [+10.82, +15.38] when answers taken from inside the reasoning are excluded. With thinking off, recovery added 36 correct answers to the frozen model and 3 to each adapter.

**4. Thinking mode added nothing measurable to the trained adapters and cost about 40% more tokens.** With recovered answers, seed 42 scored 939 with thinking on against 929 with it off, +0.65 [-1.11, +2.48]. Seed 43 scored 919 against 944, -1.63 [-3.65, +0.39]. As run, thinking on brought the parser rejections back and both adapters scored below their own thinking-off results. RL with thinking on was not run, so this measures adapters outside their training setting.

**5. Sixteen draws from the frozen model come within two to three points of one draw from an adapter.** Frozen best of 16 scores 894. The adapters' first draws lead it by +2.09 [+0.13, +3.98] and +3.06 [+1.17, +4.95]. At matched GPU cost the lead is wider. The matched-cost rule gives the frozen model as many draws as fit within 1.15 times the adapter's GPU-seconds per question and scores the selector's pick among them. That is two frozen draws against seed 42, which leads by +13.69 [+11.41, +15.97], and one frozen draw against seed 43, which leads by +22.03 [+19.62, +24.58]. In GPU time alone, training repays itself against best of 16 after about 11,500 queries (seed 42) to 13,600 (seed 43). Training moved the ceiling far less than the first draw: the count of questions answered correctly by any of 16 draws is 1,150 for the frozen model and 1,176 and 1,201 for the adapters, +1.69 [+0.13, +3.32] and +3.32 [+1.89, +4.76].

**6. Both adapters score higher on Spider, a benchmark they were not trained on.** The Spider figures are as run, with thinking off. Under the original Spider scorer the frozen model, seed 42 and seed 43 answered 287, 449 and 383 of 1,034 dev questions, gains of +15.67 [+12.96, +18.38] and +9.28 [+6.87, +11.61]. That scorer parses SQL with its own grammar and marks a statement it cannot parse as wrong. Grading by returned rows, as on BIRD, gives 594, 770 and 745, gains of +17.02 [+14.02, +20.02] and +14.60 [+11.70, +17.60]. Part of the gain is finishing with an answer: the frozen model ended without final SQL on 218 questions, the adapters on 36 and 37. The seeds differ by 6.4 points under the original scorer and 2.4 under result-set grading, against 1 point on BIRD.

**7. Three harness choices each moved measured accuracy by 15 to 30 points, comparable to the 21 to 22 points from training.** Two more had smaller effects.

| Harness choice | Arm | Correct answers | Points |
| --- | --- | --- | ---: |
| Schema placed in the prompt | Frozen | 191 to 601 of 1,534 at the 6-turn limits; 378 to 603 at the 12-turn limits | +26.7, +14.7 |
| Parser tolerant of SQL quoted in reasoning | Frozen, thinking on | 533 to 777 of 1,534, which reverses the thinking comparison | +15.9 |
| Result-set grade in place of the original scorer | Frozen, Spider | 287 to 594 of 1,034 on the same draws | +29.7 |
| 6-turn limits in place of 12-turn limits | Adapters, trained at the 12-turn limits | 926 to 829 and 941 to 866; the frozen model is unchanged at 603 and 600 | -6.3, -4.9 |
| Grading on an idle machine | Three BIRD arms, thinking off | 9 of the 3,900 first-draw answers the parser accepted changed from incorrect to correct, all on six questions in one database | +0.2 |

## What this suggests for a similar project

These are inferences from one model, one task and one harness.

- Audit the harness before spending GPU time on training. Sorting a sample of no-answer outputs by cause would have exposed the parser rule early. A second scorer and an idle-machine regrade catch scorer and timeout errors.
- Price training against the cheap alternatives. Here a tolerant parser and thinking mode reached 743 to 777 correct at about 881 tokens per question. Training reached 929 to 944 at 725 to 744.
- Compare against best of k as well as against a single draw. The seed-42 adapter leads by 21 points against one frozen draw, by 13.7 at matched GPU cost, and by 2 against 16 draws.
- Query volume decides between training and sampling. Below roughly 11,500 to 13,600 queries, frozen best of 16 used less total GPU time; above it the adapter did. Dollar cost and latency were not measured.
- Serve an adapter in the configuration it was trained in. Switching thinking on cost 40% more tokens for no measurable gain, and halving the turn and token limits cost 75 to 97 correct answers.
- Thinking and training did not stack. Thinking was worth 6.78 to 9.00 points to the frozen model and nothing measurable to the adapters. Training may have supplied what thinking gave the frozen model; that explanation was not tested.

## Recompute every table on a laptop

The published files hold one grade per question and draw. Two scripts rebuild the counts, paired differences and bootstrap intervals from them. They need Python 3.11 or newer and the standard library: no GPU, no dataset download, no packages. Each runs in about 15 seconds.

```sh
git clone https://github.com/stewtong/agentic-rl-bird.git
cd agentic-rl-bird
python3 reproduce/findings_tables.py
python3 reproduce/sampling_and_cost.py
```

The committed output of each script is in `reproduce/expected/`. These two commands print nothing when your numbers match:

```sh
python3 reproduce/findings_tables.py | diff - reproduce/expected/findings_tables.md
python3 reproduce/sampling_and_cost.py | diff - reproduce/expected/sampling_and_cost.md
```

| Script | Findings it recomputes | Input |
| --- | --- | --- |
| `reproduce/findings_tables.py` | 1 to 4, 6, and the parser, scorer and regrade rows of 7 | `evidence/bird_rescore/`, `evidence/spider/` |
| `reproduce/sampling_and_cost.py` | 5, the agent-limits row of 7, token counts for the thinking-off arms, GPU timing and the matched-cost comparisons | `evidence/bird_16draw/` |

Tokens per question in the first table are the "First-draw tokens" column of `sampling_and_cost.py` divided by 1,534.

The scripts recompute arithmetic from grades. They do not rerun inference or grading, and the published rows hold no question text, reference SQL or model output. `shasum -a 256 -c SHA256SUMS` inside `evidence/bird_rescore/`, `evidence/spider/` or `evidence/bird_16draw/` checks file integrity.

Some figures are reported and cannot be recomputed from the published rows: token counts for the thinking-on runs, training GPU-hours, the seed-43 checkpoint curve, the count of statements the original Spider scorer could not parse, the database the six regraded questions belong to, and the absence of exact question matches between Spider and BIRD. The schema comparison and the seed-42 checkpoint curve are published as aggregate counts in `evidence/aggregates/results-2026-11-20.json`.

Further checks, in increasing cost:

| Check | Command | Needs |
| --- | --- | --- |
| Analysis, pool selection and overlap utilities | `python3 -m unittest discover -s tests`, `python3 tests/check_analysis.py`, `python3 tests/check_pool_selector.py` | Python 3.11 or newer |
| Launch and timing guards | `python3 -m unittest discover -s native -p test_launch_gated.py`, `python3 -m unittest discover -s native -p test_timing_slot_guard.py` | Python 3.11 or newer |
| Execution scorer | `python3 scripts/fetch_bird_scorer.py`, `pip install func_timeout`, then `python3 -m unittest discover -s tests` | Network access |
| Training and evaluation on GPUs | [`native/README.md`](native/README.md) | Four RTX PRO 6000 cards, the Miles container, BIRD data |

The GPU path is partial. The repository includes the agent, reward, data preparation and launchers for an earlier setup with the 6-turn limits. The code changes behind the update-50 runs (the 12-turn agent with the schema in the prompt, a modified trainer and a fixed kernel configuration) and the trained adapters are not included. [`docs/method.md`](docs/method.md) lists every recorded setting.

## How the experiment was run

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, thinking off unless a row says otherwise |
| Training | GRPO with LoRA in Miles: token-level loss, no KL term, clip range 0.2, group standard-deviation normalization, learning rate 1e-5 |
| Reward | 1 when the final SQL returns the reference rows, otherwise 0 |
| Batch | 16 questions per update, 8 attempts per question, 128 episodes |
| Training questions | 340 questions from the BIRD train split on which the frozen model's eight attempts gave mixed outcomes |
| Agent | Tools `describe_schema` and `run_sql`, read-only; 12-turn limits; database schema in the prompt |
| Hardware | Four PCIe RTX PRO 6000 cards for training, with rollout engines on the same cards; one card for GPU timing |
| Training cost | GPU-hours are cards times wall-clock hours. Seed 42: 19.95 through update 50, 21.20 for the whole run. Seed 43: 23.16 for the whole run. Failed launches, evaluation and setup are excluded |
| Evaluation | BIRD dev release `bird_sql_dev_20251106`, 1,534 questions; Spider 1.0 dev, 1,034 questions; temperature 1 |
| Intervals | Paired bootstrap, 10,000 resamples of questions, seed 20261120 |

## Limits

- One model, one task family and one harness were tested. Two seeds were trained to update 50. The intervals resample questions and do not capture seed-to-seed variance. A third run, seed 44, has a scored adapter at update 8 and appears in `evidence/bird_16draw/` as an early-training reference.
- The recovery rules, the thinking-on runs of the adapters and the Spider result-set grades were defined after the first results were known.
- The 12-turn limits and the schema prompt were first used in evaluations on BIRD dev and then adopted for training, so the agent setup was chosen with dev results in view.
- The 340 training questions come from BIRD train databases and the evaluation uses BIRD dev. No duplicate-question audit of the 340 against the 1,534 dev questions was run. `scripts/audit_bird_overlap.py` performs that audit given both dataset files.
- The re-scoring graded every accepted first-draw answer again on an idle machine and was set to stop an arm with more than three disagreements with the in-run grade. Seed 43 had four, so its recovered scores come from a rerun that allowed four.
- Nine regrades disagreed, on six questions. The cause was not established. Recovered answers were graded on an idle machine and as-run answers keep their in-run grades.
- The replication rule fixed before seed 43 was trained required five checks to pass. Four passed: the first-draw lead, the matched-cost comparison against the frozen model at the 12-turn limits, and both any-of-16 comparisons. The fifth, a matched-cost comparison against the frozen model at the 6-turn limits, has one timing repeat that lacks a monitoring record showing nothing else ran on the machine. Under that rule `sampling_and_cost.py` prints the label "the update-50 result is seed-dependent".
- Cost is GPU-seconds on one card. Dollar cost, single-query latency, the selector's CPU time and a matched-cost comparison for thinking on were not measured.
- Thinking on was set by a server configuration flag that the saved request settings do not record. The outputs show it was active: across draws 0 to 3 of the frozen model, 5,837 of 6,001 stored final answers contain a reasoning close tag with thinking on, against 17 of 5,979 with thinking off.
- Spider questions have no exact match in BIRD. Paraphrase overlap was not tested, and the share of the Spider gain that comes from finishing with an answer was not measured.
- Update 50 is the last scored checkpoint. Seed 42 stopped with an out-of-memory error at update 53, and seed 43 was stopped at update 50 by plan.
- A larger frozen model, a supervised fine-tuning baseline and RL with thinking on were not run.

## Repository contents

| Path | Contents |
| --- | --- |
| `reproduce/` | The two table scripts and their expected output |
| `evidence/bird_rescore/` | Six BIRD arms, draws 0 to 3: as-run grades, recovery grades, parser and regrade fields |
| `evidence/spider/` | Three Spider arms, draws 0 to 3, under both scorers |
| `evidence/bird_16draw/` | Eight BIRD arms with 16 draws each, selector picks, and GPU timing |
| `evidence/aggregates/` | Aggregate counts for the schema comparison and the seed-42 checkpoint sweep |
| `evidence/short_runs/` | Grade rows for three early one- and two-update runs, used as the worked example for `scripts/analyze_bird.py` ([method note](docs/method.md#early-short-runs)) |
| `scripts/`, `tests/` | Analysis, pool selection, overlap audit and scorer download, with CPU tests |
| `native/` | Agent, reward, data preparation, launchers and guards for the GPU path |
| `docs/` | [Method and recorded settings](docs/method.md), plus notes on each utility |

## License and attribution

The code, documentation and study records in this repository are licensed under the [Apache License 2.0](LICENSE). Third-party terms stay separate. BIRD and Spider are distributed under CC BY-SA 4.0. This repository redistributes no questions, reference SQL or database files from either benchmark. Citations, upstream licenses and pinned versions are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The trained adapters are not part of this repository.
