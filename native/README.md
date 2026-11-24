<!-- register: public operator note | reader: an engineer with GPUs who wants to run or adapt the training path | consumed: read before deciding whether to attempt a run -->
# GPU path: agent, reward and launchers for Miles

This folder holds the code that plugs the text-to-SQL agent into [Miles](https://github.com/radixark/miles). It runs the earlier setup of the study: 6 turns, 768 generated tokens per model call, and no schema in the prompt. It does not reproduce the update-50 adapters reported in the top-level README. Three pieces behind those runs are not included: the 12-turn agent with the schema prompt, a modified trainer file, and a fixed configuration for the L2-normalization backward kernel of `flash-linear-attention`.

Use it to see how the agent, reward and data were wired, or as a starting point for your own run.

## Files

| File | Purpose |
| --- | --- |
| `bird_evaluator/bird_agent.py` | The agent loop: tool-call parsing, read-only SQL tools, final-answer parsing and grading |
| `bird_evaluator/sql_tools.py` | Read-only SQLite access, schema description, final-SQL extraction, execution scoring with a 30-second limit |
| `bird_evaluator/rewards.py` | Binary execution reward for Miles |
| `bird_evaluator/trainer_checks.py` | First-step check that the trainer loaded the pinned base weights |
| `prepare_bird_data.py` | Splits BIRD train databases into training and validation, samples 1,024 candidates, and writes `training-candidates.jsonl`, `training-rows.jsonl`, `trial-rows.jsonl`, `validation.json` and `manifest.json` under `$BIRD_ROOT/data/protocol/` |
| `run_native_9b.sh` | Three-update trial launch on the 16 trial rows, with seed 42, thinking off and 768 tokens per call fixed in the script |
| `run_campaign_9b.sh` | Training launch with validation, up to `BIRD_NUM_UPDATES` updates |
| `native_eval.py` | Zero-update evaluation of the frozen model and saved adapters from one request file |
| `export_attempts.py`, `update_metrics.py` | Export per-attempt grades and per-update metrics from a finished run. `update_metrics.py` carries a length threshold specific to one study run |
| `adapter_effect.py`, `validate_tito_multiturn.py` | Checks on adapter loading and multi-turn tokenization |
| `make_format_fixture.py`, `examples/cpu-format-fixture-request.json` | Builds a one-sample fixture in the evaluation file format, and an example request for `native_eval.py` |
| `launch_gated.py` | Verifies file hashes and a validation record before starting a training launch ([note](../docs/gated-training-launch.md)) |
| `timing_slot_guard.py` | Accepts a GPU timing slot only when nothing else ran on the node ([note](../docs/timing-slot-guard.md)) |

## Software the study ran on

| Component | Version |
| --- | --- |
| Miles | `github.com/radixark/miles` at commit `fcdfb0d10ea2dd259f7f58e979a47d3246d3a70d` |
| Megatron-LM | `github.com/radixark/Megatron-LM` at commit `79fc0894d0ba57acd10a9c0da507abd1dfef3bdf`, checked out by hash; the commit is not on the default branch |
| SGLang | `github.com/sgl-project/sglang` at commit `2fdb6557fcd38b7eb6986d2e33695e4fc1044864` |
| Container image | `radixark/miles@sha256:6c1a7d471b5410f1c6f1841d305f401716e3aba7f62bdcd2acad586cfbd05fed` |
| PyTorch | `2.11.0+cu130` |
| Ray | `2.56.0` |
| Python | 3.12 in the trainer image |
| Model | `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| BIRD train data | `train.zip` from the BIRD site, last modified July 11, 2023 |

The container image was used by digest. The versions of the components inside it were recorded from the running environment and were not checked against the image again for this publication. The NVIDIA driver, CUDA runtime and host operating system were not recorded.

## Layout and environment

`run_campaign_9b.sh` reads every variable below. `run_native_9b.sh` reads only the first three, and `native_eval.py` reads the first two and takes its other settings from a request file.

| Variable | Meaning |
| --- | --- |
| `BIRD_ROOT` | Workspace root. The model is expected at `$BIRD_ROOT/models/qwen35_9b` and BIRD train data under `$BIRD_ROOT/data/` |
| `MILES_ROOT` | Miles checkout, default `/root/miles` as in the Miles image |
| `BIRD_RUN_ROOT` | New directory for this run |
| `BIRD_TRAINING_DATA`, `BIRD_VALIDATION_DATA` | Training and validation prompt rows in the format of `training-rows.jsonl` |
| `BIRD_SEED` | Training and rollout seed |
| `BIRD_ENABLE_THINKING` | `true` or `false` |
| `BIRD_ROLLOUT_MAX_RESPONSE_LEN` | Generated tokens per model call |
| `BIRD_NUM_UPDATES` | Optimizer updates, default 100 |

Before a run:

1. Download the model revision to `$BIRD_ROOT/models/qwen35_9b`. Download BIRD `train.zip` to `$BIRD_ROOT/data/train.zip` and unpack it so that `$BIRD_ROOT/data/bird_train/train/train.json` and the `train_databases` folder beside it exist. Keep the zip; the script hashes it.
2. Run `python3 scripts/fetch_bird_scorer.py` from the repository root and install `func_timeout`.
3. Run `python3 native/prepare_bird_data.py` to write the candidate rows and the validation question list.
4. Screen the candidates with the frozen model through `native_eval.py`, export the attempts, and select the mixed-outcome pool with `scripts/select_mixed_pool.py` ([note](../docs/pool-selection.md)).
5. Launch with `--parse-only` first. Both launchers print the resolved Miles arguments without starting a run. The dry run creates `$BIRD_RUN_ROOT`, and `run_campaign_9b.sh` refuses an existing directory, so use a new `BIRD_RUN_ROOT` for the real launch.

Known gaps:

- Both launchers and `native_eval.py` request eight GPUs (`--actor-num-gpus-per-node 8 --rollout-num-gpus 8`). The update-50 runs used four cards, so those flags need editing for a four-card node.
- No included script converts `validation.json` into the prompt rows that `BIRD_VALIDATION_DATA` expects.
- The launchers add `$BIRD_ROOT/python-packages/python-packages` to `PYTHONPATH`. The study kept extra Python packages there and their list was not recorded. `func_timeout` is the one this code is known to need.
- `bird_evaluator/sql_tools.py` was restructured for publication so that it loads the scorer's comparison function from the downloaded file. The timeout mechanism and return values are unchanged, and `tests/test_sql_tools.py` covers them. The restructured file was not run on GPUs.

## Tests

Two test files run without a GPU or Miles:

```sh
python3 -m unittest discover -s native -p test_launch_gated.py
python3 -m unittest discover -s native -p test_timing_slot_guard.py
```

The other test files in this folder import Miles, PyTorch or Megatron and run only inside the trainer environment.
