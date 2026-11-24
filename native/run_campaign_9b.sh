#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
parse_only=0
if [[ "${1:-}" == "--parse-only" ]]; then
  parse_only=1
  shift
fi
bird_root=${BIRD_ROOT:?Set BIRD_ROOT to the project workspace root}
miles_root=${MILES_ROOT:-/root/miles}
model_root="$bird_root/models/qwen35_9b"
run_root=${BIRD_RUN_ROOT:?Set BIRD_RUN_ROOT to a new run directory}
training_data=${BIRD_TRAINING_DATA:?Set BIRD_TRAINING_DATA to the caller-selected training JSONL}
validation_data=${BIRD_VALIDATION_DATA:?Set BIRD_VALIDATION_DATA to validation-rows.jsonl}
seed=${BIRD_SEED:?Set BIRD_SEED explicitly}
num_updates=${BIRD_NUM_UPDATES:-100}
enable_thinking=${BIRD_ENABLE_THINKING:?Set BIRD_ENABLE_THINKING to true or false}
max_response_len=${BIRD_ROLLOUT_MAX_RESPONSE_LEN:?Set the per-call response budget explicitly before recipe freeze}
[[ "$enable_thinking" == true || "$enable_thinking" == false ]] || { printf 'BIRD_ENABLE_THINKING must be true or false\n' >&2; exit 64; }
[[ "$num_updates" =~ ^[0-9]+$ && "$num_updates" -gt 0 ]] || { printf 'BIRD_NUM_UPDATES must be a positive integer\n' >&2; exit 64; }
[[ "$max_response_len" =~ ^[0-9]+$ && "$max_response_len" -gt 0 ]] || { printf 'BIRD_ROLLOUT_MAX_RESPONSE_LEN must be a positive integer\n' >&2; exit 64; }
mkdir "$run_root"
mkdir "$run_root/rollouts"

eval_config="$run_root/eval-config.json"
export BIRD_EVAL_CONFIG_DESTINATION="$eval_config"
export BIRD_PARSE_DESTINATION="$run_root/parsed-args.json"
python3 - "$validation_data" "$max_response_len" <<'PY'
import json
import os
import sys
from pathlib import Path

validation_data, max_response_len = sys.argv[1:]
config = {
    "eval": {
        "defaults": {
            "input_key": "prompt",
            "label_key": "label",
            "metadata_key": "metadata",
            "n_samples_per_eval_prompt": 1,
            "temperature": 1.0,
            "max_response_len": int(max_response_len),
        },
        "datasets": [{"name": "heldout-training-db-validation", "path": validation_data}],
    }
}
Path(os.environ["BIRD_EVAL_CONFIG_DESTINATION"]).write_text(json.dumps(config, indent=2) + "\n")
PY

export BIRD_MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export PYTHONPATH="$project_root:$bird_root/python-packages/python-packages:$miles_root:$miles_root/examples${PYTHONPATH:+:$PYTHONPATH}"
export RAY_DEDUP_LOGS=0
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1

read -r num_heads num_query_groups num_layers hidden_size ffn_hidden_size kv_channels < <(
  python3 - "$model_root" <<'PY'
import json
import sys
from pathlib import Path
config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
text = config.get("text_config", config)
assert config["model_type"] == "qwen3_5"
assert text["attn_output_gate"]
print(text["num_attention_heads"], text["num_key_value_heads"], text["num_hidden_layers"], text["hidden_size"], text["intermediate_size"], text["head_dim"])
PY
)

date -u +"native_trial_start_utc=%Y-%m-%dT%H:%M:%SZ"
printf 'native_trial_bash_pid=%s\n' "$$"
printf 'native_trial_run_root=%s\n' "$run_root"
printf 'native_trial_mode=fresh\n'

cli=(python3 "$miles_root/train.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --rollout-num-gpus 8 \
  --sglang-server-concurrency 32 \
  --disable-bias-linear \
  --qk-layernorm \
  --group-query-attention \
  --num-attention-heads "$num_heads" \
  --num-query-groups "$num_query_groups" \
  --kv-channels "$kv_channels" \
  --num-layers "$num_layers" \
  --hidden-size "$hidden_size" \
  --ffn-hidden-size "$ffn_hidden_size" \
  --normalization RMSNorm \
  --norm-epsilon 1e-6 \
  --rotary-percent 0.25 \
  --swiglu \
  --vocab-size 248320 \
  --rotary-base 10000000 \
  --attention-output-gate \
  --untie-embeddings-and-output-weights \
  --bf16 \
  --hf-checkpoint "$model_root" \
  --ref-load "$model_root" \
  --load "$model_root" \
  --save "$run_root" \
  --save-interval 1 \
  --megatron-to-hf-mode bridge \
  --colocate \
  --lora-base-cpu-backup \
  --lora-rank 32 \
  --lora-alpha 64 \
  --target-modules language_model.decoder.layers.*.self_attention.linear_qkv,language_model.decoder.layers.*.self_attention.linear_proj,language_model.decoder.layers.*.mlp.linear_fc1,language_model.decoder.layers.*.mlp.linear_fc2 \
  --no-gradient-accumulation-fusion \
  --prompt-data "$training_data" \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --apply-chat-template-kwargs "{\"enable_thinking\":$enable_thinking}" \
  --rollout-shuffle \
  --num-rollout "$num_updates" \
  --eval-interval 20 \
  --eval-config "$eval_config" \
  --rollout-batch-size 16 \
  --n-samples-per-prompt 8 \
  --global-batch-size 128 \
  --rollout-max-response-len "$max_response_len" \
  --rollout-temperature 1.0 \
  --balance-data \
  --optimizer adam \
  --lr 1e-5 \
  --lr-decay-style constant \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --kl-coef 0 \
  --kl-loss-coef 0 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --qkv-format bshd \
  --micro-batch-size 1 \
  --attention-backend flash \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.8 \
  --seed "$seed" \
  --rollout-seed "$seed" \
  --use-session-server \
  --tito-model qwen35 \
  --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate \
  --custom-agent-function-path bird_evaluator.bird_agent.run \
  --custom-rm-path bird_evaluator.rewards.reward_func \
  --custom-megatron-before-train-step-hook-path bird_evaluator.trainer_checks.before_train_step \
  --save-debug-rollout-data "$run_root/rollouts/{rollout_id}.pt" \
  "$@")

if ((parse_only)); then
  python3 - "${cli[@]:1}" <<'PY'
import dataclasses
import json
import os
import sys
from pathlib import Path
import torch
from miles.utils.arguments import parse_args
sys.argv = sys.argv[1:]
args = parse_args()
rendered = json.dumps({
    "custom_generate_function_path": args.custom_generate_function_path,
    "custom_agent_function_path": args.custom_agent_function_path,
    "generate_multi_samples": args.generate_multi_samples,
    "custom_megatron_before_train_step_hook_path": args.custom_megatron_before_train_step_hook_path,
    "num_rollout": args.num_rollout,
    "ci_test": args.ci_test,
    "qkv_format": args.qkv_format,
    "use_dynamic_batch_size": args.use_dynamic_batch_size,
    "micro_batch_size": args.micro_batch_size,
    "apply_chat_template": args.apply_chat_template,
    "apply_chat_template_kwargs": args.apply_chat_template_kwargs,
    "save_debug_rollout_data": args.save_debug_rollout_data,
    "prompt_data": args.prompt_data,
    "seed": args.seed,
    "rollout_seed": args.rollout_seed,
    "start_rollout_id": args.start_rollout_id,
    "eval_interval": args.eval_interval,
    "eval_datasets": [dataclasses.asdict(item) for item in args.eval_datasets],
    "skip_eval_before_train": args.skip_eval_before_train,
    "rollout_max_response_len": args.rollout_max_response_len,
    "rollout_max_prompt_len": args.rollout_max_prompt_len,
    "rollout_max_context_len": args.rollout_max_context_len,
    "eval_max_prompt_len": args.eval_max_prompt_len,
    "eval_max_context_len": args.eval_max_context_len,
    "load": args.load,
    "ref_load": args.ref_load,
    "save": args.save,
    "lora_adapter_path": args.lora_adapter_path,
    "tensor_model_parallel_size": args.tensor_model_parallel_size,
    "seq_length": args.seq_length,
    "max_position_embeddings": args.max_position_embeddings,
    "parse_observer_pid": os.getpid(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_cuda_available": torch.cuda.is_available(),
}, indent=2)
Path(os.environ["BIRD_PARSE_DESTINATION"]).write_text(rendered + "\n")
print(rendered)
PY
else
  "${cli[@]}"
fi

date -u +"native_trial_end_utc=%Y-%m-%dT%H:%M:%SZ"
