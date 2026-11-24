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
mkdir -p "$run_root/rollouts"

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
  --load "$run_root" \
  --save "$run_root" \
  --save-interval 1 \
  --megatron-to-hf-mode bridge \
  --colocate \
  --lora-base-cpu-backup \
  --lora-rank 32 \
  --lora-alpha 64 \
  --target-modules language_model.decoder.layers.*.self_attention.linear_qkv,language_model.decoder.layers.*.self_attention.linear_proj,language_model.decoder.layers.*.mlp.linear_fc1,language_model.decoder.layers.*.mlp.linear_fc2 \
  --no-gradient-accumulation-fusion \
  --prompt-data "$bird_root/data/protocol/trial-rows.jsonl" \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --apply-chat-template-kwargs '{"enable_thinking":false}' \
  --rollout-shuffle \
  --num-rollout 3 \
  --rollout-batch-size 16 \
  --n-samples-per-prompt 8 \
  --global-batch-size 128 \
  --rollout-max-response-len 768 \
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
  --seed 42 \
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
import json
import sys
from miles.utils.arguments import parse_args
sys.argv = sys.argv[1:]
args = parse_args()
print(json.dumps({
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
}, indent=2))
PY
else
  "${cli[@]}"
fi

date -u +"native_trial_end_utc=%Y-%m-%dT%H:%M:%SZ"
