#!/usr/bin/env python3
"""Launch matched Miles zero-update evaluation arms from one explicit request."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess


SOURCE_ROOT = Path(__file__).resolve().parent
DEFAULT_MILES_ROOT = Path("/root/miles")
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TARGET_MODULES = (
    "language_model.decoder.layers.*.self_attention.linear_qkv,"
    "language_model.decoder.layers.*.self_attention.linear_proj,"
    "language_model.decoder.layers.*.mlp.linear_fc1,"
    "language_model.decoder.layers.*.mlp.linear_fc2"
)
REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "condition",
    "model_root",
    "model_revision",
    "model_name",
    "dataset_name",
    "dataset_path",
    "mode",
    "enable_thinking",
    "temperature",
    "top_p",
    "top_k",
    "min_new_tokens",
    "max_response_len",
    "max_prompt_len",
    "max_context_len",
    "attempt_count",
    "rollout_seed",
    "evidence_class",
    "arms",
}


def require_bird_root() -> Path:
    raw = os.environ.get("BIRD_ROOT", "").strip()
    if not raw:
        raise RuntimeError("BIRD_ROOT is not set; set BIRD_ROOT to the project workspace root")
    return Path(raw)


def configured_model_root(bird_root: Path | None = None) -> Path:
    root = bird_root if bird_root is not None else require_bird_root()
    return root / "models" / "qwen35_9b"


def configured_miles_root() -> Path:
    raw = os.environ.get("MILES_ROOT", "").strip()
    return Path(raw) if raw else DEFAULT_MILES_ROOT


def expand_bird_root(value: str, bird_root: Path) -> str:
    root_str = str(bird_root)
    return value.replace("${BIRD_ROOT}", root_str).replace("$BIRD_ROOT", root_str)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_request(path: Path, arm: str) -> tuple[dict, dict]:
    bird_root = require_bird_root()
    model_root = configured_model_root(bird_root)
    request = copy.deepcopy(json.loads(path.read_text()))
    missing = REQUIRED_FIELDS - request.keys()
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    if request["schema_version"] != "bird-native-eval-request/v1":
        raise ValueError("unsupported manifest schema")
    if isinstance(request["model_root"], str):
        request["model_root"] = expand_bird_root(request["model_root"], bird_root)
    if isinstance(request["dataset_path"], str):
        request["dataset_path"] = expand_bird_root(request["dataset_path"], bird_root)
    if Path(request["model_root"]) != model_root or request["model_revision"] != MODEL_REVISION:
        raise ValueError("this launcher is pinned to the supplied Qwen3.5-9B model and revision")
    if arm not in {"frozen", "checkpoint"} or arm not in request["arms"]:
        raise ValueError("arm must select the manifest's frozen or checkpoint entry")
    if not isinstance(request["enable_thinking"], bool):
        raise ValueError("enable_thinking must be Boolean")
    if type(request["attempt_count"]) is not int or request["attempt_count"] <= 0:
        raise ValueError("attempt_count must be a positive integer")
    if type(request["rollout_seed"]) is not int:
        raise ValueError("rollout_seed must be an integer")
    if request["min_new_tokens"] is not None:
        raise ValueError("native eval_rollout_single_dataset does not consume min_new_tokens; leave it null")
    if type(request["max_response_len"]) is not int or request["max_response_len"] <= 0:
        raise ValueError("max_response_len must be a positive integer")
    if request["max_prompt_len"] is not None and (
        type(request["max_prompt_len"]) is not int or request["max_prompt_len"] <= 0
    ):
        raise ValueError("max_prompt_len must be null or a positive integer")
    selected = request["arms"][arm]
    if isinstance(selected.get("output_root"), str):
        selected["output_root"] = expand_bird_root(selected["output_root"], bird_root)
    if isinstance(selected.get("adapter_path"), str):
        selected["adapter_path"] = expand_bird_root(selected["adapter_path"], bird_root)
    if not selected.get("output_root"):
        raise ValueError("selected arm requires output_root")
    if arm == "checkpoint":
        adapter = Path(selected.get("adapter_path", ""))
        if not adapter.is_dir():
            raise ValueError("checkpoint arm requires an existing adapter_path directory")
        for name in ("adapter_config.json", "adapter_model.bin", "adapter_megatron_tp0_pp0.pt"):
            if not (adapter / name).is_file():
                raise ValueError(f"checkpoint adapter is missing {name}")
    elif selected.get("adapter_path") is not None:
        raise ValueError("frozen arm must not name an adapter")
    return request, selected


def model_shape(model_root: Path | None = None) -> dict[str, int]:
    root = model_root if model_root is not None else configured_model_root()
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", config)
    assert config["model_type"] == "qwen3_5"
    assert text["attn_output_gate"]
    return {
        "num_heads": text["num_attention_heads"],
        "num_query_groups": text["num_key_value_heads"],
        "num_layers": text["num_hidden_layers"],
        "hidden_size": text["hidden_size"],
        "ffn_hidden_size": text["intermediate_size"],
        "kv_channels": text["head_dim"],
    }


def optional_cli(cli: list[str], flag: str, value) -> None:
    if value is not None:
        cli.extend([flag, str(value)])


def build_cli(
    request: dict,
    arm: str,
    selected: dict,
    output_root: Path,
    eval_config: Path,
    model_root: Path | None = None,
    miles_root: Path | None = None,
) -> list[str]:
    resolved_model_root = model_root if model_root is not None else configured_model_root()
    resolved_miles_root = miles_root if miles_root is not None else configured_miles_root()
    shape = model_shape(resolved_model_root)
    cli = [
        sys.executable,
        str(resolved_miles_root / "train.py"),
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", "8",
        "--rollout-num-gpus", "8",
        "--sglang-server-concurrency", "32",
        "--disable-bias-linear",
        "--qk-layernorm",
        "--group-query-attention",
        "--num-attention-heads", str(shape["num_heads"]),
        "--num-query-groups", str(shape["num_query_groups"]),
        "--kv-channels", str(shape["kv_channels"]),
        "--num-layers", str(shape["num_layers"]),
        "--hidden-size", str(shape["hidden_size"]),
        "--ffn-hidden-size", str(shape["ffn_hidden_size"]),
        "--normalization", "RMSNorm",
        "--norm-epsilon", "1e-6",
        "--rotary-percent", "0.25",
        "--swiglu",
        "--vocab-size", "248320",
        "--rotary-base", "10000000",
        "--attention-output-gate",
        "--untie-embeddings-and-output-weights",
        "--bf16",
        "--hf-checkpoint", str(resolved_model_root),
        "--load", str(resolved_model_root),
        "--save", str(output_root / "checkpoints"),
        "--save-interval", "1",
        "--megatron-to-hf-mode", "bridge",
        "--colocate",
        "--no-gradient-accumulation-fusion",
        "--prompt-data", request["dataset_path"],
        "--input-key", "prompt",
        "--label-key", "label",
        "--metadata-key", "metadata",
        "--apply-chat-template-kwargs", json.dumps({"enable_thinking": request["enable_thinking"]}),
        "--rollout-shuffle",
        "--num-rollout", "0",
        "--eval-interval", "1",
        "--eval-config", str(eval_config),
        "--rollout-batch-size", "16",
        "--n-samples-per-prompt", "8",
        "--global-batch-size", "128",
        "--rollout-max-response-len", str(request["max_response_len"]),
        "--rollout-temperature", str(request["temperature"]),
        "--balance-data",
        "--optimizer", "adam",
        "--lr", "1e-5",
        "--lr-decay-style", "constant",
        "--lr-decay-iters", "1",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.98",
        "--kl-coef", "0",
        "--kl-loss-coef", "0",
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--recompute-granularity", "full",
        "--recompute-method", "uniform",
        "--recompute-num-layers", "1",
        "--qkv-format", "bshd",
        "--micro-batch-size", "1",
        "--attention-backend", "flash",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--rollout-num-gpus-per-engine", "1",
        "--sglang-mem-fraction-static", "0.8",
        "--seed", str(request["rollout_seed"]),
        "--rollout-seed", str(request["rollout_seed"]),
        "--use-session-server",
        "--tito-model", "qwen35",
        "--custom-generate-function-path", "miles.rollout.generate_hub.agentic_tool_call.generate",
        "--custom-agent-function-path", "bird_evaluator.bird_agent.run",
        "--custom-rm-path", "bird_evaluator.rewards.reward_func",
        "--save-debug-rollout-data", str(output_root / "retained" / "{rollout_id}.pt"),
    ]
    optional_cli(cli, "--eval-max-prompt-len", request["max_prompt_len"])
    optional_cli(cli, "--eval-max-context-len", request["max_context_len"])
    cli.extend([
        "--lora-base-cpu-backup",
        "--lora-rank", "32",
        "--lora-alpha", "64",
        "--target-modules", TARGET_MODULES,
    ])
    if arm == "checkpoint":
        cli.extend(["--lora-adapter-path", selected["adapter_path"], "--debug-skip-weight-update"])
    return cli


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=("frozen", "checkpoint"), required=True)
    parser.add_argument("--parse-only", action="store_true")
    args = parser.parse_args()
    try:
        bird_root = require_bird_root()
        model_root = configured_model_root(bird_root)
        miles_root = configured_miles_root()
        request, selected = load_request(args.manifest, args.arm)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"native_eval: error: {exc}", file=sys.stderr)
        return 2
    output_root = Path(selected["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    eval_config = output_root / "eval-config.json"
    defaults = {
        "input_key": "prompt",
        "label_key": "label",
        "metadata_key": "metadata",
        "n_samples_per_eval_prompt": request["attempt_count"],
        "temperature": request["temperature"],
        "top_p": request["top_p"],
        "top_k": request["top_k"],
        "max_response_len": request["max_response_len"],
    }
    defaults = {key: value for key, value in defaults.items() if value is not None}
    metadata_overrides = {
        "eval_run_id": request["run_id"],
        "eval_arm": args.arm,
        "eval_condition": request["condition"],
        "eval_training_seed": selected.get("training_seed"),
        "eval_mode": request["mode"],
        "eval_model_revision": request["model_revision"],
        "eval_adapter_path": selected.get("adapter_path"),
        "eval_attempt_count": request["attempt_count"],
    }
    config = {
        "eval": {
            "defaults": defaults,
            "datasets": [{
                "name": request["dataset_name"],
                "path": request["dataset_path"],
                "metadata_overrides": metadata_overrides,
            }],
        }
    }
    eval_config.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    cli = build_cli(
        request,
        args.arm,
        selected,
        output_root,
        eval_config,
        model_root=model_root,
        miles_root=miles_root,
    )
    receipt = {
        "schema_version": "bird-native-eval-launch/v1",
        "created_utc": utc_now(),
        "launcher_pid": os.getpid(),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "arm": args.arm,
        "mode": "parse-only" if args.parse_only else "native-zero-update-evaluation",
        "command": cli,
        "load_path": str(model_root),
        "adapter_path": selected.get("adapter_path"),
        "save_path": str(output_root / "checkpoints"),
        "retained_eval_path": str(output_root / "retained" / "eval_0.pt"),
        "note": "num_rollout=0 initializes models/optimizer and publishes weights, but executes no actor/critic train calls.",
    }
    (output_root / "launch-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    env = os.environ.copy()
    env["BIRD_MODEL_REVISION"] = MODEL_REVISION
    env["PYTHONPATH"] = os.pathsep.join([
        str(SOURCE_ROOT),
        str(bird_root / "python-packages" / "python-packages"),
        str(miles_root),
        str(miles_root / "examples"),
    ])
    env["RAY_DEDUP_LOGS"] = "0"
    env["MILES_EXPERIMENTAL_ROLLOUT_REFACTOR"] = "1"
    env["BIRD_PARSE_DESTINATION"] = str(output_root / "parsed-args.json")
    if not args.parse_only:
        print(json.dumps(receipt, indent=2), flush=True)
        os.execve(cli[0], cli, env)
    env["CUDA_VISIBLE_DEVICES"] = ""
    parse_code = r'''
import dataclasses, json, os, sys, torch
from pathlib import Path
from miles.utils.arguments import parse_args
args = parse_args()
result = {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_cuda_available": torch.cuda.is_available(),
    "num_rollout": args.num_rollout,
    "start_rollout_id": args.start_rollout_id,
    "eval_interval": args.eval_interval,
    "hf_checkpoint": args.hf_checkpoint,
    "load": args.load,
    "save": args.save,
    "ref_load": args.ref_load,
    "lora_rank": args.lora_rank,
    "lora_alpha": args.lora_alpha,
    "lora_base_cpu_backup": args.lora_base_cpu_backup,
    "target_modules": args.target_modules,
    "lora_adapter_path": args.lora_adapter_path,
    "tensor_model_parallel_size": args.tensor_model_parallel_size,
    "rollout_seed": args.rollout_seed,
    "lr_decay_iters": args.lr_decay_iters,
    "debug_skip_weight_update": args.debug_skip_weight_update,
    "use_distributed_post": args.use_distributed_post,
    "apply_chat_template": args.apply_chat_template,
    "apply_chat_template_kwargs": args.apply_chat_template_kwargs,
    "custom_agent_function_path": args.custom_agent_function_path,
    "custom_rm_path": args.custom_rm_path,
    "save_debug_rollout_data": args.save_debug_rollout_data,
    "eval_max_prompt_len": args.eval_max_prompt_len,
    "eval_max_context_len": args.eval_max_context_len,
    "eval_datasets": [dataclasses.asdict(item) for item in args.eval_datasets],
}
rendered = json.dumps(result, indent=2, default=str)
Path(os.environ["BIRD_PARSE_DESTINATION"]).write_text(rendered + "\n")
print(rendered)
'''
    result = subprocess.run(
        [sys.executable, "-c", parse_code, *cli[2:]],
        env=env,
        text=True,
        capture_output=True,
    )
    (output_root / "parse-stdout.txt").write_text(result.stdout)
    (output_root / "parse-stderr.txt").write_text(result.stderr)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
