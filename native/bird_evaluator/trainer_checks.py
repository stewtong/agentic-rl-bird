"""First-step proof that native Megatron loaded the pinned HF base tensors."""

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open


CHECKS = (
    ("language_model.embedding.word_embeddings.weight", "embed_tokens.weight", (slice(0, 4), slice(0, 8))),
    ("language_model.decoder.layers.3.self_attention.linear_proj.to_wrap.weight", "model.language_model.layers.3.self_attn.o_proj.weight", (slice(0, 4), slice(0, 8))),
)


def tensor_fingerprint(tensor):
    packed = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(packed).hexdigest(),
    }


def unique_suffix(names, suffix):
    matches = [name for name in names if name == suffix or name.endswith("." + suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {suffix!r}, found {matches}")
    return matches[0]


def source_slice(model_root, source_suffix, selection):
    index = json.loads((model_root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    source_name = unique_suffix(list(weight_map), source_suffix)
    with safe_open(str(model_root / weight_map[source_name]), framework="pt", device="cpu") as handle:
        return source_name, handle.get_slice(source_name)[selection]


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    if rollout_id != 0 or step_id != 0:
        return
    model_root = Path(args.hf_checkpoint)
    parameters = [param for chunk in model for param in chunk.named_parameters()]
    names = [name for name, _ in parameters]
    rank = dist.get_rank() if dist.is_initialized() else 0
    checks = []
    for trainer_suffix, source_suffix, selection in CHECKS:
        trainer_name = unique_suffix(names, trainer_suffix)
        trainer_tensor = dict(parameters)[trainer_name].detach()[selection]
        source_name, source_tensor = source_slice(model_root, source_suffix, selection)
        trainer_fp = tensor_fingerprint(trainer_tensor)
        source_fp = tensor_fingerprint(source_tensor)
        checks.append({
            "trainer_parameter": trainer_name,
            "hf_tensor": source_name,
            "slice": "rows 0:4, columns 0:8",
            "layout": "direct row-major; TP1/PP1; no transpose or shard offset",
            "trainer": trainer_fp,
            "hf": source_fp,
            "exact_match": trainer_fp == source_fp,
        })
    mismatch = torch.tensor(int(not all(item["exact_match"] for item in checks)), device="cuda")
    if dist.is_initialized():
        dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
    result = {
        "schema_version": "loaded-base-fidelity/v1",
        "status": "passed" if mismatch.item() == 0 else "failed",
        "model": "Qwen/Qwen3.5-9B",
        "revision": os.environ["BIRD_MODEL_REVISION"],
        "hf_checkpoint": str(model_root),
        "capture": "actual Megatron trainer parameters at first pre-optimizer hook",
        "trainer_rank_reporter": rank,
        "distributed_mismatch_present": bool(mismatch.item()),
        "checks": checks,
    }
    if rank == 0:
        destination = Path(args.save) / "base-weight-check.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n")
    if mismatch.item():
        raise RuntimeError("Loaded trainer base tensor does not match the pinned HF snapshot")
