"""Load a saved LoRA and measure its fixed-input effect without PEFT."""

import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer


def tensor_sha256(tensor):
    packed = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(packed).hexdigest()


def adapter_pairs(state):
    pairs = {}
    for key, tensor in state.items():
        match = re.fullmatch(r"(?:base_model\.model\.)?(.+)\.lora_([AB])\.weight", key)
        if not match:
            raise RuntimeError(f"Unexpected adapter key: {key}")
        pairs.setdefault(match.group(1), {})[match.group(2)] = tensor
    if not pairs or any(set(weights) != {"A", "B"} for weights in pairs.values()):
        raise RuntimeError("Adapter does not contain complete nonzero-key LoRA A/B pairs")
    return pairs


def resolve_modules(model, pairs):
    modules = dict(model.named_modules())
    resolved = {}
    for exported_name in pairs:
        candidates = [exported_name]
        if exported_name.startswith("model."):
            candidates.append(exported_name.removeprefix("model."))
        matches = [name for candidate in candidates for name in modules if name == candidate or name.endswith("." + candidate)]
        matches = list(dict.fromkeys(matches))
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely bind {exported_name!r}; matches={matches}")
        resolved[matches[0]] = pairs[exported_name]
    return resolved


def attach_adapters(model, resolved, config):
    scale = config["lora_alpha"] / config["r"]
    records = []
    handles = []
    for module_name, weights in sorted(resolved.items()):
        module = dict(model.named_modules())[module_name]
        a = weights["A"].to(device=model.device, dtype=model.dtype)
        b = weights["B"].to(device=model.device, dtype=model.dtype)

        def hook(module, args, output, a=a, b=b):
            update = F.linear(F.linear(args[0], a), b) * scale
            return output + update.to(dtype=output.dtype)

        handles.append(module.register_forward_hook(hook))
        records.append({
            "module": module_name,
            "a_sha256": tensor_sha256(a),
            "b_sha256": tensor_sha256(b),
            "b_nonzero": bool(torch.count_nonzero(b).item()),
        })
    return handles, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=Path)
    parser.add_argument("adapter_root", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    fixed_text = "Read-only SQLite task. Return exactly one SQL query that counts every row in a table named items."
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    input_ids = tokenizer(fixed_text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_root,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda().eval()
    with torch.inference_mode():
        base_logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits.detach().clone()
        state = torch.load(args.adapter_root / "adapter_model.bin", map_location="cpu", weights_only=True)
        config = json.loads((args.adapter_root / "adapter_config.json").read_text())
        pairs = adapter_pairs(state)
        resolved = resolve_modules(model, pairs)
        handles, loaded = attach_adapters(model, resolved, config)
        trained_logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits.detach().clone()
        for handle in handles:
            handle.remove()
        restored_base_logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits.detach().clone()
        delta = (trained_logits.float() - base_logits.float()).abs()
        restored_delta = (restored_base_logits.float() - base_logits.float()).abs()
        base_top = base_logits[0, -1].float().topk(8).indices.cpu().tolist()
        trained_top = trained_logits[0, -1].float().topk(8).indices.cpu().tolist()
        result = {
            "schema_version": "loaded-adapter-effect/v1",
            "status": "passed" if any(item["b_nonzero"] for item in loaded) and delta.max().item() > restored_delta.max().item() else "failed",
            "model": "Qwen/Qwen3.5-9B",
            "loader": "same AutoModelForImageTextToText instance, BF16, text-only input, adapters as Linear forward deltas",
            "adapter_file": str(args.adapter_root / "adapter_model.bin"),
            "adapter_config": config,
            "saved_adapter_tensor_count": len(state),
            "loaded_adapter_pair_count": len(loaded),
            "all_saved_pairs_bound_and_loaded": len(state) == 2 * len(loaded),
            "loaded_nonzero_b_tensor_count": sum(item["b_nonzero"] for item in loaded),
            "loaded_tensors": loaded,
            "fixed_input": {
                "text": fixed_text,
                "token_ids_sha256": hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest(),
                "token_count": input_ids.shape[1],
            },
            "base_logits_sha256": tensor_sha256(base_logits),
            "trained_logits_sha256": tensor_sha256(trained_logits),
            "restored_base_logits_sha256": tensor_sha256(restored_base_logits),
            "max_absolute_logit_delta": delta.max().item(),
            "mean_absolute_logit_delta": delta.mean().item(),
            "l2_logit_delta": delta.square().sum().sqrt().item(),
            "base_top8_token_ids": base_top,
            "trained_top8_token_ids": trained_top,
            "top8_argmax_changed": base_top[0] != trained_top[0],
            "restored_base_max_absolute_jitter": restored_delta.max().item(),
            "restored_base_mean_absolute_jitter": restored_delta.mean().item(),
            "adapter_effect_exceeds_restored_base_jitter": delta.max().item() > restored_delta.max().item(),
        }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "passed" or not result["all_saved_pairs_bound_and_loaded"]:
        raise RuntimeError("Trained adapter did not load completely and change fixed-input logits")
    print(json.dumps({key: value for key, value in result.items() if key != "loaded_tensors"}, indent=2))


if __name__ == "__main__":
    main()
