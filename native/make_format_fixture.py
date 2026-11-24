#!/usr/bin/env python3
"""Round-trip one real training Sample through the native eval debug writer."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from miles.ray.rollout.debug_data import save_debug_rollout_data
from miles.utils.types import Sample


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-rollout", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw = torch.load(args.training_rollout, map_location="cpu", weights_only=False)
    selected = next(row for row in raw["samples"] if row["index"] == args.sample_index)
    sample = Sample.from_dict(selected)
    metadata = sample.metadata
    if type(metadata.get("execution_correct")) is not bool:
        raise ValueError("fixture requires an explicit real Boolean domain EX grade")
    question_id = metadata["question_id"]
    source_rows = [json.loads(line) for line in args.training_dataset.read_text().splitlines() if line.strip()]
    source_row = next(row for row in source_rows if row["metadata"]["question_id"] == question_id)
    subset_path = args.output_dir / "training-format-row.jsonl"
    subset_path.write_text(json.dumps(source_row) + "\n")
    retained_template = args.output_dir / "retained" / "{rollout_id}.pt"
    save_debug_rollout_data(
        SimpleNamespace(save_debug_rollout_data=str(retained_template)),
        {"training-format-fixture": {"samples": [sample]}},
        rollout_id=0,
        evaluation=True,
    )
    result = {
        "schema_version": "bird-native-format-fixture/v1",
        "evidence_class": "training-format-fixture",
        "benchmark_evaluation": False,
        "model_validation": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": torch.cuda.is_available(),
        "source_training_rollout": str(args.training_rollout),
        "source_training_rollout_sha256": sha256(args.training_rollout),
        "source_sample_index": args.sample_index,
        "question_id": question_id,
        "database_id": metadata.get("db_id"),
        "split": metadata.get("split"),
        "execution_correct": metadata["execution_correct"],
        "writer": "miles.ray.rollout.debug_data.save_debug_rollout_data(evaluation=True)",
        "retained_path": str(args.output_dir / "retained" / "eval_0.pt"),
        "dataset_path": str(subset_path),
    }
    (args.output_dir / "fixture-provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
