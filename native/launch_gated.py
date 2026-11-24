"""Validate a pinned gated-training launch and start its native wrapper once."""

import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")

import argparse
import hashlib
import json
import os
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def pinned_file(pin, label):
    path = Path(pin["path"])
    require(path.is_absolute() and path.is_file(), f"{label} file is missing or relative")
    require(digest(path) == pin["sha256"], f"{label} hash differs from manifest")
    return path


def prove_workers(manifest, proof):
    expected_devices = manifest["devices"]
    workers = proof["workers"]
    ranks = [worker["rank"] for worker in workers]
    require(
        expected_devices and len(set(expected_devices)) == len(expected_devices)
        and all(isinstance(device, str) and device for device in expected_devices)
        and all(type(rank) is int for rank in ranks)
        and sorted(ranks) == list(range(len(expected_devices))),
        "worker proof must cover every rank on a unique configured device",
    )
    for worker in workers:
        rank = worker["rank"]
        require(worker["device_uuid"] == expected_devices[rank], f"rank {rank} device differs")
        require(
            worker["actor_sha256"] == manifest["actor"]["sha256"]
            and worker["kernel_sha256"] == manifest["kernel"]["sha256"],
            f"rank {rank} actor or kernel hash differs",
        )
        require(
            worker["seal_installed_by_production_path"] is True
            and worker["supported_case"]["q_positive_finite"] is True
            and worker["supported_case"]["k_positive_finite"] is True
            and type(worker["supported_case"]["fixed_delta"]) is int
            and worker["supported_case"]["fixed_delta"] == 2
            and type(worker["negative_case"]["fixed_delta"]) is int
            and worker["negative_case"]["fixed_delta"] == 0,
            f"rank {rank} did not prove the fixed backward branch and unsupported control",
        )


def prove_validation(manifest, proof):
    endpoints = manifest["evaluation_endpoints"]
    require(proof["evaluation_endpoints"] == endpoints, "rehearsal used different endpoints")
    require(endpoints["validation"], "validation needs an explicit endpoint allocation")
    occupied = set()
    for endpoint in endpoints["validation"] + endpoints["milestones"]:
        domain, port = endpoint["bind_domain"], endpoint["port"]
        require(isinstance(domain, str) and domain, "endpoint needs a host/network domain")
        require(type(port) is int and 0 < port < 65536, "invalid endpoint port")
        require((domain, port) not in occupied, f"evaluation port collision on {domain}:{port}")
        occupied.add((domain, port))

    require(
        proof["watcher_ready"] is True
        and proof["watcher_host"] == manifest["validation_watcher_host"],
        "validation watcher is absent or on the wrong host",
    )
    require(
        proof["published_via_watcher"] is True
        and proof["fresh_gate_decision"] == "CONTINUE"
        and proof["late_gate_decision"] == "STOP",
        "validation rehearsal must publish through the watcher and exercise both gate decisions",
    )
    available = manifest["gate_window_seconds"]
    elapsed = proof["checkpoint_to_receipt_seconds"]
    require(
        type(available) in (int, float) and type(elapsed) in (int, float)
        and 0 < elapsed < available < float("inf"),
        "validation receipt did not fit the configured gate window",
    )


def validate(manifest_path, worker_path, validation_path):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    job_root = Path(manifest["job_root"])
    require(job_root.is_absolute() and job_root == job_root.resolve() and job_root.is_dir(),
            "job root must be an existing, resolved absolute directory")
    run_root = job_root / "native"
    claim = job_root / "gated-launch.claim"
    require(not os.path.lexists(run_root) and not os.path.lexists(claim),
            "native run root or launch claim already exists")

    for label in ("actor", "kernel", "launcher"):
        pinned_file(manifest[label], label)
    require(os.access(manifest["launcher"]["path"], os.X_OK), "launcher is not executable")
    argv = manifest["launcher"]["args"]
    require(isinstance(argv, list) and all(isinstance(arg, str) and "\0" not in arg for arg in argv),
            "launcher args must be a list of strings")

    workers = json.loads(worker_path.read_bytes())
    validation = json.loads(validation_path.read_bytes())
    for proof in (workers, validation):
        require(proof["manifest_sha256"] == manifest_sha, "proof belongs to a different launch manifest")
    prove_workers(manifest, workers)
    prove_validation(manifest, validation)
    return manifest, manifest_sha, job_root, run_root, claim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--worker-proof", type=Path, required=True)
    parser.add_argument("--validation-proof", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="validate without claiming or launching")
    args = parser.parse_args()
    try:
        manifest, manifest_sha, job_root, run_root, claim = validate(
            args.manifest, args.worker_proof, args.validation_proof
        )
        if args.check:
            print(f"Launch checks passed for {job_root}")
            return 0
        with claim.open("x") as receipt:
            json.dump({"manifest_sha256": manifest_sha, "pid": os.getpid()}, receipt)
            receipt.write("\n")
        launcher = manifest["launcher"]
        env = dict(os.environ, BIRD_JOB_ROOT=str(job_root), BIRD_RUN_ROOT=str(run_root))
        os.execve(launcher["path"], [launcher["path"], *launcher["args"]], env)
    except (KeyError, TypeError, AttributeError, ValueError, OSError) as exc:
        print(f"Training launch blocked: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
