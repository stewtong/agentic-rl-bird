"""Guard one isolated timing slot: claim the node before launch, judge isolation after.

preflight: checks the slot manifest, a live isolation sampler and the attempt registry,
           writes a per-slot guard marker, and claims the node lock.
close:     reads the sampler's attributed samples and the native interval, writes the
           slot disposition, and releases the node lock.

A slot is accepted only when sampling covered the whole native interval without a
gap longer than the configured maximum and no non-exempt work ran on the node during that
interval. A failed original needs one isolated rerun; a failed rerun leaves the slot
unresolved. The registry refuses a repeated original and any third attempt.
"""

import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

EXEMPT = {"native_job", "sampler", "sampler_transport", "baseline_daemon"}
MAX_ATTEMPTS = 2
MARKER = "timing-slot-guard.json"
DISPOSITION = "timing-slot-disposition.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc(text):
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    require(value.tzinfo is not None, f"timestamp without timezone: {text}")
    return value.astimezone(timezone.utc)


def load(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def absolute(path, label):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve(), f"{label} must be a resolved absolute path")
    return path


def write_new(path, data):
    """Write JSON to a fresh file atomically; fail if the file already exists."""
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x") as out:
        json.dump(data, out, indent=1)
        out.write("\n")
    try:
        os.link(temp, path)
    finally:
        temp.unlink()


def check_attempt(slot, registry):
    slug, attempt = slot["slug"], slot["attempt"]
    require(type(attempt) is int and 1 <= attempt <= MAX_ATTEMPTS,
            "attempt must be 1 (original) or 2 (the one isolated rerun)")
    if attempt == 1:
        return
    original_root = absolute(slot["rerun_of_job_root"], "rerun_of_job_root")
    original, _ = load(original_root / DISPOSITION)
    require(original["slot"] == slug and original["attempt"] == 1,
            "rerun must point at this slot's original attempt")
    require(original["status"] == "needs_rerun", "the original attempt did not need a rerun")
    require((registry / f"{slug}.attempt1").exists(), "original attempt is not in the registry")


def preflight(slot_path, sampler_path, now=None):
    slot, slot_sha = load(slot_path)
    sampler, _ = load(sampler_path)
    now = now or datetime.now(timezone.utc)
    job_root = absolute(slot["job_root"], "job root")
    require(job_root.is_dir(), "job root must be an existing directory")
    require(not os.path.lexists(job_root / MARKER), "slot already has a guard marker; use a fresh root")
    registry = absolute(slot["attempt_registry"], "attempt registry")
    require(registry.is_dir(), "attempt registry directory is missing")
    check_attempt(slot, registry)
    for field in ("max_sample_age_seconds", "max_gap_seconds"):
        require(type(slot[field]) in (int, float) and slot[field] > 0, f"{field} must be positive")

    require(sampler["node"] == slot["node"], "isolation sampler runs on a different node")
    require(sampler["running"] is True, "isolation sampler is not running")
    age = (now - utc(sampler["last_sample_utc"])).total_seconds()
    require(0 <= age <= slot["max_sample_age_seconds"], f"last isolation sample is {age:.0f} s old")
    require(sampler["gpu_processes"] == [], "other GPU processes are present on the node")
    foreign = [row for row in sampler["latest_rows"] if row["disposition"] not in EXEMPT]
    require(not foreign, f"non-exempt work on the node: {foreign[:3]}")

    lock = absolute(slot["node_lock"], "node lock")
    record = {"slot": slot["slug"], "slot_sha256": slot_sha, "attempt": slot["attempt"],
              "job_root": str(job_root), "node": slot["node"], "node_lock": str(lock),
              "max_gap_seconds": slot["max_gap_seconds"],
              "sampler_first_sample_utc": sampler["first_sample_utc"],
              "sampler_last_sample_utc": sampler["last_sample_utc"], "preflight_utc": now.isoformat()}
    write_new(job_root / MARKER, record)
    claimed = []
    try:
        write_new(lock, {"slot": slot["slug"], "job_root": str(job_root), "pid": os.getpid(),
                         "claimed_utc": now.isoformat()})
        claimed.append(lock)
        write_new(registry / f"{slot['slug']}.attempt{slot['attempt']}",
                  {"job_root": str(job_root), "slot_sha256": slot_sha, "claimed_utc": now.isoformat()})
    except BaseException:
        for path in claimed:
            path.unlink()
        (job_root / MARKER).unlink()
        raise
    return record


def judge(marker, samples, native):
    exit_code = native["exit_code"]
    require(type(exit_code) is int, "native exit code must be an integer")
    first_bar = utc(native["first_bar_utc"])
    start, finish = utc(native["started_utc"]), utc(native["finished_utc"])
    require(utc(marker["preflight_utc"]) <= start <= first_bar <= finish,
            "native interval must start after preflight and contain the first generation bar")
    times = [utc(sample["utc"]) for sample in samples]
    require(times, "no isolation samples")
    inside = [t for t in times if start <= t <= finish]
    edges = sorted({start, finish, *inside})
    largest_gap = max((b - a).total_seconds() for a, b in zip(edges, edges[1:]))
    covers_first_bar = times[0] <= first_bar
    covers_finish = times[-1] >= finish
    gap_ok = largest_gap <= marker["max_gap_seconds"]
    work = [dict(row, utc=sample["utc"]) for sample in samples
            if start <= utc(sample["utc"]) <= finish
            for row in sample["rows"] + sample["gpu_processes"]
            if row["disposition"] not in EXEMPT]
    clean = covers_first_bar and covers_finish and gap_ok and not work and exit_code == 0
    if clean:
        status = "accepted"
    elif marker["attempt"] < MAX_ATTEMPTS:
        status = "needs_rerun"
    else:
        status = "unresolved"
    return {"slot": marker["slot"], "attempt": marker["attempt"], "status": status,
            "covers_first_bar": covers_first_bar, "covers_finish": covers_finish,
            "largest_gap_seconds": largest_gap, "first_sample_utc": samples[0]["utc"],
            "first_bar_utc": native["first_bar_utc"], "work_rows": work, "native_exit_code": exit_code}


def close(job_root, samples_path, native_path):
    job_root = absolute(job_root, "job root")
    marker, marker_sha = load(job_root / MARKER)
    require(marker["job_root"] == str(job_root), "guard marker belongs to another job root")
    lock = Path(marker["node_lock"])
    held = json.loads(lock.read_text())
    require(held["slot"] == marker["slot"] and held["job_root"] == marker["job_root"],
            "node lock is held by another slot")
    samples, samples_sha = load(samples_path)
    native, native_sha = load(native_path)
    result = judge(marker, sorted(samples, key=lambda s: utc(s["utc"])), native)
    result.update(marker_sha256=marker_sha, samples_sha256=samples_sha, native_sha256=native_sha)
    write_new(job_root / DISPOSITION, result)
    lock.unlink()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--slot", type=Path, required=True)
    pre.add_argument("--sampler-status", type=Path, required=True)
    end = sub.add_parser("close")
    end.add_argument("--job-root", type=Path, required=True)
    end.add_argument("--samples", type=Path, required=True)
    end.add_argument("--native", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            record = preflight(args.slot, args.sampler_status)
            print(f"Slot {record['slot']} attempt {record['attempt']} claimed the node")
            return 0
        result = close(args.job_root, args.samples, args.native)
        print(f"Slot {result['slot']} attempt {result['attempt']}: {result['status']}")
        return 0 if result["status"] == "accepted" else 3
    except (KeyError, TypeError, AttributeError, ValueError, OSError) as exc:
        print(f"Timing slot blocked: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
