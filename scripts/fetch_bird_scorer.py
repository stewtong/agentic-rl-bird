#!/usr/bin/env python3
"""Fetch and SHA-256-verify the pinned BIRD Mini-Dev evaluation scorer files."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")

import argparse
import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path


UPSTREAM_COMMIT = "f9d2750c9b53639820c9d47f6d7a5e5025c780ab"
BASE_URL = f"https://raw.githubusercontent.com/bird-bench/mini_dev/{UPSTREAM_COMMIT}/evaluation"
DEFAULT_DEST = (
    Path(__file__).resolve().parents[1] / "vendor" / "bird_mini_dev" / "evaluation"
)
PINNED_FILES = (
    (
        "evaluation_ex.py",
        "da1bbcd4530be83692d7c650c814ea9704bb710d0c953eb75d02ccb38233cf89",
    ),
    (
        "evaluation_utils.py",
        "f6943d249caac5aeaef9bce21d43dbf29dcef85a0c965a76df032a9542f308bf",
    ),
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch_verified(filename: str, expected_sha256: str, dest_dir: Path, force: bool) -> str:
    target = dest_dir / filename
    if not force and target.is_file():
        existing_digest = sha256_bytes(target.read_bytes())
        if existing_digest == expected_sha256:
            return "verified-existing"

    url = f"{BASE_URL}/{filename}"
    request = urllib.request.Request(url, headers={"User-Agent": "agentic-rl-bird-fetch/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"failed to download {filename} from {url}: {exc}") from exc

    actual_digest = sha256_bytes(payload)
    if actual_digest != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {filename}: expected {expected_sha256}, got {actual_digest}"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{filename}.{os.getpid()}.tmp")
    try:
        tmp_path.write_bytes(payload)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Destination directory (default: vendor/bird_mini_dev/evaluation)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already match the pinned SHA-256",
    )
    args = parser.parse_args(argv)
    try:
        for filename, expected_sha256 in PINNED_FILES:
            status = fetch_verified(filename, expected_sha256, args.dest, args.force)
            print(f"{status}: {args.dest / filename} sha256={expected_sha256}")
    except (RuntimeError, OSError) as exc:
        print(f"fetch_bird_scorer: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
