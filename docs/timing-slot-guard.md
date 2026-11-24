# Guarding isolated timing slots

Cost comparisons use native seconds per draw from isolated timing slots, and a slot counts only if nothing else ran on the node. In an earlier evaluation run that repeated the update-50 evaluation on seed 43, two of fifteen slots could not be used. One original started its isolation sampling 287 s after the first generation bar. Its rerun overlapped a 30 s busy loop that an operator ran on the node. The rules allow one rerun, so the cost reading that depended on that slot stayed open. Earlier timing slots kept no per-slot guard marker.

`native/timing_slot_guard.py` wraps each slot in two steps. It needs only the standard library.

```sh
python3 native/timing_slot_guard.py preflight --slot slot.json --sampler-status sampler.json
# launch the native timing job for this slot
python3 native/timing_slot_guard.py close --job-root /abs/job --samples samples.json --native native.json
```

## Before launch: `preflight`

`slot.json` holds:
- `slug`;
- `job_root` (absolute, resolved, existing);
- `attempt`: 1, or 2 for the one isolated rerun, which also names the original's `rerun_of_job_root`;
- `node`;
- `node_lock` and `attempt_registry`, both absolute paths on the node;
- `max_sample_age_seconds`;
- `max_gap_seconds`, the longest allowed gap between isolation samples during the native interval.

`sampler.json` is the isolation sampler's current status. It holds `node`, `running`, `first_sample_utc`, `last_sample_utc`, `gpu_processes` and `latest_rows`, which are the latest attributed process rows.

Preflight blocks the launch in any of these cases:
- the sampler is stopped, on another node, or its last sample is too old;
- another GPU process is present;
- any latest row is non-exempt work;
- the node lock is already held;
- the job root already has a guard marker;
- the registry already holds this slug's attempt;
- the attempt is a third one;
- a rerun points at an original whose disposition is not `needs_rerun`.

When the checks pass, preflight writes `timing-slot-guard.json` into the job root, then claims the node lock and the registry entry. The marker file is the per-slot guard marker. If either claim fails, preflight removes what it wrote.

While the lock exists, run nothing else on the node: no tests, diagnostics, milestone scoring or exports. Any process the sampler attributes to work other than the native job, the sampler, its transport or a baseline daemon counts as overlap. An unknown process also counts as overlap.

## After the native exit: `close`

`samples.json` lists the sampler's attributed samples. Each has `utc`, `rows` and `gpu_processes`, and every process row carries a `disposition`. `native.json` holds `started_utc`, `first_bar_utc`, `finished_utc` and `exit_code` from the native exit receipt and driver log.

Close first checks that the marker belongs to this job root and that the node lock is still held by this slot. It then writes `timing-slot-disposition.json` and releases the lock. A slot is accepted only when all of these hold:
- the native interval started after preflight;
- sampling began before the first generation bar and continued past the native finish;
- no gap between samples during the interval exceeds `max_gap_seconds`;
- no non-exempt process or GPU process row appears during the interval;
- the native exit code is the integer 0.

A failed original needs one isolated rerun in a fresh root. A failed rerun is unresolved, and both measurements are reported. The command exits 3 for any slot that is not accepted.

The sampler and its process attribution stay outside this guard. Start the sampler before preflight, and keep it running across supervisor restarts.

## Regression tests

Run `python3 -B -m unittest discover -s native -p test_timing_slot_guard.py -v` from the repo root. The CPU tests replay three observed isolation failures: late sampling, operator work during a rerun, and an unknown process. They also cover:
- sampler gaps and early stops;
- unattributed GPU processes;
- a stale or stopped sampler;
- a lock held by another slot;
- a reused job root;
- a repeated original or a third attempt;
- a rerun without a failed original;
- a non-integer exit code.
