# Launching training with a fixed kernel and external validation

Use `native/launch_gated.py` to start a native training wrapper that depends on a fixed backward kernel and a validation receipt. Run it in the trainer environment, where it can hash the actor, kernel, and native wrapper files that the launcher will use.

```sh
python3 native/launch_gated.py --manifest launch.json --worker-proof workers.json --validation-proof validation.json --check
python3 native/launch_gated.py --manifest launch.json --worker-proof workers.json --validation-proof validation.json
```

`--check` leaves the run untouched. The launch command repeats the checks, claims the job root once, and invokes the pinned executable with its pinned arguments. It sets `BIRD_JOB_ROOT` to the staged job directory and `BIRD_RUN_ROOT` to its `native` child. The native wrapper creates that child. Preserve the claim and logs on failure and use a fresh root for any approved retry.

## Pin the launch

The manifest is a JSON object with these fields:

| Field | Value |
|---|---|
| `job_root` | Absolute, resolved, existing staging directory with no `native` child or launch claim |
| `actor`, `kernel`, `launcher` | Absolute `path` and `sha256` for each file in the trainer environment. The executable launcher also has an `args` array. |
| `devices` | Ordered GPU UUIDs, indexed by worker rank |
| `evaluation_endpoints` | `validation` and `milestones` arrays of objects with `bind_domain` and integer `port`. Use the same canonical host/network domain for all processes sharing a TCP bind space. Include rendezvous and evaluator service ports. Set `milestones` to an empty array if milestone scoring starts after validation exits. |
| `validation_watcher_host` | Host identity that runs the validation watcher and can reach the trainer checkpoint, evaluator, and gate receipt destination |
| `gate_window_seconds` | Usable checkpoint-to-publication budget after reserving time for gate observation and normal timing variance |

Both proof files include `manifest_sha256`, the digest of the exact manifest bytes. Generate them under the pinned launcher and environment for the fresh attempt. Archive the underlying logs and measured receipts with the proof.

## Prove the worker and validation paths

1. **Exercise the failing path.** For a candidate memory fix, use fresh private autotuning caches so a warm cache cannot skip the failing benchmark. Compare stock and fixed backward passes at representative shapes and memory pressure. Record which path executed, gradient agreement, and where any allocation failed.
2. **Prove each production worker.** Use the production actor import and actual backward call in the trainer environment on each configured GPU. Write a `workers` array with `rank`, `device_uuid`, `actor_sha256`, `kernel_sha256` and `seal_installed_by_production_path` from each worker. Record `supported_case.fixed_delta` from the fixed branch counter, with `q_positive_finite` and `k_positive_finite` from the gradients. The L2-normalization check makes two fixed calls on the supported input and zero calls on the unsupported input, recorded as `negative_case.fixed_delta`. Keep the candidate actor and kernel installed through launch and verify their hashes again after any restoration or restaging.
3. **Rehearse validation through the real watcher.** Verify the watcher runs on the host with the actual executable and data paths. Give it the validation ports through receipt publication, then release those ports for milestone scoring. A concurrent evaluator needs its own endpoints. Confirm port availability and ownership on the actual evaluation host before launch. Publish a fixture receipt through the real watcher to a rehearsal root and exercise timely and late receipts against the actual gate. Record the effective `evaluation_endpoints`, `watcher_host`, `watcher_ready`, `published_via_watcher`, `fresh_gate_decision`, `late_gate_decision` and measured `checkpoint_to_receipt_seconds`. Arm the live watcher for the new root and verify its binding before invoking the guard.

The guard checks the proof against the manifest and rehashes the installed files. The proof producers and launcher still own live process readiness, port ownership, and the exact-process stop check. Keep the native gate fail-closed if scoring fails or a receipt lands late. On failure, preserve the original result and signal only the verified owned process or listener.

## Regression tests

Run `python3 -B -m unittest discover -s native -p test_launch_gated.py -v` from the repo root. The CPU tests verify that a restored stock actor, an unexercised fixed branch, a shared rendezvous port, a misplaced watcher, and a late or stale receipt all block the launch boundary. They also verify that a passing proof allows one launch.
