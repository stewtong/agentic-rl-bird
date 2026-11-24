#!/usr/bin/env python3

import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")

import argparse
from collections import Counter
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw', type=Path)
    parser.add_argument('--label', required=True)
    parser.add_argument('--expected', type=int, required=True)
    parser.add_argument('--questions', type=int, required=True)
    parser.add_argument('--attempts', type=int, required=True)
    parser.add_argument('--optimizer-complete', action='store_true')
    parser.add_argument('--training-update', action='store_true')
    parser.add_argument('--attempts-out', type=Path, required=True)
    parser.add_argument('--metric-out', type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.raw, map_location='cpu', weights_only=True)
    rows = payload['samples']
    question_counts = Counter()
    duplicate_indexes = []
    seen_indexes = set()
    issues = []
    attempts = []
    total_generated = 0
    truncated = 0
    tool_total = 0
    turn_total = 0
    domains = []
    native_rewards = []
    for sample in rows:
        index = sample['index']
        question_id = sample['metadata'].get('question_id')
        assert isinstance(question_id, str) and question_id
        question_counts[question_id] += 1
        if index in seen_indexes:
            duplicate_indexes.append(index)
        seen_indexes.add(index)
        metadata = sample['metadata']
        status = sample['status']
        is_truncated = status == 'truncated'
        truncated += is_truncated
        mask = sample.get('loss_mask')
        logprobs = sample.get('rollout_log_probs')
        if mask is None:
            issues.append('missing_native_loss_mask')
            generated = None
            mask_consistent = False
        else:
            generated = sum(mask)
            mask_consistent = len(mask) == len(logprobs) == sample['response_length']
            if not mask_consistent:
                issues.append('mask_span_length_mismatch')
        correct = metadata.get('execution_correct')
        graded = type(correct) is bool
        if not graded:
            issues.append('missing_explicit_boolean_domain_grade')
        else:
            domains.append(correct)
        reward = sample.get('reward')
        if type(reward) in (int, float):
            native_rewards.append(reward)
        calls = metadata.get('calls') or []
        turns = metadata.get('num_turns')
        tool_total += len(calls)
        turn_total += turns if isinstance(turns, int) else 0
        if generated is not None:
            total_generated += generated
        attempts.append({
            'question_id': question_id,
            'attempt_ordinal': index % args.attempts,
            'sample_index': index,
            'grade': correct if graded else None,
            'generated_tokens_from_native_loss_mask': generated,
            'truncated': is_truncated,
            'native_sample_status': status,
            'turn_count': turns,
            'tool_call_count': len(calls),
            'mask_logprob_span_length_consistent': mask_consistent,
            'weight_versions': list(sample.get('weight_versions') or []),
        })
    missing_grades = sum(issue == 'missing_explicit_boolean_domain_grade' for issue in issues)
    missing_masks = sum(issue in ('missing_native_loss_mask', 'mask_span_length_mismatch') for issue in issues)
    assert len(rows) == args.expected, (len(rows), args.expected)
    assert not duplicate_indexes
    first_index = min(seen_indexes)
    assert seen_indexes == set(range(first_index, first_index + args.expected))
    assert len(question_counts) == args.questions
    assert all(count == args.attempts for count in question_counts.values())
    mean = total_generated / len(rows) if len(rows) else None
    fraction = truncated / len(rows)
    stat = args.raw.stat()
    result = {
        'schema_version': 'seed42-update-metrics/v1',
        'label': args.label,
        'native_payload_rollout_id': payload['rollout_id'],
        'raw_path': str(args.raw), 'raw_sha256': sha(args.raw), 'raw_bytes': stat.st_size,
        'raw_mtime_utc': datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        'optimizer_complete_observed_separately': args.optimizer_complete,
        'sample_index_min': first_index, 'sample_index_max': max(seen_indexes),
        'attempts': len(rows), 'questions': len(question_counts), 'attempts_per_question': args.attempts,
        'total_generated_tokens_from_native_loss_mask': total_generated,
        'mean_generated_tokens_per_attempt': mean,
        'truncated_attempt_count': truncated,
        'truncation_fraction': fraction,
        'tool_call_count': tool_total, 'tool_calls_mean_per_attempt': tool_total / len(rows),
        'turn_total': turn_total, 'turns_mean_per_attempt': turn_total / len(rows),
        'explicit_boolean_domain_grades': len(domains),
        'domain_ex_correct': sum(domains),
        'domain_ex_correct_rate_over_graded': sum(domains) / len(domains) if domains else None,
        'native_reward_mean_not_used_as_domain_ex': sum(native_rewards) / len(native_rewards) if native_rewards else None,
        'missing_explicit_boolean_domain_grades': missing_grades,
        'missing_or_inconsistent_masks': missing_masks,
        'stop_mean_strict_greater_than': 291.71029411764704,
        'mean_threshold_exceeded': args.training_update and mean > 291.71029411764704,
        'truncation_threshold_strictly_exceeded': fraction > 0.25,
        'stop_for_missing_grade_or_mask_corruption': missing_masks > 0 or missing_grades > 0,
        'portable_attempt_file': str(args.attempts_out), 'portable_attempt_file_sha256': sha(args.attempts_out) if args.attempts_out.exists() else None,
    }
    args.attempts_out.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in attempts))
    result['portable_attempt_file_sha256'] = sha(args.attempts_out)
    args.metric_out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
