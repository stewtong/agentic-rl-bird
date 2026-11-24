import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--root', default=os.environ.get('BIRD_ROOT'))
args = parser.parse_args()
if not args.root:
    print('prepare_bird_data.py: BIRD_ROOT is not set; set BIRD_ROOT or pass --root <path>', file=sys.stderr)
    raise SystemExit(2)
root = Path(args.root)
train_root = root / 'data/bird_train/train'
out = root / 'data/protocol'
out.mkdir(parents=True, exist_ok=True)
questions = json.loads((train_root / 'train.json').read_text())
database_root = train_root / 'train_databases'
if not database_root.exists():
    database_root = train_root / 'databases'
assert database_root.is_dir()
databases = sorted(path.name for path in database_root.iterdir() if path.is_dir())
assert len(databases) >= 2
ranked = sorted(databases, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
validation_count = max(1, (len(databases) + 9) // 10)
validation_names = ranked[:validation_count]
assert validation_names and len(validation_names) < len(databases)
assert set(validation_names) != set(databases)
validation = [row for row in questions if row['db_id'] in validation_names]
candidates = [row for row in questions if row['db_id'] not in validation_names]
random.Random(42).shuffle(candidates)
candidates = candidates[:1024]
(out / 'training-candidates.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in candidates))
(out / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')


def prompt_for(row):
    return (
        'Read-only SQLite database task. Inspect schemas and produce the correct final SQL.\n'
        f'Database ID: {row["db_id"]}\n'
        f'Question: {row["question"]}\n'
        f'Evidence: {row.get("evidence", "")}\n'
        'Use only read-only queries, then give exactly one ```sql code block.'
    )


def data_row(index, row):
    db_id = row['db_id']
    path = database_root / db_id / f'{db_id}.sqlite'
    assert path.is_file(), path
    return {
        'question_id': f'train-{index:04d}',
        'prompt': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': prompt_for(row),
                    }
                ],
            }
        ],
        'label': '',
        'metadata': {
            'question_id': f'train-{index:04d}',
            'db_id': db_id,
            'db_path': str(path),
            'gold_sql': row['SQL'],
            'split': 'train',
        },
    }


training_rows = [data_row(index, row) for index, row in enumerate(candidates, 1)]
trial_rows = training_rows[:16].copy()
(out / 'training-rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in training_rows))
(out / 'trial-rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in trial_rows))
manifest = {
    'schema_version': 'bird-data-protocol/v1',
    'dataset_url': 'https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip',
    'dataset_etag': '1FBA2891BD215A42CE1004736AB7F47F-400',
    'archive_sha256': hashlib.sha256((root / 'data/train.zip').read_bytes()).hexdigest(),
    'dataset_last_modified': '2023-07-11T06:13:29Z',
    'database_count': len(databases),
    'database_partition': 'Ascending SHA256(db_id), first ceil(10%) validation databases. dev/Mini-Dev excluded completely.',
    'validation_databases': validation_names,
    'train_databases': sorted(set(databases) - set(validation_names)),
    'source_question_count': len(questions),
    'validation_question_count': len(validation),
    'candidate_question_count': len(training_rows),
    'candidate_seed': 42,
    'trial_question_ids': [row['metadata']['question_id'] for row in trial_rows],
    'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'no_reference_solutions_executed': True,
}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
