"""Verify exported teacher-format sessions without a model or network."""
import argparse
import hashlib
import json
from pathlib import Path

try:
    from .dataset import SessionDataset, labels_from_verified_spans, requests_for_session
except ImportError:
    from dataset import SessionDataset, labels_from_verified_spans, requests_for_session


def verify(root, expected_count=None):
    root = Path(root)
    dataset = SessionDataset(root / 'data/sessions.jsonl')
    if expected_count is not None:
        assert len(dataset) == expected_count
    expanded = [t for row in dataset.rows for t in requests_for_session(row)]
    saved = [json.loads(line) for line in (root / 'data/input_target.jsonl').read_text().splitlines()]
    assert expanded == saved, 'PREFIX_OR_TARGET_MISMATCH'
    for turn in expanded:
        assert set(turn['input']) == {'messages', 'tools'}
        assert turn['input']['messages'][-1]['role'] in {'user', 'tool'}
        assert turn['target']['role'] == 'assistant'
    if len(dataset):
        item = dataset[0]
        item['messages'][0]['content'] = 'MUTATED'
        assert dataset[0]['messages'][0]['content'] != 'MUTATED'
    # Interface calibration only, not a student's actual chat-template spans.
    mask = labels_from_verified_spans(list(range(14)), [(4, 7, 1.0), (10, 12, 0.2)])
    assert mask['labels'] == [-100]*4 + [4, 5, 6] + [-100]*3 + [10, 11] + [-100]*2
    assert mask['loss_weights'][10:12] == [0.2, 0.2]
    for name, expected in json.loads((root / 'artifact_hashes.json').read_text()).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    return {'sessions': len(dataset), 'assistant_turns': len(expanded),
            'prefix_and_target_alignment': 'PASS', 'artifact_hashes': 'PASS',
            'student_template_tested': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--expected-count', type=int)
    a = p.parse_args()
    print(json.dumps(verify(a.root, a.expected_count)))
