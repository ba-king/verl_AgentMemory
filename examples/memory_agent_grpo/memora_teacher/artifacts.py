"""Portable artifacts. Credentials and machine-specific paths are not config."""
import hashlib
import json
from pathlib import Path

from ..memora_sft_dev.teacher_rollout_v1.interface import canonical, strict_json

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent


def read(path):
    return strict_json(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def runtime_files():
    # Only dependencies used here; no historical result directories or credentials.
    files = list(PACKAGE.glob('*.py')) + list((PACKAGE/'prompts').glob('*.txt'))
    for directory in ('memory_agent/action', 'memory_agent/bank', 'memory_agent/executor',
                      'teacher_pilot/travel_v0/structured_record',
                      'teacher_pilot/travel_v0/structured_record_v2',
                      'teacher_pilot/travel_v0/single_action',
                      'teacher_pilot/travel_v0/read_consolidate'):
        files += list((PROJECT/directory).glob('*.py'))
    for name in ('memora_sft_dev/teacher_rollout_v1/interface.py',
                 'memora_sft_dev/census_replay.py', 'memora_sft_dev/structured_adapter.py',
                 'memora_sft_dev/document_identity.py', 'memora_sft_dev/native_normalization.py',
                 'audits/stage3_reward_feasibility/data.py',
                 'teacher_pilot/travel_v0/memory_runtime.py',
                 'teacher_pilot/travel_v0/artifacts.py'):
        files.append(PROJECT/name)
    return {str(p.relative_to(PROJECT)): sha(p) for p in sorted(set(files))}


def verify_prepared(root):
    root = Path(root)
    manifest = read(root/'manifest.json')
    if manifest['runtime_sha256'] != runtime_files():
        raise ValueError('PREPARED_CODE_OR_PROMPT_CHANGED')
    for name, expected in manifest['public_artifact_sha256'].items():
        if sha(root/name) != expected:
            raise ValueError('PREPARED_INPUT_CHANGED: ' + name)
    return manifest
