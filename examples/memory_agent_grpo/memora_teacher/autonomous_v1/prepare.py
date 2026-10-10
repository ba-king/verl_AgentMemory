"""Derive a no-hint public capsule from a sealed paired input, without new tasks."""
import difflib
from pathlib import Path

from .. import artifacts as common
from ..artifacts import PROJECT, read, save, sha

PACKAGE = Path(__file__).resolve().parent


def runtime_files():
    files = list(PACKAGE.glob('*.py')) + [PACKAGE/'prompt.txt']
    return common.runtime_files() | {str(p.relative_to(PROJECT)): sha(p) for p in files}


def verify(prepared):
    prepared = Path(prepared)
    manifest = read(prepared/'manifest.json')
    if manifest['runtime_sha256'] != runtime_files():
        raise ValueError('AUTONOMOUS_CODE_CHANGED')
    for name, expected in manifest['public_artifact_sha256'].items():
        if sha(prepared/name) != expected:
            raise ValueError('AUTONOMOUS_INPUT_CHANGED: ' + name)
    return manifest


def prepare(paired_prepared, out):
    paired_prepared, out = Path(paired_prepared), Path(out)
    if out.exists():
        raise FileExistsError('REFUSE_OVERWRITE')
    prior = common.verify_prepared(paired_prepared)
    for name, expected in prior['audit_artifact_sha256'].items():
        if sha(paired_prepared/name) != expected:
            raise ValueError('PAIRED_REFERENCE_CHANGED')
    original = read(paired_prepared/'public/sessions.json')
    sessions = [{'current_session': item['current_session']} for item in original]
    old_prompt = (paired_prepared/'public/prompt.txt').read_text()
    new_prompt = (PACKAGE/'prompt.txt').read_text()
    # All action definitions, field conventions and domain rules remain exact.
    marker = 'ADD arguments:'
    if old_prompt.split(marker, 1)[1] != new_prompt.split(marker, 1)[1]:
        raise ValueError('FROZEN_ACTION_CONVENTIONS_CHANGED')
    out.mkdir(parents=True)
    for name in ('public/initial_bank.json', 'config.json', 'review_only/references.json'):
        dest = out/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((paired_prepared/name).read_bytes())
    save(out/'public/sessions.json', sessions)
    (out/'public/prompt.txt').write_text(new_prompt)
    (out/'review_only/prompt.diff').write_text(''.join(difflib.unified_diff(
        old_prompt.splitlines(keepends=True), new_prompt.splitlines(keepends=True),
        fromfile='label_conditioned', tofile='autonomous')))
    manifest = {'version': 'memora_teacher_autonomous_v1', 'chain': prior['chain'],
        'session_ids': prior['session_ids'], 'label_conditioned': False,
        'operation_selection_evaluated': True, 'max_writes_per_session': 1,
        'first_yield_is_valid': True, 'label_based_execution_gate': False,
        'initial_bank_origin': prior['initial_bank_origin'],
        'full_prefix_semantically_certified': False,
        'paired_source_manifest_sha256': sha(paired_prepared/'manifest.json'),
        'paired_initial_bank_identical': sha(out/'public/initial_bank.json') == sha(paired_prepared/'public/initial_bank.json'),
        'paired_config_identical': sha(out/'config.json') == sha(paired_prepared/'config.json'),
        'paired_current_sessions_identical': [i['current_session'] for i in sessions] == [i['current_session'] for i in original],
        'action_conventions_identical': True, 'runtime_sha256': runtime_files(),
        'public_artifact_sha256': {n: sha(out/n) for n in ('public/initial_bank.json',
            'public/sessions.json', 'public/prompt.txt', 'config.json')},
        'reference_sha256': sha(out/'review_only/references.json'), 'model_calls': 0}
    save(out/'manifest.json', manifest)
    return manifest
