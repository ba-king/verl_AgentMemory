"""Artifact paths and exact, append-only request/decision recording."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REPO = next(p for p in HERE.parents if (p / 'pyproject.toml').is_file() and (p / 'examples/memory_agent_grpo').is_dir())
WORKSPACE = REPO.parent
ARENA = WORKSPACE / 'MemoryArena'
DATA = PROJECT / 'docs/memoryarena_synthetic_data'
ACCEPTANCE = DATA / 'acceptance_v1'
TRAVEL = DATA / 'family_audits/travel'
IDS = ['travel_syn_2026091307_008', 'travel_syn_2026091307_000', 'travel_syn_2026091307_016']
ARMS = ['no_memory', 'oracle_memory', 'split_teacher']
FROZEN = PROJECT / 'runs/stage3_reward_feasibility_20260911_r1/formal_cpu/retrieval_config.json'
PROVIDER = PROJECT / 'runs/v4pro_tasks61_113_full_packet_20260909_r1/config.json'


def text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(text(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(text(value) + '\n')


def public_session(public, index):
    s = public['sessions'][index]
    value = {'session_id': s['session_id'], 'instruction': s['instruction']}
    if index == 0:
        value['initial_observation'] = public['initial_observation']
    return value


def selected_public():
    found = {r['episode_id']: r for r in rows(TRAVEL / 'public_episodes.jsonl')}
    return {eid: found[eid] for eid in IDS}
