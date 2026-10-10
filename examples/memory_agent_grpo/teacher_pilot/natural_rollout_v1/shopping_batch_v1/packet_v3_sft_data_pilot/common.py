import importlib
import sys
from pathlib import Path
from ..packet_v3_scope_contract_validation.common import read, save, sha, digest, no_network
from ..packet_v3_scope_contract_validation.common import frozen_hashes as previous_hashes

HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parent / 'packet_v3_scope_contract_validation'
PROJECT = HERE.parents[3]
DATA = PROJECT / 'docs/memoryarena_synthetic_data'
ARENA = PROJECT.parents[2] / 'MemoryArena'
REQUEST = Path('/Users/lizizheng/.codex/attachments/4b593961-35c5-43bd-93be-d088d4f4707f/pasted-text.txt')


def frozen_hashes():
    result = previous_hashes()
    seal = CONTRACT / 'FINAL_VALIDATION.json'
    for rel, expected in read(seal)['artifacts_sha256'].items():
        path = CONTRACT / rel
        assert sha(path) == expected, str(path)
        result[str(path)] = expected
    result[str(seal)] = sha(seal)
    return result


def rows(path):
    import json
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save_rows(path, values):
    import json
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))


def constraints():
    # Reuse the old pure deterministic task semantics without changing files or
    # invoking their main()/writing to development output directories.
    path = str(DATA / 'acceptance_v1')
    sys.path.insert(0,path)
    try:
        module=importlib.import_module('shopping_constraints')
        assert Path(module.__file__).resolve()==Path(path)/'shopping_constraints.py'
        return module
    finally:
        sys.path.remove(path)
