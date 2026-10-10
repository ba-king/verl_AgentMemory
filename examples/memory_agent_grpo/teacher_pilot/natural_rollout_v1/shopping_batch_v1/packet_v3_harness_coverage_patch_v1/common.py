from pathlib import Path
from ..packet_v3_sft_data_pilot.common import read, save, sha, digest, rows, no_network, frozen_hashes as prior_hashes

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / 'packet_v3_sft_data_pilot'
OLD = HERE.parent / 'packet_v3_scope_contract_validation'
V3 = HERE.parent / 'packet_interface_v3_minimal_validation'
REQUEST = Path('/Users/lizizheng/.codex/attachments/8269f715-8826-43f7-93af-452e92b37e4a/pasted-text.txt')


def frozen_hashes():
    result = prior_hashes()
    seal = PILOT / 'FINAL_VALIDATION.json'
    for relative, expected in read(seal)['artifacts_sha256'].items():
        path = PILOT / relative
        assert sha(path) == expected, str(path)
        result[str(path)] = expected
    result[str(seal)] = sha(seal)
    return result


def locked_episodes():
    seal = read(PILOT / 'FINAL_VALIDATION.json')['artifacts_sha256']
    for name in ('tasks/public_episodes.jsonl', 'tasks/generation_manifest.json', 'data_split_manifest.json'):
        assert sha(PILOT / name) == seal[name], name
    assert read(PILOT/'tasks/generation_manifest.json')['seed'] == 2026091503
    result = rows(PILOT/'tasks/public_episodes.jsonl')
    assert len(result) == 6 and all(len(r['sessions']) == 6 for r in result)
    return result
