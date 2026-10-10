from pathlib import Path
from ..packet_v3_live_data_pilot_v1.common import (
    HERE as OLD, PILOT, PROJECT, ARENA, read, save, sha, digest, no_network,
    frozen_hashes as old_frozen_hashes, locked_episodes,
)

HERE = Path(__file__).resolve().parent
REQUEST = Path('/Users/lizizheng/.codex/attachments/facbf7da-2033-4303-b3cb-845a1abd2480/pasted-text.txt')


def rows(path):
    import json
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, values):
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(v, ensure_ascii=False, sort_keys=True) + '\n' for v in values))


def frozen_hashes():
    result = old_frozen_hashes()
    seal = OLD / 'FINAL_VALIDATION.json'
    for relative, expected in read(seal)['artifacts_sha256'].items():
        path = OLD / relative
        assert sha(path) == expected, path
        result[str(path)] = expected
    result[str(seal)] = sha(seal)
    return result


def source_capsules():
    """Public-only input allowlist. Never opens scored session/episode summaries."""
    ts = rows(OLD / 'packet_transitions.jsonl')
    audits = rows(OLD / 'packet_audit.jsonl')
    assert len(ts) == len(audits) == 36
    out = []
    for t, audit in zip(ts, audits, strict=True):
        folder = Path(t['request_ref']).parent.parent
        assert folder.is_relative_to(OLD / 'raw_rollouts') and folder.parent.name == 'packet'
        public = read(folder / 'public_prefix.json')
        trusted = read(folder / 'trusted_registry.json')
        contract = read(folder / 'contract.json')
        assert digest(public) == t['public_cutoff_hash']
        out.append({'transition': t, 'old_audit': audit, 'public': public,
                    'trusted': trusted, 'contract': contract, 'folder': str(folder)})
    return out
