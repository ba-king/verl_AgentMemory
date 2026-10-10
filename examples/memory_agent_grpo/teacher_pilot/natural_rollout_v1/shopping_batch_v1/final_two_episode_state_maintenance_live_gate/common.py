from pathlib import Path
from ..packet_v3_canonical_two_episode_gate.common import (
    PROJECT, ARENA, PILOT, IDS, selected, read, save, sha, digest, rows, write_rows, GateStop,
)
from ..mandatory_resolve_auditor_patch_and_archived_gate_v1.common import HERE as ARCHIVED

HERE = Path(__file__).resolve().parent
REQUEST = Path('/Users/lizizheng/.codex/attachments/cea06557-4037-4d9d-b277-56118422e5a3/pasted-text.txt')


def schedule():
    return [(eid, i) for start in (0, 3) for eid in IDS for i in range(start, start + 3)]
