from pathlib import Path
from ..packet_v3_canonical_two_episode_gate.common import (
    HERE as BASE, PROJECT, ARENA, read, save, rows, sha, digest, selected as prior_selected,
)

HERE = Path(__file__).resolve().parent
REQUEST = Path('/Users/lizizheng/.codex/attachments/66701677-5977-4517-b5fa-9b1402770678/pasted-text.txt')
IDS = ['packet_v3_trainpilot_s2026091503_material_cumulative_000']

def selected():
    return [e for e in prior_selected() if e['episode_id'] in IDS]
