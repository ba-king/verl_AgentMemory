from pathlib import Path
from ..final_two_episode_state_maintenance_live_gate.common import (
    PROJECT, ARENA, PILOT, IDS, selected, read, save, sha, digest, rows, write_rows, GateStop,
)
from ..final_two_episode_state_maintenance_live_gate.common import HERE as SOURCE

HERE = Path(__file__).resolve().parent
REQUEST = Path('/Users/lizizheng/.codex/attachments/d6ce9850-9be1-40a8-976c-f2b2f2c37e6e/pasted-text.txt')


def schedule():
    return [(eid, i) for eid in IDS for i in range(6)]
