from pathlib import Path
from ..packet_v3_canonical_two_episode_gate.common import (
    HERE as TWO_EPISODE, CALIBRATION, PROJECT, ARENA, read, save, rows, sha,
    digest, write_rows, locked_episodes,
)
from ..packet_v3_material_final_dev_gate.common import HERE as MATERIAL

HERE=Path(__file__).resolve().parent
REQUEST=Path('/Users/lizizheng/.codex/attachments/b27f345b-661e-49ce-8230-24421c15264e/pasted-text.txt')
IDS=['packet_v3_trainpilot_s2026091503_material_cumulative_000',
     'packet_v3_trainpilot_s2026091503_mixed_cumulative_000']
ARCHIVE_ROOTS={IDS[0]:MATERIAL,IDS[1]:TWO_EPISODE}
