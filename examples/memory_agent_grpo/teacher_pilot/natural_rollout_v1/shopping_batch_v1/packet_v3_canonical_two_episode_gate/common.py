from pathlib import Path
from ..packet_v3_data_pipeline_convergence_v1.common import (
    HERE as CALIBRATION, OLD as PRIOR_LIVE, PILOT, PROJECT, ARENA,
    read, save, sha, digest, rows, write_rows, no_network, locked_episodes,
)

HERE=Path(__file__).resolve().parent
REQUEST=Path('/Users/lizizheng/.codex/attachments/987f3aa2-999e-4750-9718-fee7d4e9f2e9/pasted-text.txt')
IDS=['packet_v3_trainpilot_s2026091503_material_cumulative_000',
     'packet_v3_trainpilot_s2026091503_mixed_cumulative_000']


def selected():
    episodes={e['episode_id']:e for e in locked_episodes()}
    return [episodes[eid] for eid in IDS]


class GateStop(RuntimeError):
    pass
