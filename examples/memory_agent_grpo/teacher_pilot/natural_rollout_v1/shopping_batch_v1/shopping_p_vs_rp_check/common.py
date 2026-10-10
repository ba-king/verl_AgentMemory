from pathlib import Path
from ..packet_v3_local_admission_and_utility_dev_v1.common import (
    HERE as BASELINE, IDS, PROJECT, ARENA, read, save, sha, digest, rows,
)

HERE = Path(__file__).resolve().parent
REQUEST = HERE / 'AUTHORIZATION.md'


def selected():
    return read(HERE / 'selected_public.json')


def schedule():
    return [(eid, i) for eid in IDS for i in range(6)]
