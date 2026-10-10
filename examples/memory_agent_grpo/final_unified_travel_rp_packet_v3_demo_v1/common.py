"""Isolated output paths; original demo artifacts are read-only."""
from pathlib import Path
from teacher_pilot.travel_v0.artifacts import read, rows, save, append, sha, digest, text, PROJECT, ARENA, DATA, public_session

HERE = Path(__file__).resolve().parent
EID = 'travel_syn_2026091307_009'
SPLIT = 'FINAL_CONFIG_TEACHER_DEMO_ONLY'
OLD = PROJECT / 'teacher_demo_package'
TASK = OLD / 'travel/travel_demo_02'
RAW = OLD / 'runtime/travel/raw_rollouts' / EID
BATCH = PROJECT / 'teacher_pilot/natural_rollout_v1/shopping_batch_v1'
REQUEST = Path('/Users/lizizheng/.codex/attachments/974e8e19-2247-4dcc-8a26-4121c104b8aa/pasted-text.txt')


def verify():
    lock = read(HERE / 'preflight/live_lock.json')
    for path, expected in lock['files'].items():
        if sha(path) != expected:
            raise RuntimeError('FROZEN_FILE_CHANGED: ' + path)
    return lock


def offline():
    import socket
    def deny(*args, **kwargs):
        raise RuntimeError('OFFLINE_NETWORK_FORBIDDEN')
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
