from travel_lifecycle_fix_and_gated_rerun_v1.common import (
    Path, read, rows, save, append, sha, digest, text, PROJECT, ARENA, DATA,
    EID, SPLIT, OLD, TASK, RAW, BATCH, ARCHIVE, public_session, offline,
    verify as verify_lifecycle)

HERE = Path(__file__).resolve().parent
LIFECYCLE = PROJECT/'travel_lifecycle_fix_and_gated_rerun_v1'
REQUEST = Path('/Users/lizizheng/.codex/attachments/c5090b0c-407a-41ab-a056-b7422f5ff08d/pasted-text.txt')


def verify():
    verify_lifecycle()
    lock=read(HERE/'implementation_lock.json')
    for p,h in lock['files'].items():
        if sha(p)!=h:raise RuntimeError('FROZEN_FILE_CHANGED: '+p)
    return lock
