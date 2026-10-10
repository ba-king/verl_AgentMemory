from final_unified_travel_rp_packet_v3_demo_v1.common import (
    Path, read, rows, save, append, sha, digest, text, PROJECT, ARENA, DATA,
    EID, SPLIT, OLD, TASK, RAW, BATCH, public_session, offline, verify as verify_old)

HERE = Path(__file__).resolve().parent
ARCHIVE = PROJECT/'final_unified_travel_rp_packet_v3_demo_v1'
REQUEST = Path('/Users/lizizheng/.codex/attachments/9b4d417b-141c-4072-ad35-db8500b4bc4b/pasted-text.txt')


def verify():
    verify_old()
    lock = read(HERE/'implementation_lock.json')
    for path, expected in lock['files'].items():
        if sha(path) != expected:
            raise RuntimeError('FROZEN_FILE_CHANGED: '+path)
    return lock
