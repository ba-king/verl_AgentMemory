import hashlib
import json
from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import patch
import socket

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'packet_interface_v2_validation'
REQUEST = Path('/Users/lizizheng/.codex/attachments/afc75685-73b2-410c-a4af-ce106cd6abaf/pasted-text.txt')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def verify_frozen():
    files = {}
    for rel, h in read(SOURCE / 'FINAL_VALIDATION.json')['artifacts_sha256'].items():
        p = SOURCE / rel
        assert sha(p) == h, str(p)
        files[str(p)] = h
    files[str(SOURCE / 'FINAL_VALIDATION.json')] = sha(SOURCE / 'FINAL_VALIDATION.json')
    for p, h in read(SOURCE / 'manifest.json')['prior_hashes'].items():
        assert sha(p) == h, p
        files[p] = h
    return files


@contextmanager
def no_network():
    with ExitStack() as stack:
        for owner, name in [(socket.socket, 'connect'), (socket.socket, 'connect_ex'),
                            (socket, 'create_connection'), (socket, 'getaddrinfo')]:
            stack.enter_context(patch.object(owner, name, side_effect=AssertionError('OFFLINE_NETWORK_FORBIDDEN')))
        yield
