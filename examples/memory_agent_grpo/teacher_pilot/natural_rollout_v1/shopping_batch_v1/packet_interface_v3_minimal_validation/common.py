from pathlib import Path
from ..packet_binding_regression_v1.common import read, save, sha, digest, no_network, verify_frozen

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'packet_binding_regression_v1'
V2 = HERE.parent / 'packet_interface_v2_validation'
REQUEST = Path('/Users/lizizheng/.codex/attachments/c75713a2-7cc8-4666-879e-cd9fa83999eb/pasted-text.txt')


def frozen_hashes():
    hashes = verify_frozen()
    for rel, expected in read(V1 / 'FINAL_VALIDATION.json')['artifacts_sha256'].items():
        path = V1 / rel
        assert sha(path) == expected, str(path)
        hashes[str(path)] = expected
    hashes[str(V1 / 'FINAL_VALIDATION.json')] = sha(V1 / 'FINAL_VALIDATION.json')
    return hashes
