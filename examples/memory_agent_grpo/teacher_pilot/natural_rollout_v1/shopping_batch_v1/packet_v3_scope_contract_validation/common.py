from pathlib import Path
from ..packet_interface_v3_minimal_validation.common import read, save, sha, digest, no_network
from ..packet_interface_v3_minimal_validation.common import frozen_hashes as prior_hashes

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / 'packet_interface_v3_minimal_validation'
BATCH = HERE.parent
REQUEST = Path('/Users/lizizheng/.codex/attachments/db9f96d2-d3db-4c7e-88be-ebb8e62c83f2/pasted-text.txt')


def frozen_hashes():
    hashes = prior_hashes()
    final = V3 / 'FINAL_VALIDATION.json'
    for relative, expected in read(final)['artifacts_sha256'].items():
        path = V3 / relative
        assert sha(path) == expected, str(path)
        hashes[str(path)] = expected
    hashes[str(final)] = sha(final)
    return hashes
