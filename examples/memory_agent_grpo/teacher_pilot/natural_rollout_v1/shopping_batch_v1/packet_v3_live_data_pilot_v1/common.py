from pathlib import Path
from ..packet_v3_harness_coverage_patch_v1.common import read, save, sha, digest, no_network, locked_episodes, PILOT, OLD, V3
from ..packet_v3_harness_coverage_patch_v1.common import frozen_hashes as prior_hashes

HERE=Path(__file__).resolve().parent
PATCH=HERE.parent/'packet_v3_harness_coverage_patch_v1'
PROJECT=HERE.parents[3]
ARENA=PROJECT.parents[2]/'MemoryArena'
REQUEST=Path('/Users/lizizheng/.codex/attachments/edaedfa0-abab-4312-b3fe-a78f8cfeb8bc/pasted-text.txt')


def frozen_hashes():
    result=prior_hashes();seal=PATCH/'FINAL_VALIDATION.json'
    for relative,expected in read(seal)['artifacts_sha256'].items():
        p=PATCH/relative;assert sha(p)==expected,str(p);result[str(p)]=expected
    result[str(seal)]=sha(seal)
    return result
