from pathlib import Path
from ..working_packet_diagnostic_v1.common import BASE, PRIOR, read, save, sha, digest, append
from ..working_packet_diagnostic_v1.common import frozen_prior as earlier_prior
from ..policy_labeler_v2_regression.config import verify as verify_memory

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'working_packet_diagnostic_v1'


def frozen_prior():
    result = earlier_prior()
    for rel, h in read(V1 / 'FINAL_VALIDATION.json')['artifacts_sha256'].items():
        path = V1 / rel
        assert sha(path) == h, str(path)
        result[str(path)] = h
    result[str(V1 / 'FINAL_VALIDATION.json')] = sha(V1 / 'FINAL_VALIDATION.json')
    return result


def verify(catalog=False):
    identity = verify_memory(catalog=catalog)
    if (HERE / 'manifest.json').exists():
        m = read(HERE / 'manifest.json')
        for group in ('runtime_hashes', 'prior_hashes', 'capsule_hashes'):
            for path, h in m[group].items():
                assert sha(path) == h, (group, path)
        assert m['answer_system_prompt'] == read(BASE / 'manifest.json')['answer_system_prompt']
    return identity
