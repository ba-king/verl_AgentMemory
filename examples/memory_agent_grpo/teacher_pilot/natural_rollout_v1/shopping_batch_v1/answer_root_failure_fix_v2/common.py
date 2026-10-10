from pathlib import Path
from ..answer_generator_fix_v1.trajectory import read, save, sha, digest, norm, body, page_key, actions

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'answer_generator_fix_v1'
BASE=HERE.parent/'policy_labeler_v2_regression'
IDS=[f'shopping_joint_v1_{kind}_cumulative_{i:03d}' for kind in ('material','mixed') for i in (0,1)]


def verify_prior_artifacts():
    result={}
    for root,name in [(BASE,'FINAL_VALIDATION.json'),(PRIOR,'FINAL_REPORT_VALIDATION.json')]:
        for rel,value in read(root/name)['artifacts_sha256'].items():
            assert sha(root/rel)==value,('PRIOR_CHANGED',str(root/rel))
            result[str(root/rel)]=value
        result[str(root/name)]=sha(root/name)
    return result
