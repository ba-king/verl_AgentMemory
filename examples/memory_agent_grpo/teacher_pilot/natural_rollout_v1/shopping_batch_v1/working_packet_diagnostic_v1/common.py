from pathlib import Path
from ..answer_root_failure_fix_v2.common import BASE, PRIOR, IDS, read, save, sha, digest
from ..policy_labeler_v2_regression.config import verify as verify_memory

HERE=Path(__file__).resolve().parent
ROOT_V2=HERE.parent/'answer_root_failure_fix_v2'
REQUEST=Path('/Users/lizizheng/.codex/attachments/07d6f6a2-cddb-4fad-ae7e-760288c7aca9/pasted-text.txt')


def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(__import__('json').dumps(value,ensure_ascii=False)+'\n')


def frozen_prior():
    from ..answer_root_failure_fix_v2.common import verify_prior_artifacts
    result=verify_prior_artifacts()
    for rel,h in read(ROOT_V2/'FINAL_REPORT_VALIDATION.json')['artifacts_sha256'].items():
        assert sha(ROOT_V2/rel)==h
        result[str(ROOT_V2/rel)]=h
    result[str(ROOT_V2/'FINAL_REPORT_VALIDATION.json')]=sha(ROOT_V2/'FINAL_REPORT_VALIDATION.json')
    return result


def verify(catalog=False):
    identity=verify_memory(catalog=catalog)
    if (HERE/'manifest.json').exists():
        m=read(HERE/'manifest.json')
        for group in ('runtime_hashes','prior_hashes','public_capsule_hashes'):
            for path,h in m[group].items():assert sha(path)==h,(group,path)
        assert m['answer_system_prompt']==read(BASE/'manifest.json')['answer_system_prompt']
    return identity
