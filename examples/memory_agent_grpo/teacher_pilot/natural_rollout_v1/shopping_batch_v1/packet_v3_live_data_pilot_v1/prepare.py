"""Phase B runtime lock and explicit existing authorization, before requests."""
import io
import json
import unittest
from .common import HERE,REQUEST,PILOT,read,save,sha,frozen_hashes,no_network,locked_episodes
from ..pilot4_v1.freeze import verify as original_verify,candidate_config,RUN_CONFIG
from ....travel_v0.provider import model_config,credential_ready
from ..packet_interface_v2_validation.interface import PROMPT
from ..packet_interface_v3_minimal_validation.interface_v3 import TOOLS


def verify():
    phase=read(HERE/'hardening/PHASE_A_SEAL.json');assert phase['passed']
    for relative,h in phase['artifacts_sha256'].items():assert sha(HERE/relative)==h,relative
    locked_episodes()
    original_verify()
    if (HERE/'live_manifest.json').exists():
        m=read(HERE/'live_manifest.json')
        for relative,h in m['runtime_hashes'].items():assert sha(HERE/relative)==h,relative
        assert {k:v for k,v in model_config().items() if k!='key_file'}==m['model']
        assert candidate_config()==m['retrieval']


def main():
    if (HERE/'live_manifest.json').exists():raise RuntimeError('REFUSE_SECOND_PREPARE')
    with no_network():
        verify();assert frozen_hashes()==read(HERE/'frozen_before.json')
        original_verify(catalog=True)
        from . import test_runtime
        log=io.StringIO();r=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_runtime))
        (HERE/'runtime_offline_tests.log').write_text(log.getvalue())
        assert r.wasSuccessful(),'RUNTIME_OFFLINE_TESTS_FAILED'
        cfg=model_config();assert credential_ready(cfg),'MISSING_CONFIGURED_CREDENTIAL'
        for name in ('packet_transitions.jsonl','packet_audit.jsonl','packet_local_gold_candidates.jsonl','packet_rejected_candidates.jsonl'):
            path=HERE/name
            if not path.exists():path.write_text('')
        save(HERE/'data_split_manifest.json',read(PILOT/'data_split_manifest.json'))
        save(HERE/'live_manifest.json',{'run_id':'FINAL_HARNESS_HARDENING_AND_PACKET_V3_LIVE_PILOT',
            'request_path':str(REQUEST),'request_sha256':sha(REQUEST),
            'episode_ids':[e['episode_id'] for e in locked_episodes()],'seed':2026091503,
            'runtime_hashes':{str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*.py'))},
            'model':{k:v for k,v in cfg.items() if k!='key_file'},'retrieval':candidate_config(),
            'answer_runtime':RUN_CONFIG.to_dict(),'packet_prompt_exact_frozen':PROMPT,'packet_tools_exact_frozen':TOOLS,
            'max_answer_calls_per_session':30,'max_bank_calls_per_session':32,'max_packet_calls_per_session':30,
            'max_calls_per_episode':552,'max_episodes':6,'semantic_retries':0,'HTTP_retries':0,
            'packet_endpoint':'https://api.deepseek.com/beta/chat/completions',
            'answer_bank_endpoint':'https://api.deepseek.com/chat/completions',
            'packet_input_registry':'Public ref/name/observed-ID registry inside latest_observation. No requirement/proof/verdict supplied.',
            'answer_interface':'Packet only plus unchanged task/public history. No separate raw retrieved memory.',
            'protocol_failure':'Stop that session Packet stream; no repair or repeat. Frozen session runner may continue next session.',
            'semantic_admission':'Offline candidate/readiness diagnostic only, consistent with prior request-level runtime; no new blocking of native actions.',
            'packet_schedule':'INIT before first Answer; UPDATE before next Answer only after actual public event; no extra same-state calls; no post-terminal update.',
            'pilot_stop':'Evaluate at episode boundary, N>=10: schema<90% OR binding>20% OR gold<30%. Infrastructure stops immediately.',
            'mapping_denominator':'All need occurrences in applied packets per actual Packet request (including retained needs); mapped=GATING+LOCAL. Also report newly added needs separately.',
            'gold_audit':'Conservative atomic literal/price audit. Unknown free prose/derivations/removal/unresolved-list semantics reject as audit coverage, not Teacher error.',
            'no_human_review':True,'training':False})
        save(HERE/'authorization.json',{'source':str(REQUEST),'sha256':sha(REQUEST),
            'scope':'User explicitly requests Phase B on same six episodes only after Phase A PASS.',
            'model_data':'Current public task/observations, legitimate retrieved memories, previous Packet and public ref registry only.',
            'endpoints':['https://api.deepseek.com/chat/completions','https://api.deepseek.com/beta/chat/completions'],
            'max_model_requests':3312,'retries':0})
    print('LIVE_PREPARED; Phase A PASS; runtime tests',r.testsRun,'; no model requests',flush=True)


if __name__=='__main__':main()
