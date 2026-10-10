"""Static acceptance, immutable V1 evidence, four IDs and unchanged runtime locks."""
import difflib
import io
import json
import socket
import unittest
from pathlib import Path
from unittest.mock import patch
from ..pilot4_v1 import freeze
from ....travel_v0.artifacts import read, save, rows, sha, digest
from ....travel_v0.provider import model_config, credential_ready
from ....travel_v0.structured_record import policy as frozen_policy, record_renderer
from .policy import RETENTION_GUIDANCE

HERE = Path(__file__).resolve().parent
BATCH = HERE.parent
BASELINE = BATCH / 'pilot4_resume_v1/combined'
IDS = tuple(freeze.IDS)
REQUEST = Path('/Users/lizizheng/.codex/attachments/25042d39-7822-4e4c-b9ce-cceb7c983550/pasted-text.txt')
# Fixed before V2 requests. These are development diagnostics, not a statistical
# equivalence/noninferiority claim from one run per episode.
JUDGMENT = {
    'clear_mixed_retention_min_rate': 0.75,
    'minimum_mixed_eligible_for_scale': 3,
    'clear_material_min_success': 6,
    'clear_material_min_chains': 5,
    'material_clear_regression_success_loss': 2,
    'material_bank_bloat_ratio': 1.5,
    'material_answer_mean_context_growth_ratio': 1.5,
    'material_rank_median_degradation': 1,
    'notes': 'Predeclared diagnostic flags; also report raw counts, provenance, rank availability and side effects. No post-hoc tuning or resampling.'}
RUNTIME_FILES = ('__init__.py', 'policy.py', 'config.py', 'execute.py', 'launch.py',
                 'labeler.py', 'datasets.py', 'test_labeler.py', 'test_policy.py', 'metrics.py')


def prior_hashes():
    result = {}
    for root in (BATCH / 'pilot4_v1', BATCH / 'pilot4_resume_v1', BATCH / 'memory_credit_and_retention_audit_v1'):
        v = read(root / 'FINAL_VALIDATION.json')
        result.update({str(root / p): h for p, h in v['artifacts_sha256'].items()})
        result[str(root / 'FINAL_VALIDATION.json')] = sha(root / 'FINAL_VALIDATION.json')
    return result


def verify(catalog=False):
    identity = freeze.verify(catalog=catalog)
    manifest_path = HERE / 'manifest.json'
    if manifest_path.exists():
        m = read(manifest_path)
        assert tuple(m['episode_ids']) == IDS and m['seed'] == 0
        assert m['judgment'] == JUDGMENT
        for p, h in m['runtime_hashes'].items():
            assert sha(p) == h, ('V2_SOURCE_CHANGED', p)
        assert sha(freeze.HERE / 'selected_public.json') == m['v1_selected_public_sha256']
        assert read(HERE / 'selected_public.json') == read(freeze.HERE / 'selected_public.json')
        assert m['model'] == {k: v for k, v in model_config().items() if k != 'key_file'}
        assert m['memory_consolidate_prompt'] == frozen_policy.CONSOLIDATE_PROMPT + RETENTION_GUIDANCE
        gate = read(HERE / 'V2_STATIC_ACCEPTANCE.json')
        assert gate['status'] == 'V2_STATIC_ACCEPTANCE_PASS' and all(gate['gates'].values())
    return identity


def verify_priors():
    hashes = read(HERE / 'prior_artifact_hashes.json') if (HERE / 'prior_artifact_hashes.json').exists() else prior_hashes()
    for p, h in hashes.items():
        assert sha(p) == h, ('PRIOR_ARTIFACT_CHANGED', p)
    return len(hashes)


def prepare():
    assert not (HERE / 'manifest.json').exists(), 'REFUSE_SECOND_PREPARE'
    assert not (HERE / 'launch_started.json').exists()
    from . import test_labeler, test_policy
    from .datasets import export_corpus
    gates = {}; errors = []; identity = None
    try:
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('STATIC_HTTP_FORBIDDEN')):
            identity = freeze.verify(catalog=True)
            hashes = prior_hashes()
            assert all(sha(p) == h for p, h in hashes.items())
            save(HERE / 'prior_artifact_hashes.json', hashes)
            suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_labeler, test_policy))
            log = io.StringIO(); result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
            (HERE / 'static_tests.log').write_text(log.getvalue())
            gates['labeler_negative_and_known_case_tests'] = result.wasSuccessful()
            if not result.wasSuccessful():
                raise AssertionError('OFFLINE_TEST_FAILURE')
            relabel = export_corpus(BASELINE, HERE / 'v1_relabel_counterfactual', counterfactual=True)
            known = {(c['episode_id'], c['source_decision']) for c in rows(BASELINE / 'analysis/cross_session_success_chains.jsonl')}
            recovered = [w for w in relabel['write_results'] if (w['episode_id'], w['decision_id']) in known and w['label'] == 'WRITE_POSITIVE_STRICT']
            gates['five_known_adds_recovered'] = len(recovered) == 5
            snap = read(BATCH / 'frozen_config_snapshot.json')
            gates['schema_unchanged'] = record_renderer.record_schema() == snap['record_schema']
            gates['action_set_and_arguments_unchanged'] = frozen_policy.action_schema(['RETRIEVE', 'ADD', 'UPDATE', 'DELETE', 'YIELD'], ['EXAMPLE_VISIBLE_ID']) == snap['action_schema']
            gates['Top5_unchanged'] = freeze.candidate_config() == snap['retrieval_config']
            gates['READ_prompt_unchanged'] = frozen_policy.READ_PROMPT == snap['memory_read_prompt']
            gates['Answer_model_unchanged'] = {k: v for k, v in model_config().items() if k != 'key_file'} == snap['answer_agent']
            gates['Answer_prompt_unchanged'] = freeze.messages(freeze.ShoppingTaskInputAdapter(max_observation_chars=16000).build_subtask_input({'state': 'PUBLIC_INPUT_PLACEHOLDER'}))[0]['content'] == read(freeze.HERE / 'manifest.json')['answer_system_prompt']
            gates['scorer_environment_renderer_Bank_unchanged'] = bool(identity)
            gates['only_consolidation_retention_guidance_changed'] = frozen_policy.CONSOLIDATE_PROMPT == snap['memory_consolidate_prompt'] and 'Task-relevant grounded retention:' in RETENTION_GUIDANCE
            gates['four_episode_IDs_locked'] = [p['episode_id'] for p in read(freeze.HERE / 'selected_public.json')] == list(IDS)
            from .metrics import collect, token_counter
            baseline_metrics = collect(BASELINE, relabel, token_counter())
            baseline_mixed = next(r for r in baseline_metrics if r['episode_id'] == IDS[2])
            gates['V1_mixed_eligibility_calibrated_0_of_4'] = baseline_mixed['attribute_retention'] == {'eligible': 4, 'retained': 0, 'rate': 0.0}
            save(HERE / 'v1_metrics_with_labeler_v2.json', baseline_metrics)
            gates['public_only_projection_and_provenance'] = result.wasSuccessful()
            gates['original_labels_and_trajectories_unchanged'] = all(sha(p) == h for p, h in hashes.items())
            gates['local_credential_configured'] = credential_ready(model_config())
    except Exception as error:
        errors.append(type(error).__name__ + ': ' + str(error))
    status = 'V2_STATIC_ACCEPTANCE_PASS' if not errors and gates and all(gates.values()) else 'V2_STATIC_ACCEPTANCE_FAILED'
    save(HERE / 'V2_STATIC_ACCEPTANCE.json', {'status': status, 'gates': gates, 'errors': errors,
        'model_API_calls': 0, 'labeler_LLM_judge_calls': 0, 'fixtures_are_not_live_evidence': True})
    if status != 'V2_STATIC_ACCEPTANCE_PASS':
        raise RuntimeError(status + ': ' + repr(errors))
    old = read(freeze.HERE / 'manifest.json')
    selected = read(freeze.HERE / 'selected_public.json')
    save(HERE / 'selected_public.json', selected)
    save(HERE / 'manifest.json', {**old, 'run_id': 'MEMORY_POLICY_AND_LABELER_V2_REGRESSION',
        'episode_ids': list(IDS), 'seed': 0, 'judgment': JUDGMENT,
        'memory_consolidate_prompt': frozen_policy.CONSOLIDATE_PROMPT + RETENTION_GUIDANCE,
        'v1_selected_public_sha256': sha(freeze.HERE / 'selected_public.json'),
        'request_file': str(REQUEST), 'request_sha256': sha(REQUEST),
        'memory_policy_diff': 'memory_policy_diff.md',
        'max_total_HTTP_attempts': 1488, 'max_Memory_calls_per_session': 32, 'max_Answer_calls_per_session': 30,
        'remaining_eight_allowed': False, 'training_allowed': False,
        'counterfactual_V1_positive': len(relabel['positive']),
        'runtime_hashes': {str(HERE / p): sha(HERE / p) for p in RUNTIME_FILES},
        'base_execution_hashes': old['execution_hashes'],
        'execution_hashes': {str(HERE / p): sha(HERE / p) for p in RUNTIME_FILES}})
    save(HERE / 'authorization.json', {'source': str(REQUEST), 'source_sha256': sha(REQUEST),
        'authorization': 'User explicitly requests new paired regression on the original four IDs after static PASS, using unchanged DeepSeek V4 Pro actors and limits.',
        'endpoint': 'https://api.deepseek.com/chat/completions', 'episode_ids': list(IDS),
        'public_data': 'Current public task, actual browser observations/submission and legitimate Memory only; no hidden targets or evaluator feedback to actors.',
        'max_HTTP_attempts': 1488, 'retries': 0, 'no_remaining_eight': True})
    verify()
    print(status, 'V1 counterfactual positive:', len(relabel['positive']), flush=True)


if __name__ == '__main__': prepare()
