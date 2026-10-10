"""Verify the original locks plus the authorized native overlay, never replace them."""
from pathlib import Path
from ..config import HERE as BATCH, PROJECT, ARENA, SHOP, CANDIDATE, candidate_config
from ..native_init_fix_v1.run_preflight import locked_inputs
from ....travel_v0.artifacts import read,save,sha,rows,digest
from ....travel_v0.provider import model_config,credential_ready
from ....travel_v0.structured_record import policy,record_renderer
from examples.memory_agent_grpo.adapters.memoryarena.shopping.task_input_adapter import ShoppingTaskInputAdapter
from examples.memory_agent_grpo.adapters.memoryarena.shopping.tool_catalog import tool_definitions
from examples.memory_agent_grpo.answer_agent.prompts.builder import build_answer_agent_initial_messages
from examples.memory_agent_grpo.answer_agent.interaction_mode import AnswerAgentInteractionMode
from examples.memory_agent_grpo.answer_agent.runtime.config import AnswerAgentSubtaskRunnerConfig

HERE=Path(__file__).resolve().parent
FIX=BATCH/'native_init_fix_v1'
IDS=[f'shopping_joint_v1_{kind}_cumulative_{i:03d}' for kind in ['material','mixed'] for i in [0,1]]
RUN_CONFIG=AnswerAgentSubtaskRunnerConfig(30,1,3,AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL)


def messages(subtask):
    return [m.to_dict() for m in build_answer_agent_initial_messages(subtask,interaction_mode=RUN_CONFIG.interaction_mode)]


def verify(catalog=False):
    identity=locked_inputs()
    overlay=read(FIX/'validated_runtime_overlay.json')
    assert overlay['status']=='SHOPPING_NATIVE_ENV_READY'
    assert identity['patch_source_hashes']==overlay['authorized_native_patch']
    validation=read(FIX/'FINAL_VALIDATION.json')
    assert validation['status']=='SHOPPING_NATIVE_ENV_READY' and all(validation['gates'].values())
    for p,h in validation['artifacts_sha256'].items():assert sha(FIX/p)==h,('FIX_ARTIFACT_CHANGED',p)
    if catalog:
        for p,h in read(FIX/'launch_manifest.json')['identities']['catalog_and_index_sha256'].items():
            assert sha(p)==h,('CATALOG_INDEX_CHANGED',p)
    snapshot=read(BATCH/'frozen_config_snapshot.json')
    model=model_config();model.pop('key_file')
    assert model==snapshot['answer_agent']==snapshot['memory_teacher']
    assert policy.READ_PROMPT==snapshot['memory_read_prompt']
    assert policy.CONSOLIDATE_PROMPT==snapshot['memory_consolidate_prompt']
    assert record_renderer.record_schema()==snapshot['record_schema']
    assert policy.action_schema(['RETRIEVE','ADD','UPDATE','DELETE','YIELD'],['EXAMPLE_VISIBLE_ID'])==snapshot['action_schema']
    cohort=read(BATCH/'pilot_manifest.json')['episodes']
    assert len(cohort)==12 and all(eid in {r['episode_id'] for r in cohort} for eid in IDS)
    if (HERE/'manifest.json').exists():
        locked=read(HERE/'manifest.json')
        assert locked['episode_ids']==IDS and locked['native_overlay_sha256']==sha(FIX/'validated_runtime_overlay.json')
        for p,h in locked['execution_hashes'].items():assert sha(p)==h,('EXECUTION_CHANGED',p)
    return identity


def prepare():
    assert not (HERE/'manifest.json').exists(),'REFUSE_MANIFEST_OVERWRITE'
    identity=verify(catalog=True)
    assert credential_ready(model_config()),'MISSING_CONFIGURED_DEEPSEEK_CREDENTIAL'
    public={p['episode_id']:p for p in rows(SHOP/'public_episodes.jsonl')}
    selected=[public[eid] for eid in IDS]
    for ep in selected:assert len(ep['sessions'])==6
    subtask=ShoppingTaskInputAdapter(max_observation_chars=16000).build_subtask_input({'state':'PUBLIC_INPUT_PLACEHOLDER'})
    save(HERE/'manifest.json',dict(run_id='shopping_natural_pilot4_v1',episode_ids=IDS,denominator=4,
        frozen_identity=identity,native_overlay_sha256=sha(FIX/'validated_runtime_overlay.json'),
        frozen_retrieval=candidate_config(),model={k:v for k,v in model_config().items() if k!='key_file'},
        answer_system_prompt=messages(subtask)[0]['content'],answer_runner=RUN_CONFIG.to_dict(),
        memory_read_prompt=policy.READ_PROMPT,memory_consolidate_prompt=policy.CONSOLIDATE_PROMPT,
        memory_decisions_per_session=32,answer_responses_per_session=30,native_max_rounds=60,
        lifecycle=['READ at session_start','Answer native interaction','CONSOLIDATE at session end'],
        change_from_prior_request='Explicit user instruction: complete six sessions after semantic failure; tier by total strict successes, not prefix.',
        completed_purchase_is_not_necessarily_correct=True,semantic_retries=0,infrastructure_retries=0,
        max_requests_per_episode=6*(32+30),token_budget_per_episode=100000000,
        budget_note='Episode cap only sums existing per-session limits; no new semantic attempts.',
        public_submission_adapter='Literal final public Shopping observation; no synthesized attributes.',
        existing_native_environment_reused_with_fresh_client_tracking=True,
        fresh_bank_per_episode=True,fresh_conversation_per_session=True,
        no_gold_history_replacement=True,positive_requires_observed_downstream_use=True,
        execution_hashes={str(p):sha(p) for p in sorted(HERE.glob('*.py'))}))
    save(HERE/'selected_public.json',selected)
    save(HERE/'preflight.json',dict(passed=True,api_requests=0,catalog_index_verified=True,identity=identity))
    print('PILOT4_PREPARED',flush=True)


if __name__=='__main__':prepare()
