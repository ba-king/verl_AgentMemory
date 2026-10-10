"""Read and lock existing policy/config artifacts, without model API access."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
WORKSPACE=PROJECT.parents[2]
ARENA=WORKSPACE/'MemoryArena'
SHOP=PROJECT/'docs/memoryarena_synthetic_data/shopping_v1'
CANDIDATE=PROJECT/'data_generation_candidate/retrieval_config.json'
TRAVEL=PROJECT/'teacher_pilot/travel_v0'
REQUEST=Path('/Users/lizizheng/.codex/attachments/35dee927-497c-4346-9547-f0a259afc15f/pasted-text.txt')


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,value):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def append(name,value):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()


def candidate_config():
    cfg=read(CANDIDATE)
    assert cfg['final_top_k']==5 and cfg['final_top_k_frozen'] is True
    assert cfg['retriever']==dict(sparse_top_n=20,dense_top_n=20,top_k=5,rrf_constant=60,rrf_weights=[1,1])
    return cfg


def prepare():
    if (HERE/'pilot_manifest.json').exists():raise RuntimeError('REFUSE_MANIFEST_OVERWRITE')
    from ...travel_v0.structured_record import policy,record_renderer
    from ...travel_v0.provider import model_config
    cfg=candidate_config()
    publics=rows(SHOP/'public_episodes.jsonl')
    hidden={h['episode_id']:h for h in rows(SHOP/'hidden_episodes.jsonl')}
    validation=read(SHOP/'validation_stats.json')
    assert validation['episodes']==12 and all(v['valid'] for v in validation['validation'])
    selected=[]
    for template in ['material_cumulative','mixed_cumulative']:
        group=sorted([p for p in publics if hidden[p['episode_id']]['template_id']==template],key=lambda p:p['episode_id'])[:6]
        assert len(group)==6
        for p in group:
            assert len(p['sessions'])==6
            selected.append(dict(episode_id=p['episode_id'],template_type=template,session_count=6,
                source_file=str(SHOP/'public_episodes.jsonl'),source_sha256=sha(SHOP/'public_episodes.jsonl')))
    files=[*TRAVEL.rglob('*.py'),*PROJECT.joinpath('memory_agent').rglob('*.py'),
           *PROJECT.joinpath('answer_agent').rglob('*.py'),
           *PROJECT.joinpath('adapters/memoryarena/shopping').rglob('*.py'),
           *ARENA.joinpath('env/env_systems/web_shopping_env').rglob('*.py'),
           *PROJECT.joinpath('audits/stage3_reward_feasibility').glob('*.py'),
           CANDIDATE,SHOP/'public_episodes.jsonl',SHOP/'hidden_episodes.jsonl',SHOP/'source_products.jsonl',
           SHOP/'validation_stats.json',
           PROJECT/'docs/memoryarena_synthetic_data/acceptance_v1/shopping_constraints.py']
    hashes={str(p):sha(p) for p in sorted(set(files)) if '__pycache__' not in p.parts}
    # Model configuration only; do not read the key file or construct API clients.
    model=model_config();assert model['model']=='deepseek-v4-pro' and model['temperature']==0 and model['thinking']=={'type':'disabled'}
    model.pop('key_file',None)
    frozen=dict(created_utc=now(),candidate_config_path=str(CANDIDATE),retrieval_config=cfg,
        retrieval_config_sha256=sha(CANDIDATE),memory_teacher=model,answer_agent=dict(model),
        memory_read_prompt=policy.READ_PROMPT,memory_consolidate_prompt=policy.CONSOLIDATE_PROMPT,
        record_schema=record_renderer.record_schema(),action_schema=policy.action_schema(['RETRIEVE','ADD','UPDATE','DELETE','YIELD'],['EXAMPLE_VISIBLE_ID']),
        packet=False,reranker=False,architecture_changes=False,prompt_changes=False,
        scheduler_contract=['READ','Answer native WebShop interaction','actual purchase','strict validation','CONSOLIDATE','session reset; Bank only persists'],
        first_strict_failure_stops_suffix=True,one_attempt_per_episode=True,online_tuning=False,
        runtime_sources=hashes,
        preflight_limits=dict(native_create_timeout_seconds=300,overall_timeout_seconds=600,service_rss_limit_gib=24),
        note='Preflight-only preparation. No paid session runner is declared ready until the environment gate passes.')
    save('frozen_config_snapshot.json',frozen)
    config_hash=sha(HERE/'frozen_config_snapshot.json')
    for e in selected:e.update(config_hash=config_hash,retrieval_config_hash=sha(CANDIDATE))
    save('pilot_manifest.json',dict(run_id='shopping_natural_trajectory_batch_v1',created_utc=now(),
        selection_rule='For each template, ascending episode_id, first 6; locked before any model request',
        execution_condition='Native environment preflight must pass; otherwise all 12 remain NOT_RUN',
        episodes=selected,preflight_episode_id=selected[0]['episode_id'],
        request_file=str(REQUEST),request_sha256=sha(REQUEST),api_requests_at_lock=0))
    save('preflight/public_episode.json',next(p for p in publics if p['episode_id']==selected[0]['episode_id']))
    save('preflight/private_episode.json',hidden[selected[0]['episode_id']])
    save('preflight/launch_policy.json',dict(owned_service_port=36036,num_products=None,
        items_file=str(ARENA/'data/shopping/items_shuffle.json'),search_index=str(ARENA/'data/shopping/search_engine/indexes-full'),
        native_module='env.env_systems.web_shopping_env.runtime.service.launch_lite',
        human_goals=0,limit_goals=1,show_attrs=False,models_called=False,
        no_simplified_environment=True,no_catalog_reduction=True,no_oracle_purchase_substitution=True,
        preflight_actions='Public first-listed candidate; actual search/click/Features/Description/Buy Now. This is an environment smoke, not Teacher performance or strict-correct trajectory.'))
    print(json.dumps({'prepared_episodes':len(selected),'preflight_episode':selected[0]['episode_id'],'top_k':5,'api_requests':0}),flush=True)


def verify():
    manifest=read(HERE/'pilot_manifest.json');snapshot=read(HERE/'frozen_config_snapshot.json')
    assert all(e['config_hash']==sha(HERE/'frozen_config_snapshot.json') for e in manifest['episodes'])
    checks={p:sha(p)==h for p,h in snapshot['runtime_sources'].items()}
    assert all(checks.values()),[p for p,v in checks.items() if not v]
    assert candidate_config()==snapshot['retrieval_config']
    return manifest,snapshot,checks


if __name__=='__main__':prepare()
