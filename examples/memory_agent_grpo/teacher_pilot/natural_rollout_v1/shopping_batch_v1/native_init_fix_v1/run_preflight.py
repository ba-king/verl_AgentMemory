"""One owned localhost full-catalog check, gated by exact regression, zero models."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import requests

from ..config import HERE as BATCH, ARENA, PROJECT, CANDIDATE, read, sha, candidate_config
from ..preflight import stop_owned

HERE=Path(__file__).resolve().parent
NATIVE=ARENA/'env/env_systems/web_shopping_env/runtime/service/web_agent_site'
PATCHED=NATIVE/'envs/web_agent_text_env.py'
PORT=36037


def save(name,value):
    path=HERE/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def append(name,value):
    path=HERE/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')


def locked_inputs():
    snapshot=read(BATCH/'frozen_config_snapshot.json')
    checks={p:sha(p)==h for p,h in snapshot['runtime_sources'].items()}
    assert [p for p,ok in checks.items() if not ok]==[str(PATCHED)]
    manifest=read(BATCH/'pilot_manifest.json')
    assert all(e['config_hash']==sha(BATCH/'frozen_config_snapshot.json') for e in manifest['episodes'])
    assert candidate_config()==snapshot['retrieval_config']
    assert len(manifest['episodes'])==12
    # Prior batch artifacts remain immutable; record an explicit source overlay for this fix.
    old_validation=read(BATCH/'FINAL_VALIDATION.json')
    assert all(sha(BATCH/p)==h for p,h in old_validation['outputs_sha256'].items())
    return dict(unchanged_frozen_sources=sum(checks.values()),
        authorized_native_source_change=str(PATCHED),
        retrieval_config_sha256=sha(CANDIDATE),pilot_manifest_sha256=sha(BATCH/'pilot_manifest.json'),
        frozen_config_snapshot_sha256=sha(BATCH/'frozen_config_snapshot.json'),
        patch_source_hashes={str(p):sha(p) for p in [PATCHED,NATIVE/'engine/compact_goals.py',NATIVE/'engine/init_profile.py']})


def prepare():
    assert read(HERE/'equivalence.json')['status']=='PASS'
    identities=locked_inputs()
    assert not (HERE/'launch_manifest.json').exists(), 'REFUSE_PREFLIGHT_PREPARATION_OVERWRITE'
    # Full file digests; no reduced catalog or alternate index is allowed.
    root=ARENA/'data/shopping'
    paths=[root/'items_shuffle.json',*sorted((root/'search_engine/indexes-full').rglob('*'))]
    identities['catalog_and_index_sha256']={str(p):sha(p) for p in paths if p.is_file()}
    from ..retriever import CandidateHybridAdapter
    adapter=CandidateHybridAdapter(HERE/'embedding_cache')
    runtime=dict(config=adapter.backend.config,embedding_identity=adapter.encoder.identity,
                 statistics=adapter.encoder.stats,api_requests=0)
    assert runtime['embedding_identity']==read(BATCH/'preflight/retriever_runtime.json')['actual_embedding_identity']
    save('retriever_identity.json',runtime)
    save('launch_manifest.json',dict(identities=identities,num_products=None,human_goals=0,limit_goals=1,
        full_catalog=True,port=PORT,create_timeout_seconds=300,overall_timeout_seconds=600,
        max_service_rss_gib=24,api_requests=0,allowed_http='localhost native WebShop only',
        code_hashes={p.name:sha(p) for p in HERE.glob('*.py')}))
    save('preflight/public_episode.json',read(BATCH/'preflight/public_episode.json'))
    save('preflight/private_episode.json',read(BATCH/'preflight/private_episode.json'))


def service():
    import faulthandler
    import random
    from env.env_systems.web_shopping_env.runtime.service import launch_lite as native
    from env.env_systems.web_shopping_env.runtime.service.web_agent_site.engine.init_profile import phase
    faulthandler.enable();faulthandler.dump_traceback_later(45,repeat=True)
    original=native.webshop_env_server_lite.create
    def create():
        started=time.monotonic()
        with phase('native.create'):
            idx=original()
        env=native.webshop_env_server_lite.env[idx]
        goals=env.server.goals
        save('preflight/create_result.json',dict(create_pass=True,elapsed_seconds=time.monotonic()-started,
            loaded_product_count=len(env.server.all_products),selected_goals=goals,
            selected_goals_sha256=hashlib.sha256(json.dumps(goals,sort_keys=True).encode()).hexdigest(),
            final_rng_state_sha256=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            env_id=idx,selected_session=env.session,api_requests=0))
        return idx
    original_step=native.webshop_env_server_lite.step
    def step(idx,action):
        result=original_step(idx,action)
        env=native.webshop_env_server_lite.env[idx]
        state=env.server.user_sessions[env.session]
        append('preflight/native_steps.jsonl',dict(action=action,reward=result[1],done=result[2],
            session=env.session,asin=state.get('asin'),options=state.get('options'),
            state_done=state.get('done'),state_reward=state.get('reward'),
            verbose_info=state.get('verbose_info')))
        return result
    native.webshop_env_server_lite.create=create
    native.webshop_env_server_lite.step=step
    native.webshop_env_server_lite.limit_goals=1
    native.uvicorn.run(native.app,host='127.0.0.1',port=PORT)


def worker():
    # Reuse the exact prior zero-model public controller, with a new output directory/port.
    # A constructor wrapper only redirects the localhost port, never goal/action selection.
    from .. import native_worker as original
    from env.env_systems.web_shopping_env import webshop_plus_client
    Client=webshop_plus_client.WebshopPlusEnvClient
    def client(base,*args,**kwargs):
        return Client(f'http://127.0.0.1:{PORT}',*args,**kwargs)
    webshop_plus_client.WebshopPlusEnvClient=client
    original.HERE=HERE;original.save=save;original.append=append
    original.main()


def run():
    launch=read(HERE/'launch_manifest.json')
    assert locked_inputs()=={k:v for k,v in launch['identities'].items() if k!='catalog_and_index_sha256'}
    assert all(sha(HERE/p)==h for p,h in launch['code_hashes'].items())
    assert not (HERE/'preflight/started.json').exists(), 'REFUSE_SECOND_ATTEMPT'
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        PYTHONPATH=str(PROJECT.parents[1])+os.pathsep+str(ARENA),
        MEMORYARENA_WEBSHOP_DATA_ROOT=str(ARENA/'data/shopping'),
        MEMORYARENA_WEBSHOP_ITEMS_FILE=str(ARENA/'data/shopping/items_shuffle.json'),
        MEMORYARENA_WEBSHOP_SEARCH_ROOT=str(ARENA/'data/shopping/search_engine'),
        MEMORYARENA_INIT_PROFILE=str(HERE/'preflight/phases.jsonl'))
    transport=requests.Session();transport.trust_env=False
    try: transport.get(f'http://127.0.0.1:{PORT}/',timeout=1)
    except requests.RequestException: pass
    else: raise RuntimeError('PORT_OCCUPIED')
    save('preflight/started.json',dict(time=time.time(),launch_hash=sha(HERE/'launch_manifest.json')))
    service_proc=worker_proc=None;reason=None;peak=0;start=time.monotonic()
    service_log=(HERE/'preflight/service.log').open('w');worker_log=(HERE/'preflight/worker.log').open('w')
    try:
        command=[sys.executable,'-u','-m',__package__+'.run_preflight']
        service_proc=subprocess.Popen(command+['service'],cwd=PROJECT,env=env,stdout=service_log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(60):
            if service_proc.poll() is not None:raise RuntimeError('SERVICE_EXIT_BEFORE_HEALTH')
            try:
                if transport.get(f'http://127.0.0.1:{PORT}/',timeout=1).status_code==200:break
            except requests.RequestException:pass
            time.sleep(1)
        else:raise RuntimeError('SERVICE_HEALTH_TIMEOUT')
        worker_proc=subprocess.Popen(command+['worker'],cwd=PROJECT,env=env,stdout=worker_log,stderr=subprocess.STDOUT,start_new_session=True)
        save('preflight/processes.json',dict(service_pid=service_proc.pid,worker_pid=worker_proc.pid))
        tick=0
        while worker_proc.poll() is None:
            sample=subprocess.run(['/bin/ps','-o','rss=','-p',str(service_proc.pid)],capture_output=True,text=True)
            rss=int(sample.stdout.strip()) if sample.stdout.strip().isdigit() else 0
            peak=max(peak,rss)
            if tick%10==0:append('preflight/rss.jsonl',dict(elapsed_seconds=time.monotonic()-start,rss_kib=rss))
            if rss>24*1024**2:reason='RSS_GUARD_24_GIB';break
            if time.monotonic()-start>600:reason='OVERALL_TIMEOUT';break
            if service_proc.poll() is not None:reason='NATIVE_SERVICE_EXITED';break
            tick+=1;time.sleep(1)
        result=read(HERE/'preflight/worker_result.json') if (HERE/'preflight/worker_result.json').exists() else {}
        passed=reason is None and result.get('status')=='SHOPPING_ENVIRONMENT_READY'
        reason=reason or (None if passed else result.get('error_type','WORKER_FAILED'))
    except Exception as error:
        passed=False;reason=repr(error)
    finally:
        stop_owned(worker_proc);stop_owned(service_proc)
        service_log.close();worker_log.close();transport.close()
    profile_path=HERE/'preflight/phases.jsonl'
    phases=[json.loads(s) for s in profile_path.read_text().splitlines()] if profile_path.exists() else []
    goals=next((p['generated_goal_count'] for p in phases if p['phase']=='synthetic.metadata.complete'),None)
    save('preflight/result.json',dict(status='SHOPPING_NATIVE_ENV_READY' if passed else 'SHOPPING_NATIVE_INIT_FIX_FAILED',
        failure_stage=None if passed else reason,elapsed_seconds=time.monotonic()-start,
        peak_sampled_service_rss_gib=peak/1024**2,
        peak_instrumented_rss_gib=max((p['peak_rss_bytes'] for p in phases),default=0)/1024**3,
        generated_goal_count=goals,api_requests=0,api_cost=0,teacher_episodes=0,
        service_stopped=service_proc is None or service_proc.poll() is not None,
        worker_stopped=worker_proc is None or worker_proc.poll() is not None,
        frozen_identities_after=locked_inputs()))
    print(json.dumps(read(HERE/'preflight/result.json')),flush=True)


if __name__=='__main__':
    globals()[sys.argv[1]]()
