"""Own and bound one full-catalog native preflight; never dispatch model APIs."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import requests
from .config import HERE,ARENA,PROJECT,verify,read,save,append,now,sha


def stop_owned(process):
    if process is None or process.poll() is not None:return
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)


def main():
    manifest,snapshot,_=verify()
    marker=HERE/'preflight/started.json'
    if marker.exists():raise RuntimeError('REFUSE_SECOND_NATIVE_PREFLIGHT_ATTEMPT')
    env=dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTHONPATH=str(PROJECT.parents[1])+os.pathsep+str(ARENA),
        MEMORYARENA_WEBSHOP_DATA_ROOT=str(ARENA/'data/shopping'),
        MEMORYARENA_WEBSHOP_ITEMS_FILE=str(ARENA/'data/shopping/items_shuffle.json'),
        MEMORYARENA_WEBSHOP_SEARCH_ROOT=str(ARENA/'data/shopping/search_engine'))
    server_code=("import faulthandler; faulthandler.enable(); "
        "faulthandler.dump_traceback_later(45, repeat=True); "
        "from env.env_systems.web_shopping_env.runtime.service import launch_lite as native; "
        "native.webshop_env_server_lite.limit_goals=1; "
        "native.uvicorn.run(native.app, host='127.0.0.1', port=36036)")
    command=[sys.executable,'-u','-c',server_code]
    transport=requests.Session();transport.trust_env=False
    try:
        existing=transport.get('http://127.0.0.1:36036/',timeout=1)
    except requests.RequestException:pass
    else:raise RuntimeError('PREFLIGHT_PORT_ALREADY_OCCUPIED; do not touch another service')
    save('preflight/started.json',dict(time=now(),preflight_episode_id=manifest['preflight_episode_id'],
        command=command,service_cwd=str(ARENA),worker_module=__package__+'.native_worker',
        limits=snapshot['preflight_limits'],model_api_requests=0,
        orchestration_hashes={p.name:sha(p) for p in HERE.glob('*.py')},
        loopback_only=True,original_native_server_class=True,loader_code_modified=False))
    service=worker=None;reason=None;peak_kib=0;started=time.monotonic();sampled=0
    service_log=(HERE/'preflight/native_service.log').open('w')
    worker_log=(HERE/'preflight/native_worker.log').open('w')
    try:
        service=subprocess.Popen(command,cwd=ARENA,env=env,stdout=service_log,stderr=subprocess.STDOUT,start_new_session=True)
        save('preflight/processes.json',dict(service_pid=service.pid,owned_by_this_preflight=True))
        for _ in range(60):
            if service.poll() is not None:raise RuntimeError('NATIVE_SERVICE_EXITED_BEFORE_HEALTH_CHECK')
            try:
                r=transport.get('http://127.0.0.1:36036/',timeout=1)
                if r.status_code==200:break
            except requests.RequestException:pass
            time.sleep(1)
        else:raise RuntimeError('NATIVE_SERVICE_HEALTH_TIMEOUT')
        worker=subprocess.Popen([sys.executable,'-u','-m',__package__+'.native_worker'],cwd=PROJECT,
            env=env,stdout=worker_log,stderr=subprocess.STDOUT,start_new_session=True)
        save('preflight/processes.json',dict(service_pid=service.pid,worker_pid=worker.pid,owned_by_this_preflight=True))
        while worker.poll() is None:
            elapsed=time.monotonic()-started
            result=subprocess.run(['/bin/ps','-o','rss=','-p',str(service.pid)],capture_output=True,text=True)
            rss=int(result.stdout.strip()) if result.returncode==0 and result.stdout.strip().isdigit() else None
            if rss is not None:peak_kib=max(peak_kib,rss)
            if elapsed-sampled>=10:
                append('preflight/resource_samples.jsonl',dict(time=now(),elapsed_seconds=elapsed,service_pid=service.pid,rss_kib=rss))
                sampled=elapsed
            if rss and rss>snapshot['preflight_limits']['service_rss_limit_gib']*1024**2:
                reason='NATIVE_INITIALIZATION_RESOURCE_GUARD_RSS_24_GIB';break
            if elapsed>snapshot['preflight_limits']['overall_timeout_seconds']:
                reason='NATIVE_PREFLIGHT_OVERALL_TIMEOUT';break
            if service.poll() is not None:
                reason='NATIVE_SERVICE_EXITED';break
            time.sleep(1)
        worker_result=read(HERE/'preflight/worker_result.json') if (HERE/'preflight/worker_result.json').exists() else None
        if reason is None and worker_result and worker_result['status']=='SHOPPING_ENVIRONMENT_READY':
            status='SHOPPING_ENVIRONMENT_READY'
        else:
            status='SHOPPING_ENVIRONMENT_BLOCKED'
            reason=reason or (worker_result or {}).get('error_type') or 'NATIVE_WORKER_FAILED'
    except Exception as error:
        status='SHOPPING_ENVIRONMENT_BLOCKED';reason=type(error).__name__+': '+str(error)
    finally:
        # Cleanup only the two process groups created here, never other services.
        stop_owned(worker);stop_owned(service)
        service_log.close();worker_log.close();transport.close()
        _,_,checks=verify()
        save('preflight/result.json',dict(status=status,reason=reason,elapsed_seconds=time.monotonic()-started,
            peak_service_rss_gib=peak_kib/1024**2,model_api_requests=0,api_cost=0,
            owned_service_stopped=service is None or service.poll() is not None,
            owned_worker_stopped=worker is None or worker.poll() is not None,
            worker_result=read(HERE/'preflight/worker_result.json') if (HERE/'preflight/worker_result.json').exists() else None,
            last_worker_stage=read(HERE/'preflight/worker_stage.json') if (HERE/'preflight/worker_stage.json').exists() else None,
            frozen_source_checks_passed=all(checks.values()),frozen_source_file_count=len(checks),
            original_native_loader_modified=False))
    print(read(HERE/'preflight/result.json')['status'],flush=True)


if __name__=='__main__':main()
