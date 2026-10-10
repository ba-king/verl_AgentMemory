"""All-archive regression and same-proxy static savings, before any API request."""
import copy
from collections import Counter,defaultdict
from .common import *
from .implementation.renderer import *
from ..agentmemory_api_cost_audit_v1.components import Proxy,prompt_proxy


CHECKS=('AUTHORITATIVE_TASK_PRESENT_EXACTLY_ONCE','NO_NON_TASK_FIELD_CHANGED','RESTORE_EQUIVALENCE',
        'TOOL_ARGS_UNCHANGED','RAW_RETRIEVAL_UNCHANGED','PACKET_UNCHANGED','MESSAGE_ORDER_UNCHANGED')


def protected_projection(body,kind):
    keys={'raw':{'retrieved_memories','new_retrieved_memories','visible_memories','visible_retrieved_memories'},
          'packet':{'previous_packet','packet','working_packet'},
          'arguments':{'arguments','tool_calls','tool_arguments'}}[kind]
    projection=[]
    for leaf in request_leaves(body):
        if any(k in keys for k in leaf.path):projection.append((leaf.message,leaf.path,leaf.value))
    for i,m in enumerate(body['messages']):
        if kind=='arguments' and m['role']=='assistant':projection.append((i,'assistant',m))
        if kind=='raw' and str(m.get('content','')).startswith('[Retrieved Memory]'):projection.append((i,'raw',m))
        if kind=='packet' and str(m.get('content','')).startswith('[Working Packet]'):projection.append((i,'packet',m))
    return projection


def check_one(original,rendered):
    new=rendered.body;r=rendered.receipt;task=r['authoritative_task'];checks={}
    checks['AUTHORITATIVE_TASK_PRESENT_EXACTLY_ONCE']=r['complete_task_occurrences_after']==1 if task else None
    checks['RESTORE_EQUIVALENCE']=restore_task_refs(rendered)==original
    checks['TOOL_ARGS_UNCHANGED']=original.get('tools')==new.get('tools') and protected_projection(original,'arguments')==protected_projection(new,'arguments')
    checks['RAW_RETRIEVAL_UNCHANGED']=protected_projection(original,'raw')==protected_projection(new,'raw')
    checks['PACKET_UNCHANGED']=protected_projection(original,'packet')==protected_projection(new,'packet')
    checks['MESSAGE_ORDER_UNCHANGED']=[{k:v for k,v in m.items() if k!='content'} for m in original['messages']]==[{k:v for k,v in m.items() if k!='content'} for m in new['messages']]
    # Independent edit audit: evidence prefix/suffix unchanged, only literal task removed.
    values_ok=True
    for edit in r['edits']:
        before,after=edit['original_value'],edit['rendered_value']
        if edit['reason']=='EXACT_TASK_FIELD':values_ok &= before==task and after==MARKER
        elif edit['reason']=='EXACT_NATIVE_INSTRUCTION_HEADER':
            match=HEADER.match(before)
            values_ok &= bool(match and match.group(1)==task and after==before[:match.start(1)]+MARKER+before[match.end(1):])
        else:values_ok=False
    old_meta={k:v for k,v in original.items() if k!='messages'};new_meta={k:v for k,v in new.items() if k!='messages'}
    checks['NO_NON_TASK_FIELD_CHANGED']=values_ok and old_meta==new_meta and all(
        original['messages'][i]==m for i,m in enumerate(new['messages']) if i not in {e['message'] for e in r['edits']})
    return checks


def run():
    forbid_network()
    if (HERE/'live_started.json').exists():raise RuntimeError('REFUSE_REGENERATING_AFTER_LIVE')
    records=rows(AUDIT/'request_level_usage.jsonl');assert len(records)>=4777
    proxy=Proxy();output=[];stats=Counter();failures=[];savings=[]
    episodes=read(AUDIT/'trajectory_cost_analysis.json')['episodes']
    focus={(e['experiment'],e['episode']) for e in episodes if e['is_full_trajectory'] and e['experiment'] in ('shopping_p_vs_rp_check','teacher_demo_package_rp_final')}
    for i,record in enumerate(records):
        body=read(record['path'])
        try:
            rendered=render_dedup(body,role=record['role']);checks=check_one(body,rendered)
            if record['role']=='ANSWER':assert checks['AUTHORITATIVE_TASK_PRESENT_EXACTLY_ONCE'] is True
            if checks['AUTHORITATIVE_TASK_PRESENT_EXACTLY_ONCE'] is None:assert record['role']=='HEALTH'
            passed=all(v is not False for v in checks.values());assert passed
            row={'request_id':record['request_id'],'request_ref':record['path'],'role':record['role'],
                 'checks':checks,'PASS':passed,'reference_count':rendered.receipt['task_reference_count'],
                 'original_sha256':rendered.receipt['original_sha256'],'rendered_sha256':rendered.receipt['rendered_sha256'],
                 'authoritative_location':rendered.receipt['authoritative_location']}
            old=prompt_proxy(body,proxy);new=prompt_proxy(rendered.body,proxy)
            ratio=(old-new)/old if old else 0
            savings.append({'request_id':record['request_id'],'experiment':record['experiment'],'episode':record['episode'],
                'role':record['role'],'old_proxy_tokens':old,'new_proxy_tokens':new,'proxy_reduction':old-new,
                'actual_prompt_tokens':record['prompt_tokens'],'actual_completion_tokens':record['completion_tokens'],
                'actual_total_tokens':record['total_tokens'],'estimated_prompt_saved':(record['prompt_tokens'] or 0)*ratio,
                'focus_complete_RP':(record['experiment'],record['episode']) in focus})
            stats[(record['role'],'changed' if body!=rendered.body else 'identity')]+=1
        except Exception as exc:
            row={'request_id':record['request_id'],'request_ref':record['path'],'role':record['role'],'PASS':False,'error':str(exc)}
            failures.append(row)
        output.append(row)
        if (i+1)%1200==0:print(f'ARCHIVE_CHECK {i+1}/{len(records)}',flush=True)
    summary={'REQUEST_ATTEMPTS_CHECKED':len(records),'PASS_REQUESTS':sum(r['PASS'] for r in output),
        'FAIL_REQUESTS':len(failures),'STATUS':'PASS' if not failures else 'FAIL',
        'checks':{k:{'PASS':sum(r.get('checks',{}).get(k) is True for r in output),
                     'FAIL':sum(r.get('checks',{}).get(k) is False for r in output),
                     'NOT_APPLICABLE':sum(k in r.get('checks',{}) and r['checks'][k] is None for r in output)} for k in CHECKS},
        'health_exception':'Only 10 task-free infrastructure HEALTH requests: identity rendering and restoration checked; authoritative-task invariant does not apply.',
        'ANSWER_REQUESTS':sum(r['role']=='ANSWER' for r in records),'NEW_API_CALLS':0,
        'by_role_transform':{role:{state:stats[(role,state)] for state in ('changed','identity')} for role in sorted({r['role'] for r in records})},
        'failure_examples':failures,'renderer_sha256':sha(Path(__file__).with_name('implementation').joinpath('renderer.py').read_bytes())}
    save('offline_renderer_regression.json',summary);jsonl('offline_request_checks.jsonl',output)
    jsonl('static_request_savings.jsonl',savings)
    per_episode=[]
    for exp,ep in sorted(focus):
        ss=[r for r in savings if (r['experiment'],r['episode'])==(exp,ep)]
        total=sum(r['actual_total_tokens'] or 0 for r in ss);saved=sum(r['estimated_prompt_saved'] for r in ss)
        per_episode.append({'experiment':exp,'episode':ep,'requests':len(ss),'old_proxy_tokens':sum(r['old_proxy_tokens'] for r in ss),
            'new_proxy_tokens':sum(r['new_proxy_tokens'] for r in ss),'actual_recorded_total_tokens':total,
            'estimated_prompt_tokens_saved':saved,'estimated_optimized_total_tokens':total-saved,
            'estimated_total_reduction_percent':100*saved/total,
            'by_role_saved':{role:sum(r['estimated_prompt_saved'] for r in ss if r['role']==role) for role in ('ANSWER','PACKET','READ','CONSOLIDATE')}})
    total=sum(r['actual_recorded_total_tokens'] for r in per_episode);saved=sum(r['estimated_prompt_tokens_saved'] for r in per_episode)
    save('static_token_savings.json',{'method':'Same local Qwen proxy as preceding audit. Per-request actual prompt × proxy fractional reduction; recorded completion held fixed.',
        'per_complete_RP_episode':per_episode,'baseline_total_tokens':total,'estimated_prompt_tokens_saved':saved,
        'estimated_total_reduction_percent':100*saved/total,'previous_audit_percent':25.83658251063989,
        'difference_explanation':'Original JSON whitespace is preserved outside elided strings; shorter canonical marker; exact instruction fields inside current_task.initial_observation are now also elided. No Packet/raw compression.',
        'behavioral_equivalence_not_implied':True,'scope':'All-role presentation transform; online gate below tests Answer only.',
        'answer_only_rollout_total_reduction_percent':100*sum(r['by_role_saved']['ANSWER'] for r in per_episode)/total})
    save('source_hashes.json',INPUT_HASHES)
    if failures:raise RuntimeError('OFFLINE_REGRESSION_FAILED_STOP_BEFORE_API')
    return summary


if __name__=='__main__':run()
