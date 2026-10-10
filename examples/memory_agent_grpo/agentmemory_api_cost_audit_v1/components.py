"""Lossless text partition, local Qwen proxy counts, and request variants.

Component texts partition every message content character. Metadata/tool JSON is
counted separately. Counts are not DeepSeek's proprietary chat/tool billable
tokenization and are never substituted for provider usage.
"""
import copy
import re
from functools import lru_cache
from collections import defaultdict
from tokenizers import Tokenizer
from .common import *

NAMES=('system_prompt','tool_schemas','global_task_static_instruction','conversation_action_history',
       'latest_observation','raw_retrieved_memories','working_packet','trusted_registry_metadata','other')


class Proxy:
    def __init__(self):
        self.tokenizer=Tokenizer.from_file(str(TOKENIZER))
        INPUTS[str(TOKENIZER)]=sha(TOKENIZER.read_bytes())

    @lru_cache(maxsize=12000)
    def count(self,text):
        return len(self.tokenizer.encode(text,add_special_tokens=False).ids) if text else 0


def payload(body):
    for m in reversed(body.get('messages',[])):
        if m.get('role')!='user' or not isinstance(m.get('content'),str):continue
        try:x=json.loads(m['content'])
        except ValueError:continue
        if isinstance(x,dict):return x
    return {}


def task_for(record,body):
    x=payload(body);task=x.get('current_task',x.get('task'))
    if isinstance(task,str):return task
    if isinstance(task,dict) and isinstance(task.get('instruction'),str):return task['instruction']
    if record.get('session_summary_ref'):
        public=read(Path(record['session_summary_ref']).with_name('public_input.json'),{})
        if isinstance(public,dict) and isinstance(public.get('instruction'),str):return public['instruction']
    return None


def top_spans(s):
    """Return exact raw spans of top-level JSON members, including keys/separators."""
    d=json.JSONDecoder();i=len(s)-len(s.lstrip());out=[]
    if i>=len(s) or s[i]!='{':return None
    start=0;i+=1
    try:
        while True:
            while i<len(s) and s[i].isspace():i+=1
            if s[i]=='}':
                if out:out[-1]=(out[-1][0],out[-1][1]+s[start:])
                else:out.append((None,s))
                break
            key,end=d.raw_decode(s,i);i=end
            while s[i].isspace():i+=1
            if s[i]!=':':return None
            i+=1
            while s[i].isspace():i+=1
            _,i=d.raw_decode(s,i)
            while i<len(s) and s[i].isspace():i+=1
            if s[i]==',':i+=1
            out.append((key,s[start:i]));start=i
        assert ''.join(v for _,v in out)==s
        return out
    except (ValueError,IndexError,AssertionError):return None


def component_key(k):
    if k in ('current_task','task','subtask_input','instruction'):return 'global_task_static_instruction'
    if k in ('previous_packet','packet','working_packet','working_state'):return 'working_packet'
    if k in ('latest_observation','initial_observation','current_observation'):return 'latest_observation'
    if k in ('retrieved_memories','new_retrieved_memories','retrieved_memory'):return 'raw_retrieved_memories'
    if k in ('decision_schema','schema'):return 'tool_schemas'
    if k in ('session_public_history','answer_public_history','history','latest_action','latest_answer_action','public_history'):return 'conversation_action_history'
    if k and any(t in k for t in ('registry','scaffold','metadata','receipt','budget','visible_memory','available_actions','opportunity','mode','operation')):return 'trusted_registry_metadata'
    return 'other'


def partition(body,record,proxy):
    task=task_for(record,body);pieces=[]
    def add(cat,text,kind,index=None):
        pieces.append({'component':cat,'kind':kind,'text':text,'message_index':index})
    def add_with_task(cat,text,kind,index,escaped=False):
        needle=json.dumps(task,ensure_ascii=False)[1:-1] if escaped and task else task
        if needle and needle in text and cat in ('latest_observation','conversation_action_history'):
            ss=text.split(needle)
            for j,s in enumerate(ss):
                if s:add(cat,s,kind,index)
                if j<len(ss)-1:add('global_task_static_instruction',needle,'repeated_task_in_observation',index)
        else:add(cat,text,kind,index)
    messages=body.get('messages',[]);last_tool=max((i for i,m in enumerate(messages) if m['role']=='tool'),default=-1)
    content_chars=0
    for i,m in enumerate(messages):
        text=m.get('content');text=text if isinstance(text,str) else compact(text) if text is not None else ''
        content_chars+=len(text)
        if m['role']=='system':add('system_prompt',text,'system',i)
        elif text.startswith('[Working Packet]'):add('working_packet',text,'packet_message',i)
        elif text.startswith('[Retrieved Memory]'):add('raw_retrieved_memories',text,'raw_memory_message',i)
        elif m['role']=='tool':add_with_task('latest_observation' if i==last_tool else 'conversation_action_history',text,'tool_observation',i)
        elif m['role']=='assistant':add('conversation_action_history',text,'assistant_history',i)
        elif m['role']=='user' and record['role'] in ('PACKET','READ','CONSOLIDATE','READ_CONSOLIDATE') and top_spans(text):
            for key,value in top_spans(text):add_with_task(component_key(key),value,'json_member:'+str(key),i,escaped=True)
        elif '<answer_agent_subtask_json>' in text or m['role']=='user' and i==1 and record['role']=='ANSWER':
            add('global_task_static_instruction',text,'initial_task_message',i)
        else:add('other',text,'other_message',i)
        meta={k:v for k,v in m.items() if k!='content'}
        # Message role/tool-call IDs/arguments are explicit input overhead.
        add('conversation_action_history' if m['role'] in ('tool','assistant') else 'other',compact(meta),'message_metadata',i)
    if body.get('tools'):add('tool_schemas',compact(body['tools']),'tools')
    if body.get('response_format'):add('tool_schemas',compact(body['response_format']),'response_format')
    assert sum(len(p['text']) for p in pieces if p['kind'] not in ('message_metadata','tools','response_format'))==content_chars
    totals={k:0 for k in NAMES}
    for p in pieces:
        p['proxy_tokens']=proxy.count(p['text']);p['hash']=sha(p['text']);totals[p['component']]+=p['proxy_tokens']
    full_proxy=proxy.count(compact({'messages':body.get('messages',[]),**{k:body[k] for k in ('tools','response_format') if k in body}}))
    return totals,pieces,task,full_proxy


def packet_output(record,body):
    if record.get('packet_transition_ref'):
        t=read(record['packet_transition_ref']);before=t.get('previous_packet');after=t.get('applied_packet')
        if t.get('infrastructure_failure') or after is None:
            return {'status':'NO_VALID_DELTA','effective_empty':None,'delta':None,'before':before,'after':after}
        return {'status':'VALID_APPLIED','effective_empty':before==after,'delta':t.get('parsed_packet_delta'),
                'before':before,'after':after}
    response=read(record.get('response_ref',''),{}) if record.get('response_ref') else {}
    try:
        msg=response['choices'][0]['message'];tool=msg.get('tool_calls',[])
        parsed=json.loads(tool[0]['function']['arguments'] if tool else msg.get('content',''))
        delta=parsed.get('packet_delta',parsed);before=payload(body).get('previous_packet')
        # Older INIT/full rewrite responses are not certified as delta no-ops.
        if isinstance(delta,dict) and any(k in delta for k in ('add_evidence','resolve','set_ready_to_commit')):
            changes=any(v not in ([],None) for k,v in delta.items() if k not in ('set_ready_to_commit','set_unresolved'))
            ready=delta.get('set_ready_to_commit');unresolved=delta.get('set_unresolved')
            if before is None:return {'status':'PARSED_WITHOUT_PRIOR','effective_empty':None,'delta':delta,'before':None,'after':None}
            changes |= ready not in (None,'unchanged','keep') and ready not in (before.get('ready_to_commit'),str(before.get('ready_to_commit')).lower())
            changes |= unresolved is not None and unresolved!=before.get('unresolved')
            return {'status':'PARSED_NOT_REPLAYED','effective_empty':not changes,'delta':delta,'before':before,'after':None}
    except (ValueError,KeyError,IndexError,TypeError):pass
    return {'status':'NO_APPLIED_DELTA_TAPE','effective_empty':None,'delta':None,'before':None,'after':None}


def lossless_task_dedup(body,task):
    """Candidate offline rendering: retain authoritative task; elide exact copies."""
    if not task:return copy.deepcopy(body),0
    out=copy.deepcopy(body);count=0
    for m in out.get('messages',[]):
        content=m.get('content')
        if not isinstance(content,str):continue
        if m['role']=='tool' and task in content:
            n=content.count(task);m['content']=content.replace(task,'[Task unchanged; see current task.]');count+=n
        elif m['role']=='user':
            try:x=json.loads(content)
            except ValueError:continue
            if not isinstance(x,dict) or 'current_task' not in x:continue
            def walk(value,protect=False):
                nonlocal count
                if isinstance(value,str) and not protect and task in value:
                    count+=value.count(task);return value.replace(task,'[Task unchanged; see current task.]')
                if isinstance(value,list):return [walk(v,protect) for v in value]
                if isinstance(value,dict):return {k:walk(v,protect or k in ('instruction','current_task') and isinstance(v,str)) for k,v in value.items()}
                return value
            # Memory/Packet/system/task fields remain byte-for-byte values.
            # Only duplicated task literals inside observations/history qualify.
            for key in ('latest_observation','initial_observation','current_observation',
                        'environment_observation','session_public_history','answer_public_history','public_history'):
                if key in x:x[key]=walk(x[key])
            m['content']=compact(x)
    # No task elision unless a verbatim authoritative task survives somewhere.
    if count and task not in '\n'.join(str(m.get('content','')) for m in out['messages']):
        # JSON encoding means the surviving raw instruction is escaped.
        if json.dumps(task,ensure_ascii=False)[1:-1] not in '\n'.join(str(m.get('content','')) for m in out['messages']):
            return copy.deepcopy(body),0
    return (out,count) if count else (copy.deepcopy(body),0)


def packet_compact(body):
    out=copy.deepcopy(body)
    for m in out.get('messages',[]):
        s=m.get('content')
        if isinstance(s,str) and s.startswith('[Working Packet]\n'):
            try:m['content']='[Working Packet]\n'+compact(json.loads(s.split('\n',1)[1]))
            except ValueError:pass
        elif isinstance(s,str):
            try:x=json.loads(s)
            except ValueError:continue
            if isinstance(x,dict) and 'previous_packet' in x:m['content']=compact(x)
    return out


def prompt_proxy(body,proxy):
    total=0
    for m in body.get('messages',[]):
        content=m.get('content');total+=proxy.count(content if isinstance(content,str) else compact(content) if content is not None else '')
        total+=proxy.count(compact({k:v for k,v in m.items() if k!='content'}))
    for k in ('tools','response_format'):
        if k in body:total+=proxy.count(compact(body[k]))
    return total


def raw_compact(body):
    """Remove only blank lines and pretty JSON whitespace; keep fields/IDs/order."""
    out=copy.deepcopy(body)
    for m in out.get('messages',[]):
        s=m.get('content')
        if isinstance(s,str) and s.startswith('[Retrieved Memory]'):
            m['content']='\n'.join(l.strip() for l in s.splitlines() if l.strip())
    return out


def rp_overlap(body,record):
    raw='\n'.join(m.get('content','') for m in body['messages'] if str(m.get('content','')).startswith('[Retrieved Memory]'))
    pack='\n'.join(m.get('content','') for m in body['messages'] if str(m.get('content','')).startswith('[Working Packet]'))
    def grams(s):
        w=re.findall(r'\w+',s.lower());return {tuple(w[i:i+5]) for i in range(len(w)-4)}
    a,b=grams(raw),grams(pack)
    wire=read(Path(record['path']).parent.parent.parent/'rp_wiring'/f"{Path(record['path']).parent.name}.json",{})
    facts,duplicates=0,0
    def norm(s):return re.sub(r'[^a-z0-9.]','',str(s).lower())
    aliases={'remaining_budget':'remaining_budget_after_purchase','price':'price','material':'material','color':'color'}
    if wire:
        hits={h['memory_id']:h.get('record',{}) for r in wire.get('actual_read_receipts',[]) for h in r.get('hits',[])}
        for e in wire.get('packet_sent_to_answer',{}).get('evidence',[]):
            mem=[ref.split('@')[0] for ref in e.get('source_refs',[]) if ref.startswith('M')]
            if not mem or '=' not in e.get('fact',''):continue
            key,value=e['fact'].split('=',1)
            if key not in aliases:continue # Derived public mappings are not duplicate raw facts.
            facts+=1
            if any(aliases[key]==f.get('path') and norm(value)==norm(f.get('value'))
                   for mid in mem for f in hits.get(mid,{}).get('fields',[])):duplicates+=1
    return {'text_5gram_jaccard':len(a&b)/len(a|b) if a|b else None,
        'raw_5grams_in_packet':len(a&b)/len(a) if a else None,
        'packet_literal_memory_facts_checked':facts,'same_provenance_literal_field_values_repeated':duplicates}
