"""Direct V3 tool output, incremental public inputs, immutable tapes, no repair."""
import copy
import hashlib
import json
from .common import save,digest,HERE
from . import audit
from .hardening.shopping import build
from ..packet_interface_v2_validation.transport import PacketClient as PriorPacketClient
from ..packet_interface_v2_validation.interface import messages as frozen_messages
from ..packet_interface_v3_minimal_validation import interface_v3 as interface
from ..packet_interface_v3_minimal_validation.trusted_registry import build as registry
from ....travel_v0.artifacts import append


class PacketClient(PriorPacketClient):
    def request(self,inp):
        return {**{k:self.config[k] for k in ('model','temperature','max_tokens','thinking','stream')},
                'messages':frozen_messages(inp),'tools':interface.TOOLS,
                'tool_choice':{'type':'function','function':{'name':'emit_packet_delta'}},'parallel_tool_calls':False}


def retrieved(memory):
    out=[];seen=set();receipts=[]
    for row in memory.decisions:
        if not row.get('answer_memory_evidence_delivered'):continue
        result=row['retrieval_result'];assert result['final_k']==5
        receipts.append(copy.deepcopy(result))
        for hit in row['answer_memory_evidence_delivered']['hits']:
            version=hashlib.sha256(hit['content'].encode()).hexdigest();ref=hit['memory_id']+'@'+version[:12]
            if ref not in seen:
                out.append({'memory_id':hit['memory_id'],'version_sha256':version,'source_ref':ref,'content':hit['content']});seen.add(ref)
    return out,receipts


def public_input(public,previous,trusted):
    events=public['public_prefix_events_only'];event=events[-1] if events else public['initial_observation']
    # Public reference labels are Harness input metadata, not task requirements,
    # audit verdicts or added Packet fields. No previous raw observations replay.
    subjects={k:{a:v for a,v in value.items() if a in {'name','visible_asins'}}
              for k,value in trusted['subject_registry'].items()}
    return {'mode':'PACKET','operation':'UPDATE_PACKET' if events else 'INIT_PACKET',
        'current_task':public['current_task'],'previous_packet':copy.deepcopy(previous),
        'latest_action':copy.deepcopy(event.get('action')),
        'latest_observation':{'event_id':event['event_id'],'observation':event['observation'],
                              'public_subject_registry':subjects},
        'retrieved_memories':copy.deepcopy(public['actual_retrieved_memories'] if not events else [])}


class PacketSession:
    def __init__(self,episode,session,current,memory,budget,dest,histories):
        self.eid=episode;self.sid=session;self.dest=dest;self.histories=histories
        self.memories,self.retrieval_receipts=retrieved(memory)
        self.public={'current_task':current['instruction'],
            'initial_observation':{'event_id':'e000','observation':current['initial_observation']},
            'actual_retrieved_memories':self.memories,'public_prefix_events_only':[]}
        self.contract=build(current['instruction']);self.packet=interface.empty_packet()
        self.client=PacketClient(budget);self.last_cutoff=None;self.rows=[];self.failed=False
        save(dest/'packet/retrieval_receipts.json',self.retrieval_receipts)

    def event(self,event):
        # Called only after a real native execution. Terminal purchase is not a
        # future Packet input: no Packet call is made after the session ends.
        self.public['public_prefix_events_only'].append({'event_id':f'e{len(self.public["public_prefix_events_only"])+1:03d}',
            'action':copy.deepcopy(event['action']),'observation':event['observation']})

    def update(self):
        if self.failed:raise RuntimeError('PACKET_STREAM_ALREADY_STOPPED')
        cutoff=len(self.public['public_prefix_events_only'])
        if self.last_cutoff==cutoff:return self.packet
        trusted=registry(copy.deepcopy(self.public),cutoff)
        inp=public_input(self.public,self.packet,trusted);dest=self.dest/'packet'/f'e{cutoff:03d}'
        row={'episode_id':self.eid,'session_id':self.sid,'event_id':f'e{cutoff:03d}',
            'mode':'PACKET','operation':inp['operation'],'input':inp,
            'previous_packet':copy.deepcopy(self.packet),'schema_valid':False,'wrapper_valid':False,
            'apply_valid':False,'http_failure':False,'malformed':False,'raw_response':None,
            'parsed_packet_delta':None,'applied_packet':None,'public_cutoff_hash':digest(self.public),
            'task_hash':digest(self.public['current_task']),'request_ref':str(dest/'api/request.json'),
            'response_ref':str(dest/'api/response.json'),'trusted_registry_ref':str(dest/'trusted_registry.json'),
            'manual_repair':False,'semantic_retries':0}
        save(dest/'public_prefix.json',self.public);save(dest/'trusted_registry.json',trusted)
        save(dest/'contract.json',self.contract)
        error=None
        try:
            body=self.client.call(inp,dest/'api');row['raw_response']=body;row['usage']=body.get('usage',{})
            call=body.get('choices',[{}])[0].get('message',{}).get('tool_calls',[{}])[0]
            try:row['wrapper_valid']=set(json.loads(call.get('function',{}).get('arguments','null')))=={'packet_delta'}
            except (ValueError,TypeError):pass
            delta=interface.parse(body);row.update(schema_valid=True,parsed_packet_delta=delta)
            new=interface.apply(self.packet,delta,trusted['source_ownership_registry'],init=cutoff==0)
            replay=interface.apply(copy.deepcopy(self.packet),copy.deepcopy(delta),trusted['source_ownership_registry'],init=cutoff==0)
            row.update(apply_valid=True,applied_packet=new,replay_equal=new==replay)
            self.packet=new
        except Exception as exc:
            error=type(exc).__name__+': '+str(exc);row['error']=error
            row['http_failure']=(dest/'api/error.json').exists()
            row['malformed']='MALFORMED_JSON' in error
            self.failed=True
        result=audit.evaluate(row,self.public,trusted,self.contract,self.histories)
        row['audit_ref']=str(dest/'audit.json');row['output_packet_hash']=digest(row['applied_packet'])
        save(dest/'audit.json',result);save(dest/'transition.json',row)
        append(HERE/'packet_transitions.jsonl',row)
        append(HERE/'packet_audit.jsonl',{'episode_id':self.eid,'session_id':self.sid,'event_id':row['event_id'],**result})
        candidate={'episode_id':self.eid,'session_id':self.sid,'event_id':row['event_id'],'mode':'PACKET',
            'input':inp,'target_delta':row['parsed_packet_delta'],'previous_packet_hash':digest(row['previous_packet']),
            'output_packet_hash':row['output_packet_hash'],'task_hash':row['task_hash'],'public_cutoff_hash':row['public_cutoff_hash'],
            'source_refs':list(trusted['source_ownership_registry']),'subject_refs':list(trusted['subject_registry']),
            'transition_subtypes':result['transition_subtypes'],'transition_local_verdict':result['transition_local_verdict'],
            'full_packet_verdict':result['full_packet_verdict'],'inherited_errors':result['inherited_errors'],
            'annotation_origin':result['annotation_origin'],'teacher_generated':True,'validator_status':result.get('validator_status','FAIL'),
            'review_status':result['review_status'],'validator_version':'bounded_public_v1','schema_version':'PACKET_V3_FROZEN',
            'reason_codes':result['reason_codes'],'coverage_reasons':result['coverage_reasons'],'audit_ref':row['audit_ref']}
        append(HERE/('packet_local_gold_candidates.jsonl' if result['gold_candidate'] else 'packet_rejected_candidates.jsonl'),candidate)
        self.rows.append(row);self.last_cutoff=cutoff
        if error:raise RuntimeError('PACKET_STREAM_STOP: '+error)
        return self.packet

    def close(self):self.client.transport.close()
