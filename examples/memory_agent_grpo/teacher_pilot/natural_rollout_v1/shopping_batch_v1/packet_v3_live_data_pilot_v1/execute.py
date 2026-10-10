"""One natural attempt per locked episode with frozen actors/native execution."""
import copy
import json
import time
from .common import HERE,PILOT,read,save,locked_episodes
from .packet_runtime import PacketSession
from .prepare import verify
from ..pilot4_v1 import boundary as original_boundary
from ..pilot4_v1.execute import RecordingAnswer as OriginalAnswer, infrastructure_error
from ..pilot4_v1.freeze import RUN_CONFIG,messages
from ..policy_labeler_v2_regression.policy import ShoppingMemorySessionV2
from ..retriever import CandidateHybridAdapter
from ....travel_v0.provider import Budget
from ....travel_v0.structured_record.bank import StructuredBank
from ....travel_v0.artifacts import rows,append
from examples.memory_agent_grpo.answer_agent.providers.deepseek import DeepSeekAnswerAgentClient
from examples.memory_agent_grpo.answer_agent.runtime.subtask_runner import AnswerAgentSubtaskRunner
from examples.memory_agent_grpo.adapters.memoryarena.shopping.tool_catalog import tool_definitions


class FreshBoundary(original_boundary.Boundary):
    def __init__(self,base_url):
        # Original reset/native/scorer implementation, fresh private task data
        # used ONLY by the environment/scorer. Never passed to Packet or actors.
        self.base_url=base_url;self.client=None;self.actual=[]
        self.hidden={e['episode_id']:e for e in rows(PILOT/'tasks/private/hidden_episodes.jsonl')}
        self.products={p['asin']:p for p in rows(PILOT/'tasks/private/source_products.jsonl')}


class Answer(OriginalAnswer):
    def __init__(self,dest,budget,packet):
        super().__init__(dest,budget,[]);self.packet=packet

    def generate(self,conversation,definitions):
        packet=self.packet.update()
        visible=DeepSeekAnswerAgentClient._validate_and_serialize_conversation(conversation)
        assert not any(m['role']=='user' and str(m.get('content','')).startswith('[Retrieved Memory]') for m in visible)
        visible.append({'role':'user','content':'[Working Packet]\n'+json.dumps(packet,ensure_ascii=False,separators=(',',':'))})
        tools=DeepSeekAnswerAgentClient._validate_and_serialize_tools(definitions)
        body,self.last_meta=self.client.call(visible,tools)
        save(self.dest/'answer_exact_conversation.json',visible)
        return DeepSeekAnswerAgentClient._parse_response(json.dumps(body).encode())


class Executor(original_boundary.RecordingExecutor):
    def execute(self,call):
        result=super().execute(call)
        self.answer.packet.event(self.events[-1])
        return result


def episode(public,retriever,boundary):
    eid=public['episode_id'];directory=HERE/'raw_rollouts'/eid
    if directory.exists():raise RuntimeError('REFUSE_SECOND_EPISODE_ATTEMPT')
    directory.mkdir(parents=True)
    summary={'episode_id':eid,'type':'mixed' if '_mixed_' in eid else 'material','sessions':[],
             'planned_sessions':6,'semantic_retries':0,'status':'RUNNING'}
    boundary.begin_episode(eid);bank=StructuredBank(retriever)
    budget=Budget(HERE/'budgets'/f'{eid}.json',552,100000000)
    histories={}
    for index,public_session in enumerate(public['sessions']):
        sid=public_session['session_id'];dest=directory/sid;dest.mkdir()
        session={'session_id':sid,'strict_validation':{'passed':False,'reason':'not_purchased'},
            'termination_reason':'not_started','memory_decisions':0,'answer_calls':0,'packet_calls':0,
            'infrastructure_invalid':False}
        memory=answer=packet=None
        try:
            verify()
            subtask=boundary.reset(public_session,dest,index)
            current={'session_id':sid,'instruction':public_session['instruction'],'initial_observation':subtask.subtask_input}
            conversation=messages(subtask)
            memory=ShoppingMemorySessionV2('packet_v3_live_data_pilot_v1',eid,current,dest,bank,budget,conversation)
            save(dest/'bank_before.json',bank.snapshot())
            memory.opportunity('session_start')
            packet=PacketSession(eid,sid,current,memory,budget,dest,copy.deepcopy(histories))
            answer=Answer(dest,budget,packet)
            executor=Executor(boundary,memory,answer,dest)
            runtime=AnswerAgentSubtaskRunner(client=answer,executor=executor,config=RUN_CONFIG)
            result=runtime.run(subtask,tool_definitions=tool_definitions())
            save(dest/'answer_runner_result.json',result.to_dict())
            session['answer_runner_status']=result.status.value
            session['answer_error_code']=result.error_code.value if result.error_code else None
            session['termination_reason']='purchase_completed' if boundary.client.purchased_asins else 'answer_ended_without_purchase'
            if packet.failed:session['termination_reason']='packet_stream_stopped_no_retry'
            # Frozen consolidation consumes only real current public events.
            memory.opportunity('after_purchase' if boundary.client.purchased_asins else 'session_ended_without_purchase')
        except Exception as exc:
            session.update(error=type(exc).__name__+': '+str(exc),termination_reason='session_error',
                           infrastructure_invalid=infrastructure_error(exc))
            save(dest/'error.json',{'error':session['error'],'infrastructure':session['infrastructure_invalid']})
        finally:
            if boundary.client is not None and boundary.dest==dest:
                try:session.update(boundary.score())
                except Exception as exc:session.update(infrastructure_invalid=True,scoring_error=repr(exc))
            if memory:
                memory.finish(session);session['memory_decisions']=len(memory.decisions)
                for row in memory.decisions:append(dest/'memory_decisions.jsonl',row)
                memory.client.transport.close()
            if answer:session['answer_calls']=answer.client.count;answer.client.transport.close()
            if packet:
                session['packet_calls']=len(packet.rows)
                if any(r['http_failure'] for r in packet.rows):session['infrastructure_invalid']=True
                packet.close()
            events=rows(dest/'answer_events.jsonl') if (dest/'answer_events.jsonl').exists() else []
            histories[f'P{index+1}']=[{'event_id':e['event_id'],'action':e['action'],'observation':e['observation']} for e in events]
            save(dest/'bank_after.json',bank.snapshot());save(dest/'session_summary.json',session)
            summary['sessions'].append(session)
            save(directory/'episode_summary.json',summary)
            print(json.dumps({'episode_id':eid,'session_id':sid,'answer':session['answer_calls'],
                'packet':session['packet_calls'],'memory':session['memory_decisions'],
                'termination':session['termination_reason'],'infrastructure':session['infrastructure_invalid']}),flush=True)
        if session['infrastructure_invalid']:
            summary['status']='INFRASTRUCTURE_INVALID';break
    if summary['status']=='RUNNING':summary['status']='COMPLETED_NATURAL_ATTEMPT'
    summary['successful_purchases']=sum(s['strict_validation']['passed'] for s in summary['sessions'])
    summary['native_purchases']=sum(s.get('purchase_completed',False) for s in summary['sessions'])
    save(directory/'episode_summary.json',summary)
    return summary


def stop_reason():
    transitions=rows(HERE/'packet_transitions.jsonl');audits=rows(HERE/'packet_audit.jsonl');n=len(transitions)
    if n<10:return None
    schema=sum(r['schema_valid'] for r in transitions)/n
    binding=sum(bool(set(r['reason_codes'])&{'SUBJECT_BINDING','UNKNOWN_SUBJECT_REF'}) for r in audits)/n
    gold=sum(r['gold_candidate'] for r in audits)/n
    triggers=[]
    if schema<.9:triggers.append('SCHEMA_VALID_BELOW_90_PERCENT')
    if binding>.2:triggers.append('SUBJECT_BINDING_ABOVE_20_PERCENT')
    if gold<.3:triggers.append('LOCAL_GOLD_YIELD_BELOW_30_PERCENT')
    return {'denominator':n,'schema_valid_rate':schema,'binding_failure_rate':binding,
            'gold_yield':gold,'triggers':triggers} if triggers else None


def main(base_url):
    verify()
    if (HERE/'live_started.json').exists():raise RuntimeError('REFUSE_SECOND_LIVE_ATTEMPT')
    save(HERE/'live_started.json',{'time':time.time(),'locked_episodes':[e['episode_id'] for e in locked_episodes()]})
    retriever=CandidateHybridAdapter(HERE/'embedding_cache')
    save(HERE/'retriever_identity.json',{'config':retriever.backend.config,'embedding':retriever.encoder.identity})
    boundary=FreshBoundary(base_url);summaries=[];stop=None
    try:
        for public in locked_episodes():
            summary=episode(public,retriever,boundary);summaries.append(summary)
            save(HERE/'progress.json',{'episodes':summaries})
            if summary['status']=='INFRASTRUCTURE_INVALID':stop={'triggers':['INFRASTRUCTURE_INVALID']};break
            stop=stop_reason()
            if stop:break
    finally:
        boundary.close()
        save(HERE/'completed.json',{'episodes':summaries,'stop':stop,
            'remaining_not_run':[e['episode_id'] for e in locked_episodes()[len(summaries):]],
            'single_attempt':True,'semantic_retries':0})


if __name__=='__main__':
    import sys
    main(sys.argv[1])
