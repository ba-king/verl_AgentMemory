"""Compose frozen actors/executor. Six sessions per episode, one natural attempt."""
import copy
import json
import time
from .freeze import HERE,IDS,RUN_CONFIG,verify,messages
from .memory import ShoppingMemorySession
from .boundary import Boundary,RecordingExecutor
from ..retriever import CandidateHybridAdapter
from ....travel_v0.provider import FixedClient,Budget
from ....travel_v0.structured_record.bank import StructuredBank
from ....travel_v0.artifacts import read,save,append,rows
from examples.memory_agent_grpo.answer_agent.providers.deepseek import DeepSeekAnswerAgentClient
from examples.memory_agent_grpo.answer_agent.runtime.subtask_runner import AnswerAgentSubtaskRunner
from examples.memory_agent_grpo.adapters.memoryarena.shopping.tool_catalog import tool_definitions


class RecordingAnswer:
    def __init__(self,dest,budget,delivery_messages):
        self.client=FixedClient('answer',dest/'api',budget)
        self.delivery_messages=delivery_messages;self.dest=dest;self.last_meta={}

    def generate(self,conversation,definitions):
        visible=DeepSeekAnswerAgentClient._validate_and_serialize_conversation(conversation)
        # Frozen structured renderer delivered raw records after READ YIELD.
        visible[2:2]=copy.deepcopy(self.delivery_messages)
        tools=DeepSeekAnswerAgentClient._validate_and_serialize_tools(definitions)
        body,self.last_meta=self.client.call(visible,tools)
        save(self.dest/'answer_exact_conversation.json',visible)
        return DeepSeekAnswerAgentClient._parse_response(json.dumps(body).encode())


def infrastructure_error(error):
    import requests
    if isinstance(error,requests.RequestException):return True
    text=str(error)
    return any(s in text for s in ['HTTP_','PREDECLARED_','UNEXPECTED_RESPONSE_MODEL','CLIENT_INITIALIZATION_FAILED',
        'RESET_FAILED','ConnectionError','ReadTimeout','SSLError','internal_server_error'])


def run_episode(public,retriever,boundary):
    eid=public['episode_id'];directory=HERE/'episodes'/eid
    assert not directory.exists(),'REFUSE_SECOND_EPISODE_ATTEMPT'
    directory.mkdir(parents=True)
    summary=dict(episode_id=eid,type='mixed' if '_mixed_' in eid else 'material',sessions=[],
        status='RUNNING',semantic_retries=0,planned_sessions=6)
    boundary.begin_episode(eid);bank=StructuredBank(retriever)
    budget=Budget(HERE/'budgets'/f'{eid}.json',372,100000000)
    for index,public_session in enumerate(public['sessions']):
        sid=public_session['session_id'];dest=directory/sid;dest.mkdir()
        session=dict(session_id=sid,strict_validation={'passed':False,'reason':'not_purchased'},
            termination_reason='not_started',memory_decisions=0,answer_calls=0,infrastructure_invalid=False)
        memory=answer=None
        try:
            verify()
            subtask=boundary.reset(public_session,dest,index)
            current={'session_id':sid,'instruction':public_session['instruction'],'initial_observation':subtask.subtask_input}
            conversation=messages(subtask)
            memory=ShoppingMemorySession('shopping_natural_pilot4_v1',eid,current,dest,bank,budget,conversation)
            save(dest/'bank_before.json',bank.snapshot())
            memory.opportunity('session_start')
            # Only actual READ deliveries, not bank snapshots or private scorer state.
            answer=RecordingAnswer(dest,budget,conversation[2:])
            executor=RecordingExecutor(boundary,memory,answer,dest)
            runtime=AnswerAgentSubtaskRunner(client=answer,executor=executor,config=RUN_CONFIG)
            result=runtime.run(subtask,tool_definitions=tool_definitions())
            save(dest/'answer_runner_result.json',result.to_dict())
            session['answer_runner_status']=result.status.value
            session['answer_error_code']=result.error_code.value if result.error_code else None
            session['termination_reason']='purchase_completed' if boundary.client.purchased_asins else 'answer_ended_without_purchase'
            # No evaluator feedback is passed into CONSOLIDATE.
            memory.opportunity('after_purchase' if boundary.client.purchased_asins else 'session_ended_without_purchase')
        except Exception as error:
            session.update(error=type(error).__name__+': '+str(error),
                infrastructure_invalid=infrastructure_error(error),termination_reason='session_error')
            save(dest/'error.json',dict(type=type(error).__name__,message=str(error),infrastructure=session['infrastructure_invalid']))
        finally:
            # Score only after all actor calls for this session; keep sealed output private.
            if boundary.client is not None and boundary.dest==dest:
                try:session.update(boundary.score())
                except Exception as error:
                    session.update(infrastructure_invalid=True,scoring_error=repr(error))
            if memory:
                memory.finish(session)
                session['memory_decisions']=len(memory.decisions)
                for row in memory.decisions:append(dest/'memory_decisions.jsonl',row)
                memory.client.transport.close()
            if answer:
                session['answer_calls']=answer.client.count;answer.client.transport.close()
            save(dest/'bank_after.json',bank.snapshot())
            save(dest/'session_summary.json',session)
            summary['sessions'].append(session)
            if session['infrastructure_invalid']:summary['status']='INFRASTRUCTURE_INVALID'
            save(directory/'episode_summary.json',summary)
            print(json.dumps(dict(episode_id=eid,session=sid,strict=session['strict_validation']['passed'],
                memory=session['memory_decisions'],answer=session['answer_calls'],
                infrastructure=session['infrastructure_invalid'])),flush=True)
        if session['infrastructure_invalid']:break
    if summary['status']!='INFRASTRUCTURE_INVALID':summary['status']='COMPLETED_NATURAL_ATTEMPT'
    summary['successful_purchases']=sum(s['strict_validation']['passed'] for s in summary['sessions'])
    summary['native_purchases']=sum(s.get('purchase_completed',False) for s in summary['sessions'])
    save(directory/'episode_summary.json',summary)
    return summary


def main(base_url):
    verify()
    assert not (HERE/'live_started.json').exists(),'REFUSE_SECOND_LIVE_PILOT'
    save(HERE/'live_started.json',dict(started=time.time(),episode_ids=IDS,paid_requests_before=0))
    retriever=CandidateHybridAdapter(HERE/'embedding_cache')
    save(HERE/'live_retriever_identity.json',dict(config=retriever.backend.config,embedding=retriever.encoder.identity))
    boundary=Boundary(base_url);summaries=[]
    try:
        for public in read(HERE/'selected_public.json'):
            summary=run_episode(public,retriever,boundary);summaries.append(summary)
            save(HERE/'progress.json',dict(episodes=summaries,embedding_statistics=retriever.encoder.stats))
            if summary['status']=='INFRASTRUCTURE_INVALID':break
    finally:
        boundary.close()
        save(HERE/'completed.json',dict(episodes=summaries,remaining_not_run=IDS[len(summaries):],
            frozen_verified=bool(verify()),embedding_statistics=retriever.encoder.stats))


if __name__=='__main__':
    import sys
    main(sys.argv[1])
