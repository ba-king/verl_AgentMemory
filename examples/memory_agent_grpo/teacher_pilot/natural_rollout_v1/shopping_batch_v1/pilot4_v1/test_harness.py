"""Offline calibration only; never counted as live Teacher evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from .memory import current_state,ShoppingMemorySession
from .freeze import RUN_CONFIG
from ....travel_v0.structured_record.bank import StructuredBank
from ....travel_v0.structured_record import policy
from ....travel_v0.single_action import session
from ....travel_v0.artifacts import read


class FakeClient:
    def __init__(self,*args):self.config={};self.count=0
    def call(self,*args):
        self.count+=1
        return {'choices':[{'finish_reason':'stop','message':{'content':'{"action":"YIELD","arguments":{}}'}}]},{}


class HarnessTests(unittest.TestCase):
    def test_modes_use_only_current_public_and_literal_submission(self):
        bank=StructuredBank(None)
        current={'session_id':'p1','instruction':'PUBLIC_TASK'}
        event={'event_id':'p1:a1','action':{'tool':'click','arguments':{'item':'Buy Now'}},
            'observation':'Purchase #1: PUBLIC_ASIN ($12.34)'}
        before=copy.deepcopy(event)
        state=current_state(bank,current,[event],30,{'stage':'after_purchase','index':2})
        self.assertEqual(state['decision_mode'],'CONSOLIDATE')
        self.assertEqual(state['submitted_result']['public_text'],event['observation'])
        self.assertEqual(event,before)
        self.assertNotIn('private',json.dumps(state))
        self.assertEqual(policy.messages_for(state)[0]['content'].split('\nCurrent available actions:')[0],policy.CONSOLIDATE_PROMPT)
        read_state=current_state(bank,current,[],32,{'stage':'session_start','index':1})
        self.assertEqual(read_state['available_actions'],['YIELD'])
        self.assertNotIn('submitted_result',read_state)

    def test_frozen_single_action_loop_runs_with_shopping_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(session,'JsonTeacherClient',FakeClient):
                memory=ShoppingMemorySession('fixture','fixture',{'session_id':'p1','instruction':'public'},
                    Path(directory),StructuredBank(None),None,[])
            memory.opportunity('session_start')
            memory.public_event({'event_id':'p1:a1','action':{'tool':'click','arguments':{'item':'Buy Now'}},'observation':'ACTUAL_PUBLIC_RECEIPT'})
            memory.opportunity('after_purchase')
            self.assertEqual([r['parsed_memory_action']['action'] for r in memory.decisions],['YIELD','YIELD'])
            self.assertEqual(memory.decisions[-1]['student_input']['submitted_result']['public_text'],'ACTUAL_PUBLIC_RECEIPT')
            self.assertEqual(len(memory.bank.snapshot()['records']),0)

    def test_existing_answer_runner_receives_only_actual_delivery(self):
        from . import execute
        from examples.memory_agent_grpo.answer_agent.runtime import AnswerAgentTaskToolExecutionResult
        from examples.memory_agent_grpo.answer_agent.tool_protocol.observation import ToolObservation,ToolObservationStatus
        from examples.memory_agent_grpo.adapters.memoryarena.shopping.task_input_adapter import ShoppingTaskInputAdapter
        class Client:
            def __init__(self,*args):pass
            def call(self,messages,tools):
                self.messages=messages
                return {'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
                    'tool_calls':[{'id':'a1','type':'function','function':{'name':'click','arguments':'{"item":"Buy Now"}'}}]}}]},{}
        class Executor:
            def execute(self,call):
                return AnswerAgentTaskToolExecutionResult(ToolObservation(call.tool_call_id,call.name,
                    ToolObservationStatus.SUCCESS,'PUBLIC_RECEIPT'),True)
        with tempfile.TemporaryDirectory() as directory,patch.object(execute,'FixedClient',Client):
            delivery=[{'role':'user','content':'[Retrieved Memory]\nLEGITIMATE_MEMORY'}]
            client=execute.RecordingAnswer(Path(directory),None,delivery)
            runner=execute.AnswerAgentSubtaskRunner(client=client,executor=Executor(),config=RUN_CONFIG)
            subtask=ShoppingTaskInputAdapter(max_observation_chars=16000).build_subtask_input({'state':'PUBLIC_PAGE'})
            result=runner.run(subtask,tool_definitions=execute.tool_definitions())
            self.assertEqual(result.status.value,'COMPLETED')
            self.assertEqual(client.client.messages[2],delivery[0])
            self.assertEqual(result.to_dict()['executed_tool_count'],1)

    def test_actual_wrong_history_is_not_replaced_by_gold(self):
        from .boundary import strict_purchase
        # A purchase may be wrong, yet the next score must depend on the actual price.
        from .freeze import SHOP
        from ....travel_v0.artifacts import rows
        hidden=next(h for h in rows(SHOP/'hidden_episodes.jsonl') if h['episode_id'].endswith('material_cumulative_000'))
        products={p['asin']:p for p in rows(SHOP/'source_products.jsonl')}
        from shopping_constraints import cents
        gold=hidden['sessions'][0]['valid_answers_or_targets'][0]
        wrong=next(a for a in hidden['sessions'][0]['structured_goal']['candidate_asins'] if a!=gold)
        history=[{'asin':wrong,'price_cents':cents(products[wrong])}]
        saved=copy.deepcopy(history)
        strict_purchase(hidden['sessions'][1]['structured_goal'],products,history,hidden['total_cap_cents'],{'asin':None,'price_cents':None})
        self.assertEqual(history,saved)
        self.assertEqual(history[0]['asin'],wrong)


if __name__=='__main__':unittest.main()
