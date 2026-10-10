"""Offline wiring calibration. Synthetic tapes are never live results."""
import copy
import unittest
from .common import read,OLD
from .packet_runtime import PacketClient,public_input
from . import audit
from ..packet_interface_v2_validation.interface import PROMPT
from ..packet_interface_v3_minimal_validation import interface_v3 as v3
from ..packet_v3_harness_coverage_patch_v1.test_shopping_adapter import task
from .hardening.shopping import build


class RuntimeTests(unittest.TestCase):
    def test_request_uses_real_v3_tools_and_exact_frozen_prompt(self):
        cfg={'model':'deepseek-v4-pro','temperature':0,'max_tokens':4096,'thinking':{'type':'disabled'},'stream':False}
        client=PacketClient.__new__(PacketClient);client.config=cfg
        p={'current_task':task(),'initial_observation':{'event_id':'e000','observation':'initial'},
           'actual_retrieved_memories':[],'public_prefix_events_only':[]}
        t={'subject_registry':{'T0':{'name':'Current task'},'C1':{'name':'Visible candidate'}}}
        r=client.request(public_input(p,v3.empty_packet(),t))
        self.assertEqual(r['messages'][0]['content'],PROMPT);self.assertEqual(r['tools'],v3.TOOLS)
        self.assertIn('subject_ref',r['tools'][0]['function']['parameters']['properties']['packet_delta']['properties']['add_evidence']['items']['required'])

    def test_incremental_input_no_old_raw_history(self):
        p={'current_task':task(),'initial_observation':{'event_id':'e000','observation':'OLD_INITIAL'},
           'actual_retrieved_memories':[{'content':'OLD_RAW_MEMORY'}],
           'public_prefix_events_only':[{'event_id':'e001','observation':'OLD_PAGE','action':{}},
                                        {'event_id':'e002','observation':'CURRENT','action':{}}]}
        inp=public_input(p,v3.empty_packet(),{'subject_registry':{}})
        self.assertEqual(inp['latest_observation']['observation'],'CURRENT');self.assertEqual(inp['retrieved_memories'],[])
        self.assertNotIn('OLD_PAGE',str(inp));self.assertNotIn('OLD_RAW_MEMORY',str(inp));self.assertNotIn('OLD_INITIAL',str(inp))

    def test_long_free_prose_cannot_be_partially_certified(self):
        t={'source_ownership_registry':{},'allowed_claims_up_to_event':[]}
        e={'subject_ref':'C1','source_refs':[],'fact':'Candidate C1 has Material: "glass". It is the best and costs $500.'}
        self.assertEqual(audit.interpret(e,t),[])

    def test_literal_value_support_not_just_existing_ref(self):
        t={'source_ownership_registry':{'e001':{'public_text':'Material: Glass'}},
           'allowed_claims_up_to_event':[{'subject_ref':'C1','predicate':'material','value':'glass','source_ref':'e001'}]}
        e={'subject_ref':'C1','source_refs':['e001'],'fact':'Material: iron'}
        claim=audit.interpret(e,t)[0];self.assertFalse(audit.supported(e,claim,t))
        e['fact']='Material: Glass';self.assertTrue(audit.supported(e,audit.interpret(e,t)[0],t))

    def test_invalid_protocol_cannot_be_candidate(self):
        row={'schema_valid':False,'apply_valid':False,'previous_packet':v3.empty_packet(),
             'applied_packet':None,'parsed_packet_delta':None}
        r=audit.evaluate(row,{}, {}, {}, {})
        self.assertFalse(r['gold_candidate']);self.assertEqual(r['reason_codes'],['SCHEMA'])

    def test_noop_empty_init_is_omission(self):
        d={'add_needs':[],'add_evidence':[],'remove_evidence':[],'resolve':[],'add_pending':[],
           'remove_pending':[],'add_failed_attempts':[],'remove_failed_attempts':[],
           'set_unresolved':[],'set_ready_to_commit':'false'}
        p=v3.empty_packet();c=build(task())
        row={'schema_valid':True,'apply_valid':True,'previous_packet':p,'applied_packet':p,
             'parsed_packet_delta':d,'operation':'INIT_PACKET','replay_equal':True}
        t={'subject_registry':c['subject_registry'],'source_ownership_registry':{},
           'allowed_claims_up_to_event':[],'allowed_claims_from_latest_event':[]}
        public={'actual_retrieved_memories':[],'public_prefix_events_only':[]}
        r=audit.evaluate(row,public,t,c,{})
        self.assertFalse(r['gold_candidate']);self.assertIn('OMISSION',r['reason_codes'])
