import copy
import unittest
from .common import LIFECYCLE,read
from .implementation.stale_evidence_auditor import claim,audit
from .implementation.packet_state_maintenance import Client,STATE_MAINTENANCE_RULE,PROMPT
from final_unified_travel_rp_packet_v3_demo_v1.packet import V3,Client as OldClient
from teacher_pilot.travel_v0.provider import model_config


def packet(facts):
    p=V3.empty_packet();p['needs']=[{'id':'N1','text':'public requirement','scope_refs':['S1','S2']},
                                  {'id':'N2','text':'other requirement','scope_refs':['S1']}]
    p['progress']['pending']=['N1','N2']
    p['evidence']=[{'fact':f,'need_id':n,'subject_ref':s,'source_refs':['e001']} for n,s,f in facts]
    return p


def delta():
    return {'add_needs':[],'add_evidence':[],'remove_evidence':[],'resolve':[],
        'add_pending':[],'remove_pending':[],'add_failed_attempts':[],
        'remove_failed_attempts':[],'set_unresolved':[],'set_ready_to_commit':'keep'}


class MaintenanceTests(unittest.TestCase):
    def test_A_direct_contradiction_and_conceptual_retraction(self):
        p=packet([('N1','S1','No candidates retrieved yet'),('N1','S1','16 candidates retrieved')])
        a=[claim(p,0,'candidate_evidence_available',False),claim(p,1,'candidate_evidence_available',True)]
        self.assertEqual(audit(p,a)['counts']['ACTIVE_STATE_CONTRADICTION'],1)
        d=delta();d['remove_evidence']=[0]
        out=V3.apply(p,d,{'e001'},init=False)
        self.assertEqual(len(out['evidence']),1)

    def test_B_compatible_enrichment(self):
        p=packet([('N1','S1','price=880'),('N1','S1','minimum_nights=1')])
        original=copy.deepcopy(p)
        self.assertFalse(audit(p,[claim(p,0,'price',880),claim(p,1,'minimum_nights',1)])['findings'])
        self.assertEqual(p,original)

    def test_C_corrected_value_requires_old_removed(self):
        p=packet([('N1','S1','price=900'),('N1','S1','price=880')])
        truth={'need_id':'N1','subject_ref':'S1','dimension':'price','value':880,'source_refs':['e002']}
        a=[claim(p,0,'price',900),claim(p,1,'price',880)]
        self.assertEqual(audit(p,a,[truth])['counts']['STALE_EVIDENCE_NOT_RETRACTED'],1)
        d=delta();d['remove_evidence']=[0];out=V3.apply(p,d,{'e001','e002'},init=False)
        self.assertEqual(out['evidence'][0]['fact'],'price=880')

    def test_D_different_subjects(self):
        p=packet([('N1','S1','hotel=A'),('N1','S2','hotel=B')])
        self.assertFalse(audit(p,[claim(p,0,'hotel','A'),claim(p,1,'hotel','B')])['findings'])

    def test_E_different_dimensions(self):
        p=packet([('N1','S1','city=Phoenix'),('N1','S1','price=880')])
        self.assertFalse(audit(p,[claim(p,0,'city','Phoenix'),claim(p,1,'price',880)])['findings'])

    def test_F_unresolved_cleanup(self):
        p=packet([('N1','S1','candidate evidence available')]);p['unresolved']=['candidate evidence missing']
        a=[claim(p,0,'candidate_evidence_available',True),
           claim(p,0,'candidate_evidence_available',False,collection='unresolved',need_id='N1',subject_ref='S1')]
        self.assertEqual(audit(p,a)['counts']['ACTIVE_STATE_CONTRADICTION'],1)
        out=V3.apply(p,delta(),{'e001'},init=False)
        self.assertEqual(out['unresolved'],[])

    def test_distinct_missing_requirement_kept(self):
        p=packet([('N1','S1','price=880')]);p['unresolved']=['room type still missing']
        a=[claim(p,0,'price',880),claim(p,0,'room_type_available',False,collection='unresolved',need_id='N1',subject_ref='S1')]
        self.assertFalse(audit(p,a)['findings'])

    def test_same_subject_different_need_not_merged(self):
        p=packet([('N1','S1','price=880'),('N2','S1','price=900')])
        self.assertFalse(audit(p,[claim(p,0,'price',880),claim(p,1,'price',900)])['findings'])

    def test_unreviewed_prose_not_claimed_clean(self):
        p=packet([('N1','S1','opaque prose')]);self.assertEqual(audit(p,[])['unreviewed'],[['evidence',0]])

    def test_no_automatic_packet_or_delta_mutation(self):
        p=packet([('N1','S1','price=880'),('N1','S1','price=900')]);old=copy.deepcopy(p)
        result=audit(p,[claim(p,0,'price',880),claim(p,1,'price',900)])
        self.assertEqual(p,old);self.assertFalse(result['corrective_delta_generated'])

    def test_policy_change_is_only_system_suffix(self):
        inp=read(LIFECYCLE/'phase2/multi_event_packet_live/source_capsule.json')['input']
        new=object.__new__(Client);new.config=model_config()
        old=object.__new__(OldClient);old.config=model_config()
        a,b=new.request(inp),old.request(inp)
        self.assertEqual(a['messages'][0]['content'],b['messages'][0]['content']+STATE_MAINTENANCE_RULE)
        a['messages'][0]=b['messages'][0];self.assertEqual(a,b)
        self.assertNotIn('Phoenix',STATE_MAINTENANCE_RULE)
        self.assertNotIn('Morgan',STATE_MAINTENANCE_RULE)

    def test_real_bank_restore_matches_sealed_json_exactly(self):
        import json
        from types import SimpleNamespace
        from .common import rows,text,digest
        from teacher_pilot.travel_v0.structured_record.bank import StructuredBank
        bank=StructuredBank(SimpleNamespace(backend=SimpleNamespace(config={'top_k':3})))
        for row in rows(LIFECYCLE/'phase2/s1_durable_retention_live/memory_decisions.jsonl'):
            a=row['parsed_memory_action']
            if a['action'] in {'ADD','UPDATE','DELETE'}:bank.execute(a)
        old=read(LIFECYCLE/'phase2/s1_durable_retention_live/bank_after.json')
        self.assertEqual(json.loads(text(bank.snapshot())),old)
        self.assertEqual(digest(bank.snapshot()),digest(old))

    def test_http_wire_compaction_preserves_all_model_values(self):
        import json
        from .transport import compact_wire
        inp=read(LIFECYCLE/'phase2/multi_event_packet_live/source_capsule.json')['input']
        client=object.__new__(Client);client.config=model_config()
        request=client.request(inp)
        encoded=compact_wire(json.dumps(request,ensure_ascii=False).encode())
        self.assertEqual(json.loads(encoded),request)
        self.assertLessEqual(len(encoded)+512*len(request['messages'])+request['max_tokens']+8192,40000)


if __name__=='__main__':unittest.main()
