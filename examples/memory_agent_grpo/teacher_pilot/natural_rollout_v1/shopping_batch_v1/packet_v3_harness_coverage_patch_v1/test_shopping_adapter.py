"""Real locked tasks for parsing; explicitly synthetic public proofs for unit tests.

No fixture is a rollout, catalog observation, Teacher output or training example.
"""
import copy
import json
import unittest
from decimal import Decimal
from .common import locked_episodes, digest
from .harness_contract.shopping import build, CURRENT, PREVIOUS, PREVIOUS_COLOR, BUDGET, MAPPING, ANCHOR, validate_need_against_contract
from .harness_contract.shopping_proof import history_receipt, certify_history, audit_need
from .harness_contract.validation import evaluate_packet
from ..packet_interface_v3_minimal_validation.interface_v3 import PACKET_SCHEMA
from jsonschema import Draft202012Validator


def task(episode=0, session=0):
    return locked_episodes()[episode]['sessions'][session]['instruction']


def blank():
    return {'needs':[], 'evidence':[], 'progress':{'resolved':[], 'pending':[], 'failed_attempts':[]},
            'unresolved':[], 'ready_to_commit':False}


def public_history_unit_fixture(subject, material='wood', color='bronze'):
    """Synthetic public event grammar only, intentionally not actual catalog data."""
    asin='B00000000'+subject[1:]
    literals='Material: '+material
    if color is not None: literals+='; Color: '+color
    return [
        {'event_id':f'unit:{subject}:1','action':{'tool':'search','arguments':{'query':'unit candidate'}},'observation':asin},
        {'event_id':f'unit:{subject}:2','action':{'tool':'click','arguments':{'item':asin}},'observation':'Unit page'},
        {'event_id':f'unit:{subject}:3','action':{'tool':'click','arguments':{'item':'Features'}},'observation':literals},
        {'event_id':f'unit:{subject}:4','action':{'tool':'click','arguments':{'item':'Buy Now'}},'observation':f'Purchase #1: {asin} ($1.00)'}]


def history_unit_fixture(c, color='bronze', material='wood'):
    histories={s:public_history_unit_fixture(s,material,color) for s in c['previous_subjects']}
    memories=[]
    for s in c['previous_subjects']:
        fields={'purchased_item_id':'B00000000'+s[1:],'price_paid':'1.00','material':material,
                'color':color or '', 'total_budget':c['budget_cap'],
                'remaining_budget_after_purchase':str(Decimal(c['budget_cap'])-int(s[1:]))}
        content=f'Subject: "Product {s[1:]} purchase"\n'+'\n'.join(f'- {k} = {json.dumps(v)}' for k,v in fields.items())
        h=digest(content);memories.append({'source_ref':f'UNIT_{s}@{h[:12]}','version_sha256':h,'content':content})
    return histories,memories


def anchor_positive_unit_fixture():
    c=build(task()); p=blank(); field=c['current_field']
    p['needs']=[{'id':'a','text':c['requirements'][ANCHOR]['public_basis'],'scope_refs':['T0']},
                {'id':'b','text':'Establish remaining budget from the public cap.','scope_refs':['T0']},
                {'id':'c','text':f"Determine each candidate's literal {field} label and price.",'scope_refs':c['current_subjects']}]
    p['evidence']=[{'need_id':'a','subject_ref':'T0','fact':c['requirements'][ANCHOR]['public_basis'],'source_refs':['task']},
                   {'need_id':'b','subject_ref':'T0','fact':f'Remaining budget for Product 1 is ${c["budget_cap"]}.','source_refs':['task']}]
    trusted={'allowed_claims_up_to_event':[]}; factual={'literal_interpretations':{},'derived_claims':[]}
    for i,s in enumerate(c['current_subjects'],1):
        for dim,value in ((field,c['initial_anchor']),('price',str(i)+'.00')):
            fact=f'UNIT TEST ONLY {s} {dim} = {value}.';ref=f'unit_{s}_{dim}'
            p['evidence'].append({'need_id':'c','subject_ref':s,'fact':fact,'source_refs':[ref]})
            claim={'predicate':dim,'value':value,'kind':'literal'}
            factual['literal_interpretations'][digest(fact)]={'text':fact,'claims':[claim]}
            trusted['allowed_claims_up_to_event'].append({**claim,'subject_ref':s,'source_ref':ref})
    p['progress']['resolved']=['a','b','c'];p['ready_to_commit']=True
    return {'annotation_origin':'SYNTHETIC_ENGINEERING_FIXTURE_NOT_LIVE_NOT_TRAINING',
            'contract':c,'packet':p,'trusted':trusted,'factual':factual,
            'history':{'memory_certificates':{},'public_history_receipts':{}}}


def eval_anchor(f):
    return evaluate_packet(f['packet'],f['contract'],f['trusted'],f['factual'],f['history'])


class ShoppingAdapterTests(unittest.TestCase):
    def test_t1_initial_material_anchor(self):
        c=build(task());self.assertEqual(c['initial_anchor'],'glass')
        self.assertIsNone(c['previous_subject']);self.assertIsNone(c['previous_field'])
        self.assertEqual(c['previous_subjects'],[]);self.assertNotIn('P0',c['subject_registry'])
        self.assertEqual(set(c['requirements']),{CURRENT,BUDGET,ANCHOR})
        self.assertEqual(c['current_subjects'],['C1','C2','C3','C4'])

    def test_t2_same_template_color_anchor(self):
        t=task().replace('current item is Material.','current item is Color.').replace('with Material label "glass"','with Color label "grey"')
        c=build(t);self.assertEqual(c['current_field'],'color');self.assertEqual(c['initial_anchor'],'gray')
        self.assertNotIn(MAPPING,c['requirements'])

    def test_t3_empty_budget_plus_nonvacuous_readiness(self):
        f=anchor_positive_unit_fixture();r=eval_anchor(f)
        self.assertTrue(r['requirement_results'][BUDGET]['satisfied'])
        self.assertEqual(r['factual_receipts']['b']['remaining'],f['contract']['budget_cap'])
        self.assertEqual(r['factual_receipts']['b']['budget_route'],'ALL_PREVIOUS_PRICES')
        self.assertTrue(r['global_ready_allowed'])
        self.assertEqual(r['decision_verifier']['verifier_only_subject_ref'],'C4')
        Draft202012Validator(PACKET_SCHEMA).validate(f['packet'])
        f['packet']['evidence']=[e for e in f['packet']['evidence'] if e['need_id']!='c']
        self.assertFalse(eval_anchor(f)['global_ready_allowed'])

    def test_t3_untrusted_and_wrong_empty_budget_rejected(self):
        for kind in ('wrong_value','untrusted_source','wrong_subject'):
            f=anchor_positive_unit_fixture();e=f['packet']['evidence'][1]
            if kind=='wrong_value':e['fact']='Remaining budget for Product 1 is $0.00.'
            if kind=='untrusted_source':e['source_refs']=['invented']
            if kind=='wrong_subject':e['subject_ref']='C1'
            self.assertFalse(eval_anchor(f)['requirement_results'][BUDGET]['satisfied'],kind)

    def test_t4_previous_material_and_current_color(self):
        c=build(task(0,1));self.assertEqual(c['previous_subject'],'P1');self.assertEqual(c['previous_field'],'material')
        self.assertIn(PREVIOUS,c['requirements'])
        c=build(task(3,1));self.assertEqual((c['previous_field'],c['current_field']),('material','color'))

    def test_t5_real_previous_color_to_material(self):
        for episode in (3,4,5):
            for session in (2,4):
                c=build(task(episode,session))
                self.assertEqual((c['previous_subject'],c['previous_field'],c['current_field']),(f'P{session}','color','material'))
                self.assertIn(PREVIOUS_COLOR,c['requirements']);self.assertNotIn(ANCHOR,c['requirements'])
        c=build(task(3,2));self.assertEqual(c['mapping'],{'bronze':'faux leather','purple':'polyester'})

    def test_t6_wrong_previous_subject_text_and_scope(self):
        c=build(task(3,2));n={'id':'h','text':'Determine the Color label of the item actually purchased for Product 2.','scope_refs':['P2']}
        self.assertEqual(validate_need_against_contract(n,c)['classification'],'CERTIFIED_GATING')
        for bad in ({**n,'scope_refs':['P1']},{**n,'text':n['text'].replace('Product 2','Product 1')}):
            self.assertEqual(validate_need_against_contract(bad,c)['classification'],'SCOPE_CONTRACT_VIOLATION')

    def test_t7_color_receipt_requires_purchased_literal_color(self):
        c=build(task(3,2));h,m=history_unit_fixture(c);audit=certify_history(c,m,h)
        previous=c['previous_subject'];ref=next(k for k,v in audit['memory_certificates'].items() if v['subject_ref']==previous)
        n={'id':'h','text':'Determine the Color label of the item actually purchased for Product 2.','scope_refs':['P2']}
        cert=validate_need_against_contract(n,c);p=blank();p['needs']=[n]
        p['evidence']=[{'need_id':'h','subject_ref':'P2','source_refs':[ref],'fact':'Product 2 Color is "bronze".'}]
        run=lambda a:audit_need(n,cert,p,c,{'allowed_claims_up_to_event':[]},{'literal_interpretations':{}},a)
        self.assertTrue(run(audit)['previous_ok'])
        p['evidence'][0]['fact']='Product 2 Material is "wood".'
        self.assertFalse(run(audit)['previous_ok'])
        p['evidence'][0]['fact']='Product 2 Color is "bronze".'
        for label,mutate in (
            ('missing literal',lambda x:x.replace('; Color: bronze','')),
            ('different literal',lambda x:x.replace('Color: bronze','Color: purple'))):
            altered=copy.deepcopy(h)
            for e in altered['P2']:e['observation']=mutate(e['observation'])
            self.assertFalse(run(certify_history(c,m,altered))['previous_ok'],label)
        wrong_ref=next(k for k,v in audit['memory_certificates'].items() if v['subject_ref']=='P1')
        p['evidence'][0]['source_refs']=[wrong_ref]
        self.assertFalse(run(audit)['previous_ok'])

    def test_t7_wrong_attribute_proposal_rejected(self):
        c=build(task(3,2));n={'id':'h','text':'Determine the Material label of the item actually purchased for Product 2.','scope_refs':['P2']}
        self.assertIn('PREVIOUS_ATTRIBUTE_MISMATCH',validate_need_against_contract(n,c)['reason_codes'])

    def test_color_mapping_and_budget_reach_ready_with_public_proofs(self):
        f=anchor_positive_unit_fixture();c=build(task(3,2));f['contract']=c
        h,m=history_unit_fixture(c);f['history']=certify_history(c,m,h)
        ref=next(k for k,v in f['history']['memory_certificates'].items() if v['subject_ref']=='P2')
        balance=f['history']['memory_certificates'][ref]['remaining']
        f['packet']['needs'][0]={'id':'a','text':"Determine Product 2's actual purchased Color label to select the allowed Product 3 Material.",'scope_refs':['P2']}
        f['packet']['needs'][1]={'id':'b','text':'Establish remaining budget from a certified cumulative state.','scope_refs':['P2']}
        f['packet']['evidence'][0]={'need_id':'a','subject_ref':'P2','source_refs':[ref,'task'],
                                   'fact':'Product 2 Color is "bronze", so the public rule allows current Material "faux leather".'}
        f['packet']['evidence'][1]={'need_id':'b','subject_ref':'P2','source_refs':[ref],
                                   'fact':f'Remaining budget after Product 2 is ${balance}.'}
        # Rewrite ONLY synthetic unit receipts/claims, never real task/Bank.
        for e in f['packet']['evidence'][2:]:
            if ' material = ' in e['fact']:
                old=e['fact'];e['fact']=old.replace('glass','faux leather')
                interpretation=f['factual']['literal_interpretations'].pop(digest(old))
                interpretation['text']=e['fact'];interpretation['claims'][0]['value']='faux leather'
                f['factual']['literal_interpretations'][digest(e['fact'])]=interpretation
        for claim in f['trusted']['allowed_claims_up_to_event']:
            if claim['predicate']=='material':claim['value']='faux leather'
        Draft202012Validator(PACKET_SCHEMA).validate(f['packet'])
        result=eval_anchor(f)
        self.assertTrue(result['requirement_results'][PREVIOUS_COLOR]['satisfied'])
        self.assertTrue(result['requirement_results'][MAPPING]['satisfied'])
        self.assertTrue(result['global_ready_allowed'])
        self.assertEqual(result['decision_verifier']['verifier_only_subject_ref'],'C4')
        for cert in f['history']['memory_certificates'].values(): cert['color_certified']=False
        self.assertFalse(eval_anchor(f)['global_ready_allowed'])

    def test_t8_unknown_and_ambiguous_templates_fail_closed(self):
        variants=[task(3,2).replace('Use the Color','Use the Finish'),
                  task().replace('current item is Material.','current item is Finish.'),
                  task().replace('highest-priced','lowest-priced'),
                  task(3,2).replace('Previous Color "bronze"','Previous Material "bronze"'),
                  task(3,2).replace('applies only to Product 3','applies only to Product 4'),
                  task(3,2).replace('purchased for Product 2','purchased for Product 3'),
                  task()+'\nAdditional constraint: buy the red item.',
                  task().replace('Only candidates','Use the Material label of the item actually purchased for Product 0.\nOnly candidates')]
        for t in variants:
            with self.assertRaises(ValueError):build(t)
        with self.assertRaisesRegex(ValueError,'UNSUPPORTED_PUBLIC_TASK_CONTRACT'):build(variants[0])
        with self.assertRaises(TypeError):build({'task':task(),'hidden':'ignored'})

    def test_t9_candidate_exact_set_anchor_and_dependency(self):
        for t in (task(),task(3,2)):
            c=build(t);n={'id':'c','text':f"Determine each candidate's literal {c['current_field']} label and price.",'scope_refs':c['current_subjects']}
            self.assertEqual(validate_need_against_contract(n,c)['classification'],'CERTIFIED_GATING')
            for refs in (['C1','C2','C3'],['C1','C2','C3','C4','T0']):
                self.assertEqual(validate_need_against_contract({**n,'scope_refs':refs},c)['classification'],'SCOPE_CONTRACT_VIOLATION')
        f=anchor_positive_unit_fixture();f['packet']['evidence']=[e for e in f['packet']['evidence'] if e['subject_ref']!='C4']
        self.assertFalse(eval_anchor(f)['global_ready_allowed'])

    def test_anchor_requires_public_fact_not_phantom_mapping(self):
        f=anchor_positive_unit_fixture();f['packet']['evidence'][0]['fact']='Only candidates with Material label "wood" are eligible.'
        self.assertFalse(eval_anchor(f)['requirement_results'][ANCHOR]['satisfied'])
        f=anchor_positive_unit_fixture();f['packet']['evidence'][0]['source_refs']=['invented']
        self.assertFalse(eval_anchor(f)['requirement_results'][ANCHOR]['satisfied'])

    def test_unknown_need_never_guessed(self):
        c=build(task());n={'id':'x','text':'all candidates perfect and ready','scope_refs':c['current_subjects']}
        self.assertEqual(validate_need_against_contract(n,c)['classification'],'UNRESOLVED_REQUIREMENT_MAPPING')

    def test_full_locked36_public_only(self):
        cs=[build(s['instruction']) for e in locked_episodes() for s in e['sessions']]
        self.assertEqual(len(cs),36);self.assertEqual(sum(c['previous_subject'] is None for c in cs),6)
        self.assertEqual(sum(c['previous_field']=='color' for c in cs),6)
        self.assertTrue(all(len(c['current_subjects'])==4 for c in cs))

    def test_color_not_inferred_from_title_or_other_item(self):
        events=public_history_unit_fixture('P2',color=None)
        events[1]['observation']='Bronze Unit Item'
        self.assertIsNone(history_receipt(events)['color'])
        events[2]['observation']='Frame Color: bronze; Material: wood'
        self.assertIsNone(history_receipt(events)['color'])
        events[2]['observation']='Color: bronze; Color: purple'
        self.assertIsNone(history_receipt(events)['color'])

    def test_color_provenance_ignores_private_keys(self):
        events=public_history_unit_fixture('P2');before=history_receipt(events)
        for e in events:e.update(hidden_answer='purple',scorer=1,future={'color':'purple'})
        self.assertEqual(before,history_receipt(events))
