"""Compile already trusted Shopping requirements to unchanged V3 needs."""
from jsonschema import Draft202012Validator
from ..common import digest
from ...packet_v3_data_pipeline_convergence_v1 import language as L
from ...packet_interface_v3_minimal_validation import interface_v3 as V3

def build_packet_scaffold(contract):
    if contract['authority']!='HARNESS_TASK_CONTRACT':raise ValueError('UNTRUSTED_CONTRACT')
    c=contract;field=c['current_field'];previous=c['previous_subject']
    if field not in {'material','color'}:raise ValueError('UNSUPPORTED_FIELD')
    specs=[]
    if previous:
        if c['previous_field'] not in {'material','color'}:raise ValueError('UNSUPPORTED_PREVIOUS_FIELD')
        specs += [(f"PREVIOUS_ATTRIBUTE|subject={previous}|field={c['previous_field']}",[previous]),
                  (f"PUBLIC_MAPPING|subject={previous}|from={c['previous_field']}|to={field}",[previous])]
    else:
        specs += [(f"INITIAL_ANCHOR|field={field}|allowed={c['initial_anchor']}",['T0'])]
    specs += [(f"CURRENT_FULL|field={field}|dimensions={field},price",list(c['current_subjects'])),
              ('REMAINING_BUDGET|basis='+('prior_purchases' if previous else 'public_cap'),['T0'])]
    needs=[{'id':f'N{i}','text':text,'scope_refs':scope} for i,(text,scope) in enumerate(specs,1)]
    packet=V3.empty_packet();packet['needs']=needs;packet['progress']['pending']=[n['id'] for n in needs]
    Draft202012Validator(V3.PACKET_SCHEMA).validate(packet)
    sidecar={};certifications=[]
    for need in needs:
        cert=L.certify(need,c,legacy=False)
        if cert['classification']!='CERTIFIED_GATING' or len(cert['requirements'])!=1:
            raise ValueError('SCAFFOLD_CERTIFICATION_FAILED: '+str(cert))
        sidecar[need['id']]=cert['requirements'][0];certifications.append(cert)
    if len(sidecar)!=len(c['requirements']) or set(sidecar.values())!=set(c['requirements']):
        raise ValueError('SCAFFOLD_EXACT_REQUIREMENT_COVERAGE_FAILED')
    current=c['requirements'][L.CURRENT]
    expected={(s,d) for s in current['required_scope'] for d in current['required_dimensions']}
    actual={(s['subject_ref'],s['dimension']) for cert in certifications
            for s in cert['contributions'].get(L.CURRENT,[])}
    if actual!=expected:raise ValueError('SCAFFOLD_SLOT_COVERAGE_FAILED')
    return {'packet':packet,'need_requirement_sidecar':sidecar,
            'scaffold_digest':digest(packet),'contract_digest':digest(c),
            'certifications':certifications}

def guard_delta(delta,scaffold,contract):
    """Additional scaffold admission boundary; never repairs or rewrites delta."""
    duplicates=[];wrong=[];local=[]
    if delta is None:return {'duplicate_gating_needs':[],'wrong_gating_role_creation':[], 'local_needs':[], 'passed':False}
    required_kinds={L.parse(n['text'])['kind'] for n in scaffold['packet']['needs']}
    for n in delta['add_needs']:
        try:kind=L.parse(n['text'])['kind']
        except ValueError:wrong.append(n);continue
        if kind in {'LOCAL_INSPECT','LOCAL_RETRY'}:
            local.append({'need':n,'certification':L.certify(n,contract,legacy=False)})
            continue
        if kind in required_kinds or kind=='CURRENT_FRAGMENT':duplicates.append(n)
        else:wrong.append(n)
    return {'duplicate_gating_needs':duplicates,'wrong_gating_role_creation':wrong,
            'local_needs':local,'passed':not duplicates and not wrong}
