"""Canonical admission and explicit current obligations. No posthoc annotations.

Source value uncertainty, language coverage and proven falsehood are separate.
Local transitions do not inherit unrelated errors as new mistakes.
"""
import copy
from collections import defaultdict
from . import language as lang, proofs
from .common import digest


def subtypes(previous,packet,delta):
    labels=[]
    for key,name in {'add_needs':'ADD_NEED','add_evidence':'ADD_EVIDENCE','remove_evidence':'REMOVE_EVIDENCE',
                     'resolve':'RESOLVE','add_pending':'REOPEN','add_failed_attempts':'FAILED_ATTEMPT'}.items():
        if delta[key]:labels.append(name)
    if delta['remove_pending']:labels.append('REMOVE_PENDING')
    if delta['remove_failed_attempts']:labels.append('REMOVE_FAILED_ATTEMPT')
    if packet['unresolved']!=previous['unresolved']:labels.append('UNRESOLVED_CHANGE')
    if packet['ready_to_commit']!=previous['ready_to_commit']:labels.append('READINESS_CHANGE')
    return labels or ['EMPTY_DELTA']


def atom_key(e,r):
    return (e['subject_ref'],r['claim']['predicate'],r['claim']['value']) if r['claim'] else None


def obligation_audit(previous,packet,atoms,trusted,c,h,init=False):
    known={atom_key(e,r) for e,r in zip(packet['evidence'],atoms) if r['status']=='PASS'}
    old_results=[proofs.evidence(e,trusted,c,h) for e in previous['evidence']]
    old_known={atom_key(e,r) for e,r in zip(previous['evidence'],old_results) if r['status']=='PASS'}
    groups=defaultdict(set)
    for r in trusted['allowed_claims_from_latest_event']:
        if r['subject_ref'] in c['current_subjects'] and r['predicate'] in {c['current_field'],'price'}:
            val=proofs.shop.money(r['value']) if r['predicate']=='price' else lang.canonical(r['value'])
            groups[r['subject_ref'],r['predicate']].add(val)
    if init:
        if not c['previous_subjects']:
            groups['T0','remaining_budget'].add(c['budget_cap'])
            groups['T0','allowed_'+c['current_field']].add(c['initial_anchor'])
        else:
            prior=proofs.prior_value(c,h)
            if prior is not None:
                groups[c['previous_subject'],c['previous_field']].add(prior)
                if prior in c['mapping']:groups['T0','allowed_'+c['current_field']].add(c['mapping'][prior])
            prices={r['subject_ref']:r['price'] for r in h['memory_certificates'].values() if r['price_certified']}
            balance=proofs.core.remaining_from_verified_prices(c['budget_cap'],c['previous_subjects'],prices)
            last=f"P{c['current_step']-1}"
            balances={r['remaining'] for r in h['memory_certificates'].values() if r['subject_ref']==last and r['cumulative_certified']}
            if balance is None and len(balances)==1:balance=next(iter(balances))
            if balance is not None:groups['T0','remaining_budget'].add(balance)
    obligations=[];unresolved=[];omissions=[];corrections=[]
    for (subject,pred),values in groups.items():
        if len(values)!=1:
            unresolved.append({'subject':subject,'predicate':pred,'reason':'Multiple public values at this cutoff; no hidden winner/field selection.', 'values':sorted(values)})
            continue
        val=next(iter(values));key=(subject,pred,val)
        if key in old_known:continue
        ob={'subject':subject,'predicate':pred,'value':val,'satisfied':key in known}
        obligations.append(ob)
        if key not in known:
            # Unknown legacy prose is uncertainty rather than demonstrated absence.
            if any(e['subject_ref']==subject and r['claim'] is None for e,r in zip(packet['evidence'],atoms)):
                unresolved.append({**ob,'reason':'Uninterpreted fact may already carry this value.'})
            else:omissions.append(ob)
        old_wrong=[(e,r) for e,r in zip(previous['evidence'],old_results) if r['claim'] and e['subject_ref']==subject and r['claim']['predicate']==pred and r['claim']['value']!=val]
        retained_wrong=any(e['subject_ref']==subject and r['claim'] and r['claim']['predicate']==pred and r['claim']['value']!=val for e,r in zip(packet['evidence'],atoms))
        if old_wrong and (key not in known or retained_wrong):corrections.append(ob)
    status='PROVEN_OMISSION' if omissions else 'AUDIT_UNRESOLVED' if unresolved else 'REQUIRED_CHANGE_SATISFIED' if obligations else 'NO_REQUIRED_CHANGE'
    return {'status':status,'obligations':obligations,'proven_omissions':omissions,'unresolved':unresolved,'correction_failures':corrections}


def relevance(e,cert,c):
    if cert['classification'] not in {'CERTIFIED_GATING','LOCAL_ONLY'}:return None
    try:claim=lang.fact(e['fact'])
    except ValueError:return None
    role=cert.get('role');pred=claim['predicate'];s=e['subject_ref']
    if pred=='comparison_ready':return s=='T0' and role=='CURRENT_FULL'
    if role in {'CURRENT_FULL','CURRENT_FRAGMENT'}:return s in cert['scope_refs'] and pred in {c['current_field'],'price'}
    if role=='PREVIOUS':return s==c['previous_subject'] and pred==c['previous_field']
    if role in {'MAPPING','ANCHOR'}:return s=='T0' and pred=='allowed_'+c['current_field']
    if role=='BUDGET':return (s=='T0' and pred=='remaining_budget') or (s in c['previous_subjects'] and pred=='price')
    if role=='LOCAL':return s in cert['scope_refs'] and pred in {'material','color','price'}
    return None


def unresolved_check(text,known,trusted,c,h):
    try:r=lang.unresolved(text)
    except ValueError:return 'UNRESOLVED_LANGUAGE_UNSUPPORTED'
    if r['subject'] not in trusted['subject_registry']:return 'UNKNOWN_SUBJECT_REF'
    values={v for s,p,v in known if s==r['subject'] and p==r['predicate']}
    if r['predicate']=='identity':
        identified=bool(trusted['subject_registry'][r['subject']].get('visible_asins'))
        if identified:return 'STALE_UNRESOLVED'
    elif r['kind'] in {'MISSING','UNKNOWN'} and values:return 'STALE_UNRESOLVED'
    elif r['kind']=='CONFLICT' and len(values)<2:
        raw={x['value'] for x in trusted['allowed_claims_up_to_event'] if x['subject_ref']==r['subject'] and x['predicate']==r['predicate']}
        if len(raw)<2:return 'UNSUPPORTED_CONFLICT'
    return None


def evaluate(row,public,trusted,contract,histories):
    previous=row['previous_packet'];packet=row['applied_packet'];delta=row.get('parsed_packet_delta')
    base={'schema_valid':row['schema_valid'],'apply_valid':row['apply_valid'],
        'transition_local_verdict':'FAIL','full_packet_verdict':'NOT_APPLIED','admission_verdict':'FAIL',
        'global_ready_allowed':False,'ready_result':'NOT_APPLIED','inherited_errors':[],
        'reason_codes':[],'coverage_reasons':[],'review_status':'AUTO_REJECT',
        'annotation_origin':'CANONICAL_PUBLIC_AUDIT_V1','teacher_generated':True,'gold_candidate':False,
        'need_mapping':[],'need_text_audit':[],'transition_subtypes':[],
        'factual_auditable':0,'factual_total':0,'repair_types':[]}
    if not row['schema_valid'] or not row['apply_valid']:
        base['reason_codes']=['SCHEMA' if not row['schema_valid'] else 'APPLY'];return base
    c=contract;h=proofs.history(c,public['actual_retrieved_memories'],histories)
    gate,atoms=proofs.evaluate_packet(packet,c,trusted,h)
    old_atoms=[proofs.evidence(e,trusted,c,h) for e in previous['evidence']]
    reasons=[];coverage=[];inherited=[];full_issues=[];repairs=[]
    touched=set(delta['resolve'])|set(delta['add_pending'])|{n['id'] for n in delta['add_needs']}
    old_evidence={digest(e) for e in previous['evidence']};certs={r['need_id']:r for r in gate['need_certifications']}
    if row['operation']=='INIT_PACKET' and not packet['needs']:reasons.append('OMISSION')
    for n in packet['needs']:
        cert=certs[n['id']]
        issue=None
        if cert['classification']=='SCOPE_CONTRACT_VIOLATION':issue={'need_id':n['id'],'reason_codes':cert['reason_codes']}
        elif cert['classification']=='UNRESOLVED_REQUIREMENT_MAPPING':issue={'need_id':n['id'],'reason_codes':['HARNESS_PROPOSAL_LANGUAGE_COVERAGE']}
        if issue:
            full_issues.append(issue)
            if n['id'] not in touched:inherited.append(issue)
            elif cert['classification']=='UNRESOLVED_REQUIREMENT_MAPPING':coverage.append('HARNESS_PROPOSAL_LANGUAGE_COVERAGE')
            else:reasons.extend(cert['reason_codes'])
    for e,a in zip(packet['evidence'],atoms):
        codes=list(a['reason_codes']);rel=relevance(e,certs[e['need_id']],c)
        if rel is False:codes.append('EVIDENCE_NEED_SCOPE')
        if codes:
            issue={'evidence_hash':digest(e),'reason_codes':codes};full_issues.append(issue)
            if digest(e) in old_evidence:inherited.append(issue)
            else:
                coverage.extend(x for x in codes if x in {'FACTUAL_AUDIT_UNRESOLVED','SOURCE_SUPPORT_UNPROVEN','SOURCE_OWNERSHIP_UNRESOLVED'})
                reasons.extend(x for x in codes if x not in {'FACTUAL_AUDIT_UNRESOLVED','SOURCE_SUPPORT_UNPROVEN','SOURCE_OWNERSHIP_UNRESOLVED'})
    obligation=obligation_audit(previous,packet,atoms,trusted,c,h,row['operation']=='INIT_PACKET')
    if obligation['proven_omissions']:reasons.append('PROVEN_OMISSION')
    if obligation['unresolved']:coverage.append('OMISSION_AUDIT_UNRESOLVED')
    if obligation['correction_failures']:reasons.append('CORRECTION')
    for p in gate['premature_resolves']:
        if p['need_id'] in delta['resolve']:reasons.append('NEED_SCOPE')
    # History/anchor/budget assertions cannot be resolved without proof either.
    for nid in delta['resolve']:
        cert=certs[nid]
        if cert['classification']=='CERTIFIED_GATING':
            for rid in cert['requirements']:
                if c['requirements'][rid]['kind']!='slots' and not gate['requirement_results'][rid]['satisfied']:
                    reasons.append('UNPROVEN_RESOLVE')
    known={atom_key(e,a) for e,a in zip(packet['evidence'],atoms) if a['status']=='PASS'}
    for text in packet['unresolved']:
        code=unresolved_check(text,known,trusted,c,h)
        if code:
            if code=='UNRESOLVED_LANGUAGE_UNSUPPORTED':
                if text not in previous['unresolved']:coverage.append(code)
                else:inherited.append({'reason_codes':[code],'text':text})
            elif text in previous['unresolved'] and obligation['status']=='NO_REQUIRED_CHANGE':
                inherited.append({'reason_codes':[code],'text':text})
            else:reasons.append(code)
    events={e['event_id']:e for e in public['public_prefix_events_only']}
    latest=public['public_prefix_events_only'][-1] if events else None
    if latest and proofs.public_failure(latest):
        if not any(latest['event_id'] in f['source_refs'] for f in packet['progress']['failed_attempts']):reasons.append('FAILED_ATTEMPT_OMISSION')
    for f in delta['add_failed_attempts']:
        try:parsed=lang.failed(f['text'])
        except ValueError:coverage.append('FAILED_ATTEMPT_LANGUAGE_UNSUPPORTED');continue
        matches=[events[r] for r in f['source_refs'] if r in events and proofs.public_failure(events[r])]
        expected_tool='search' if parsed['kind']=='FAILED_SEARCH' else 'click';arg='keywords' if expected_tool=='search' else 'item'
        exact=[e for e in matches if e['action']['tool']==expected_tool and e['action']['arguments'].get(arg)==parsed['value']]
        if not exact or parsed['subject'] not in c['current_subjects']:reasons.append('FAILED_ATTEMPT')
        else:
            # Query may be a fragment, but the intended subject must be present
            # in the public action or bound active-page provenance, not guessed.
            name=c['subject_registry'][parsed['subject']]
            subject_ok=any((expected_tool=='search' and proofs.shop.norm(parsed['value'])==proofs.shop.norm(name)) or
                parsed['subject'] in trusted['source_ownership_registry'][e['event_id']]['subject_refs'] for e in exact)
            if not subject_ok:coverage.append('FAILED_ATTEMPT_SUBJECT_UNRESOLVED')
    # Removal is justified by redundancy or currently established supersession.
    for idx in delta['remove_evidence']:
        old_e=previous['evidence'][idx];old_a=old_atoms[idx]
        key=atom_key(old_e,old_a)
        duplicates=key in known if key else False
        corrected=bool(key and any(s==key[0] and p==key[1] and v!=key[2] for s,p,v in known))
        if corrected and old_a['status']!='PASS':repairs.append('WRONG_VALUE_REPAIR')
        elif duplicates:pass
        else:coverage.append('REMOVAL_JUSTIFICATION_AUDIT_UNRESOLVED')
    if delta['remove_failed_attempts']:coverage.append('FAILED_ATTEMPT_REMOVAL_UNRESOLVED')
    if packet['ready_to_commit']:
        if not gate['global_ready_allowed']:
            if any(c['classification']=='UNRESOLVED_REQUIREMENT_MAPPING' for c in certs.values()) or any(a['status']=='UNRESOLVED' for a in atoms):
                coverage.append('READINESS_PROOF_UNRESOLVED')
            else:reasons.append('READINESS')
        if packet['unresolved']:reasons.append('READINESS_WITH_UNRESOLVED')
    elif gate['global_ready_allowed']:reasons.append('READINESS_FALSE_DESPITE_PROOF')
    if row.get('replay_equal') is not True:reasons.append('DETERMINISTIC_REPLAY')
    local='FAIL' if reasons else 'UNRESOLVED' if coverage else 'PASS'
    full='FAIL' if full_issues or reasons else 'UNRESOLVED' if coverage or inherited else 'PASS'
    base.update(transition_local_verdict=local,full_packet_verdict=full,
        admission_verdict='PASS' if local=='PASS' else 'FAIL',gold_candidate=local=='PASS',
        review_status='AUTO_PASS' if local=='PASS' else 'AUTO_REJECT',
        inherited_errors=inherited,reason_codes=sorted(set(reasons)),coverage_reasons=sorted(set(coverage)),
        need_mapping=gate['need_certifications'],need_text_audit=[lang.text_audit(n,c) for n in packet['needs']],
        contract_audit=gate,history_audit=h,evidence_audit=atoms,obligation_audit=obligation,
        global_ready_allowed=gate['global_ready_allowed'],ready_result=gate['ready_result'],
        full_packet_errors=full_issues,transition_subtypes=subtypes(previous,packet,delta),
        factual_auditable=sum(a['claim'] is not None for a in atoms),factual_total=len(atoms),
        repair_types=sorted(set(repairs)),validator_status=local)
    return base
