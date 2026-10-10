"""Conservative public-cutoff audit. Unknown entailment never becomes gold.

Atomic interpretations are admitted only for the explicit forms below or exact
literal source quotations. Free prose/derived claims outside this finite audit
are labelled audit coverage, not silently accepted or called Teacher errors.
No private task, target, scorer or future event is an input.
"""
import copy
import re
from decimal import Decimal
from .common import digest
from .hardening.shopping import CURRENT, BUDGET, canonical, money, norm
from .hardening.need_text_consistency import audit as audit_text
from .hardening.validation import evaluate_packet
from .hardening.shopping_proof import certify_history
from ..packet_interface_v3_minimal_validation.trusted_registry import binding_check


def history_audit(contract, memories, histories):
    try:return certify_history(contract,memories,histories)
    except (ValueError,KeyError,TypeError) as exc:
        return {'memory_certificates':{},'public_history_receipts':{},'audit_unavailable':str(exc)}


def interpret(e, trusted):
    text=e['fact'];subject=e['subject_ref'];ownership=trusted['source_ownership_registry']
    sources=[ownership[r]['public_text'] for r in e['source_refs'] if r in ownership]
    # Exact full fact quote, or finite single-attribute assertion. No partial
    # recognition of a long sentence that might contain a second false claim.
    patterns=[r'(Material|Color):\s*([^;\n]+?)\.?',
              r'(?:Candidate )?(C\d+) (?:has|is) (Material|Color):?\s*"([^"]+)"\.',
              r'(?:Candidate )?(C\d+) (?:has )?price \$(\d+(?:\.\d+)?)\.']
    claims=[]
    m=re.fullmatch(patterns[0],text,re.I)
    if m:claims=[{'kind':'literal','predicate':m[1].lower(),'value':canonical(m[2])}]
    m=re.fullmatch(patterns[1],text,re.I)
    if m and m[1].upper()==subject:claims=[{'kind':'literal','predicate':m[2].lower(),'value':canonical(m[3])}]
    m=re.fullmatch(patterns[2],text,re.I)
    if m and m[1].upper()==subject:claims=[{'kind':'literal','predicate':'price','value':format(Decimal(m[2]).normalize(),'f')}]
    if not claims and any(norm(text)==norm(s) for s in sources):
        # An exact quote may be large. It is eligible only when the complete
        # source is a single literal declaration, not a page/status summary.
        m=re.fullmatch(patterns[0],text,re.I)
        if m:claims=[{'kind':'literal','predicate':m[1].lower(),'value':canonical(m[2])}]
    return claims


def factual_registry(previous, packet, trusted):
    interpretations={}
    for e in previous['evidence']+packet['evidence']:
        claims=interpret(e,trusted)
        if claims:interpretations[digest(e['fact'])]={'text':e['fact'],'claims':claims}
    return {'literal_interpretations':interpretations,'derived_claims':[]}


def supported(e,claim,trusted):
    return any(c['subject_ref']==e['subject_ref'] and c['predicate']==claim['predicate']
               and c['value']==claim['value'] and c['source_ref'] in e['source_refs']
               for c in trusted['allowed_claims_up_to_event'])


def failure_event(public):
    events=public['public_prefix_events_only']
    if not events:return None
    latest=events[-1];text=norm(latest['observation'])
    if any(s in text for s in ('not clickable','rejected','no results','no products found')):
        return {'event_id':latest['event_id'],'kind':'public_failure'}
    if latest['action']['tool']=='search':
        for old in events[:-1]:
            if old['action']==latest['action'] and old['observation']==latest['observation']:
                return {'event_id':latest['event_id'],'kind':'exact_repeated_query_and_observation'}
    return None


def evaluate(row, public, trusted, contract, histories):
    previous=row['previous_packet'];packet=row['applied_packet'];delta=row.get('parsed_packet_delta')
    base={'schema_valid':row['schema_valid'],'apply_valid':row['apply_valid'],
          'transition_local_verdict':'FAIL','full_packet_verdict':'NOT_APPLIED','admission_verdict':'FAIL',
          'global_ready_allowed':False,'ready_result':'NOT_APPLIED','inherited_errors':[],
          'reason_codes':[],'coverage_reasons':[],'review_status':'AUTO_REJECT',
          'annotation_origin':'PROGRAMMATIC_BOUNDED_PUBLIC_AUDIT','teacher_generated':True,
          'gold_candidate':False,'need_mapping':[],'need_text_audit':[],'transition_subtypes':[]}
    if not row['schema_valid'] or not row['apply_valid']:
        base['reason_codes']=['SCHEMA' if not row['schema_valid'] else 'APPLY'];return base
    reasons=[];coverage=[];inherited=[]
    factual=factual_registry(previous,packet,trusted)
    history=history_audit(contract,public['actual_retrieved_memories'],histories)
    gate=evaluate_packet(packet,contract,trusted,factual,history)
    base.update(global_ready_allowed=gate['global_ready_allowed'],ready_result=gate['ready_result'],
                need_mapping=gate['need_certifications'],contract_audit=gate,history_audit=history)
    by_id={n['id']:n for n in packet['needs']}
    new_ids={n['id'] for n in delta['add_needs']};touched=new_ids|set(delta['resolve'])|set(delta['add_pending'])
    if row['operation']=='INIT_PACKET' and not packet['needs']:reasons.append('OMISSION')
    for cert in gate['need_certifications']:
        if cert['classification']=='SCOPE_CONTRACT_VIOLATION' and cert['need_id'] in touched:
            reasons.extend(cert['reason_codes'])
        elif cert['classification']=='UNRESOLVED_REQUIREMENT_MAPPING' and cert['need_id'] in touched:
            coverage.append('HARNESS_PROPOSAL_LANGUAGE_COVERAGE')
    for n in packet['needs']:
        result=audit_text(n,contract,history,gate['requirement_results']);base['need_text_audit'].append(result)
        if result['status']!='PASS':
            if n['id'] in touched:
                if result['mismatches']:reasons.append('NEED_TEXT_CONSISTENCY')
                if result['unverified_claims']:coverage.append('NEED_TEXT_AUDIT_UNRESOLVED')
            else:inherited.append({'code':'NEED_TEXT_CONSISTENCY' if result['mismatches'] else 'NEED_TEXT_AUDIT_UNRESOLVED','need_id':n['id']})
    old_digests={digest(e) for e in previous['evidence']}
    full_errors=[];known_claims=[]
    for index,e in enumerate(packet['evidence']):
        codes=binding_check(e,trusted['subject_registry'],trusted['source_ownership_registry'])
        claims=interpret(e,trusted)
        if not claims:codes.append('FACTUAL_AUDIT_UNRESOLVED')
        for claim in claims:
            known_claims.append((index,e,claim))
            if not supported(e,claim,trusted):codes.append('SOURCE_SUPPORT')
        if codes:
            issue={'path':f'/evidence/{index}','reason_codes':sorted(set(codes))};full_errors.append(issue)
            if digest(e) in old_digests:inherited.append(issue)
            else:
                reasons.extend(c for c in codes if c!='FACTUAL_AUDIT_UNRESOLVED')
                if 'FACTUAL_AUDIT_UNRESOLVED' in codes:coverage.append('FACTUAL_AUDIT_UNRESOLVED')
    # Correction/omission check only for source-grounded, task-relevant new
    # literals. Unknown free prose cannot prove absence: report audit coverage.
    for literal in trusted['allowed_claims_from_latest_event']:
        if literal['predicate'] not in {contract['current_field'],'price'}:continue
        matches=[(e,cl) for _,e,cl in known_claims if e['subject_ref']==literal['subject_ref'] and cl['predicate']==literal['predicate']]
        correct=any(cl['value']==literal['value'] and supported(e,cl,trusted) for e,cl in matches)
        old_bad=any(cl['value']!=literal['value'] for e in previous['evidence'] if e['subject_ref']==literal['subject_ref']
                    for cl in interpret(e,trusted) if cl['predicate']==literal['predicate'])
        if old_bad and (not correct or any(cl['value']!=literal['value'] for e,cl in matches)):
            reasons.append('CORRECTION')
        if not correct:
            if any('FACTUAL_AUDIT_UNRESOLVED' in e['reason_codes'] for e in full_errors):coverage.append('KEY_OMISSION_AUDIT_UNRESOLVED')
            else:reasons.append('OMISSION')
    # Stale free-text status requires a full interpretation; never certify it
    # solely from a keyword or from final task success.
    for e in packet['evidence']:
        if re.search(r'not yet|unknown|not observed|unresolved',e['fact'],re.I):
            if digest(e) not in old_digests:coverage.append('STALE_STATE_AUDIT_UNRESOLVED')
    failure=failure_event(public)
    if failure and not any(failure['event_id'] in a['source_refs'] for a in packet['progress']['failed_attempts']):
        reasons.append('FAILED_ATTEMPT')
    if delta['add_failed_attempts']:
        for a in delta['add_failed_attempts']:
            if not failure or failure['event_id'] not in a['source_refs']:reasons.append('FAILED_ATTEMPT')
            else:coverage.append('FAILED_ATTEMPT_TEXT_AUDIT_UNRESOLVED')
    if packet['ready_to_commit'] and not gate['global_ready_allowed']:reasons.append('READINESS')
    if not packet['ready_to_commit'] and gate['global_ready_allowed']:reasons.append('READINESS')
    reasons.extend('NEED_SCOPE' for x in gate['premature_resolves'] if x['need_id'] in delta['resolve'])
    # Removal of unrelated facts / pending/unresolved maintenance needs semantic
    # review unless the current bounded audit can establish the transition.
    if delta['remove_evidence'] or delta['remove_failed_attempts']:coverage.append('REMOVAL_JUSTIFICATION_AUDIT_UNRESOLVED')
    if packet['unresolved']!=previous['unresolved']:coverage.append('UNRESOLVED_LIST_AUDIT_UNRESOLVED')
    if row.get('replay_equal') is not True:reasons.append('DETERMINISTIC_REPLAY')
    names={'add_needs':'ADD_NEED','add_evidence':'ADD_EVIDENCE','remove_evidence':'REMOVE_EVIDENCE',
           'resolve':'RESOLVE','add_pending':'REOPEN','add_failed_attempts':'FAILED_ATTEMPT'}
    subtypes=[name for key,name in names.items() if delta[key]]
    if packet['unresolved']!=previous['unresolved']:subtypes.append('UNRESOLVED_CHANGE')
    if packet['ready_to_commit']!=previous['ready_to_commit']:subtypes.append('READINESS_CHANGE')
    if not subtypes and not delta['remove_pending'] and not delta['remove_failed_attempts']:subtypes=['EMPTY_DELTA']
    # A NOOP can be good only if no pending audit gap/required correction exists.
    local='FAIL' if reasons else 'UNRESOLVED' if coverage else 'PASS'
    full='FAIL' if full_errors or inherited or gate['scope_errors'] or (packet['ready_to_commit'] and not gate['global_ready_allowed']) else 'UNRESOLVED' if coverage else 'PASS'
    base.update(transition_local_verdict=local,admission_verdict='PASS' if local=='PASS' else 'FAIL',
        full_packet_verdict=full,inherited_errors=inherited,reason_codes=sorted(set(reasons)),
        coverage_reasons=sorted(set(coverage)),gold_candidate=local=='PASS',
        review_status='AUTO_PASS' if local=='PASS' else 'AUTO_REJECT',transition_subtypes=subtypes,
        full_packet_errors=full_errors,validator_status=local,
        limits='Finite atomic literal/price audit; complex prose/derivations/removals remain audit-unresolved, not certified by citation presence.')
    return base
