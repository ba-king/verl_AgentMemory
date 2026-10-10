"""V3 structural checks plus bounded external factual audit. Zero LLM calls.

binding_check reads only explicit refs and trusted provenance. scope_check reads
only explicit scope refs and factual completion receipts. Neither reads text.
The high-level validator never imports V1's validator or migration annotator.
"""
import copy
from .interface_v3 import apply
from .trusted_registry import binding_check
from .factual_audit import claims_for, supported, completion_receipts
from .common import digest


def scope_check(packet, receipts, subjects, resolve_ids, trusted_scope_lock=None):
    """Design A by default. Design B is an EXPLICIT optional lock experiment.

No claim is made that schema authenticates the correctness of model-created
scope sets or the truth of supplied completion receipts.
"""
    errors = []
    for need in packet['needs']:
        nid = need['id']; scope = need['scope_refs']
        if not scope or len(scope) != len(set(scope)):
            errors.append({'code': 'INVALID_SCOPE_REFS', 'need_id': nid})
        if set(scope) - set(subjects):
            errors.append({'code': 'UNKNOWN_SUBJECT_REF', 'need_id': nid,
                           'unknown_refs': sorted(set(scope) - set(subjects))})
        if trusted_scope_lock is not None and nid in trusted_scope_lock and set(scope) != set(trusted_scope_lock[nid]):
            errors.append({'code': 'SCOPE_CONTRACT_VIOLATION', 'need_id': nid,
                           'declared': scope, 'trusted': trusted_scope_lock[nid]})
        if nid not in receipts:
            if nid in resolve_ids:
                errors.append({'code': 'COMPLETION_AUDIT_UNAVAILABLE', 'need_id': nid})
            continue
        missing = [s for s in scope if not receipts[nid].get(s, {}).get('satisfied', False)]
        if nid in resolve_ids and missing:
            errors.append({'code': 'NEED_SCOPE', 'need_id': nid, 'missing_scope_refs': missing})
        if packet['ready_to_commit'] and missing:
            errors.append({'code': 'READINESS_GATE', 'need_id': nid, 'missing_scope_refs': missing})
    return errors


def validate(fixture, delta, trusted_scope_lock=None):
    previous = fixture['previous_packet_v3']; trusted = fixture['trusted']; audit = fixture['factual_audit']
    subjects = trusted['subject_registry']; ownership = trusted['source_ownership_registry']
    violations = []
    def fail(code, path, **detail):
        violations.append({'code': code, 'path': path, **detail})
    for i, evidence in enumerate(delta.get('add_evidence', [])):
        if 'subject_ref' in evidence:
            for code in binding_check(evidence, subjects, ownership):
                fail(code, f'/add_evidence/{i}', subject_ref=evidence['subject_ref'], source_refs=evidence['source_refs'])
    try:
        packet = apply(previous, delta, ownership)
    except Exception as error:
        fail('DELTA_STRUCTURE', '/', detail=str(error))
        return {'transition_local_verdict': 'FAIL', 'reason_codes': sorted({v['code'] for v in violations}),
            'violations': violations, 'packet': None, 'full_packet_verdict': 'NOT_APPLIED', 'inherited_errors': []}
    for i, evidence in enumerate(delta['add_evidence']):
        claims = claims_for(audit, evidence)
        if not claims:
            fail('FACTUAL_AUDIT_UNRESOLVED', f'/add_evidence/{i}',
                 detail='Unknown predicate/value meaning; structural binding has its separate verdict.')
            continue
        for claim in claims:
            if claim['kind'] == 'failed_search':
                if audit['observed_failure']:
                    fail('FAILED_ATTEMPT_SEPARATION', f'/add_evidence/{i}')
            elif not supported(audit, trusted, claim, evidence['source_refs']):
                fail('SOURCE_SUPPORT', f'/add_evidence/{i}', claim=claim)

    latest = [c for c in trusted['allowed_claims_from_latest_event'] if c['predicate'] != 'price']
    for literal in latest:
        def matching(records):
            return [(i, e, c) for i, e in enumerate(records) for c in (claims_for(audit, e) or [])
                if c['subject_ref'] == literal['subject_ref'] and c['predicate'] == literal['predicate']]
        old = matching(previous['evidence']); new = matching(packet['evidence'])
        def bad(rows):
            return [i for i, e, c in rows if c['kind'] == 'literal' and
                (c['value'] != literal['value'] or not supported(audit, trusted, c, e['source_refs']))]
        correct = any(c['kind'] == 'literal' and c['value'] == literal['value']
                      and supported(audit, trusted, c, e['source_refs']) for _, e, c in new)
        if bad(old) and (bad(new) or not correct):
            fail('CORRECTION_ON_NEW_LITERAL_EVIDENCE', '/evidence', latest=literal,
                 remaining_bad_indices=bad(new), correct_replacement_present=correct)
        stale_before = any(c['kind'] == 'unobserved' for _, _, c in old)
        stale_after = [i for i, _, c in new if c['kind'] == 'unobserved']
        if stale_before and (stale_after or not correct):
            fail('STALE_STATE_CLEANUP', '/evidence', subject_ref=literal['subject_ref'], remaining_indices=stale_after)

    receipts = completion_receipts(audit, trusted)
    for error in scope_check(packet, receipts, subjects, delta['resolve'], trusted_scope_lock):
        violations.append({'path': '/needs_or_readiness', **error})
    failure = audit['observed_failure']
    if failure:
        if not any(set(failure['source_refs']) <= set(a['source_refs']) for a in packet['progress']['failed_attempts']):
            fail('FAILED_ATTEMPT_SEPARATION', '/progress/failed_attempts')
        for i, attempt in enumerate(delta['add_failed_attempts']):
            if attempt['text'] != audit['allowed_failed_attempt_text'] or not set(failure['source_refs']) <= set(attempt['source_refs']):
                fail('SOURCE_SUPPORT', f'/add_failed_attempts/{i}')
    elif delta['add_failed_attempts']:
        fail('SOURCE_SUPPORT', '/add_failed_attempts', detail='No observed failed event in this fixture')

    full = []; inherited = []
    old_digests = {digest(e) for e in previous['evidence']}
    known = {(c['subject_ref'], c['predicate']) for c in trusted['allowed_claims_up_to_event'] if c['predicate'] != 'price'}
    # Inherited errors must already be evidenced BEFORE this latest event.
    known_before = {(c['subject_ref'], c['predicate']) for c in trusted['allowed_claims_up_to_event']
                    if c['predicate'] != 'price' and c['source_ref'] != fixture['event_id']}
    for i, evidence in enumerate(packet['evidence']):
        errors = []
        for code in binding_check(evidence, subjects, ownership):
            errors.append({'code': code, 'subject_ref': evidence['subject_ref']})
        for claim in claims_for(audit, evidence) or []:
            pair = claim['subject_ref'], claim['predicate']
            if claim['kind'] == 'unobserved' and pair in known:
                errors.append({'code': 'STALE_STATE_CLEANUP', 'claim': claim,
                               'known_before_latest': pair in known_before})
            elif claim['kind'] == 'literal' and not supported(audit, trusted, claim, evidence['source_refs']):
                errors.append({'code': 'SOURCE_SUPPORT', 'claim': claim})
        for error in errors:
            item = {'path': f'/evidence/{i}', **error}
            full.append(item)
            if digest(evidence) in old_digests and error.get('known_before_latest', True):
                inherited.append(copy.deepcopy(item))
    full.extend(scope_check(packet, receipts, subjects, packet['progress']['resolved'], trusted_scope_lock))
    full.extend(copy.deepcopy(violations))
    return {'transition_local_verdict': 'PASS' if not violations else 'FAIL',
        'reason_codes': sorted({v['code'] for v in violations}), 'violations': violations,
        'packet': packet, 'full_packet_verdict': 'PASS' if not full else 'FAIL',
        'full_packet_violations': full, 'inherited_errors': inherited,
        'completion_receipts': receipts,
        'audit_limits': 'Bounded factual/repair registry. Readiness checks audited candidate coverage, not all possible commitment semantics. Full PASS is scoped to these checks.'}


def run(fixtures):
    results = []
    for f in fixtures:
        actual = validate(f, f['actual_delta_v3']); oracle = validate(f, f['oracle_delta_v3'])
        wanted = f['expected_actual_delta_result']
        matches = ((actual['transition_local_verdict'] == 'PASS') == wanted['transition_local_pass']
                   and set(wanted['required_reason_codes']) <= set(actual['reason_codes']))
        expected_match = oracle['packet'] == f['expected_packet_v3']
        results.append({'case_id': f['case_id'], 'actual': actual, 'offline_oracle': oracle,
            'expected_actual_matches': matches, 'expected_packet_matches': expected_match,
            'regression_pass': matches and expected_match and oracle['transition_local_verdict'] == 'PASS'})
    return results
