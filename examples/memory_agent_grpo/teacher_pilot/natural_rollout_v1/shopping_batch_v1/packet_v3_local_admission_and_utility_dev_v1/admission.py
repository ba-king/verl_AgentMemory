"""Independent admission; original responses, packets and audits stay intact."""
from collections import Counter
from .common import read, digest

UNCERTAIN = {'SOURCE_SUPPORT_UNPROVEN', 'SOURCE_OWNERSHIP_UNRESOLVED', 'FACTUAL_AUDIT_UNRESOLVED',
             'UNRESOLVED_LANGUAGE_UNSUPPORTED', 'HARNESS_PROPOSAL_LANGUAGE_COVERAGE'}
REPAIR_COVERAGE = {'OMISSION_AUDIT_UNRESOLVED', 'REMOVAL_JUSTIFICATION_AUDIT_UNRESOLVED',
                   'FAILED_ATTEMPT_REMOVAL_UNRESOLVED'}
REPAIR_ERRORS = {'CORRECTION', 'STALE_UNRESOLVED', 'FAILED_ATTEMPT_OMISSION', 'PROVEN_RESOLVE_OMISSION'}


def admit(t, a):
    f = a.get('frozen_audit') or {}
    reasons = set(f.get('reason_codes', [])); coverage = set(f.get('coverage_reasons', []))
    protocol = all(t.get(k) for k in ('schema_valid', 'apply_valid', 'replay_equal'))
    current_checks = []
    extra_error = []; extra_uncertain = []
    if protocol:
        # Explicitly check every proposal, including a duplicate of an inherited
        # fact. A new unsupported proposal cannot hide behind its old hash.
        packet = t['applied_packet']; delta = t['parsed_packet_delta']
        for e in delta['add_evidence']:
            matches = [atom for candidate, atom in zip(packet['evidence'], f.get('evidence_audit', [])) if candidate == e]
            own_issues = [x for x in f.get('full_packet_errors', []) if x.get('evidence_hash') == digest(e)]
            codes = set(c for x in own_issues for c in x['reason_codes'])
            supported = bool(matches) and all(x['status'] == 'PASS' for x in matches) and not codes
            if not supported:
                if codes - UNCERTAIN:
                    extra_error.extend(sorted(codes - UNCERTAIN))
                else:
                    extra_uncertain.append('CURRENT_NEW_EVIDENCE_NOT_CERTIFIED')
            current_checks.append({'evidence': e, 'supported': supported, 'atomic_receipts': matches,
                                   'reason_codes': sorted(codes)})
    inherited = f.get('inherited_errors', [])
    # This status describes old issues only when the frozen CURRENT obligation
    # checks have passed. It cannot excuse a current correction/omission.
    current_obligations_pass = not reasons and not coverage and not extra_error and not extra_uncertain
    unrelated_uncertain = [x for x in inherited if set(x.get('reason_codes', [])) & UNCERTAIN] if current_obligations_pass else []
    unrelated_errors = [x for x in inherited if set(x.get('reason_codes', [])) - UNCERTAIN] if current_obligations_pass else []
    reject_uncertain = bool(coverage or extra_uncertain)
    reject_omission = bool(reasons & {'PROVEN_OMISSION', 'OMISSION'})
    reject_repair = bool(reasons & REPAIR_ERRORS)
    reject_output = bool((reasons - {'PROVEN_OMISSION', 'OMISSION'} - REPAIR_ERRORS) or extra_error or not protocol)
    guard = bool(a.get('scaffold_guard', {}).get('passed'))
    accepted = bool(protocol and a.get('local_pass') and guard and current_obligations_pass)
    uncertainty = []
    if coverage - REPAIR_COVERAGE or extra_uncertain:
        uncertainty.append('CURRENT_OUTPUT_UNCERTAIN')
    if coverage & REPAIR_COVERAGE:
        uncertainty.append('CURRENT_REPAIR_OBLIGATION_UNRESOLVED')
    if unrelated_uncertain:
        uncertainty.append('INHERITED_UNRELATED_UNCERTAIN')
    return {k: t[k] for k in ('episode_id', 'session_id', 'event_id')} | {
        'admission_version': 'LOCAL_TRANSITION_SFT_ADMISSION_V1',
        'split': 'PIPELINE_UTILITY_DEV_ONLY', 'eligible_for_formal_train': False,
        'nonempty': bool(a.get('nonempty')), 'frozen_local_pass': bool(a.get('local_pass')),
        'frozen_transition_local_verdict': f.get('transition_local_verdict'),
        'frozen_full_packet_verdict': f.get('full_packet_verdict'),
        'original_audit_ref': t.get('audit_ref'), 'original_audit_sha256': digest(a),
        'original_reason_codes': f.get('reason_codes', []), 'original_coverage_reasons': f.get('coverage_reasons', []),
        'inherited_errors': inherited, 'new_evidence_checks': current_checks,
        'uncertainty_classification': uncertainty, 'additional_current_errors': extra_error,
        'additional_current_uncertainty': extra_uncertain,
        'LOCAL_TRANSITION_SFT_CANDIDATE': accepted and bool(a.get('nonempty')),
        'local_admission_pass': accepted, 'empty_auto_pass': accepted and not a.get('nonempty'),
        'label': 'AUTO_AUDITED_LOCAL_TRANSITION_CANDIDATE' if accepted and a.get('nonempty') else 'EMPTY_LOCAL_PASS' if accepted else 'REJECTED_LOCAL_TRANSITION',
        'reject_current_output_error': reject_output or not guard,
        'reject_current_omission': reject_omission,
        'reject_required_repair_failure': reject_repair,
        'reject_current_uncertainty': reject_uncertain,
        'inherited_unrelated_error_count': len(unrelated_errors),
        'inherited_unrelated_uncertainty_count': len(unrelated_uncertain),
        'raw_artifacts_modified': False}


def summarize(rr):
    active = [r for r in rr if r['nonempty']]
    out = {'TOTAL_TRANSITIONS': len(rr), 'TOTAL_NONEMPTY': len(active),
           'LOCAL_TRANSITION_PASS': sum(r['frozen_local_pass'] for r in active),
           'LOCAL_TRANSITION_CANDIDATES': sum(r['LOCAL_TRANSITION_SFT_CANDIDATE'] for r in active),
           'EMPTY_LOCAL_PASS': sum(r['empty_auto_pass'] for r in rr)}
    for k in ('reject_current_output_error', 'reject_current_omission', 'reject_required_repair_failure', 'reject_current_uncertainty'):
        out[k.upper()] = sum(bool(r[k]) for r in active)
    for k in ('inherited_unrelated_error_count', 'inherited_unrelated_uncertainty_count'):
        out[k.upper()] = sum(r[k] for r in active)
        out[k.upper().replace('_COUNT', '_TRANSITIONS')] = sum(bool(r[k]) for r in active)
    out['LOCAL_TRANSITION_CANDIDATE_YIELD'] = out['LOCAL_TRANSITION_CANDIDATES'] / len(active) if active else None
    out['counts_may_overlap'] = True
    out['split'] = 'PIPELINE_UTILITY_DEV_ONLY'
    return out


def candidate(t, a, r):
    return {'split': 'PIPELINE_UTILITY_DEV_ONLY', 'eligible_for_formal_train': False,
            'mode': 'PACKET', 'operation': 'UPDATE_PACKET',
            **{k: t[k] for k in ('episode_id', 'session_id', 'event_id')},
            'input': read(t['input_ref']), 'target_delta': t['parsed_packet_delta'],
            'admission': r, 'response_ref': t['response_ref'], 'audit_ref': t['audit_ref'],
            'teacher_generated': True, 'manual_repair': False}
