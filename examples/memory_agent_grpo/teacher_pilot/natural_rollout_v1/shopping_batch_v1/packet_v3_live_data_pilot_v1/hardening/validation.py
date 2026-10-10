"""Shopping wiring to generic certification/aggregation and frozen V3 validation.

A future adapter supplies its own bounded proposal/proof/decision functions;
core and the Packet schema need no domain-dependent branches.
"""
from .core import evaluate
from .shopping import CURRENT, BUDGET, MAPPING, ANCHOR, generic_contract, interpret_need, decide
from .shopping_proof import audit_need
from ...packet_interface_v3_minimal_validation.validator_v3 import validate as old_validate


def evaluate_packet(packet, contract, trusted, factual, history):
    c = generic_contract(contract)
    def receipt(n, certification):
        r = audit_need(n, certification, packet, c, trusted, factual, history)
        proofs = {
            CURRENT: {'slots': r['current_slots']},
            BUDGET: {'satisfied': r['budget_ok'], 'remaining': r.get('remaining'), 'route': r['budget_route']},
            MAPPING: {'satisfied': r['mapping_ok']}, ANCHOR: {'satisfied': r['anchor_ok']}}
        if c['previous_requirement']:
            proofs[c['previous_requirement']] = {'satisfied': r['previous_ok']}
        return {'proofs': proofs, 'detail': r}
    return evaluate(packet, c, interpret_need, receipt,
                    lambda requirements, slots: decide(c, requirements, slots, history))


def validate_transition(fixture, delta):
    """Preserve the frozen local/factual vs readiness distinction verbatim."""
    old = old_validate(fixture['v3_fixture'], delta)
    if old['packet'] is None:
        return {'transition_local_verdict': 'FAIL', 'reason_codes': old['reason_codes'], 'old_v3': old, 'contract': None}
    result = evaluate_packet(old['packet'], fixture['task_contract'], fixture['v3_fixture']['trusted'],
                             fixture['v3_fixture']['factual_audit'], fixture['history_audit'])
    replaced = {'NEED_SCOPE', 'READINESS_GATE', 'COMPLETION_AUDIT_UNAVAILABLE', 'INVALID_SCOPE_REFS'}
    preserved = [v for v in old['violations'] if v['code'] not in replaced
                 and not (v['code'] == 'UNKNOWN_SUBJECT_REF' and v.get('path') == '/needs_or_readiness')]
    reasons = {v['code'] for v in preserved}
    for cert in result['scope_errors']:
        reasons.update(cert['reason_codes'])
    reasons.update(v['code'] for v in result['premature_resolves'] if v['need_id'] in delta['resolve'])
    local_reasons = sorted(reasons)
    if result['ready_result'] == 'READINESS_GATE':
        reasons.add('READINESS_GATE')
    return {'transition_local_verdict': 'PASS' if not local_reasons else 'FAIL',
        'admission_verdict': 'PASS' if not reasons else 'FAIL',
        'local_reason_codes': local_reasons, 'reason_codes': sorted(reasons),
        'preserved_factual_violations': preserved, 'contract': result, 'old_v3': old,
        'full_packet_verdict': old['full_packet_verdict'], 'inherited_errors': old['inherited_errors']}
