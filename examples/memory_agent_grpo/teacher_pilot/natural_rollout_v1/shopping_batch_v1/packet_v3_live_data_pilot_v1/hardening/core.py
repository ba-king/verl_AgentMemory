"""Domain-neutral authority and receipt aggregation; no natural-language parser.

Adapters create requirements and bounded proposal interpretations. Auditors
produce factual receipts separately. None of these structures are Packet fields.
Only resolved, admitted needs contribute proof; local needs never gate readiness.
"""
import hashlib
import json
from decimal import Decimal


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def certify_need(need, contract, proposal):
    scope = need['scope_refs']; declared = set(scope)
    errors = []
    if declared - set(contract['subject_registry']):
        errors.append('UNKNOWN_SUBJECT_REF')
    if not scope or len(scope) != len(declared):
        errors.append('INVALID_SCOPE_REFS')
    base = {'need_id': need['id'], 'need_digest': digest(need), 'scope_refs': scope,
            'authority': contract['authority'], 'task_digest': contract['task_digest']}
    if proposal is None and errors:
        return {**base, 'classification': 'SCOPE_CONTRACT_VIOLATION',
                'reason_codes': errors, 'requirements': [], 'contributions': {}}
    if proposal is None:
        return {**base, 'classification': 'UNRESOLVED_REQUIREMENT_MAPPING',
                'reason_codes': errors, 'requirements': [], 'contributions': {}}
    errors.extend(proposal.get('errors', []))
    if 'allowed_scopes' in proposal:
        if declared not in [set(s) for s in proposal['allowed_scopes']]:
            errors.append('SCOPE_CONTRACT_VIOLATION')
    if 'subset_of' in proposal and not declared <= set(proposal['subset_of']):
        errors.append('SCOPE_CONTRACT_VIOLATION')
    if errors:
        return {**base, 'classification': 'SCOPE_CONTRACT_VIOLATION', 'role': proposal['role'],
                'reason_codes': sorted(set(errors)), 'requirements': [], 'contributions': {}}
    if proposal['role'] == 'LOCAL':
        return {**base, 'classification': 'LOCAL_ONLY', 'role': 'LOCAL',
                'reason_codes': [], 'requirements': [], 'contributions': {}}
    contributions = {}
    for rid in proposal['requirements']:
        requirement = contract['requirements'][rid]
        if requirement['kind'] == 'slots':
            refs = scope if requirement.get('use_declared_scope') else requirement['required_scope']
            dimensions = proposal.get('dimensions', requirement['required_dimensions'])
            if not set(dimensions) <= set(requirement['required_dimensions']):
                raise ValueError('INVALID_ADAPTER_DIMENSIONS')
            contributions[rid] = [{'subject_ref': s, 'dimension': d} for s in refs for d in dimensions]
        else:
            contributions[rid] = [{'subject_ref': s, 'dimension': d}
                                  for s in requirement.get('proof_scope', [contract['task_subject']])
                                  for d in requirement.get('proof_dimensions', ['proof'])]
    return {**base, 'classification': 'CERTIFIED_GATING', 'role': proposal['role'],
            'reason_codes': [], 'requirements': proposal['requirements'],
            'contributions': contributions,
            'certification_means': 'Scope/proposal admissible, NOT factual completion certified'}


def aggregate(packet, contract, certifications, receipts):
    """Union independent fragment proofs without expanding their admitted scope.

Receipt format is keyed by requirement: slots, assertion or certified balance.
Factual support is an auditor obligation, never inferred from resolved flags.
"""
    by_id = {c['need_id']: c for c in certifications}
    if len(by_id) != len(certifications):
        raise ValueError('DUPLICATE_NEED_IDS')
    values = {r: {} for r in contract['requirements']}
    assertions = set(); balances = {}; routes = {}; premature = []
    for need in packet['needs']:
        cert = by_id[need['id']]
        if cert['classification'] != 'CERTIFIED_GATING' or need['id'] not in packet['progress']['resolved']:
            continue
        for rid, permitted_rows in cert['contributions'].items():
            requirement = contract['requirements'][rid]
            receipt = receipts[need['id']].get(rid, {})
            if requirement['kind'] == 'slots':
                permitted = {(r['subject_ref'], r['dimension']) for r in permitted_rows}
                actual = receipt.get('slots', [])
                available = {(r['subject_ref'], r['dimension']) for r in actual}
                if requirement.get('check_resolve_slots') and permitted - available:
                    premature.append({'code': 'NEED_SCOPE', 'need_id': need['id'],
                                      'unproved_slots': [list(k) for k in sorted(permitted - available)]})
                for r in actual:
                    key = r['subject_ref'], r['dimension']
                    if key in permitted:
                        values[rid].setdefault(key, set()).add(r['value'])
            elif requirement['kind'] == 'assertion' and receipt.get('satisfied'):
                assertions.add(rid)
            elif requirement['kind'] == 'balance' and receipt.get('satisfied'):
                balances.setdefault(rid, set()).add(receipt['remaining'])
                routes.setdefault(rid, []).append(receipt['route'])
    results = {}
    for rid, req in contract['requirements'].items():
        if req['kind'] == 'slots':
            expected = {(s, d) for s in req['required_scope'] for d in req['required_dimensions']}
            found = values[rid]
            conflicts = [list(k) for k, v in found.items() if len(v) != 1]
            results[rid] = {'satisfied': expected <= set(found) and not conflicts,
                           'required_slots': [list(k) for k in sorted(expected)],
                           'missing_slots': [list(k) for k in sorted(expected - set(found))],
                           'conflicts': conflicts}
        elif req['kind'] == 'assertion':
            results[rid] = {'satisfied': rid in assertions}
        elif req['kind'] == 'balance':
            candidates = balances.get(rid, set())
            results[rid] = {'satisfied': len(candidates) == 1, 'routes': routes.get(rid, []),
                           'remaining_values': sorted(candidates)}
        else:
            raise ValueError('UNSUPPORTED_REQUIREMENT_KIND')
    return {'requirement_results': results, 'slot_values': values, 'premature_resolves': premature}


def global_readiness(requirement_results, decision_receipt, scope_errors, premature_resolves):
    return (bool(requirement_results) and all(r['satisfied'] for r in requirement_results.values())
            and decision_receipt.get('established', False) and not scope_errors and not premature_resolves)


def remaining_from_verified_prices(cap, required_history, prices):
    """Empty history is an exact zero sum; absent nonempty history is unknown."""
    if not set(required_history) <= set(prices):
        return None
    return format(Decimal(cap) - sum((Decimal(prices[s]) for s in required_history), Decimal(0)), '.2f')


def evaluate(packet, contract, interpret_need, audit_receipt, verify_decision):
    """Adapter protocol: bounded interpretation, independent proof, final check.

All three callbacks are trusted Harness code. The first returns admissible
proposal metadata; the second returns {proofs, detail}; the third consumes only
requirement results and proved slots. No callback is a model-generated function.
"""
    certificates = [certify_need(n, contract, interpret_need(n, contract)) for n in packet['needs']]
    by_id = {c['need_id']: c for c in certificates}
    if len(by_id) != len(certificates):
        raise ValueError('DUPLICATE_NEED_IDS')
    receipts = {n['id']: audit_receipt(n, by_id[n['id']]) for n in packet['needs']}
    result = aggregate(packet, contract, certificates, {n: r['proofs'] for n, r in receipts.items()})
    requirements = result['requirement_results']
    decision = verify_decision(requirements, result['slot_values'])
    errors = [c for c in certificates if c['classification'] == 'SCOPE_CONTRACT_VIOLATION']
    premature = result['premature_resolves']
    ready = global_readiness(requirements, decision, errors, premature)
    return {'need_certifications': certificates,
        'factual_receipts': {n: r['detail'] for n, r in receipts.items()},
        'requirement_results': requirements, 'scope_errors': errors,
        'premature_resolves': premature, 'decision_verifier': decision,
        'global_ready_allowed': ready,
        'ready_result': ('READY_ACCEPTED' if ready else 'READINESS_GATE') if packet['ready_to_commit'] else 'NO_READY_PROPOSAL',
        'local_need_ids': [c['need_id'] for c in certificates if c['classification'] == 'LOCAL_ONLY'],
        'unresolved_mapping_need_ids': [c['need_id'] for c in certificates if c['classification'] == 'UNRESOLVED_REQUIREMENT_MAPPING']}
