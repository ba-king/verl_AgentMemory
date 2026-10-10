"""Conservative Travel public-support receipts for unchanged local admission.

The existing Shopping factual auditor is not a Travel verifier. This adapter
certifies exact subject-bound field claims; unproved prose/derivations remain
uncertain and cannot become positive supervision. It never filters live Packet.
"""
import json
import re
from .common import digest
from .adapter import guard, BASE_PATHS
from teacher_pilot.natural_rollout_v1.shopping_batch_v1.packet_v3_local_admission_and_utility_dev_v1.admission import admit


def atom(evidence, sources, subjects):
    codes = []
    subject = evidence['subject_ref']
    if subject not in subjects:
        codes.append('UNKNOWN_SUBJECT_REF')
    if any(ref not in sources for ref in evidence['source_refs']):
        codes.append('SOURCE_EXISTENCE')
    cited = [sources[r] for r in evidence['source_refs'] if r in sources]
    if not any(subject in s['subjects'] for s in cited):
        codes.append('SUBJECT_BINDING')
    match = re.fullmatch(r'([A-Za-z_][A-Za-z_0-9.]*)\s*=\s*(.+)', evidence['fact'].strip(), re.S)
    path, value = (match[1], match[2]) if match else (None, None)
    if value and value.startswith('"'):
        try:
            value = json.loads(value)
        except ValueError:
            value = None
    proved = False
    known_values = []
    for source in cited:
        if subject not in source['subjects']:
            continue
        record = source.get('record')
        if record and record['subject'] == subject.removeprefix('person:'):
            known_values.extend(f['value'] for f in record['fields'] if f['path'] == path)
            proved |= any(f['path'] == path and f['value'] == value for f in record['fields'])
        if source.get('bootstrap') and subject == 'person:' + source['bootstrap']['name']:
            for day in source['bootstrap']['daily_plans']:
                known_values.extend(v for k, v in day.items() if path == f"day{day['days']}.{k}" and k != 'days')
                proved |= any(path == f"day{day['days']}.{k}" and v == value for k, v in day.items() if k != 'days')
    if not proved and not codes:
        codes.append('EXACT_FIELD_DRIFT' if known_values else 'SOURCE_SUPPORT_UNPROVEN')
    return {'status': 'PASS' if proved and not codes else 'UNRESOLVED' if codes == ['SOURCE_SUPPORT_UNPROVEN'] else 'FAIL',
            'reason_codes': codes, 'subject_ref': subject, 'path': path, 'value': value,
            'scope': 'Literal historical/public bootstrap fields only; no inferred cross-person equality.'}


def evaluate(row, scaffold, sources, subjects):
    packet = row.get('applied_packet') or row['previous_packet']
    protocol = all(row.get(k) for k in ('schema_valid', 'apply_valid', 'replay_equal'))
    audits = [atom(e, sources, subjects) for e in packet['evidence']]
    before = row['previous_packet']
    old_hashes = {digest(e) for e in before['evidence']}
    errors, inherited, reasons, coverage = [], [], set(), set()
    for evidence, result in zip(packet['evidence'], audits):
        need = next((n for n in packet['needs'] if n['id'] == evidence['need_id']), None)
        if need and evidence['subject_ref'] not in need['scope_refs']:
            result['reason_codes'].append('SUBJECT_OUTSIDE_NEED_SCOPE')
            result['status'] = 'FAIL'
        if result['status'] == 'PASS':
            continue
        entry = {'evidence_hash': digest(evidence), 'reason_codes': result['reason_codes']}
        errors.append(entry)
        if digest(evidence) in old_hashes:
            inherited.append(entry)
        else:
            for code in result['reason_codes']:
                (coverage if code == 'SOURCE_SUPPORT_UNPROVEN' else reasons).add(code)
    contract = scaffold['contract']
    need_ids = {n['id'] for n in packet['needs']}
    for need in packet['needs']:
        if not need['scope_refs'] or set(need['scope_refs']) - set(subjects):
            reasons.add('SCOPE_CONTRACT_VIOLATION')
    for need in scaffold['packet']['needs']:
        if need not in packet['needs']:
            reasons.add('GLOBAL_NEED_CHANGED')
    proven = {(a['subject_ref'], a['path']): a['value'] for a in audits if a['status'] == 'PASS'}
    newly_resolved = set(packet['progress']['resolved']) - set(before['progress']['resolved'])
    resolve_receipts = []
    for nid in newly_resolved:
        kind = contract['need_kinds'].get(nid)
        proved = False
        if kind == 'BASE_FIELDS':
            proved = all((f"person:{contract['base']}", p) in proven for p in BASE_PATHS)
        elif kind == 'PRIOR_LODGING' and contract['rule_kind'] == 'copy':
            proved = (f"person:{contract['previous']}", 'day1.accommodation') in proven
        if not proved:
            coverage.add('TRAVEL_RESOLVE_PROOF_UNCERTIFIED')
        resolve_receipts.append({'need_id': nid, 'kind': kind, 'proved': proved})
    if packet['ready_to_commit']:
        # No automated complete-plan comparison certificate is fabricated from
        # a model's flag. Runtime sees the flag; SFT admission stays fail-closed.
        coverage.add('TRAVEL_COMPLETE_PLAN_READINESS_UNCERTIFIED')
    if not set(packet['progress']['pending'] + packet['progress']['resolved']) <= need_ids:
        reasons.add('INVALID_NEED_REFERENCE')
    if set(packet['progress']['resolved']) & set(packet['progress']['pending']):
        reasons.add('CONTRADICTORY_PROGRESS')
    # Conservative coverage: known required base fields cannot simply disappear.
    known_base = set()
    base = contract['base']
    for source in sources.values():
        if source.get('bootstrap', {}).get('name') == base:
            known_base.update(BASE_PATHS)
        rec = source.get('record', {})
        if rec.get('subject') == base:
            known_base.update(f['path'] for f in rec['fields'] if f['path'] in BASE_PATHS)
    missing = sorted(p for p in known_base if (f'person:{base}', p) not in proven)
    if missing:
        coverage.add('TRAVEL_BASE_COVERAGE_UNCERTIFIED')
    if packet['progress']['failed_attempts']:
        coverage.add('TRAVEL_FAILED_ATTEMPT_TEXT_UNCERTIFIED')
    if packet['unresolved']:
        coverage.add('TRAVEL_UNRESOLVED_TEXT_UNCERTIFIED')
    g = guard(row.get('parsed_packet_delta'), scaffold)
    passed = protocol and not reasons and not coverage and g['passed']
    frozen_audit = {'reason_codes': sorted(reasons), 'coverage_reasons': sorted(coverage),
        'evidence_audit': audits, 'full_packet_errors': errors, 'inherited_errors': inherited,
        'transition_local_verdict': 'PASS' if passed else 'REJECT_OR_UNCERTIFIED',
        'full_packet_verdict': 'PASS' if passed and not inherited else 'UNPROVEN',
        'backend': 'TRAVEL_EXACT_PUBLIC_SUPPORT_V1',
        'shopping_factual_auditor_reused': False,
        'resolve_receipts': resolve_receipts, 'uncertified_base_paths': missing}
    result = {'frozen_audit': frozen_audit, 'scaffold_guard': g,
              'local_pass': passed, 'nonempty': packet != before if protocol else bool(row.get('parsed_packet_delta'))}
    admitted = admit(row, result)
    admitted['split'] = 'FINAL_CONFIG_TEACHER_DEMO_ONLY'
    admitted['eligible_for_formal_train'] = False
    admitted['verifier_limit'] = 'Uncertified means insufficient automatic proof, not necessarily an incorrect Packet. Never counted as a positive.'
    return result, admitted
