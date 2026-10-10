"""Offline, field-wise grounding. No actor, scorer, catalog, eval or LLM imports.

Public evidence is scoped to the purchased subject. Verified historical facts
can be reused only when that exact Bank version is in the actual source input.
Unknown fields remain uncertain; downstream success never substitutes for truth.
"""
import copy
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

SAFE = {'OBSERVED_EXACT', 'OBSERVED_NORMALIZED', 'DERIVED_SAFE'}
MONEY_RE = r'\$\d+(?:,\d{3})*\.\d{2}'
FIELD_ROLES = {
    'purchased_item_id': 'identity', 'purchased_asin': 'identity', 'asin': 'identity',
    'item_id': 'identity', 'product_id': 'identity',
    'purchased_item_name': 'name', 'product_name': 'name', 'item_name': 'name',
    'price_paid': 'price', 'paid_price': 'price', 'purchased_item_price': 'price',
    'purchase_price': 'price', 'price': 'price',
    'purchases_made': 'purchase_count', 'required_targets': 'required_count',
    'total_expense': 'session_total',
    'total_expense_after_purchase': 'cumulative', 'cumulative_expense': 'cumulative',
    'cumulative_expense_after_purchase': 'cumulative', 'cumulative_spent': 'cumulative',
    'cumulative_spent_after_purchase': 'cumulative', 'total_spent': 'cumulative',
    'total_spent_after_purchase': 'cumulative', 'cumulative_total': 'cumulative',
    'remaining_budget': 'remaining', 'remaining_budget_after_purchase': 'remaining',
    'budget_remaining': 'remaining', 'initial_budget': 'budget', 'total_budget': 'budget',
}


def norm(x):
    return re.sub(r'\s+', ' ', str(x)).strip().casefold()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cents(value):
    # No floats, tolerance, expression eval, exponent, NaN or fractional cents.
    if not isinstance(value, str) or not re.fullmatch(r'\$?\d+(?:,\d{3})*(?:\.\d{1,2})?', value.strip()):
        raise ValueError('not_exact_nonnegative_money')
    amount = Decimal(value.strip().replace('$', '').replace(',', '')) * 100
    if amount != amount.to_integral_value():
        raise ValueError('fractional_cent')
    return int(amount)


def role(path):
    key = path.casefold().replace('.', '_')
    if key in FIELD_ROLES:
        return FIELD_ROLES[key]
    for attribute in ('material', 'color', 'size'):
        if key in {attribute, attribute + '_label', 'product_' + attribute,
                   'purchased_item_' + attribute, 'attributes_' + attribute}:
            return attribute
    return None


def source_ref(ref, pointer, excerpt):
    return {'request_ref': ref, 'outer_pointer': '/messages/1/content',
            'decode_json_then_pointer': pointer, 'excerpt': excerpt, 'visibility': 'PUBLIC_ACTUAL_INPUT'}


def actual_input(decision):
    request = json.loads(Path(decision['request_ref']).read_text())
    if request['messages'] != decision['memory_teacher_messages']:
        raise ValueError('actual_memory_input_mismatch')
    state = json.loads(request['messages'][1]['content'])
    if any(k in state for k in ('target_asin', 'hidden_target', 'strict_validation', 'future_sessions', 'native_score')):
        raise ValueError('private_state_in_actor_input')
    return state


def public_facts(state, request_ref):
    """Only actual selected-product pages and submitted public purchase receipt."""
    task = state.get('current_task', {})
    sid = task.get('session_id', '')
    instruction = task.get('instruction', '')
    submit = state.get('submitted_result', {})
    receipt = submit.get('public_text', '')
    found = re.findall(r'Purchase #\d+: (B[A-Z0-9]{9}) \((' + MONEY_RE + r')\)', receipt)
    result = {'facts': {}, 'literal_declarations': [], 'pages': [], 'session_id': sid,
              'completed_purchase': bool(submit.get('purchase_submitted')) and len(found) == 1,
              'current_attribute': None, 'source_event_id': submit.get('source_event_id'),
              'conflicts': []}
    current = re.search(r'The field for the current item is (\w+)\.', instruction)
    if current:
        result['current_attribute'] = current[1].casefold()
    if not result['completed_purchase']:
        return result
    asin, price = found[0]
    result.update(asin=asin, price_cents=cents(price))

    def add(kind, value, pointer, excerpt=None):
        proof = source_ref(request_ref, pointer, value if excerpt is None else excerpt)
        fact = {'role': kind, 'value': value, 'provenance': [proof], 'session_id': sid, 'asin': asin}
        existing = result['facts'].get(kind)
        if existing and norm(existing['value']) != norm(value):
            result['conflicts'].append(kind)
        else:
            result['facts'][kind] = fact

    add('identity', asin, '/submitted_result/public_text')
    add('price', price, '/submitted_result/public_text')
    for kind, pattern in [('session_total', r'Total Expense: (' + MONEY_RE + r')'),
                          ('purchase_count', r'Purchases Made: (\d+)'),
                          ('required_count', r'Required Targets: (\d+)')]:
        m = re.search(pattern, receipt)
        if m:
            add(kind, m[1], '/submitted_result/public_text', m[0])
    budget = re.search(r'fixed total budget for all six purchases is (' + MONEY_RE + r')', instruction)
    if budget:
        add('budget', budget[1], '/current_task/instruction', budget[0])
    active = None
    for index, event in enumerate(state.get('session_public_history', [])):
        action = event.get('action', {})
        observation = event.get('observation', '')
        if observation.startswith('Click rejected:') or event.get('execution_status') == 'error':
            continue
        tool = action.get('tool'); item = action.get('arguments', {}).get('item', '')
        if tool == 'search' or item.casefold() == 'back to search':
            active = None
        elif tool == 'click' and re.fullmatch(r'B[A-Z0-9]{9}', item):
            active = item
        marker = '[SEP] < Prev [SEP]'
        if active != asin or marker not in observation or tool != 'click':
            continue
        body = observation.split(marker, 1)[1].strip()
        pointer = f'/session_public_history/{index}/observation'
        result['pages'].append({'event_id': event['event_id'], 'item': item, 'body': body,
                                'provenance': source_ref(request_ref, pointer, body)})
        if item == asin:
            name = body.split('[SEP]', 1)[0].strip()
            add('name', name, pointer)
        if item.casefold() not in {'features', 'description'}:
            continue
        # Strip task prefix, bind active ASIN, and require a named declaration.
        # Stop at sentence punctuation before another key; retain decimal words.
        for m in re.finditer(r'(?:^|\[SEP\]\s*|\n)(Material|Color|Size):\s*([^\n]*?)(?=\s*\[SEP\]|$)', body, re.I):
            value = re.split(r'\.\s*(?=[A-Z][A-Za-z ]+:)', m[2].strip())[0].rstrip('.').strip()
            declaration = m[1] + ': ' + value
            add(m[1].casefold(), value, pointer, declaration)
            result['literal_declarations'].append({'field': m[1].casefold(), 'value': value,
                'event_id': event['event_id'], 'provenance': source_ref(request_ref, pointer, declaration)})
    return result


def ownership(record, facts, episode_id):
    if not facts['completed_purchase']:
        return {'passed': False, 'contradiction': False, 'reason': 'NO_COMPLETED_PUBLIC_PURCHASE'}
    sid = facts['session_id']; asin = facts['asin']
    index = int(sid[1:]) if re.fullmatch(r'p\d+', sid) else None
    subject = record['subject']; scope = record['scope']
    explicit_steps = re.findall(r'\bProduct[ _]*(\d+)\b|:p(\d+)\b', subject + ' ' + scope, re.I)
    if index is not None and any(int(a or b) != index for a, b in explicit_steps):
        return {'passed': False, 'contradiction': True, 'reason': 'WRONG_PRODUCT_STEP_BINDING'}
    subject_asins = re.findall(r'\bB[A-Z0-9]{9}\b', subject)
    if any(v != asin for v in subject_asins):
        return {'passed': False, 'contradiction': True, 'reason': 'WRONG_SUBJECT_ASIN'}
    ids = [f['value'] for f in record['fields'] if role(f['path']) == 'identity']
    if ids and any(v != asin for v in ids):
        return {'passed': False, 'contradiction': True, 'reason': 'WRONG_PURCHASED_IDENTITY'}
    expected_subjects = {f'{episode_id}:{sid}', asin, f'Product {index}', f'Product {index} purchase'}
    name = facts['facts'].get('name', {}).get('value')
    if name:
        expected_subjects.add(name)
    subject_ok = norm(subject) in {norm(v) for v in expected_subjects}
    namespace_ok = str(facts.get('source_event_id', '')).startswith(f'{episode_id}:{sid}:')
    return {'passed': subject_ok and namespace_ok and bool(ids or subject == asin or subject == name),
            'contradiction': False, 'reason': 'PUBLIC_PURCHASE_SUBJECT_BINDING' if subject_ok else 'SUBJECT_NOT_PROVEN',
            'public_receipt_provenance': facts['facts']['identity']['provenance']}


class ProvenanceLedger:
    """Only versions actually written successfully and independently grounded."""
    def __init__(self):
        self.versions = {}

    def register(self, result, entry):
        if not result['ownership']['passed'] or not result['schema_valid'] or not result['execution_success']:
            return
        fields = {r['path']: r for r in result['fields'] if r['label'] in SAFE}
        self.versions[(entry['memory_id'], digest(entry['record']))] = {
            'fields': fields, 'session_id': result['session_id'], 'asin': result.get('asin'),
            'source_decision_ref': result['source_decision_ref'], 'entry': copy.deepcopy(entry)}

    def visible(self, state, request_ref):
        found = []; unverified = []
        for i, hit in enumerate(state.get('retrieved_memories', [])):
            key = (hit.get('memory_id'), digest(hit.get('record')))
            old = self.versions.get(key)
            if not old or hit.get('content') != old['entry']['content']:
                unverified.append(hit.get('memory_id')); continue
            for j, f in enumerate(hit['record']['fields']):
                proof = old['fields'].get(f['path'])
                if not proof or proof['value'] != f['value']:
                    continue
                found.append({'role': proof['role'], 'value': f['value'],
                    'session_id': old['session_id'], 'asin': old['asin'],
                    'provenance': [source_ref(request_ref, f'/retrieved_memories/{i}/record/fields/{j}/value', f['value'])],
                    'original_provenance': proof['provenance'], 'original_derivation': proof.get('derivation'),
                    'original_source_decision_ref': old['source_decision_ref']})
        return found, unverified


def derive_total(facts, visible):
    sid = facts['session_id']
    if not re.fullmatch(r'p[1-9]\d*', sid):
        return None, 'UNKNOWN_PURCHASE_ORDER'
    index = int(sid[1:]); current = facts['facts'].get('price')
    if current is None:
        return None, 'CURRENT_PRICE_NOT_PUBLIC'
    # Every operand must be visible NOW. Do not reach into an invisible ledger.
    previous = {}; conflict = False
    for f in visible:
        if f['role'] != 'price' or not re.fullmatch(r'p[1-9]\d*', f['session_id']):
            continue
        j = int(f['session_id'][1:])
        if j >= index:
            continue
        if j in previous and (cents(previous[j]['value']), previous[j]['asin']) != (cents(f['value']), f['asin']):
            conflict = True
        previous[j] = f
    if conflict:
        return None, 'CONFLICTING_VERIFIED_VISIBLE_PRICES'
    if set(previous) == set(range(1, index)):
        operands = [previous[j] for j in sorted(previous)] + [current]
        return {'operation': 'MONETARY_ADDITION', 'semantic_type': 'CUMULATIVE_ACTUAL_PURCHASE_EXPENSE',
                'operands': operands, 'result_cents': sum(cents(o['value']) for o in operands)}, None
    # An independently verified visible cumulative state is a permissible operand.
    aggregates = [f for f in visible if f['role'] == 'cumulative' and f['session_id'] == f'p{index-1}']
    if aggregates and len({cents(f['value']) for f in aggregates}) == 1:
        operands = [aggregates[0], current]
        return {'operation': 'MONETARY_ADDITION', 'semantic_type': 'VERIFIED_PREFIX_PLUS_CURRENT_PURCHASE',
                'operands': operands, 'result_cents': sum(cents(o['value']) for o in operands)}, None
    return None, 'REQUIRED_HISTORICAL_OPERAND_NOT_VISIBLE_OR_NOT_VERIFIED'


def audit_write(decision, state, ledger, schema_valid, execution_success, downstream_valid=False, source_decision_ref=''):
    """Pure audit core; production wrapper authenticates request/action/Bank first."""
    action = decision['parsed_memory_action']; record = action['arguments']['record']
    ref = decision.get('request_ref', 'SYNTHETIC_FIXTURE')
    facts = public_facts(state, ref)
    own = ownership(record, facts, decision['episode_id'])
    visible, unverified = ledger.visible(state, ref)
    total, total_error = derive_total(facts, visible) if facts['completed_purchase'] else (None, 'NO_PURCHASE')
    output = []
    for f in record['fields']:
        kind = role(f['path']); label = 'UNSUPPORTED'; reason = 'UNKNOWN_FIELD_ROLE'; proof = []; derivation = None
        value = f['value']; contradiction = False
        if re.search(r'private|hidden|sealed|target_asin|gold_answer', f['path'], re.I):
            label = 'PRIVATE_LEAK'; reason = 'PRIVATE_SEMANTIC_FIELD_NOT_ALLOWED'
        elif kind in {'cumulative', 'remaining'}:
            # Derived monetary semantics are accepted only under this public contract.
            contract = state.get('current_task', {}).get('instruction', '')
            explicit_contract = ('prices actually paid for ALL earlier Products' in contract
                and 'There are no shipping fees or price-changing options.' in contract)
            if not explicit_contract:
                reason = 'ARITHMETIC_CONTRACT_NOT_PUBLIC'
            elif total is None:
                reason = total_error
            else:
                derivation = copy.deepcopy(total)
                if kind == 'remaining':
                    budget = facts['facts'].get('budget')
                    if budget is None:
                        derivation = None; reason = 'PUBLIC_BUDGET_MISSING'
                    else:
                        derivation = {'operation': 'MONETARY_SUBTRACTION', 'semantic_type': 'REMAINING_BUDGET',
                            'operands': [budget, total], 'result_cents': cents(budget['value']) - total['result_cents']}
                if derivation is not None:
                    try:
                        correct = cents(value) == derivation['result_cents']
                    except (ValueError, InvalidOperation):
                        correct = False
                    label = 'DERIVED_SAFE' if correct else 'UNSUPPORTED'
                    reason = 'EXACT_CENTS_WHITELIST_PROOF' if correct else 'ARITHMETIC_CONTRADICTION'
                    contradiction = not correct
                    proof = [source_ref(ref, '/current_task/instruction', contract)]
                    for operand in total['operands']:
                        proof.extend(operand['provenance'])
                    if kind == 'remaining' and facts['facts'].get('budget'):
                        proof.extend(facts['facts']['budget']['provenance'])
        elif kind in facts['facts'] and kind not in facts['conflicts']:
            fact = facts['facts'][kind]; proof = fact['provenance']
            if value == fact['value']:
                label = 'OBSERVED_EXACT'; reason = 'EXACT_ROLE_BOUND_PUBLIC_FACT'
            elif norm(value) == norm(fact['value']):
                label = 'OBSERVED_NORMALIZED'; reason = 'CASE_OR_WHITESPACE_ONLY'
            elif kind in {'price', 'budget', 'session_total'}:
                try:
                    equal = cents(value) == cents(fact['value'])
                except (ValueError, InvalidOperation):
                    equal = False
                label = 'OBSERVED_NORMALIZED' if equal else 'UNSUPPORTED'
                reason = 'EXACT_MONEY_FORMAT_ONLY' if equal else 'PUBLIC_VALUE_CONTRADICTION'
                contradiction = not equal
            else:
                reason = 'PUBLIC_VALUE_CONTRADICTION'; contradiction = True
        elif kind in facts['conflicts']:
            reason = 'CONFLICTING_PUBLIC_DECLARATIONS'
        elif kind:
            reason = 'NO_ROLE_BOUND_PUBLIC_PROOF'
        output.append({'path': f['path'], 'value': value, 'role': kind, 'label': label,
            'reason': reason, 'provenance': proof, 'derivation': derivation, 'contradiction': contradiction})
    grounded = bool(output) and all(f['label'] in SAFE for f in output)
    bad = own['contradiction'] or any(f['contradiction'] or f['label'] == 'PRIVATE_LEAK' for f in output)
    positive = schema_valid and execution_success and own['passed'] and grounded and downstream_valid
    verdict = ('WRITE_POSITIVE_STRICT' if positive else 'WRITE_NEGATIVE' if bad else 'WRITE_UNCERTAIN')
    return {'episode_id': decision['episode_id'], 'session_id': facts['session_id'],
        'decision_id': decision['decision_id'], 'action': action['action'], 'source_decision_ref': source_decision_ref,
        'asin': facts.get('asin'), 'schema_valid': schema_valid, 'execution_success': execution_success,
        'ownership': own, 'fields': output, 'grounded_all_fields': grounded,
        'downstream_evidence_valid': downstream_valid, 'label': verdict,
        'unverified_visible_memory_ids': unverified, 'public_fact_conflicts': facts['conflicts'],
        'no_private_scorer_used_for_grounding': True}
