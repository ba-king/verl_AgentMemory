"""Offline factual receipts from public sources, separate from scope authority.

History uses ONLY earlier sessions' executed public navigation and purchase
lines; task success/scorer/targets do not establish requirements or values.
Candidate proof uses the previous V3 bounded factual interpretation registry.
"""
import json
import re
from decimal import Decimal
from .shopping import canonical, money, norm, CURRENT, PREVIOUS, BUDGET, MAPPING, ANCHOR, generic_contract
from .core import remaining_from_verified_prices
from .core import digest
from ...packet_interface_v3_minimal_validation.factual_audit import claims_for, supported


def public_history_projection(rows):
    return [{'event_id': r['event_id'], 'action': r['action'], 'observation': r['observation']} for r in rows]


def history_receipt(events):
    active = None; seen = set(); literals = {}; purchases = []
    for event in events:
        action = event['action']; observation = event['observation']
        body = observation.split('[SEP] Back to Search [SEP]', 1)[-1]
        rejected = 'not clickable' in norm(body) or 'rejected' in norm(body)
        if action['tool'] == 'search' and not rejected:
            active = None
            seen.update(re.findall(r'\bB[A-Z0-9]{9}\b', body))
        elif action['tool'] == 'click' and not rejected:
            item = action['arguments']['item']
            if re.fullmatch(r'B[A-Z0-9]{9}', item):
                active = item if item in seen else None
            elif norm(item) == 'back to search':
                active = None
            elif norm(item) in {'features', 'description'} and active:
                for piece in body.split('[SEP]'):
                    for match in re.finditer(r'(?i)\b(Material|Color):\s*([^;\n]+)', piece):
                        if piece[:match.start()].strip().casefold().endswith(('leg', 'frame', 'base')):
                            continue
                        literals.setdefault(active, []).append({'field': match[1].lower(), 'value': canonical(match[2]),
                            'event_id': event['event_id'], 'literal_span': match[0]})
            if norm(item) == 'buy now':
                for asin, paid in re.findall(r'^\s*Purchase #\d+: (B[A-Z0-9]{9}) \(\$([\d.]+)\)', observation, re.M):
                    if asin != active:
                        raise ValueError('HISTORY_PURCHASE_OWNERSHIP_UNRESOLVED')
                    purchases.append({'asin': asin, 'price_paid': money(paid),
                        'event_id': event['event_id'], 'public_purchase_line': f'{asin} (${paid})'})
    if len(purchases) != 1:
        raise ValueError('HISTORY_NOT_EXACTLY_ONE_OBSERVED_PURCHASE')
    result = purchases[0]; values = literals.get(result['asin'], [])
    for field in ('material', 'color'):
        field_values = [{k: v for k, v in row.items() if k != 'field'} for row in values if row['field'] == field]
        labels = {v['value'] for v in field_values}
        result[field] = next(iter(labels)) if len(labels) == 1 else None
        result[field + '_sources'] = field_values
    result['basis'] = 'Actual public earlier purchase and literal page, regardless of task correctness'
    return result


def memory_fields(content):
    fields = {}
    for key, raw in re.findall(r'- (\w+) = ("(?:\\.|[^"\\])*")', content):
        fields[key] = json.loads(raw)
    return fields


def certify_history(contract, memories, histories):
    contract = generic_contract(contract)
    receipts = {subject: history_receipt(histories[subject]) for subject in contract['previous_subjects']}
    certificates = {}
    for memory in memories:
        subject_match = re.search(r'Subject:\s*"Product (\d+) purchase"', memory['content'])
        if not subject_match:
            continue
        subject = f'P{subject_match[1]}'; fields = memory_fields(memory['content'])
        if subject not in receipts:
            continue
        observed = receipts[subject]
        asin = fields.get('purchased_item_id', fields.get('asin'))
        price_ok = asin == observed['asin'] and money(fields.get('price_paid', '-1')) == observed['price_paid']
        attribute_ok = {f: price_ok and observed[f] is not None and canonical(fields.get(f, '')) == observed[f] for f in ('material', 'color')}
        through = int(subject[1:])
        summed = sum(Decimal(receipts[f'P{i}']['price_paid']) for i in range(1, through + 1))
        remaining = money(Decimal(contract['budget_cap']) - summed)
        cumulative_ok = (price_ok and money(fields.get('total_budget', '-1')) == contract['budget_cap']
                         and money(fields.get('remaining_budget_after_purchase', '-1')) == remaining)
        certificates[memory['source_ref']] = {'subject_ref': subject, 'version_sha256': memory['version_sha256'],
            'content_digest': digest(memory['content']), 'price_certified': price_ok, 'price': observed['price_paid'],
            **{f + '_certified': ok for f, ok in attribute_ok.items()},
            **{f: observed[f] for f in attribute_ok},
            'cumulative_certified': cumulative_ok, 'remaining': remaining,
            'covers_history_through': through, 'cap': contract['budget_cap'],
            'price_event': observed['event_id'],
            **{f + '_sources': observed[f + '_sources'] for f in attribute_ok},
            'cumulative_purchase_events': [receipts[f'P{i}']['event_id'] for i in range(1, through + 1)]}
    return {'public_history_receipts': receipts, 'memory_certificates': certificates}


def need_evidence(packet, need):
    return [e for e in packet['evidence'] if e['need_id'] == need['id']]


def certified_sources(evidence, history):
    return [history['memory_certificates'][r] for r in evidence['source_refs'] if r in history['memory_certificates']]


def remaining_claim(text, cap):
    """Bounded observed forms; unknown prose is not automatic budget proof."""
    expression = re.fullmatch(r'Remaining budget for Product \d+ = \$(\d+\.\d+) - \$(\d+\.\d+) = \$(\d+\.\d+)\.', text)
    if expression:
        stated_cap, spent, balance = map(Decimal, expression.groups())
        return money(balance) if money(stated_cap) == cap and stated_cap - spent == balance else None
    simple = re.fullmatch(r'Remaining budget after Product \d+(?: purchase)? is \$(\d+\.\d+)(?: \(([^()]*)\))?\.', text)
    if simple:
        balance = Decimal(simple[1]); explanation = simple[2]
        if explanation:
            budget_only = re.fullmatch(r'total budget \$(\d+\.\d+)', explanation)
            subtraction = re.fullmatch(r'total \$(\d+\.\d+) minus \$(\d+\.\d+)', explanation)
            if budget_only:
                if money(budget_only[1]) != cap: return None
            elif subtraction:
                if money(subtraction[1]) != cap or Decimal(subtraction[1]) - Decimal(subtraction[2]) != balance: return None
            else:
                return None
        return money(balance)
    return None


def audit_need(need, certificate, packet, contract, trusted, factual, history):
    """Receipts certify actual evidence, never Teacher resolved/ready booleans."""
    contract = generic_contract(contract)
    evidence = need_evidence(packet, need); refs = {r for e in evidence for r in e['source_refs']}
    output = {'current_slots': [], 'previous_ok': False, 'mapping_ok': False, 'budget_ok': False,
              'anchor_ok': False, 'budget_route': None, 'evidence_refs': sorted(refs)}
    if certificate['classification'] != 'CERTIFIED_GATING':
        return output
    # Current facts must be carried by source-grounded evidence of the same
    # structural subject. Unknown free text is not certified as arbitrary fact.
    for e in evidence:
        good_claims = [c for c in (claims_for(factual, e) or []) if supported(factual, trusted, c, e['source_refs'])]
        for claim in trusted['allowed_claims_up_to_event']:
            if claim['subject_ref'] != e['subject_ref'] or claim['source_ref'] not in e['source_refs']:
                continue
            dimension = claim['predicate']
            direct = any(c['predicate'] == dimension and c['value'] == claim['value'] for c in good_claims)
            derived = any(c['kind'] == 'derived' for c in good_claims)
            price_text = dimension == 'price' and any(Decimal(x) == Decimal(claim['value']) for x in re.findall(r'\$(\d+(?:\.\d+)?)', e['fact']))
            if direct or derived or price_text:
                output['current_slots'].append({'subject_ref': e['subject_ref'], 'dimension': dimension,
                    'value': claim['value'], 'source_ref': claim['source_ref']})
    previous = contract['previous_subject']; pf = contract['previous_field']
    previous_certs = [c for c in history['memory_certificates'].values()
                      if c['subject_ref'] == previous and c.get(str(pf) + '_certified', False)]
    # History/mapping interpretations are bounded to these public templates;
    # require the validated value to actually appear, not only a source ID.
    for e in evidence:
        for c in certified_sources(e, history):
            if c['subject_ref'] == previous and c.get(str(pf) + '_certified', False):
                text = norm(e['fact'])
                if c[pf] in text and pf in text:
                    output['previous_ok'] = True
                allowed = contract['mapping'].get(c[pf])
                if allowed and c[pf] in text and allowed in text and any(word in text for word in ('allow', 'rule', 'satisf')):
                    output['mapping_ok'] = True
    # A public mapping declaration itself may cite task/e000; the previous
    # premise must still have an independently certified memory source elsewhere
    # in the Packet, and global readiness separately demands a gating P proof.
    for e in evidence:
        if set(e['source_refs']) & {'task', 'e000'}:
            for c in previous_certs:
                if c[pf] in norm(e['fact']) and contract['mapping'].get(c[pf], '\0') in norm(e['fact']):
                    output['mapping_ok'] = True
    if ANCHOR in certificate['requirements']:
        basis = norm(contract['requirements'][ANCHOR]['public_basis'])
        output['anchor_ok'] = any(e['subject_ref'] == 'T0' and set(e['source_refs']) & {'task', 'e000'}
                                  and norm(e['fact']) == basis for e in evidence)
    if BUDGET in certificate['requirements']:
        prices = {}
        for ref in refs:
            c = history['memory_certificates'].get(ref)
            if c and c['price_certified']:
                prices[c['subject_ref']] = c['price']
        all_prices = set(prices) >= set(contract['previous_subjects'])
        calculated = remaining_from_verified_prices(contract['budget_cap'], contract['previous_subjects'], prices)
        for e in evidence:
            # The finite remaining-budget claim form must contain the exact
            # derived balance. A spurious $500 with real citations is rejected.
            text = e['fact']
            if 'remaining budget' not in norm(text):
                continue
            claimed = remaining_claim(text, contract['budget_cap'])
            if not contract['previous_subjects']:
                simple = re.fullmatch(r'Remaining budget for Product 1 is \$(\d+\.\d{2})\.', text)
                if simple: claimed = money(simple[1])
                if e['subject_ref'] != 'T0' or not set(e['source_refs']) & {'task', 'e000'}:
                    continue
            if all_prices and claimed == calculated:
                output.update(budget_ok=True, budget_route='ALL_PREVIOUS_PRICES', remaining=calculated)
            for c in certified_sources(e, history):
                if c['subject_ref'] == previous and c['cumulative_certified'] and claimed == c['remaining']:
                    # Prefer route B when a single cumulative source suffices.
                    if not output['budget_ok'] or len(refs) == 1:
                        output.update(budget_ok=True, budget_route='CERTIFIED_CUMULATIVE_STATE', remaining=c['remaining'])
    return output
