"""Bounded public Shopping grammar and proposal interpretation, outside Packet.

Only the existing six-item synthetic template is accepted. No Teacher output,
catalog target, history or scorer is an argument to build(). Unknowns fail closed.
"""
import re
from decimal import Decimal
from .core import digest, certify_need

CURRENT = 'REQ_CURRENT_CANDIDATES'
PREVIOUS = 'REQ_PREVIOUS_PRODUCT_MATERIAL'  # Preserve old receipt identity.
PREVIOUS_COLOR = 'REQ_PREVIOUS_PRODUCT_COLOR'
BUDGET = 'REQ_REMAINING_BUDGET'
MAPPING = 'REQ_PUBLIC_MAPPING'
ANCHOR = 'REQ_CURRENT_LABEL_CONSTRAINT'
ATTRIBUTES = ('Material', 'Color')
PREFIX = (
    'Buy a six-item indoor furnishing and decoration bundle, one item per Product in order. '
    'The fixed total budget for all six purchases is ${cap}. '
    'Before each purchase, subtract the prices actually paid for ALL earlier Products from this cap. '
    'There are no shipping fees or price-changing options. '
    'These label mappings are my bundle preferences, not physical compatibility claims. '
    'Use only literal Material: or Color: declarations on Features or Description pages; '
    'ignore case, treating gray/grey and aluminium/aluminum as equal. '
    'Keep all other label words, including percentages. Search listed candidate names individually.\n\n')
RANKING = ('Among listed candidates that satisfy BOTH the label rule and the remaining total budget, '
           'buy the highest-priced item. Break a price tie by ascending exact product name (case-sensitive).')


def norm(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def canonical(text):
    text = norm(text).strip(' .\"\'')
    return {'grey': 'gray', 'aluminium': 'aluminum'}.get(text, text)


def money(text):
    return format(Decimal(str(text).lstrip('$')), '.2f')


def unsupported():
    raise ValueError('UNSUPPORTED_PUBLIC_TASK_CONTRACT')


def build(task):
    if not isinstance(task, str):
        raise TypeError('PUBLIC_TASK_STRING_REQUIRED')
    cap = re.search(r'fixed total budget for all six purchases is \$(\d+\.\d{2})\.', task)
    if cap is None or not task.startswith(PREFIX.format(cap=cap[1])):
        unsupported()
    body = task[len(PREFIX.format(cap=cap[1])):]
    match = re.fullmatch(r'Product ([1-6]):\nThe field for the current item is (Material|Color)\.\n'
                         r'(.+)\n' + re.escape(RANKING) + r'\nAvailable Options:\n(.+)', body, re.S)
    if match is None:
        unsupported()
    step = int(match[1]); field = match[2]; rule = match[3]
    option_lines = match[4].splitlines()
    if not option_lines or any(not line.startswith('- ') or not line[2:].strip() for line in option_lines):
        raise ValueError('AMBIGUOUS_PUBLIC_OPTIONS')
    names = [line[2:] for line in option_lines]
    if len(names) != len(set(names)):
        raise ValueError('AMBIGUOUS_PUBLIC_OPTIONS')
    previous = prev_field = None; mapping = {}; anchor = None; public_mapping = []
    if step == 1:
        m = re.fullmatch(r'Only candidates with (Material|Color) label "([^"\n]+)" are eligible\.', rule)
        if m is None or m[1] != field:
            unsupported()
        anchor = canonical(m[2])
        if not anchor:
            unsupported()
    else:
        lines = rule.splitlines()
        previous_match = re.fullmatch(
            r'Use the (Material|Color) label of the item actually purchased for Product ([1-6])\. '
            r'This complete rule applies only to Product ([1-6]):', lines[0])
        if (previous_match is None or int(previous_match[2]) >= step
                or int(previous_match[3]) != step or len(lines) < 2):
            unsupported()
        prev_field = previous_match[1].lower(); previous = f'P{previous_match[2]}'
        for line in lines[1:]:
            m = re.fullmatch(r'- Previous (Material|Color) "([^"\n]+)" allows current (Material|Color) '
                             r'"([^"\n]+)"; all other current labels are forbidden\.', line)
            if m is None or m[1].lower() != prev_field or m[3] != field:
                unsupported()
            key = canonical(m[2]); value = canonical(m[4])
            if not key or not value or key in mapping:
                raise ValueError('AMBIGUOUS_PUBLIC_MAPPING')
            mapping[key] = value; public_mapping.append(line)
    refs = [f'C{i}' for i in range(1, len(names) + 1)]
    history = [f'P{i}' for i in range(1, step)]
    requirements = {
        CURRENT: {'kind': 'slots', 'required_scope': refs, 'required_dimensions': [field.lower(), 'price'],
                  'use_declared_scope': True, 'check_resolve_slots': True,
                  'scope_policy': 'EXACT_FULL_OR_CERTIFIED_FRAGMENTS',
                  'public_basis': ['Search listed candidate names individually.', RANKING, 'Available Options']},
        BUDGET: {'kind': 'balance', 'proof_routes': ['ALL_PREVIOUS_PRICES', 'CERTIFIED_CUMULATIVE_STATE'],
                 'required_history': history, 'cap': money(cap[1]),
                 'empty_history_balance': money(cap[1]) if not history else None,
                 'public_basis': ['Before each purchase, subtract the prices actually paid for ALL earlier Products from this cap.',
                                  'There are no shipping fees or price-changing options.']}}
    previous_id = None
    if previous:
        previous_id = PREVIOUS if prev_field == 'material' else PREVIOUS_COLOR
        requirements[previous_id] = {'kind': 'assertion', 'required_scope': [previous],
            'required_dimensions': [prev_field], 'proof_scope': [previous], 'proof_dimensions': [prev_field],
            'public_basis': lines[0].split(' This complete rule', 1)[0]}
        requirements[MAPPING] = {'kind': 'assertion', 'required_previous_subject': previous,
            'previous_field': prev_field, 'current_field': field.lower(), 'public_mapping': mapping,
            'public_basis': public_mapping}
    else:
        requirements[ANCHOR] = {'kind': 'assertion', 'required_scope': ['T0'],
            'current_field': field.lower(), 'allowed_labels': [anchor], 'public_basis': rule}
    return {'authority': 'HARNESS_TASK_CONTRACT', 'task_digest': digest(task), 'public_task': task,
        'task_subject': 'T0', 'current_step': step, 'previous_subject': previous,
        'previous_field': prev_field, 'previous_requirement': previous_id, 'current_field': field.lower(),
        'budget_cap': money(cap[1]), 'current_subjects': refs, 'previous_subjects': history,
        'subject_registry': {**dict(zip(refs, names)), **{s: f'Previous {s}' for s in history}, 'T0': 'Current public task'},
        'mapping': mapping, 'initial_anchor': anchor, 'requirements': requirements,
        'limits': 'Exact existing public six-item highest-price synthetic template only.'}


def generic_contract(contract):
    """Compatibility input for old archived contracts; authority remains public."""
    return contract if 'task_subject' in contract else build(contract['public_task'])


def match_proposal(text, contract):
    c = generic_contract(contract); text = norm(text)
    field = c['current_field']; step = c['current_step']; pf = c['previous_field']
    prev = c['previous_subject']; prior_id = c['previous_requirement']
    rule_id = MAPPING if prev else ANCHOR
    local = re.fullmatch(r'(?:inspect candidate (c\d+)(?: (?:material|color))?|inspect (c\d+) (?:material|color)|verify identity of (c\d+) after ambiguous search)\.?', text)
    if local:
        return {'role': 'LOCAL', 'declared_subjects_in_text': [next(x for x in local.groups() if x).upper()]}
    fragment = re.fullmatch(r'inspect candidates (c\d+(?: and c\d+)*) for the task comparison\.', text)
    if fragment:
        return {'role': 'CURRENT_FRAGMENT', 'declared_subjects_in_text': [s.upper() for s in fragment[1].split(' and ')],
                'dimensions': [field, 'price'], 'requirements': [CURRENT]}
    if text in {f"determine each candidate's literal {field} label and price.",
                f"identify each candidate's literal {field} label and price.",
                f"identify each listed candidate's literal {field} label and price."}:
        return {'role': 'CURRENT_FULL', 'dimensions': [field, 'price'], 'requirements': [CURRENT]}
    if text == f'{field} labels of each product {step} candidate':
        return {'role': 'CURRENT_FULL', 'dimensions': [field], 'requirements': [CURRENT]}
    if text == f'prices of each product {step} candidate':
        return {'role': 'CURRENT_FULL', 'dimensions': ['price'], 'requirements': [CURRENT]}
    comparisons = {
        'which candidate satisfies both label rule and remaining budget, and is highest-priced (tie-break by ascending exact name)',
        f'determine which candidates satisfy the {field} rule and remaining budget, then select highest-priced (tie by ascending name).'}
    for previous, allowed in c['mapping'].items():
        comparisons.add(f'identify highest-priced candidate with {field} "{allowed}" within remaining budget, tie-broken by ascending exact name.')
        comparisons.add(f'identify which listed candidates satisfy the product {step} {field} rule (previous {pf} "{previous}" allows current {field} "{allowed}") and the remaining budget, and select the highest-priced (tie-break by ascending exact name).')
    if text in comparisons or re.fullmatch(r'select highest-priced candidate satisfying both the '+field+r' rule \([a-z ]+\) and remaining budget \$[\d.]+\.', text):
        return {'role': 'CURRENT_FULL', 'dimensions': [field, 'price'], 'requirements': [CURRENT, rule_id]}
    prior_forms = [
        r'establish product (\d+) purchased item(?:, price, and (material|color) label| and its (material|color) label)\.',
        r'determine the (material|color) label of the item actually purchased for product (\d+)\.',
        r'(material|color) label of product (\d+) actually purchased',
        r"determine product (\d+)'s actual purchased (material|color) label to select the allowed product (\d+) (material|color)\."]
    for pattern in prior_forms:
        if re.fullmatch(pattern, text):
            numbers = re.findall(r'product (\d+)', text)
            attrs = re.findall(r'\b(material|color)\b', text)
            errors = []
            if not prev or f'P{numbers[0]}' != prev: errors.append('SCOPE_CONTRACT_VIOLATION')
            if not pf or attrs[0] != pf: errors.append('PREVIOUS_ATTRIBUTE_MISMATCH')
            if len(numbers) > 1 and (int(numbers[1]) != step or attrs[-1] != field):
                errors.append('SCOPE_CONTRACT_VIOLATION')
            return {'role': 'PREVIOUS', 'requirements': [prior_id] + ([MAPPING] if len(numbers)>1 else []), 'errors': errors}
    if (re.fullmatch(r'(?:establish|determine) remaining budget after product \d+(?: purchase)?\.', text)
        or re.fullmatch(r'remaining budget after product \d+ purchase', text)
        or re.fullmatch(r'determine the remaining budget for product \d+ after subtracting prices paid for products \d+ and \d+\.', text)
        or text in {'establish remaining budget from the prior purchase prices.',
                    'establish remaining budget from a certified cumulative state.',
                    'establish remaining budget from the public cap.'}):
        return {'role': 'BUDGET', 'requirements': [BUDGET]}
    mapping = re.fullmatch(r'(?:determine allowed (material|color) label for product (\d+) given product (\d+) (material|color)\.|applicable product (\d+) label rule given product (\d+) (material|color))', text)
    if mapping:
        numbers = re.findall(r'product (\d+)', text); attrs = re.findall(r'\b(material|color)\b', text)
        errors = []
        if not prev or f'P{numbers[-1]}' != prev or int(numbers[0]) != step:
            errors.append('SCOPE_CONTRACT_VIOLATION')
        if attrs[-1] != pf: errors.append('PREVIOUS_ATTRIBUTE_MISMATCH')
        if len(attrs)>1 and attrs[0] != field: errors.append('SCOPE_CONTRACT_VIOLATION')
        return {'role': 'MAPPING', 'requirements': [MAPPING], 'errors': errors}
    if not prev and text in {f'determine allowed {field} label for product {step}.', norm(c['requirements'][ANCHOR]['public_basis'])}:
        return {'role': 'ANCHOR', 'requirements': [ANCHOR]}
    return None


def interpret_need(need, contract):
    c = generic_contract(contract); p = match_proposal(need['text'], c)
    if p is not None:
        role = p['role']
        if role == 'CURRENT_FULL': p['allowed_scopes'] = [c['current_subjects']]
        elif role in {'CURRENT_FRAGMENT', 'LOCAL'}:
            p['allowed_scopes'] = [p['declared_subjects_in_text']]; p['subset_of'] = c['current_subjects']
        elif role == 'PREVIOUS': p['allowed_scopes'] = [[c['previous_subject']]]
        elif role == 'MAPPING': p['allowed_scopes'] = [[c['previous_subject']], ['T0']]
        elif role == 'ANCHOR': p['allowed_scopes'] = [['T0']]
        elif role == 'BUDGET':
            p['allowed_scopes'] = [c['previous_subjects'], ['T0']]
            if c['previous_subject']: p['allowed_scopes'].append([c['previous_subject']])
    return p


def validate_need_against_contract(need, contract):
    c = generic_contract(contract)
    return certify_need(need, c, interpret_need(need, c))


def decide(contract, requirement_results, slot_values, history):
    """Verifier-only comparison using public proofs; never an Answer instruction."""
    c = generic_contract(contract); decision = {'established': False}
    if not all(r['satisfied'] for r in requirement_results.values()): return decision
    allowed = c['initial_anchor']
    if c['previous_subject']:
        pf = c['previous_field']
        labels = {r[pf] for r in history['memory_certificates'].values()
                  if r['subject_ref'] == c['previous_subject'] and r.get(pf+'_certified')}
        if len(labels) != 1: return decision
        allowed = c['mapping'].get(next(iter(labels)))
    balance = Decimal(requirement_results[BUDGET]['remaining_values'][0]); eligible = []
    slots = slot_values[CURRENT]
    for subject in c['current_subjects']:
        label = next(iter(slots[(subject, c['current_field'])])); price = Decimal(next(iter(slots[(subject, 'price')])))
        if label == allowed and price <= balance:
            eligible.append((price, c['subject_registry'][subject], subject))
    if eligible:
        decision = {'established': True, 'verifier_only_subject_ref': sorted(eligible, key=lambda x: (-x[0], x[1]))[0][2],
                    'basis': 'Public literal/price proofs, budget and public mapping; no sealed answer'}
    return decision
