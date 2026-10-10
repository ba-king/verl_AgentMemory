"""Bounded Shopping claim audit, independent of proposal-role mapping.

This is a candidate-admission audit, not a wording normalizer or new parser.
Unknown factual values remain unverified; absence of a numeric/label claim is
valid. Source receipts, not Teacher evidence or eventual success, supply values.
"""
import re
from decimal import Decimal
from .shopping import canonical, generic_contract, norm, BUDGET

MISMATCH='NEED_TEXT_CONTRACT_MISMATCH'
UNVERIFIED='NEED_TEXT_AUDIT_UNRESOLVED'


def audit(need, contract, history=None, requirement_results=None):
    c=generic_contract(contract);text=norm(need['text'])
    step=c['current_step'];prev=c['previous_subject'];pf=c['previous_field'];cf=c['current_field']
    errors=[];unknown=[];checked=[];bound_product_spans=[]
    def check(kind,actual,expected):
        checked.append({'kind':kind,'actual':actual,'expected':expected})
        if expected is None:unknown.append({'kind':kind,'actual':actual})
        elif actual!=expected:errors.append({'code':MISMATCH,'kind':kind,'actual':actual,'expected':expected})
    # Product references with explicit temporal/decision roles. Do not use N-id.
    product_patterns=[
        (r'item actually purchased for product (\d+)',prev),
        (r'product (\d+)(?:\'s)? (?:actual purchased|purchased item)',prev),
        (r'(?:material|color) label of product (\d+) actually purchased',prev),
        (r'given product (\d+)',prev),
        (r'after product (\d+)',f'P{step-1}' if step>1 else None),
        (r'(?:each|every|listed) product (\d+) candidate',f'P{step}'),
        (r'product (\d+) (?:material|color|label) rule',f'P{step}'),
        (r'allowed product (\d+) (?:material|color)',f'P{step}'),
        (r'(?:remaining budget|allowed (?:material|color) label) for product (\d+)',f'P{step}')]
    for pattern,expected in product_patterns:
        for m in re.finditer(pattern,text):
            check('PRODUCT_INDEX','P'+m[1],expected)
            bound_product_spans.append(m.span(1))
    for m in re.finditer(r'product (\d+)',text):
        if m.span(1) not in bound_product_spans:
            # Plural historical sums have their own price receipts; an opaque
            # numbered assertion is not silently accepted as a current fact.
            unknown.append({'kind':'PRODUCT_ROLE_UNRESOLVED','actual':'P'+m[1]})
    prev_patterns=[r'previous (material|color)',r'determine the (material|color) label of the item actually purchased',
                   r'actual purchased (material|color) label',r'purchased item(?:, price, and| and its) (material|color) label',
                   r'(material|color) label of product \d+ actually purchased',r'given product \d+ (material|color)']
    previous_attr_spans=set()
    for pattern in prev_patterns:
        for m in re.finditer(pattern,text):
            check('PREVIOUS_ATTRIBUTE',m[1],pf);previous_attr_spans.add(m.span(1))
    for pattern in [r'current (material|color)',r"candidate(?:'s|s')? literal (material|color)",
                    r'(material|color) labels of each product \d+ candidate',
                    r'allowed (material|color) label for product',r'allowed product \d+ (material|color)']:
        for m in re.finditer(pattern,text):check('CURRENT_ATTRIBUTE',m[1],cf)
    certs=(history or {}).get('memory_certificates',{})
    prior_values={r.get(pf) for r in certs.values() if r.get('subject_ref')==prev and r.get(str(pf)+'_certified')}
    prior_value=next(iter(prior_values)) if len(prior_values)==1 else None
    allowed=c['initial_anchor'] if prev is None else c['mapping'].get(prior_value)
    explicit_prev=re.search(r'previous (material|color) "([^"]+)"',text)
    conditional_label=None
    if explicit_prev:
        pvalue=canonical(explicit_prev[2])
        if pvalue not in c['mapping']:
            check('PUBLIC_MAPPING_KEY',pvalue,'one of '+repr(sorted(c['mapping'])))
        else:
            conditional_label=c['mapping'][pvalue]
        if prior_value is not None:check('PREVIOUS_LABEL',pvalue,prior_value)
    for m in re.finditer(r'\b(material|color)(?: label)?\s*"([^"]+)"',text):
        value=canonical(m[2]);attribute=m[1]
        is_prior=text[max(0,m.start()-9):m.start()].endswith('previous ')
        if is_prior:continue  # checked as a mapping premise above
        historical=bool(re.search(r'(?:actually purchased|purchased item)',text)) and not re.search(r'current|candidate',text)
        expected_attr=pf if historical else cf
        check('PREVIOUS_ATTRIBUTE' if historical else 'CURRENT_ATTRIBUTE',attribute,expected_attr)
        target=prior_value if historical else (conditional_label or allowed)
        check('PREVIOUS_LABEL' if historical else 'ALLOWED_LABEL',value,target)
    # A label-free instruction carries no implied static or dynamic value.
    balances=(requirement_results or {}).get(BUDGET,{}).get('remaining_values',[])
    if not c['previous_subjects']:
        balances=[c['budget_cap']]
    elif not balances:
        last=f'P{step-1}'
        balances=list({r['remaining'] for r in certs.values() if r.get('subject_ref')==last and r.get('cumulative_certified')})
    remaining=balances[0] if len(balances)==1 else None
    seen_money=[]
    for m in re.finditer(r'(remaining budget|total budget|budget cap)\s*(?:=|is|of|:)?\s*\$(\d+(?:\.\d+)?)',text):
        actual=format(Decimal(m[2]),'.2f');expected=c['budget_cap'] if m[1]!='remaining budget' else remaining
        check('BUDGET',actual,expected);seen_money.append(m.span(2))
    for m in re.finditer(r'\$(\d+(?:\.\d+)?)',text):
        if m.span(1) not in seen_money:unknown.append({'kind':'NUMERIC_ROLE_UNRESOLVED','actual':m[1]})
    return {'need_id':need['id'],'need_text':need['text'],'status':'FAIL' if errors else 'UNRESOLVED' if unknown else 'PASS',
            'reason_codes':([MISMATCH] if errors else [])+([UNVERIFIED] if unknown else []),
            'mismatches':errors,'unverified_claims':unknown,'checked_claims':checked,
            'does_not_rewrite_teacher_text':True}
