"""Finite Shopping text grammar; no Packet fields or scope-authority changes."""
import re
from decimal import Decimal, InvalidOperation
from ..packet_v3_live_data_pilot_v1.hardening import core, shopping as old

CURRENT, BUDGET, MAPPING, ANCHOR = old.CURRENT, old.BUDGET, old.MAPPING, old.ANCHOR
build = old.build
canonical = old.canonical

FORMS = {
 'CURRENT_FULL': ('field','dimensions'),
 'CURRENT_FRAGMENT': ('subjects','field','dimensions'),
 'PREVIOUS_ATTRIBUTE': ('subject','field'),
 'PUBLIC_MAPPING': ('subject','from','to'),
 'INITIAL_ANCHOR': ('field','allowed'),
 'REMAINING_BUDGET': ('basis',),
 'LOCAL_INSPECT': ('subject','purpose'),
 'LOCAL_RETRY': ('subject','purpose'),
}


def parse(text):
    parts = text.split('|'); kind = parts[0]
    if kind not in FORMS:raise ValueError('CANONICAL_GRAMMAR_INVALID')
    if len(parts)-1 != len(FORMS[kind]):raise ValueError('CANONICAL_GRAMMAR_INVALID')
    result = {'kind':kind}
    for part, key in zip(parts[1:],FORMS[kind]):
        if not part.startswith(key+'=') or not part[len(key)+1:] or '\n' in part:
            raise ValueError('CANONICAL_GRAMMAR_INVALID')
        result[key] = part[len(key)+1:]
    return result


def proposal(need, contract, legacy=True):
    c=old.generic_contract(contract);text=need['text']
    if '|' not in text and not any(text.startswith(k) for k in FORMS):
        return old.interpret_need(need,c) if legacy else None
    try:p=parse(text)
    except ValueError:
        return {'role':'INVALID_CANONICAL','requirements':[], 'errors':['CANONICAL_GRAMMAR_INVALID']}
    k=p['kind'];errors=[];result={'role':k,'requirements':[], 'errors':errors}
    def check(actual,expected,code='NEED_TEXT_CONTRACT_MISMATCH'):
        if actual!=expected:errors.append(code)
    if k in {'CURRENT_FULL','CURRENT_FRAGMENT'}:
        check(p['field'],c['current_field'])
        dims=p['dimensions'].split(',')
        if len(dims)!=len(set(dims)) or not set(dims)<={c['current_field'],'price'}:
            errors.append('CANONICAL_DIMENSIONS_INVALID')
        if k=='CURRENT_FULL':
            check(dims,[c['current_field'],'price']);scope=c['current_subjects']
        else:
            scope=p['subjects'].split(',')
            if not scope or len(scope)!=len(set(scope)) or not set(scope)<=set(c['current_subjects']):
                errors.append('UNKNOWN_SUBJECT_REF')
        result.update(requirements=[CURRENT],allowed_scopes=[scope],dimensions=dims)
    elif k=='PREVIOUS_ATTRIBUTE':
        check(p['subject'],c['previous_subject']);check(p['field'],c['previous_field'])
        result.update(role='PREVIOUS',requirements=[c['previous_requirement']] if c['previous_requirement'] else [],allowed_scopes=[[c['previous_subject']]])
    elif k=='PUBLIC_MAPPING':
        check(p['subject'],c['previous_subject']);check(p['from'],c['previous_field']);check(p['to'],c['current_field'])
        result.update(role='MAPPING',requirements=[MAPPING] if MAPPING in c['requirements'] else [],allowed_scopes=[[c['previous_subject']],['T0']])
    elif k=='INITIAL_ANCHOR':
        check(c['previous_subject'],None);check(p['field'],c['current_field']);check(canonical(p['allowed']),c['initial_anchor'])
        result.update(role='ANCHOR',requirements=[ANCHOR] if ANCHOR in c['requirements'] else [],allowed_scopes=[['T0']])
    elif k=='REMAINING_BUDGET':
        check(p['basis'],'prior_purchases' if c['previous_subjects'] else 'public_cap')
        scopes=[['T0']]
        if c['previous_subjects']:scopes.extend([c['previous_subjects'],[c['previous_subject']]])
        result.update(role='BUDGET',requirements=[BUDGET],allowed_scopes=scopes)
    else:
        if p['subject'] not in c['current_subjects']:errors.append('UNKNOWN_SUBJECT_REF')
        if p['purpose'] not in ({'identity','attribute'} if k=='LOCAL_INSPECT' else {'search'}):errors.append('CANONICAL_PURPOSE_INVALID')
        result.update(role='LOCAL',allowed_scopes=[[p['subject']]])
    return result


def certify(need, contract, legacy=True):
    return core.certify_need(need,contract,proposal(need,contract,legacy))


def text_audit(need,contract):
    if '|' not in need['text']:
        from ..packet_v3_live_data_pilot_v1.hardening.need_text_consistency import audit
        return audit(need,contract)
    p=proposal(need,contract,legacy=False)
    cert=core.certify_need(need,contract,p)
    return {'need_id':need['id'],'need_text':need['text'],'status':'FAIL' if cert['reason_codes'] else 'PASS',
            'reason_codes':cert['reason_codes'],'mismatches':cert['reason_codes'],'unverified_claims':[],
            'does_not_rewrite_teacher_text':True}


def fact(text):
    """One atomic assertion. Unknown prose is never partially accepted."""
    m=re.fullmatch(r'(material|color|price|remaining_budget|allowed_material|allowed_color|comparison_ready)=([^|\n=]+)',text)
    if not m:raise ValueError('FACTUAL_AUDIT_UNRESOLVED')
    pred,value=m.groups()
    if pred in {'price','remaining_budget'}:
        if not re.fullmatch(r'-?\d+(?:\.\d{1,2})?',value):raise ValueError('CANONICAL_VALUE_INVALID')
        amount=Decimal(value)
        if pred=='price' and amount<0:raise ValueError('CANONICAL_VALUE_INVALID')
        value=format(amount,'.2f')
    elif pred=='comparison_ready':
        if value!='true':raise ValueError('CANONICAL_VALUE_INVALID')
    else:
        if value != value.strip() or not value:raise ValueError('CANONICAL_VALUE_INVALID')
        value=canonical(value)
    return {'predicate':pred,'value':value}


def unresolved(text):
    m=re.fullmatch(r'(MISSING|UNKNOWN|CONFLICT)\|subject=(T0|C\d+|P\d+)\|predicate=(material|color|price|identity|remaining_budget|allowed_material|allowed_color)',text)
    if not m:raise ValueError('UNRESOLVED_LANGUAGE_UNSUPPORTED')
    return dict(zip(('kind','subject','predicate'),m.groups()))


def failed(text):
    m=re.fullmatch(r'(FAILED_SEARCH|FAILED_CLICK)\|subject=(C\d+)\|(query|target)=(.+)',text)
    if not m or '\n' in text:raise ValueError('FAILED_ATTEMPT_LANGUAGE_UNSUPPORTED')
    kind,subject,key,value=m.groups()
    if key!=('query' if kind=='FAILED_SEARCH' else 'target'):raise ValueError('FAILED_ATTEMPT_LANGUAGE_UNSUPPORTED')
    return {'kind':kind,'subject':subject,'value':value}
