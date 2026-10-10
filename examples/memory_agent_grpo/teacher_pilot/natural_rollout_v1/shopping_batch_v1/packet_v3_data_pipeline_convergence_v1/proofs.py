"""Public-only atomic evidence receipts, independent of Teacher conclusions."""
import re
import hashlib
from decimal import Decimal
from . import language as lang
from ..packet_v3_live_data_pilot_v1.hardening import core, shopping as shop
from ..packet_v3_live_data_pilot_v1.hardening.shopping_proof import history_receipt, memory_fields
from ..packet_interface_v3_minimal_validation.trusted_registry import binding_check


def history(contract, memories, histories):
    """Independent subjects; one absent purchase cannot erase other receipts.

    Existing rendered records have price OR price_paid. Accept either only when
    it agrees with the actual public purchase; conflicting aliases fail closed.
    No Bank change, no certification solely from a record's stated value.
    """
    observed={};unavailable={};certs={}
    for subject in contract['previous_subjects']:
        try:observed[subject]=history_receipt(histories.get(subject,[]))
        except (ValueError,KeyError,TypeError) as exc:unavailable[subject]=str(exc)
    for memory in memories:
        version=hashlib.sha256(memory['content'].encode()).hexdigest()
        if memory.get('version_sha256')!=version or not memory['source_ref'].endswith('@'+version[:12]):
            unavailable[memory['source_ref']]='MEMORY_VERSION_MISMATCH';continue
        m=re.search(r'Subject:\s*"Product (\d+) purchase"',memory['content'])
        if not m or 'P'+m[1] not in observed:continue
        subject='P'+m[1];receipt=observed[subject];fields=memory_fields(memory['content'])
        ids=[fields[k] for k in ('asin','purchased_item_id') if k in fields]
        prices=[shop.money(fields[k]) for k in ('price','price_paid') if k in fields]
        price_ok=bool(ids and prices) and set(ids)=={receipt['asin']} and set(prices)=={receipt['price_paid']}
        attrs={f:price_ok and receipt[f] is not None and lang.canonical(fields.get(f,''))==receipt[f] for f in ('material','color')}
        through=int(m[1]);prefix=[f'P{i}' for i in range(1,through+1)]
        balance=core.remaining_from_verified_prices(contract['budget_cap'],prefix,{s:r['price_paid'] for s,r in observed.items()})
        explicit_balance=fields.get('remaining_budget_after_purchase')
        # Cap comes from the public task. A redundant total_budget Bank field
        # is not required; if present it must agree with that public cap.
        cap_ok='total_budget' not in fields or shop.money(fields['total_budget'])==contract['budget_cap']
        cumulative=price_ok and cap_ok and balance is not None and explicit_balance is not None and shop.money(explicit_balance)==balance
        certs[memory['source_ref']]={'subject_ref':subject,'price_certified':price_ok,'price':receipt['price_paid'],
            **{f:receipt[f] for f in attrs},**{f+'_certified':ok for f,ok in attrs.items()},
            'cumulative_certified':cumulative,'remaining':balance,'covers_history_through':through,
            'version_sha256':memory['version_sha256'],'public_purchase_event':receipt['event_id']}
    return {'memory_certificates':certs,'public_history_receipts':observed,'unavailable':unavailable}


def prior_value(c,h):
    p,f=c['previous_subject'],c['previous_field']
    values={r[f] for r in h['memory_certificates'].values() if r['subject_ref']==p and r.get(str(f)+'_certified')}
    return next(iter(values)) if len(values)==1 else None


def derived_support(e,claim,c,h):
    """Support must be carried by this claim's refs, not an uncited audit fact."""
    pred,val=claim['predicate'],claim['value'];refs=set(e['source_refs'])
    certs=[r for ref,r in h['memory_certificates'].items() if ref in refs]
    if pred=='remaining_budget':
        if e['subject_ref']!='T0' or not refs & {'task','e000'}:return False
        prices={r['subject_ref']:r['price'] for r in certs if r['price_certified']}
        exact=core.remaining_from_verified_prices(c['budget_cap'],c['previous_subjects'],prices)
        if exact==val:return True
        last=f"P{c['current_step']-1}"
        return any(r['subject_ref']==last and r['cumulative_certified'] and r['remaining']==val for r in certs)
    if pred.startswith('allowed_'):
        if e['subject_ref']!='T0' or pred!='allowed_'+c['current_field'] or not refs & {'task','e000'}:return False
        if not c['previous_subject']:return val==c['initial_anchor']
        f=c['previous_field']
        return any(r['subject_ref']==c['previous_subject'] and r.get(f+'_certified') and c['mapping'].get(r[f])==val for r in certs)
    return False


def evidence(e,trusted,c,h,comparison=False):
    codes=binding_check(e,trusted['subject_registry'],trusted['source_ownership_registry'])
    try:claim=lang.fact(e['fact'])
    except ValueError as exc:return {'status':'UNRESOLVED','reason_codes':codes+[str(exc)],'claim':None}
    pred,val=claim['predicate'],claim['value'];supported=False
    if pred in {'material','color','price'}:
        if e['subject_ref'].startswith('C'):
            supported=any(r['subject_ref']==e['subject_ref'] and r['predicate']==pred
                and (shop.money(r['value']) if pred=='price' else lang.canonical(r['value']))==val
                and r['source_ref'] in e['source_refs'] for r in trusted['allowed_claims_up_to_event'])
        elif e['subject_ref'].startswith('P'):
            supported=any(r['subject_ref']==e['subject_ref'] and r.get(pred+'_certified') and r[pred]==val
                          for ref,r in h['memory_certificates'].items() if ref in e['source_refs'])
    elif pred=='comparison_ready':
        # Recomputed independently from grounded Packet premises below.
        supported=comparison and e['subject_ref']=='T0' and 'task' in e['source_refs']
    else:supported=derived_support(e,claim,c,h)
    if not supported:
        # A known contradictory cited value is a factual error. Missing proof
        # (e.g. unavailable earlier purchase) remains evidence insufficiency.
        contradicted=False
        if pred in {'material','color','price'} and e['subject_ref'].startswith('C'):
            cited=[r for r in trusted['allowed_claims_up_to_event'] if r['subject_ref']==e['subject_ref'] and r['predicate']==pred and r['source_ref'] in e['source_refs']]
            contradicted=bool(cited)
        elif pred=='remaining_budget' and not c['previous_subjects']:
            contradicted=e['subject_ref']=='T0' and bool(set(e['source_refs']) & {'task','e000'}) and val!=c['budget_cap']
        elif pred.startswith('allowed_') and not c['previous_subject']:
            contradicted=e['subject_ref']=='T0' and bool(set(e['source_refs']) & {'task','e000'}) and (pred!='allowed_'+c['current_field'] or val!=c['initial_anchor'])
        codes.append('SOURCE_SUPPORT' if contradicted else 'SOURCE_SUPPORT_UNPROVEN')
    return {'status':'PASS' if not codes else 'FAIL','reason_codes':sorted(set(codes)),'claim':claim}


def evaluate_packet(packet,c,trusted,h):
    atoms=[evidence(e,trusted,c,h) for e in packet['evidence']]
    def receipt(n,cert):
        proofs={};slots=[]
        own=[(e,r) for e,r in zip(packet['evidence'],atoms) if e['need_id']==n['id'] and r['status']=='PASS']
        for e,r in own:
            cl=r['claim'];pred=cl['predicate']
            if e['subject_ref'] in c['current_subjects'] and pred in {c['current_field'],'price'}:
                slots.append({'subject_ref':e['subject_ref'],'dimension':pred,'value':cl['value'],'source_ref':e['source_refs'][0]})
        proofs[lang.CURRENT]={'slots':slots}
        balances=[r['claim']['value'] for _,r in own if r['claim']['predicate']=='remaining_budget']
        proofs[lang.BUDGET]={'satisfied':len(set(balances))==1,'remaining':balances[0] if balances else None,'route':'EXPLICIT_CITED_PUBLIC_ARITHMETIC'}
        allowed=any(r['claim']['predicate']=='allowed_'+c['current_field'] for _,r in own)
        proofs[lang.MAPPING]={'satisfied':allowed};proofs[lang.ANCHOR]={'satisfied':allowed}
        if c['previous_requirement']:
            proofs[c['previous_requirement']]={'satisfied':any(e['subject_ref']==c['previous_subject'] and r['claim']['predicate']==c['previous_field'] for e,r in own)}
        return {'proofs':proofs,'detail':{'atomic_receipts':own}}
    gate=core.evaluate(packet,c,lambda n,c:lang.proposal(n,c),receipt,
                       lambda requirements,slots:shop.decide(c,requirements,slots,h))
    # A comparison claim must cite the actual premise refs that certify the
    # comparison; mentioning task alone is never enough.
    premise_refs={r for e,a in zip(packet['evidence'],atoms) if a['status']=='PASS' for r in e['source_refs']}
    for e,a in zip(packet['evidence'],atoms):
        if a['claim'] and a['claim']['predicate']=='comparison_ready':
            fresh=evidence(e,trusted,c,h,comparison=gate['global_ready_allowed'] and premise_refs<=set(e['source_refs']))
            a.update(fresh)
    return gate,atoms


def public_failure(event):
    """Repetition itself is not a failure. Query text/title is never evidence."""
    if not event:return False
    ob=event['observation'].split('[SEP] Back to Search [SEP]',1)[-1].strip()
    # Explicit environment result markers, not mere identical page hashes.
    return bool(re.search(r'(?i)(?:\bno (?:products|results)(?: were)? found\b|\bno results\b|\b0 results\b|\bnot clickable\b|^\s*(?:error: ?)?(?:action |click |search )?rejected\b)',ob))
