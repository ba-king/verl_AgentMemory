"""Travel public contract -> subject/path obligations, never generated values.

Future PUBLIC instructions can establish dependency scope, never future results.
Proof values are read only from the present bootstrap and actual submission.
The optional hook is absent for existing Shopping adapters, leaving them intact.
"""
import copy
from final_unified_travel_rp_packet_v3_demo_v1.adapter import build, FIELDS
from teacher_pilot.travel_v0.read_consolidate.modes import public_submission


def optional_obligations(adapter, **kwargs):
    hook = getattr(adapter, 'get_durable_obligations', None)
    return hook(**kwargs) if hook else {'obligations':[], 'public_facts':{}}


def fields_of(plans):
    result = {}
    for day in plans:
        number = day['days']
        for field in FIELDS:
            if field in day:
                if not isinstance(day[field], str):
                    raise ValueError('NONLITERAL_PUBLIC_FIELD')
                result[f'day{number}.{field}'] = day[field]
    return result


def get_durable_obligations(current, history, future_public_sessions):
    contract = build(current)['contract']  # rejects private input keys
    referenced = {}
    for future in future_public_sessions:
        if set(future) != {'session_id','instruction'}:
            raise ValueError('FUTURE_PUBLIC_INSTRUCTION_ONLY')
        fc = build(future)['contract']
        for subject in (fc['base'], fc['previous']):
            if subject:
                referenced.setdefault(subject, []).append(future['session_id'])
    sources = []
    base = current.get('initial_observation', {}).get('base_person')
    if base and base['name'] in referenced:
        sources.append((base['name'], base['daily_plans'], 'current_public_bootstrap',
                        'public baseline state referenced by remaining public session contracts'))
    # No fabricated absent result. Failed sessions have no submission source.
    if any('final_itinerary' in e.get('action', {}) for e in history):
        submission = public_submission(history)
        if submission.get('traveler') != contract['traveler']:
            raise ValueError('ACTUAL_SUBMISSION_SUBJECT_MISMATCH')
        sources.append((submission['traveler'], submission.get('daily_plans', []),
                        submission['source_event_id'], 'actual subject-specific submitted state'))
    obligations, facts = [], {}
    for subject, plans, source, reason in sources:
        values = fields_of(plans)
        if not values:
            continue
        oid = f'durable:{current["session_id"]}:{subject}'
        obligations.append({'obligation_id':oid,'subject':subject,'required_paths':list(values),
            'source_refs':[source],'reason':reason,
            'public_dependency_sessions':referenced.get(subject, [])})
        facts[oid] = {'subject':subject,'values':copy.deepcopy(values),'source_refs':[source]}
    return {'obligations':obligations, 'public_facts':facts,
            'hidden_reference_used':False,'future_result_used':False}
