"""Public provenance construction, independent of Teacher Packet or its text.

This bounded Shopping adapter reuses the audited public navigation extractor.
It never reads the old fixture's source_subject_map. Unknown ownership remains
unknown, including a search whose query names a candidate but does not find it.
The registries are audit/harness data, not new Packet schema fields.
"""
import copy
import re
from ..packet_binding_regression_v1.public_evidence import ledger


def public_projection(export, cutoff):
    return {
        'current_task': export['current_task'],
        'initial_observation': copy.deepcopy(export['initial_observation']),
        'actual_retrieved_memories': copy.deepcopy(export['actual_retrieved_memories']),
        'public_prefix_events_only': copy.deepcopy(export['public_prefix_events_only'][:cutoff]),
    }


def build(public_input, cutoff):
    allowed = {'current_task', 'initial_observation', 'actual_retrieved_memories', 'public_prefix_events_only'}
    if set(public_input) != allowed:
        raise ValueError('TRUSTED_INPUT_NOT_PUBLIC_PROJECTION')
    data = ledger(public_input, cutoff)
    task = public_input['current_task']
    current = int(re.search(r'^Product (\d+):', task, re.M)[1])
    subjects = {sid: {**copy.deepcopy(e), 'established_by': 'task/Available Options'}
                for sid, e in data['entities'].items()}
    subjects['T0'] = {'name': 'Current public task / bundle decision', 'established_by': 'task'}
    for index in range(1, current):
        subjects[f'P{index}'] = {'name': f'Previous Product {index} purchase state',
            'established_by': 'task: ALL earlier Products before the current Product index'}
    ownership = {}
    memories = {m['source_ref']: m for m in public_input['actual_retrieved_memories']}
    for ref, source in data['source_subject_map'].items():
        owners = list(source['subject_refs'])
        basis = 'Actual public search result identity and executed click/page continuity'
        if ref == 'task':
            owners = ['T0']; basis = 'Public current task'
        elif ref == 'e000':
            owners = ['T0']; basis = 'Initial public session context; not a candidate attribute page'
        elif ref in memories:
            match = re.search(r'Subject:\s*"Product (\d+) purchase"', memories[ref]['content'])
            owners = [f'P{match[1]}'] if match and f'P{match[1]}' in subjects else []
            basis = 'Visible versioned retrieved record subject header; not certification of its factual value'
        ownership[ref] = {'ownership': 'known' if owners else 'unknown',
            'subject_refs': owners, 'public_text': source['public_text'], 'basis': basis,
            'page_kind': source['kind']}
        if ref in memories:
            ownership[ref]['version_sha256'] = memories[ref]['version_sha256']
    return {'subject_registry': subjects, 'source_ownership_registry': ownership,
        'allowed_claims_up_to_event': copy.deepcopy(data['allowed_claims_up_to_event']),
        'allowed_claims_from_latest_event': copy.deepcopy(data['allowed_claims_from_latest_event']),
        'public_projection': copy.deepcopy(public_input),
        'cutoff': cutoff}


def binding_check(evidence, subjects, ownership):
    """Structural binding ONLY. Deliberately never accesses fact/need text."""
    errors = []
    subject = evidence['subject_ref']
    if subject not in subjects:
        errors.append('UNKNOWN_SUBJECT_REF')
    refs = evidence['source_refs']
    absent = [ref for ref in refs if ref not in ownership]
    if absent:
        errors.append('SOURCE_EXISTENCE')
    unknown = [ref for ref in refs if ref in ownership and ownership[ref]['ownership'] != 'known']
    if unknown:
        errors.append('SOURCE_OWNERSHIP_UNRESOLVED')
    owners = {sid for ref in refs if ref in ownership and ownership[ref]['ownership'] == 'known'
              for sid in ownership[ref]['subject_refs']}
    # A derived fact may also cite another subject's premises (e.g. prior budget).
    # At least one known source must bind the asserted subject; value entailment
    # must separately justify the complete claim using the cited premises.
    if not absent and not unknown and subject in subjects and subject not in owners:
        errors.append('SUBJECT_BINDING')
    return sorted(set(errors))
