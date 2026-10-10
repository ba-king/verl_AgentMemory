"""Public-only hotel-chain task contract; unchanged generic Packet V3 types.

No Bank, hidden reference, DB or future-session input is accepted here. Literal
instruction clauses become obligations, never candidate/answer conclusions.
"""
import copy
import re
from jsonschema import Draft202012Validator
from .common import digest
from teacher_pilot.natural_rollout_v1.shopping_batch_v1.packet_interface_v3_minimal_validation import interface_v3 as V3

FIELDS = ('current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation')
BASE_PATHS = [f'day{d}.{f}' for d in range(1, 4) for f in FIELDS
              if not (d in (1, 2) and f == 'accommodation')]


def build(current):
    if set(current) - {'session_id', 'instruction', 'initial_observation'}:
        raise ValueError('NONPUBLIC_TASK_PROJECTION')
    task = current['instruction']
    opening = re.match(r'I am ([^.]+)\. I am joining ([^.]+) for this three-day trip to ([^.]+)\.', task)
    if not opening:
        raise ValueError('TRAVEL_PACKET_ADAPTER_UNSUPPORTED_OPENING')
    traveler, base, city = opening.groups()
    literal = f"Keep every unmentioned entry from {base}'s confirmed itinerary. Make a separate booking for this single adult."
    if literal not in task:
        raise ValueError('TRAVEL_PACKET_ADAPTER_UNSUPPORTED_BASE_RULE')
    lodging = next((s for s in task.splitlines() if s.startswith('For the accommodation on both nights,')), None)
    copy_rule = re.fullmatch(r"For the accommodation on both nights, use the same choice as ([^.]+)'s first-night accommodation\.", lodging or '')
    relative = re.fullmatch(r"For the accommodation on both nights, choose an accommodation in (.+) with minimum nights at most (\d+) and maximum occupancy at least (\d+) with the lowest nightly price strictly above that of ([^.]+)'s first-night accommodation\. It must also have the same room type as that earlier booking\.", lodging or '')
    anchor = re.fullmatch(r'For the accommodation on both nights, choose an accommodation in (.+) with room type (.+), minimum nights at most (\d+) and maximum occupancy at least (\d+) with the lowest nightly price that is at least \$(\d+(?:\.\d+)?)\.', lodging or '')
    if not any((copy_rule, relative, anchor)):
        raise ValueError('TRAVEL_PACKET_ADAPTER_UNSUPPORTED_LODGING_RULE')
    previous = copy_rule[1] if copy_rule else relative[4] if relative else None
    tie = 'Break an equal-price tie by ascending exact database name (case-sensitive).'
    if (relative or anchor) and tie not in task:
        raise ValueError('MISSING_PUBLIC_TIE_RULE')
    ending = f"Output only the complete three-day plan in the standard === {traveler}'s Plan === format."
    allowed_lines = [opening[0], literal, lodging, ending] + ([tie] if not copy_rule else [])
    if task.splitlines() != allowed_lines[:3] + ([tie] if not copy_rule else []) + [ending]:
        raise ValueError('UNCOVERED_PUBLIC_TASK_CLAUSE')
    subjects = {'T0': {'name': 'Current public Travel task'},
                f'person:{base}': {'name': base, 'role': 'base traveler'},
                f'person:{traveler}': {'name': traveler, 'role': 'current traveler'},
                f'lodgings:{city}': {'name': f'Public accommodations in {city}', 'role': 'candidate collection'}}
    if previous:
        subjects[f'person:{previous}'] = {'name': previous, 'role': 'referenced actual prior traveler'}
    specifications = [
        ('BASE_FIELDS', f"Obtain {base}'s independently subject-bound confirmed state for the unchanged itinerary fields: {', '.join(BASE_PATHS)}. Preserve exact values; a different traveler's equal values do not establish this base state.", [f'person:{base}']),
    ]
    if previous:
        specifications.append(('PRIOR_LODGING', f"Obtain the actual first-night accommodation submitted for {previous}." +
            (' Also establish its public room type and nightly price for the stated relative-price comparison.' if relative else ''), [f'person:{previous}']))
    specifications.append(('LODGING_RULE', lodging + (' ' + tie if not copy_rule else ''),
                           ['T0', f'lodgings:{city}'] + ([f'person:{previous}'] if previous else [])))
    specifications.append(('COMPLETE_PLAN', f"Prepare a separate single-adult proposal for {traveler}, three ordered days, with all seven fields per day ({', '.join(FIELDS)}). Retain the required {base} fields and satisfy the current lodging rule on both nights. Submit through the unchanged SUBMIT_PLAN interface with exact field strings.", [f'person:{traveler}']))
    base_public = current.get('initial_observation', {}).get('base_person')
    if base_public:
        if base_public['name'] != base:
            raise ValueError('BOOTSTRAP_SUBJECT_MISMATCH')
        # The existing bootstrap query is a public inherited trip constraint,
        # not a source of hidden later-session needs or selected hotel facts.
        specifications.append(('PUBLIC_TRIP_CONSTRAINTS', 'Respect the public confirmed trip constraints: ' + base_public['query'], ['T0', f'person:{base}']))
    needs = [{'id': f'N{i}', 'text': content, 'scope_refs': scope}
             for i, (_, content, scope) in enumerate(specifications, 1)]
    packet = V3.empty_packet()
    packet['needs'] = needs
    packet['progress']['pending'] = [n['id'] for n in needs]
    Draft202012Validator(V3.PACKET_SCHEMA).validate(packet)
    contract = {'authority': 'HARNESS_TASK_CONTRACT', 'domain': 'travel', 'task_digest': digest(task),
        'session_id': current['session_id'], 'traveler': traveler, 'base': base, 'city': city,
        'previous': previous, 'rule_kind': 'copy' if copy_rule else 'relative' if relative else 'anchor',
        'subject_registry': subjects, 'public_lodging_clause': lodging,
        'base_paths': BASE_PATHS, 'covered_public_clauses': allowed_lines,
        'need_kinds': {n['id']: spec[0] for n, spec in zip(needs, specifications)}}
    return {'contract': contract, 'packet': packet, 'scaffold_digest': digest(packet),
            'input_keys': sorted(current), 'hidden_reference_used': False}


def guard(delta, scaffold):
    if delta is None:
        return {'passed': False, 'reason': 'NO_DELTA'}
    # Model may add genuinely local inspection/retry needs; it never rewrites
    # deterministic global obligations. Admission checks do not repair output.
    additions = delta['add_needs']
    wrong = [n for n in additions if not n['text'].startswith(('LOCAL_INSPECT', 'LOCAL_RETRY'))]
    return {'passed': not wrong, 'wrong_gating_role_creation': wrong}
