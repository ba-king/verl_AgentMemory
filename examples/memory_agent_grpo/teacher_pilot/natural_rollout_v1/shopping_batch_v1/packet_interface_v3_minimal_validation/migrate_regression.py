"""One-time OFFLINE V2-to-V3 annotation; not evidence of live V3 generation.

Legacy text/subject and need/scope annotations are used here to label the newly
required fields. They are not copied into the V3 validator input. Existing fact
strings, delta operations, removal indices and event cutoffs stay unchanged.
"""
import copy
import json
import re
from .common import HERE, V1, read, save, sha, digest
from .trusted_registry import public_projection, build as build_registry
from ..packet_binding_regression_v1.fixtures_build import identify


def migrate(old, export):
    cutoff = int(old['event_id'][1:])
    public = public_projection(export, cutoff)
    trusted = build_registry(public, cutoff)
    candidates = {k: v for k, v in trusted['subject_registry'].items() if k.startswith('C')}
    log = []

    def subject_of(evidence):
        annotated = old['fact_annotations'].get(digest(evidence['fact']), {}).get('claims', [])
        refs = {c['subject_ref'] for c in annotated if c['subject_ref'] in candidates}
        if len(refs) == 1:
            subject = next(iter(refs)); basis = 'Existing offline single-subject claim annotation'
        elif evidence['fact'].startswith('Among listed candidates, only'):
            subject = identify(evidence['fact'], candidates)
            basis = 'Offline migration: comparison statement asserted about the named winner, not a new entity'
        elif evidence['fact'].startswith('Product 4 candidates include:'):
            subject = 'T0'; basis = 'Offline migration: public task candidate-set statement'
        else:
            subject = identify(evidence['fact'], candidates)
            basis = 'Offline migration: bounded explicit candidate name annotation'
            if subject is None:
                histories = {s for r in evidence['source_refs']
                    for s in trusted['source_ownership_registry'][r]['subject_refs'] if s.startswith('P')}
                subject = next(iter(histories)) if len(histories) == 1 else 'T0'
                basis = 'Offline migration: prior purchase state or current bundle aggregate'
        if subject not in trusted['subject_registry']:
            raise ValueError('MIGRATION_CANNOT_ESTABLISH_SUBJECT')
        log.append({'kind': 'subject_annotation', 'fact_digest': digest(evidence['fact']),
                    'subject_ref': subject, 'basis': basis})
        return subject

    def migrate_evidence(evidence):
        return {**copy.deepcopy(evidence), 'subject_ref': subject_of(evidence)}

    def migrate_need(need):
        if need['id'] in old['need_scopes']:
            scope = list(old['need_scopes'][need['id']]['scope_refs'])
            basis = 'Offline reviewed required-all scope from original need; NOT automatic Teacher-scope certification'
        else:
            # Scope labeling during migration only; runtime checks never parse text.
            history = sorted(set(re.findall(r'Product (\d+)', need['text'])))
            scope = [f'P{x}' for x in history if f'P{x}' in trusted['subject_registry']]
            scope = scope or ['T0']
            basis = 'Offline migration annotation of public historical/task premise'
        log.append({'kind': 'scope_annotation', 'need_id': need['id'], 'scope_refs': scope, 'basis': basis})
        return {**copy.deepcopy(need), 'scope_refs': scope}

    def packet(value):
        out = copy.deepcopy(value)
        out['needs'] = [migrate_need(n) for n in value['needs']]
        out['evidence'] = [migrate_evidence(e) for e in value['evidence']]
        return out

    def delta(value):
        out = copy.deepcopy(value)
        out['add_needs'] = [migrate_need(n) for n in value['add_needs']]
        out['add_evidence'] = [migrate_evidence(e) for e in value['add_evidence']]
        return out

    prev = packet(old['previous_packet']); actual = delta(old['actual_delta'])
    oracle = delta(old['gold_delta']); expected = packet(old['expected_gold_packet'])
    # Factual interpretation keeps ONLY predicate/value/kind, never a text->subject map.
    # Identity comes from the new structural field for both support and repair.
    literals = {key: {'text': row['text'], 'claims': [
        {k: c[k] for k in ('predicate', 'value', 'kind')} for c in row['claims']]}
        for key, row in old['fact_annotations'].items()}
    derived = copy.deepcopy(old['public_derived_claims'])
    for claim in derived:
        if claim['subject_ref'] == 'decision':
            claim['subject_ref'] = claim['value'].upper()
    # Predicate criteria remain an explicitly EXTERNAL factual audit. They say
    # what constitutes completion, never WHICH subjects a need requires.
    criteria = {nid: {'required_predicates': scope['required_predicates'],
        'origin': 'Bounded factual-completion annotation; no scope/subject set'}
        for nid, scope in old['need_scopes'].items()}
    result = {'case_id': old['case_id'], 'event_id': old['event_id'],
        'previous_packet_v3': prev, 'actual_delta_v3': actual, 'oracle_delta_v3': oracle,
        'expected_packet_v3': expected, 'trusted': trusted,
        'factual_audit': {'literal_interpretations': literals,
            'derived_claims': derived, 'need_completion_criteria': criteria,
            'observed_failure': old['observed_failure'],
            'allowed_failed_attempt_text': old['allowed_failed_attempt_text']},
        'expected_actual_delta_result': copy.deepcopy(old['expected_actual_delta_result']),
        'migration_log': log,
        'provenance': {'original_fixture': str(V1 / 'fixtures' / (old['case_id'] + '.json')),
            'original_fixture_sha256': sha(V1 / 'fixtures' / (old['case_id'] + '.json')),
            'case_export': old['source_provenance']['case_export'],
            'case_export_sha256': sha(old['source_provenance']['case_export']),
            'original_response_ref': old['source_provenance']['original_response_ref'],
            'cutoff': cutoff, 'real_v3_model_response': False,
            'oracle_status': 'Codex-authored offline oracle/regression target, not independently human-certified gold'},
        'scope_contract_mode': 'A_DECLARED_MODEL_SCOPE; migration annotations fixed for regression, not runtime-certified scope truth'}
    return result


def build_all():
    fixtures = []
    for path in sorted((V1 / 'fixtures').glob('*.json')):
        old = read(path); export = read(old['source_provenance']['case_export'])
        fixture = migrate(old, export)
        save(HERE / 'fixtures' / path.name, fixture)
        fixtures.append(fixture)
    return fixtures
