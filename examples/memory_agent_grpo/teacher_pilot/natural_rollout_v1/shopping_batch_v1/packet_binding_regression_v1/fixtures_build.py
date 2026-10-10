"""Extract real tapes and declare inspectable, bounded offline oracle targets.

fact_annotations and need_scopes are explicitly an OFFLINE semantic sidecar.
They do not exist in v2's model output. The validator cannot accept an unseen
free-text claim merely because these annotated fixtures happened to pass.
"""
import copy
import json
import re
from .common import HERE, SOURCE, read, save, sha, digest
from .public_evidence import ledger, derivations, atom, canonical, norm, page

PLAN = [('ready_true_p2', i, False) for i in (6, 7, 10, 11, 15)] + [
    ('ready_true_p3', i, True) for i in (7, 11, 15)] + [
    ('longterm_p4', 3, False), ('longterm_p6', 3, False), ('root_C', 14, False)]


def identify(fact, entities):
    """Reviewed fixture aliases only, never used to parse new model proposals."""
    aliases = ['Xiaomei', 'Lywins', 'Heavy Duty', 'Generetic', 'Hawthorne', 'Sure Fit Deluxe',
        '1 Light Wall Sconce', 'Deco 79', 'Noble House', 'CHDHALTD', 'Christopher Knight',
        'Tobias', 'Brika', 'Furniking', '3-Tier Bakers Rack', 'End Table Snack Tray', 'Room Essentials']
    for alias in aliases:
        candidates = [sid for sid, entity in entities.items() if alias in entity['name']]
        if len(candidates) == 1 and alias in fact:
            return candidates[0]
    return None


def annotation(fact, entities):
    """Finite annotation templates for the actual strings; not general entailment."""
    sid = identify(fact, entities); claims = []; tag = 'context_paraphrase'
    if fact.startswith('Among listed candidates, only'):
        sid = next(s for s, e in entities.items() if 'Hawthorne' in e['name'])
        claims = [atom('decision', 'winner', sid, 'derived')]; tag = 'public_comparison'
    elif sid:
        if any(x in fact for x in ('not yet been observed', 'not yet confirmed', 'not yet opened')):
            predicate = 'color' if 'Color' in fact and 'Material' not in fact else 'material'
            claims.append(atom(sid, predicate, 'unobserved', 'unobserved')); tag = 'current_absence'
        elif 'has literal Material' in fact or 'has Material:' in fact or 'shows literal' in fact:
            match = re.search(r"(?:has (?:literal )?Material\s*:?\s*[\"']?|shows literal\s*'Material:\s*)([A-Za-z ]+)", fact)
            assert match, fact
            value = re.split(r' on its | \(from | \(Features| and ', match[1])[0].strip()
            claims.append(atom(sid, 'material', value)); tag = 'literal_assertion'
            color = re.search(r"'Color:\s*([^']+)'", fact)
            if color:
                claims.append(atom(sid, 'color', color[1]))
        elif 'satisfies the Product' in fact and 'within remaining budget' in fact:
            claims = [atom(sid, 'eligible', 'true', 'derived')]; tag = 'public_comparison'
        elif 'does NOT satisfy' in fact:
            claims = [atom(sid, 'eligible', 'false', 'derived')]; tag = 'public_comparison'
        elif 'none of the listed results exactly matches' in fact:
            claims = [atom(sid, 'query_outcome', 'no_exact_match', 'failed_search')]; tag = 'search_absence'
    return {'text': fact, 'claims': claims, 'annotation_kind': tag,
        'provenance': 'Offline Codex regression annotation of this exact existing string. Not independent human certification and not runtime data.'}


def neutral(previous):
    return {'add_needs': [], 'add_evidence': [], 'remove_evidence': [], 'resolve': [],
        'add_pending': [], 'remove_pending': [], 'add_failed_attempts': [],
        'remove_failed_attempts': [], 'set_unresolved': copy.deepcopy(previous['unresolved']),
        'set_ready_to_commit': 'keep'}


def build_one(export, export_name, index, expected_pass):
    tape = export['packet_trajectory'][index]
    assert tape['event_index'] == index
    prefix = ledger(export, index)
    old = copy.deepcopy(tape['input']['previous_packet']); actual = copy.deepcopy(tape['target_delta'])
    event_id = f'e{index:03d}'; latest = prefix['public_events'][-1]
    assert latest['event_id'] == event_id and latest['action'] == tape['input']['latest_action']
    assert latest['observation'] == tape['input']['latest_observation']['observation']
    catalog = {}
    for evidence in old['evidence'] + actual['add_evidence']:
        catalog[digest(evidence['fact'])] = annotation(evidence['fact'], prefix['entities'])
    # Only needs that assert current candidate coverage/comparison need a sidecar.
    if export_name == 'ready_true_p2':
        scopes = {'N4': ['material', 'price'], 'N5': ['material', 'price']}
    elif export_name == 'ready_true_p3':
        scopes = {'N3': ['material', 'price']}
    elif export_name == 'longterm_p4':
        scopes = {'N4': ['material'], 'N5': ['price'], 'N6': ['material', 'price']}
    elif export_name == 'longterm_p6':
        scopes = {'N3': ['material', 'price'], 'N4': ['material', 'price']}
    else:
        scopes = {'N4': ['color', 'price'], 'N5': ['color', 'price']}
    need_scopes = {nid: {'scope_refs': list(prefix['entities']), 'required_predicates': predicates,
        'closure': 'all_listed_subjects', 'basis': next(n['text'] for n in old['needs'] if n['id'] == nid),
        'annotation_only_not_present_in_v2': True} for nid, predicates in scopes.items()}
    required_predicates = ['color', 'price'] if export_name == 'root_C' else ['material', 'price']
    required = [{'subject_ref': s, 'predicate': p} for s in prefix['entities'] for p in required_predicates]
    gold = neutral(old); removals = set(); expected_add = []; expected_failed = []
    required_codes = []
    if expected_pass:
        gold = copy.deepcopy(actual)
        removals = {digest(old['evidence'][i]) for i in actual['remove_evidence']}
        expected_add = copy.deepcopy(actual['add_evidence'])
    elif export_name == 'ready_true_p2' and index == 6:
        required_codes = ['SOURCE_SUPPORT']
    elif export_name == 'ready_true_p2' and index in (7, 11, 15):
        subject = prefix['source_subject_map'][event_id]['subject_refs'][0]
        latest_literal = next(c for c in prefix['allowed_claims_from_latest_event'] if c['predicate'] == 'material')
        for evidence in old['evidence']:
            claims = catalog[digest(evidence['fact'])]['claims']
            if any(c['subject_ref'] == subject and c['predicate'] == 'material' for c in claims):
                removals.add(digest(evidence))
        fact = f"Candidate '{prefix['entities'][subject]['name']}' has literal Material: {latest_literal['value']}."
        expected_add = [{'need_id': 'N4', 'fact': fact, 'source_refs': [event_id]}]
        catalog[digest(fact)] = {'text': fact, 'claims': [atom(subject, 'material', latest_literal['value'])],
            'annotation_kind': 'literal_assertion', 'provenance': 'Offline oracle/regression target from this actual latest literal; not human-certified gold.'}
        required_codes = ['SUBJECT_BINDING', 'SOURCE_SUPPORT'] if index == 7 else ['CORRECTION_ON_NEW_LITERAL_EVIDENCE']
        if index == 11:
            gold['set_ready_to_commit'] = 'false'
            gold['add_pending'] = ['N4', 'N5']
    elif export_name == 'ready_true_p2' and index == 10:
        gold['set_ready_to_commit'] = 'false'
        required_codes = ['NEED_SCOPE', 'READINESS_GATE']
    elif export_name in {'longterm_p4', 'longterm_p6'}:
        expected_add = copy.deepcopy(actual['add_evidence'])
        subject = prefix['source_subject_map'][event_id]['subject_refs'][0]
        for evidence in old['evidence']:
            if any(c['subject_ref'] == subject and c['kind'] == 'unobserved'
                   for c in catalog[digest(evidence['fact'])]['claims']):
                removals.add(digest(evidence))
        required_codes = ['NEED_SCOPE']
    elif export_name == 'root_C':
        previous_event = prefix['public_events'][-2]
        assert latest['action'] == previous_event['action']
        assert norm(page(latest['observation'])) == norm(page(previous_event['observation']))
        # Move the already-failed query out of ordinary evidence; no new query.
        for evidence in old['evidence']:
            if 'e013' in evidence['source_refs'] and 'End Table Snack Tray' in evidence['fact']:
                removals.add(digest(evidence))
        expected_failed = [{'text': 'The same End Table query at e013 and e014 returned the same public results without the exact requested candidate; the attempt made no progress.',
                            'source_refs': ['e013', 'e014']}]
        required_codes = ['FAILED_ATTEMPT_SEPARATION']
    if not expected_pass:
        gold['remove_evidence'] = [i for i, e in enumerate(old['evidence']) if digest(e) in removals]
        gold['add_evidence'] = expected_add
        gold['add_failed_attempts'] = expected_failed
    # Expected state constructed from semantic edit declarations, independently
    # of frozen index-based apply(). Mutant deletion indices are tested later.
    expected = copy.deepcopy(old)
    expected['evidence'] = [copy.deepcopy(e) for e in old['evidence'] if digest(e) not in removals] + expected_add
    expected['progress']['failed_attempts'] += expected_failed
    for nid in gold['resolve']:
        if nid in expected['progress']['pending']:
            expected['progress']['pending'].remove(nid)
        if nid not in expected['progress']['resolved']:
            expected['progress']['resolved'].append(nid)
    for nid in gold['add_pending']:
        if nid in expected['progress']['resolved']:
            expected['progress']['resolved'].remove(nid)
        if nid not in expected['progress']['pending']:
            expected['progress']['pending'].append(nid)
    expected['unresolved'] = copy.deepcopy(gold['set_unresolved'])
    if gold['set_ready_to_commit'] != 'keep':
        expected['ready_to_commit'] = gold['set_ready_to_commit'] == 'true'
    out = {'case_id': export_name + '_' + event_id, 'event_id': event_id,
        'episode_id': export['case']['episode_id'], 'session_id': export['case']['session_id'],
        'current_task': export['current_task'], 'previous_packet': old,
        'latest_public_event': latest, 'runtime_input': copy.deepcopy(tape['input']),
        'source_subject_map': prefix['source_subject_map'],
        'entity_catalog': prefix['entities'], 'allowed_claims_from_latest_event': prefix['allowed_claims_from_latest_event'],
        'allowed_claims_up_to_event': prefix['allowed_claims_up_to_event'],
        'public_derived_claims': derivations(export, prefix),
        'fact_annotations': catalog, 'need_scopes': need_scopes, 'required_candidate_coverage': required,
        'actual_delta': actual,
        'expected_actual_delta_result': {'transition_local_pass': expected_pass, 'required_reason_codes': required_codes},
        'gold_delta': gold, 'expected_gold_packet': expected,
        'invariants_checked': ['SOURCE_EXISTENCE', 'SOURCE_SUPPORT', 'SUBJECT_BINDING',
            'CORRECTION_ON_NEW_LITERAL_EVIDENCE', 'NEED_SCOPE', 'READINESS_GATE',
            'STALE_STATE_CLEANUP', 'FAILED_ATTEMPT_SEPARATION'],
        'observed_failure': {'event_id': event_id, 'source_refs': ['e013', 'e014'], 'kind': 'repeated_query_same_state'} if export_name == 'root_C' else None,
        'allowed_failed_attempt_text': expected_failed[0]['text'] if expected_failed else None,
        'source_provenance': {'case_export': str(SOURCE / 'case_exports' / (export_name + '.json')),
            'export_sha256': sha(SOURCE / 'case_exports' / (export_name + '.json')),
            'original_response_ref': tape['response_ref'], 'response_sha256': sha(tape['response_ref']),
            'prefix_cutoff': index, 'future_event_count_in_fixture': 0},
        'notes': ['Offline oracle/regression target written by Codex; not independent human-certified gold.',
            'Each fixture starts from its actual recorded previous Packet, not a previous oracle output.',
            'Transition-local correctness is separate from inherited full-Packet errors.',
            'Current goal/scope annotations are fixture-only sidecars. They are not supplied to a Packet model or added to frozen v2.',
            'Coverage is the literal all-candidates need in these examples; this is not a universal demand to browse every candidate in every task.',
            '45 previous candidates and 69 rejected candidates remain untouched; no promotion to training gold.']}
    return out


def build():
    fixtures = []
    for name, index, passed in PLAN:
        export = read(SOURCE / 'case_exports' / (name + '.json'))
        fixture = build_one(export, name, index, passed)
        save(HERE / 'fixtures' / (fixture['case_id'] + '.json'), fixture)
        fixtures.append(fixture)
    (HERE / 'oracle_transitions.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in fixtures))
    return fixtures
