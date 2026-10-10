import copy
import json
import unittest
from .common import HERE, V1, read
from .interface_v3 import project_v2
from .trusted_registry import binding_check, build, public_projection
from .migrate_regression import migrate
from .factual_audit import completion_receipts
from .validator_v3 import validate, run, scope_check


def fixture(name):
    return read(HERE / 'fixtures' / (name + '.json'))


def scope_truncation_probe():
    original = fixture('ready_true_p2_e010')
    mutant = copy.deepcopy(original)
    # Lywins is fully observed at e003. Deliberately omit the other subjects.
    for need in mutant['previous_packet_v3']['needs']:
        if need['id'] in {'N4', 'N5'}:
            need['scope_refs'] = ['C1']
    lock = {n['id']: n['scope_refs'] for n in original['previous_packet_v3']['needs'] if n['id'] in {'N4', 'N5'}}
    declared_only = validate(mutant, mutant['actual_delta_v3'])
    locked = validate(mutant, mutant['actual_delta_v3'], trusted_scope_lock=lock)
    return {'case_id': original['case_id'], 'mutation': 'N4/N5 scope C1,C2,C3,C4 -> C1; delta unchanged',
        'design_A': {'verdict': declared_only['transition_local_verdict'], 'reasons': declared_only['reason_codes']},
        'design_B_explicit_reference_lock': {'verdict': locked['transition_local_verdict'], 'reasons': locked['reason_codes']},
        'lock_provenance': 'Reviewed original public need/candidate list, injected ONLY for this mutation diagnostic; not a production task parser',
        'full_original_actual': validate(original, original['actual_delta_v3'])['reason_codes'],
        'demonstrates': 'Structural required-all enforcement cannot authenticate a Teacher-created required set.'}


class V3RegressionTests(unittest.TestCase):
    def test_eleven_regressions(self):
        rows = run([read(p) for p in sorted((HERE / 'fixtures').glob('*.json'))])
        self.assertEqual(len(rows), 11)
        self.assertTrue(all(r['regression_pass'] for r in rows))

    def test_actual_and_oracle_projection_preserves_v2(self):
        for path in (HERE / 'fixtures').glob('*.json'):
            f = read(path); old = read(V1 / 'fixtures' / path.name)
            for newkey, oldkey in [('previous_packet_v3', 'previous_packet'), ('expected_packet_v3', 'expected_gold_packet')]:
                self.assertEqual(project_v2(f[newkey]), old[oldkey])
            for newkey, oldkey in [('actual_delta_v3', 'actual_delta'), ('oracle_delta_v3', 'gold_delta')]:
                self.assertEqual(project_v2(f[newkey], True), old[oldkey])

    def test_subject_only_mutation(self):
        f = fixture('ready_true_p2_e007'); e = copy.deepcopy(f['oracle_delta_v3']['add_evidence'][0])
        t = f['trusted']; check = lambda x: binding_check(x, t['subject_registry'], t['source_ownership_registry'])
        self.assertEqual(e['subject_ref'], 'C2') # In this actual task C2 is Xiaomei.
        self.assertEqual(check(e), [])
        e['subject_ref'] = 'C3'
        self.assertEqual(check(e), ['SUBJECT_BINDING'])

    def test_binding_does_not_access_fact_text(self):
        f = fixture('ready_true_p2_e007'); t = f['trusted']
        class StructuralOnly(dict):
            def __getitem__(self, key):
                if key in {'fact', 'need_id'}:
                    raise AssertionError('BINDING_READ_TEXT')
                return super().__getitem__(key)
        e = StructuralOnly(subject_ref='C2', source_refs=['e007'])
        self.assertEqual(binding_check(e, t['subject_registry'], t['source_ownership_registry']), [])
        e['subject_ref'] = 'C3'
        self.assertIn('SUBJECT_BINDING', binding_check(e, t['subject_registry'], t['source_ownership_registry']))

    def test_unknown_subject(self):
        f = fixture('ready_true_p2_e007'); d = copy.deepcopy(f['oracle_delta_v3'])
        d['add_evidence'][0]['subject_ref'] = 'C999'
        self.assertIn('UNKNOWN_SUBJECT_REF', validate(f, d)['reason_codes'])

    def test_unknown_ownership_is_not_existence(self):
        f = fixture('ready_true_p2_e007'); t = f['trusted']
        t['source_ownership_registry']['eXYZ'] = {'ownership': 'unknown', 'subject_refs': [], 'public_text': 'Title unavailable'}
        e = {'subject_ref': 'C2', 'source_refs': ['eXYZ']}
        self.assertEqual(binding_check(e, t['subject_registry'], t['source_ownership_registry']), ['SOURCE_OWNERSHIP_UNRESOLVED'])

    def test_absent_source_rejected(self):
        f = fixture('ready_true_p2_e007'); d = copy.deepcopy(f['oracle_delta_v3'])
        d['add_evidence'][0]['source_refs'] = ['e999']
        self.assertIn('SOURCE_EXISTENCE', validate(f, d)['reason_codes'])

    def test_binding_pass_not_fact_support(self):
        f = fixture('ready_true_p2_e006'); t = f['trusted']
        e = f['actual_delta_v3']['add_evidence'][0]
        self.assertEqual(binding_check(e, t['subject_registry'], t['source_ownership_registry']), [])
        self.assertIn('SOURCE_SUPPORT', validate(f, f['actual_delta_v3'])['reason_codes'])

    def test_unseen_fact_does_not_break_structural_binding(self):
        f = fixture('ready_true_p2_e007'); d = copy.deepcopy(f['oracle_delta_v3'])
        d['add_evidence'][0]['fact'] = 'It is iron.'
        r = validate(f, d)
        self.assertNotIn('SUBJECT_BINDING', r['reason_codes'])
        self.assertIn('FACTUAL_AUDIT_UNRESOLVED', r['reason_codes'])

    def test_legacy_subject_scope_sidecars_absent(self):
        for path in (HERE / 'fixtures').glob('*.json'):
            f = read(path)
            for key in ('fact_annotations', 'source_subject_map', 'need_scopes', 'required_candidate_coverage'):
                self.assertNotIn(key, f)
            for annotation in f['factual_audit']['literal_interpretations'].values():
                for claim in annotation['claims']:
                    self.assertNotIn('subject_ref', claim)
            for criterion in f['factual_audit']['need_completion_criteria'].values():
                self.assertNotIn('scope_refs', criterion)
                self.assertNotIn('subject_ref', criterion)

    def test_validator_ignores_migration_log_and_expected_labels(self):
        f = fixture('ready_true_p2_e007'); result = validate(f, f['actual_delta_v3'])
        f['migration_log'] = []; f['case_id'] = 'other'; f['expected_actual_delta_result'] = {}
        for key in ('fact_annotations', 'source_subject_map', 'need_scopes'):
            f[key] = 'MUST_NOT_READ'
        f['oracle_delta_v3'] = {}; f['expected_packet_v3'] = {}
        self.assertEqual(result, validate(f, f['actual_delta_v3']))

    def test_need_scope_never_reads_text(self):
        f = fixture('longterm_p4_e003'); baseline = validate(f, f['actual_delta_v3'])
        for need in f['previous_packet_v3']['needs']:
            need['text'] = 'opaque text with no scope words'
        self.assertEqual(baseline['reason_codes'], validate(f, f['actual_delta_v3'])['reason_codes'])
        self.assertIn('NEED_SCOPE', baseline['reason_codes'])

    def test_scope_refs_not_need_text_control_gate(self):
        packet = {'needs': [{'id': 'X', 'scope_refs': ['C1', 'C2', 'C3', 'C4']}], 'ready_to_commit': False}
        subjects = {f'C{i}': {} for i in range(1, 5)}
        receipts = {'X': {'C1': {'satisfied': True}}}
        errors = scope_check(packet, receipts, subjects, ['X'])
        self.assertEqual(errors[0]['code'], 'NEED_SCOPE')
        packet['needs'][0]['scope_refs'] = ['C1']
        self.assertEqual(scope_check(packet, receipts, subjects, ['X']), [])

    def test_scope_truncation_is_real_unresolved_risk(self):
        r = scope_truncation_probe()
        self.assertEqual(r['design_A']['verdict'], 'PASS')
        self.assertEqual(r['design_B_explicit_reference_lock']['verdict'], 'FAIL')
        self.assertIn('SCOPE_CONTRACT_VIOLATION', r['design_B_explicit_reference_lock']['reasons'])

    def test_unknown_and_empty_scope_rejected(self):
        f = fixture('longterm_p4_e003')
        for scope, code in [(['C999'], 'UNKNOWN_SUBJECT_REF'), ([], 'INVALID_SCOPE_REFS')]:
            altered = copy.deepcopy(f); altered['previous_packet_v3']['needs'][3]['scope_refs'] = scope
            self.assertIn(code, validate(altered, altered['actual_delta_v3'])['reason_codes'])

    def test_ready_and_resolve_independent(self):
        f = fixture('ready_true_p2_e010'); d = copy.deepcopy(f['actual_delta_v3'])
        d['resolve'] = []
        self.assertIn('READINESS_GATE', validate(f, d)['reason_codes'])
        d = copy.deepcopy(f['actual_delta_v3']); d['set_ready_to_commit'] = 'false'
        self.assertIn('NEED_SCOPE', validate(f, d)['reason_codes'])

    def test_good_local_and_inherited_hawthorne_separate(self):
        for event in (7, 11, 15):
            f = fixture(f'ready_true_p3_e{event:03d}'); r = validate(f, f['actual_delta_v3'])
            self.assertEqual(r['transition_local_verdict'], 'PASS')
            self.assertEqual(r['full_packet_verdict'], 'FAIL')
            self.assertTrue(any(e['code'] == 'STALE_STATE_CLEANUP' and e['claim']['subject_ref'] == 'C1'
                                for e in r['inherited_errors']))

    def test_correct_value_does_not_excuse_remaining_wrong(self):
        f = fixture('ready_true_p2_e015'); d = copy.deepcopy(f['oracle_delta_v3'])
        d['remove_evidence'] = d['remove_evidence'][:1]
        self.assertIn('CORRECTION_ON_NEW_LITERAL_EVIDENCE', validate(f, d)['reason_codes'])

    def test_remove_without_replacement_fails(self):
        f = fixture('ready_true_p2_e011'); d = copy.deepcopy(f['oracle_delta_v3']); d['add_evidence'] = []
        self.assertIn('CORRECTION_ON_NEW_LITERAL_EVIDENCE', validate(f, d)['reason_codes'])

    def test_no_future_or_private_projection(self):
        old = read(V1 / 'fixtures' / 'ready_true_p2_e007.json')
        export = read(old['source_provenance']['case_export']); expected = migrate(old, export)
        mutated = copy.deepcopy(export)
        for e in mutated['public_prefix_events_only'][7:]:
            e['observation'] = 'FUTURE_SENTINEL'
        mutated['packet_trajectory'] = 'DO_NOT_READ_RESPONSE_TAPE_OR_FUTURE'
        mutated['scorer'] = {'private_target': 'DO_NOT_READ'}
        self.assertEqual(expected, migrate(old, mutated))
        for subject in ('C3', 'C4'):
            self.assertEqual(expected['trusted']['subject_registry'][subject]['visible_asins'], [])

    def test_registry_does_not_read_old_source_map(self):
        old = read(V1 / 'fixtures' / 'ready_true_p2_e007.json'); export = read(old['source_provenance']['case_export'])
        first = migrate(old, export)['trusted']
        old['source_subject_map'] = {'e007': {'subject_refs': ['C999']}}
        self.assertEqual(first, migrate(old, export)['trusted'])

    def test_unknown_page_continuity_not_guessed_from_title(self):
        f = fixture('ready_true_p2_e007'); public = copy.deepcopy(f['trusted']['public_projection'])
        public['public_prefix_events_only'][5]['action']['arguments']['item'] = 'B000000000'
        registry = build(public, 7)['source_ownership_registry']
        self.assertEqual(registry['e007']['ownership'], 'unknown')

    def test_search_query_subject_not_page_owner(self):
        f = fixture('root_C_e014'); t = f['trusted']
        self.assertEqual(t['source_ownership_registry']['e014']['ownership'], 'unknown')
        self.assertIn('FAILED_ATTEMPT_SEPARATION', validate(f, f['actual_delta_v3'])['reason_codes'])
        self.assertEqual(validate(f, f['oracle_delta_v3'])['transition_local_verdict'], 'PASS')

    def test_factual_completion_not_scope_authentication(self):
        f = fixture('longterm_p4_e003'); receipts = completion_receipts(f['factual_audit'], f['trusted'])
        self.assertTrue(receipts['N4']['C1']['satisfied'])
        self.assertFalse(receipts['N4']['C2']['satisfied'])
        del f['previous_packet_v3']['needs']
        self.assertEqual(receipts, completion_receipts(f['factual_audit'], f['trusted']))


if __name__ == '__main__':
    from .common import no_network
    with no_network():
        unittest.main()
