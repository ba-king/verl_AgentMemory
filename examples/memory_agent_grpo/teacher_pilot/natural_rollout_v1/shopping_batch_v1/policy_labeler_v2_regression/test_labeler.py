"""Offline calibration; altered copies are synthetic tests, never trajectory data."""
import copy
import json
import socket
import unittest
from pathlib import Path
from unittest.mock import patch
from .datasets import authenticate_write, evaluate_corpus
from .labeler import ProvenanceLedger, audit_write, cents

ROOT = Path(__file__).resolve().parent.parent / 'pilot4_resume_v1/combined'


class LabelerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.no_network = patch.object(socket.socket, 'connect', side_effect=AssertionError('OFFLINE_TEST_HTTP_FORBIDDEN'))
        cls.no_network.start()
        cls.chains = [json.loads(s) for s in (ROOT / 'analysis/cross_session_success_chains.jsonl').read_text().splitlines()]
        cls.first = json.loads(Path(cls.chains[0]['source_decision_ref']).read_text())
        cls.second = json.loads(Path(cls.chains[1]['source_decision_ref']).read_text())

    @classmethod
    def tearDownClass(cls):
        cls.no_network.stop()

    def fixture(self):
        ledger = ProvenanceLedger()
        s, entry = authenticate_write(self.first)
        r = audit_write(self.first, s, ledger, True, True, True, self.chains[0]['source_decision_ref'])
        ledger.register(r, entry)
        d = copy.deepcopy(self.second)
        state, _ = authenticate_write(d)
        return d, state, ledger

    def run_fixture(self, d, s, ledger, downstream=True):
        return audit_write(d, s, ledger, True, True, downstream, 'SYNTHETIC_CHANGED_COPY_NOT_LIVE')

    def field(self, d, path):
        return next(f for f in d['parsed_memory_action']['arguments']['record']['fields'] if f['path'] == path)

    def assert_blocked(self, result):
        self.assertNotEqual(result['label'], 'WRITE_POSITIVE_STRICT')

    def test_five_known_live_adds_recovered_and_no_automatic_terminal_credit(self):
        r = evaluate_corpus(ROOT)
        known = {(c['episode_id'], c['source_decision']) for c in self.chains}
        writes = [w for w in r['write_results'] if (w['episode_id'], w['decision_id']) in known]
        self.assertEqual(len(writes), 5)
        self.assertTrue(all(w['label'] == 'WRITE_POSITIVE_STRICT' for w in writes))
        old = [json.loads(s) for s in (ROOT / 'analysis/positive_sft_candidates.jsonl').read_text().splitlines()]
        self.assertFalse(any((w['episode_id'], w['decision_id']) in known for w in old))
        self.assertEqual(r['positive_by_action'], {'ADD': 5, 'RETRIEVE': 5})
        terminal = next(w for w in r['write_results'] if w['episode_id'] == self.first['episode_id'] and w['decision_id'] == 'p6_d003')
        self.assertTrue(terminal['grounded_all_fields'])
        self.assertEqual(terminal['label'], 'WRITE_UNCERTAIN')

    def test_correct_cumulative_and_remaining_exact(self):
        d, s, ledger = self.fixture()
        r = self.run_fixture(d, s, ledger)
        self.assertEqual(r['label'], 'WRITE_POSITIVE_STRICT')
        for path in ('total_expense_after_purchase', 'remaining_budget_after_purchase'):
            row = next(f for f in r['fields'] if f['path'] == path)
            self.assertEqual(row['label'], 'DERIVED_SAFE')
            self.assertTrue(row['provenance'])
            self.assertEqual(row['derivation']['result_cents'], cents(row['value']))

    def test_one_cent_wrong_fails(self):
        d, s, ledger = self.fixture()
        f = self.field(d, 'total_expense_after_purchase')
        f['value'] = f"${(cents(f['value']) + 1) // 100}.{(cents(f['value']) + 1) % 100:02d}"
        r = self.run_fixture(d, s, ledger)
        self.assertEqual(r['label'], 'WRITE_NEGATIVE')
        self.assertTrue(any(f['reason'] == 'ARITHMETIC_CONTRADICTION' for f in r['fields']))

    def test_invisible_operand_fails_even_if_in_ledger(self):
        d, s, ledger = self.fixture(); s['retrieved_memories'] = []
        self.assert_blocked(self.run_fixture(d, s, ledger))

    def test_hidden_private_operand_fails(self):
        d, s, ledger = self.fixture()
        s['private_scorer_operands'] = s.pop('retrieved_memories')
        self.assert_blocked(self.run_fixture(d, s, ledger))

    def test_unproven_retrieved_operand_fails(self):
        d, s, _ = self.fixture()
        self.assert_blocked(self.run_fixture(d, s, ProvenanceLedger()))

    def test_poisoned_retrieved_version_fails(self):
        d, s, ledger = self.fixture()
        s['retrieved_memories'][0]['record']['fields'][2]['value'] = '$999.99'
        self.assert_blocked(self.run_fixture(d, s, ledger))

    def test_wrong_literal_even_present_in_other_candidate_fails(self):
        d, s, ledger = self.fixture()
        self.field(d, 'material_label')['value'] = 'Aluminum'
        self.assertEqual(self.run_fixture(d, s, ledger)['label'], 'WRITE_NEGATIVE')

    def test_subject_mismatch_fails_despite_right_asin_field(self):
        d, s, ledger = self.fixture()
        d['parsed_memory_action']['arguments']['record']['subject'] = d['episode_id'] + ':p99'
        r = self.run_fixture(d, s, ledger)
        self.assertFalse(r['ownership']['passed']); self.assertEqual(r['label'], 'WRITE_NEGATIVE')

    def test_downstream_success_alone_never_positive(self):
        d, s, ledger = self.fixture()
        d['parsed_memory_action']['arguments']['record']['fields'].append({'path': 'unobserved_claim', 'value': 'certified best'})
        self.assert_blocked(self.run_fixture(d, s, ledger, downstream=True))

    def test_no_downstream_credit_even_with_grounded_fields(self):
        d, s, ledger = self.fixture()
        self.assert_blocked(self.run_fixture(d, s, ledger, downstream=False))

    def test_private_field_rejected(self):
        d, s, ledger = self.fixture()
        d['parsed_memory_action']['arguments']['record']['fields'].append({'path': 'sealed_target_asin', 'value': 'B000000000'})
        self.assertEqual(self.run_fixture(d, s, ledger)['label'], 'WRITE_NEGATIVE')

    def test_name_token_not_literal_declaration(self):
        d, s, ledger = self.fixture()
        # Remove all item literal pages but retain names and current task text.
        s['session_public_history'] = [e for e in s['session_public_history'] if e['action'].get('arguments', {}).get('item') not in {'Features', 'Description'}]
        r = self.run_fixture(d, s, ledger)
        self.assert_blocked(r)
        self.assertEqual(next(f for f in r['fields'] if f['path'] == 'material_label')['reason'], 'NO_ROLE_BOUND_PUBLIC_PROOF')

    def test_non_whitelist_math_is_rejected(self):
        for text in ('49.9901', '49.99+280.57', 'eval(330.56)', 'NaN', '1e2', '-1', 330.56):
            with self.assertRaises(ValueError): cents(text)

    def test_unknown_arithmetic_contract_stays_uncertain(self):
        d, s, ledger = self.fixture()
        s['current_task']['instruction'] = s['current_task']['instruction'].replace('There are no shipping fees or price-changing options.', '')
        self.assert_blocked(self.run_fixture(d, s, ledger))


if __name__ == '__main__': unittest.main()
