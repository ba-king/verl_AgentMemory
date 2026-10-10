"""Verify parser, atomic Bank behavior, visibility and recorded action tapes."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from .bank import StructuredBank
from . import policy
from ..structured_record import policy as old_policy
from ..structured_record.bank import StructuredBank as OldBank
from ..single_action.protocol import ProtocolError
from ..single_action import session


def response(action):
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(action)}}]}


def record():
    return {'subject': 'user', 'scope': 'project', 'fields': [
        {'path': 'budget', 'value': '1200000'}, {'path': 'timeline', 'value': '30 months'},
        {'path': 'stakeholders', 'value': '["A","B"]'}]}


def update(**changes):
    args = {'memory_id': 'M001', 'set_fields': [{'path': 'budget', 'value': '1000000'}], 'remove_fields': []}
    args.update(changes)
    return {'action': 'UPDATE', 'arguments': args}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.bank = StructuredBank(None)
        self.bank.execute({'action': 'ADD', 'arguments': {'record': record()}})

    def test_valid_edit_preserves_other_fields_and_logs_actual_delta(self):
        target = update()
        schema = policy.action_schema(['ADD', 'UPDATE', 'DELETE', 'YIELD'], ['M001'])
        parsed, _ = policy.parse_decision(response(target), schema)
        receipt, _ = self.bank.execute(parsed)
        self.assertEqual(receipt['status'], 'SUCCEEDED')
        after = self.bank.snapshot()['records'][0]['record']
        self.assertEqual(after['fields'][1:], record()['fields'][1:])
        self.assertEqual(self.bank.actions[-1]['action'], target)
        self.assertEqual(self.bank.receipts[-1]['record']['record'], after)

    def test_invalid_edit_is_atomic_including_logs_and_receipts(self):
        for target in [update(remove_fields=['missing']), update(remove_fields=['budget']),
                       update(set_fields=[], remove_fields=['budget', 'timeline', 'stakeholders']),
                       update(memory_id='M999')]:
            before = copy.deepcopy((self.bank.snapshot(), self.bank.actions, self.bank.receipts))
            result, _ = self.bank.execute(target)
            self.assertEqual(result['status'], 'REJECTED')
            self.assertEqual((self.bank.snapshot(), self.bank.actions, self.bank.receipts), before)

    def test_field_delete_and_entire_record_delete_differ(self):
        result, _ = self.bank.execute(update(set_fields=[], remove_fields=['timeline']))
        self.assertEqual(result['status'], 'SUCCEEDED')
        self.assertEqual(len(self.bank.snapshot()['records']), 1)
        self.assertNotIn('timeline', [f['path'] for f in self.bank.records['M001']['fields']])
        self.bank.execute({'action': 'DELETE', 'arguments': {'memory_id': 'M001'}})
        self.assertFalse(self.bank.snapshot()['records'])

    def test_session_reset_still_requires_record_visibility(self):
        before = self.bank.snapshot()
        self.bank.begin_session()
        result, _ = self.bank.execute(update())
        self.assertEqual(result['code'], 'ID_NOT_VISIBLE_IN_CURRENT_SESSION')
        self.assertEqual(self.bank.snapshot(), before)

    def test_legacy_update_rejected_only_by_v2(self):
        target = {'action': 'UPDATE', 'arguments': {'memory_id': 'M001', 'record': record()}}
        old, new = (p.action_schema(['UPDATE'], ['M001']) for p in (old_policy, policy))
        self.assertEqual(old_policy.parse_decision(response(target), old)[0], target)
        with self.assertRaises(ProtocolError):
            policy.parse_decision(response(target), new)
        self.assertEqual(policy.READ_PROMPT, old_policy.READ_PROMPT)
        self.assertNotIn('ADD and UPDATE now take', policy.CONSOLIDATE_PROMPT)
        for name in ['ADD', 'DELETE', 'RETRIEVE', 'YIELD']:
            self.assertEqual(policy.action_schema([name], ['M001']), old_policy.action_schema([name], ['M001']))

    def test_single_action_session_consumes_update_then_yield_offline(self):
        class TapeClient:
            config = {'model': 'offline_test_no_http'}
            count = 0
            def call(self, messages):
                self.count += 1
                state = json.loads(messages[1]['content'])
                assert state['decision_mode'] == 'CONSOLIDATE'
                action = update() if self.count == 1 else {'action': 'YIELD', 'arguments': {}}
                return response(action), {'token_usage': {}, 'latency': 0}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(session, 'JsonTeacherClient', return_value=TapeClient()):
                obj = policy.StructuredSession('offline', 'offline', {'session_id': 's1', 'instruction': 'public'},
                                               Path(tmp), self.bank, None, [])
            # Simulate an already exposed record; no retrieval/model is tested here.
            self.bank.visible_ids.add('M001')
            self.bank.visible = list(self.bank.snapshot()['records'])
            obj.history = [{'event_id': 'public_submit', 'action': {'final_itinerary': "=== User's Plan ===\nDay 1:\nCurrent City: Place\n"},
                            'observation': {'step': 1, 'final': 'public'}}]
            obj.opportunity('after_final_itinerary')
            self.assertEqual(len(obj.decisions), 2)
            self.assertEqual(self.bank.records['M001']['fields'][0]['value'], '1000000')
            tape = json.loads((Path(tmp) / 'decisions/s1_d001.json').read_text())
            self.assertEqual(tape['parsed_memory_action'], update())

    def test_v1_bank_is_rejected_before_a_request(self):
        # Guard against mixing a new parser with an old replacement-only Bank.
        obj = object.__new__(policy.StructuredSession)
        obj.bank = OldBank(None)
        with self.assertRaisesRegex(TypeError, 'requires_matching_bank'):
            obj.opportunity('after_final_itinerary')


if __name__ == '__main__':
    unittest.main()
