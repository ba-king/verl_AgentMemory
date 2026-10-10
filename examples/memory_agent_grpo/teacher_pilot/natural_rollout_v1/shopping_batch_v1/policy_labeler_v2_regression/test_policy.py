"""Offline check of the single permitted prompt delta and the frozen action loop."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import policy
from ..pilot4_v1.memory import current_state
from ..pilot4_v1.test_harness import FakeClient
from ....travel_v0.single_action import session
from ....travel_v0.structured_record.bank import StructuredBank
from ....travel_v0.structured_record import policy as frozen


class PolicyTests(unittest.TestCase):
    def test_read_exactly_frozen_and_only_cons_addition(self):
        bank = StructuredBank(None); current = {'session_id': 'p1', 'instruction': 'public task'}
        for stage in ('session_start', 'after_purchase', 'session_ended_without_purchase'):
            state = current_state(bank, current, [], 30, {'stage': stage, 'index': 1})
            saved = copy.deepcopy(state); old = frozen.messages_for(state); new = policy.messages_for(state)
            self.assertEqual(state, saved); self.assertEqual(new[1], old[1])
            if stage == 'session_start': self.assertEqual(new, old)
            else:
                self.assertEqual(new[0]['content'].replace(policy.RETENTION_GUIDANCE, '', 1), old[0]['content'])
                self.assertEqual(new[0]['content'].count(policy.RETENTION_GUIDANCE), 1)

    def test_same_single_action_loop_and_no_automatic_write(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(session, 'JsonTeacherClient', FakeClient):
            memory = policy.ShoppingMemorySessionV2('ignored', 'fixture', {'session_id': 'p1', 'instruction': 'public'},
                Path(directory), StructuredBank(None), None, [])
            memory.opportunity('session_start')
            memory.public_event({'event_id': 'fixture:p1:a1', 'action': {'tool': 'click', 'arguments': {'item': 'Buy Now'}}, 'observation': 'ACTUAL_PUBLIC_RECEIPT'})
            memory.opportunity('after_purchase')
            self.assertEqual([r['parsed_memory_action']['action'] for r in memory.decisions], ['YIELD', 'YIELD'])
            self.assertEqual(len(memory.bank.snapshot()['records']), 0)


if __name__ == '__main__': unittest.main()
