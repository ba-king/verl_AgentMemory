"""Portable method checks; synthetic fixtures, no model or experiment archives."""
import copy
import unittest

from ..teacher_pilot.natural_rollout_v1.shopping_batch_v1.packet_interface_v2_validation.test_offline import delta, response
from ..teacher_pilot.natural_rollout_v1.shopping_batch_v1.packet_interface_v3_minimal_validation import interface_v3 as packet


class MethodHandoffTests(unittest.TestCase):
    def test_v3_delta_preserves_subject_and_scope(self):
        initial = packet.apply(packet.empty_packet(), delta(
            add_needs=[{'id': 'N1', 'text': 'Previous subject state', 'scope_refs': ['person:avery']}],
            add_pending=['N1'], set_unresolved=['State missing']), ['task'], init=True)
        evidence = {'need_id': 'N1', 'fact': 'Avery location Paris',
                    'source_refs': ['M001@v1'], 'subject_ref': 'person:avery'}
        change = delta(add_evidence=[evidence], resolve=['N1'], set_ready_to_commit='true')
        parsed = packet.parse(response({'packet_delta': change}))
        updated = packet.apply(initial, parsed, ['task', 'M001@v1'])
        self.assertEqual(updated['needs'][0]['scope_refs'], ['person:avery'])
        self.assertEqual(updated['evidence'], [evidence])
        self.assertEqual(updated['progress']['pending'], [])
        self.assertTrue(updated['ready_to_commit'])
        self.assertEqual(initial['evidence'], [])

    def test_v3_rejects_missing_provenance_without_mutation(self):
        initial = packet.empty_packet()
        before = copy.deepcopy(initial)
        with self.assertRaises(ValueError):
            packet.apply(initial, delta(
                add_needs=[{'id': 'N1', 'text': 'State', 'scope_refs': ['person:avery']}],
                add_evidence=[{'need_id': 'N1', 'fact': 'Not observed',
                               'source_refs': ['future_event'], 'subject_ref': 'person:avery'}]), ['task'])
        self.assertEqual(initial, before)


if __name__ == '__main__':
    unittest.main()
