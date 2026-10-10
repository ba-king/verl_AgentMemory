"""Portable CPU tests: no recorded runs, external dataset, model, or HTTP."""
import json
import tempfile
import unittest
from pathlib import Path

from .dataset import labels_from_verified_spans
from ..memora_teacher.weekly_pro_yield_v2.interface import apply_response_v2
from ..memora_sft_dev.teacher_rollout_v1.interface import restore, snapshot


def response(name, arguments, call_id):
    return {'choices': [{'finish_reason': 'tool_calls', 'message': {
        'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': call_id, 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps(arguments)}}]}}]}


class HandoffTests(unittest.TestCase):
    def test_partial_delete_retains_record_and_delete_removes_it(self):
        bank = restore({'records': [], 'next_id': 1})
        seen = set()
        created = apply_response_v2(response('ADD', {'record': {
            'subject': 'user', 'scope': 'document: demo', 'fields': [
                {'path': 'data.title', 'value': 'Demo'},
                {'path': 'data.notes', 'value': 'Remove only this field'}]}}, 'a'), bank, False, seen)
        memory_id = created['receipt']['record']['memory_id']
        changed = apply_response_v2(response('UPDATE', {
            'memory_id': memory_id, 'set_fields': [], 'remove_fields': ['data.notes']}, 'u'), bank, False, seen)
        self.assertEqual(changed['receipt']['code'], 'UPDATED')
        self.assertEqual(snapshot(bank)['records'][0]['record']['fields'],
                         [{'path': 'data.title', 'value': 'Demo'}])
        removed = apply_response_v2(response('DELETE', {'memory_id': memory_id}, 'd'), bank, False, seen)
        self.assertEqual(removed['receipt']['code'], 'DELETED')
        self.assertEqual(snapshot(bank)['records'], [])

    def test_yield_wire_does_not_mutate_bank(self):
        bank = restore({'records': [], 'next_id': 1})
        before = snapshot(bank)
        result = apply_response_v2(response('YIELD', {'status': 'done'}, 'y'), bank, False, set())
        self.assertEqual(result['action'], {'action': 'YIELD', 'arguments': {}})
        self.assertEqual(result['receipt']['code'], 'YIELDED')
        self.assertEqual(snapshot(bank), before)
        with self.assertRaises(ValueError):
            apply_response_v2(response('YIELD', {'status': 'DONE'}, 'bad'), bank, False, set())
        self.assertEqual(snapshot(bank), before)

    def test_input_and_tool_tokens_are_not_supervised(self):
        mask = labels_from_verified_spans([7]*12, [(5, 8, 1.0), (10, 12, 0.2)])
        self.assertEqual(mask['labels'], [-100]*5 + [7]*3 + [-100]*2 + [7]*2)
        self.assertEqual(mask['loss_weights'], [0.0]*5 + [1.0]*3 + [0.0]*2 + [0.2]*2)


if __name__ == '__main__':
    unittest.main()
