"""Synthetic cases calibrate mechanics only; never counted as live evidence."""
import copy
import json
import unittest
from .interface import apply, parse, empty_packet, ProtocolError, answer_request, messages


def delta(**updates):
    value = {'add_needs': [], 'add_evidence': [], 'remove_evidence': [], 'resolve': [],
        'add_pending': [], 'remove_pending': [], 'add_failed_attempts': [],
        'remove_failed_attempts': [], 'set_unresolved': [], 'set_ready_to_commit': 'false'}
    return {**value, **updates}


def response(value):
    return {'choices': [{'finish_reason': 'tool_calls', 'message': {'content': '',
        'tool_calls': [{'type': 'function', 'function': {'name': 'emit_packet_delta',
            'arguments': json.dumps(value)}}]}}]}


class InterfaceTests(unittest.TestCase):
    def test_empty_delta_no_fact_creation(self):
        self.assertEqual(apply(empty_packet(), delta(), ['task'], init=True), empty_packet())

    def test_init_and_causal_update(self):
        p = apply(empty_packet(), delta(add_needs=[{'id': 'N1', 'text': 'Prior person state'}],
            add_pending=['N1'], set_unresolved=['Missing state']), ['task'], init=True)
        d = delta(add_evidence=[{'need_id': 'N1', 'fact': 'Morgan location Paris',
            'source_refs': ['M001@abc']}], resolve=['N1'], set_ready_to_commit='true')
        q = apply(p, d, ['task', 'M001@abc'])
        self.assertEqual(q['evidence'], d['add_evidence'])
        self.assertTrue(q['ready_to_commit'])
        self.assertEqual(p['evidence'], [])
        self.assertEqual(q['progress']['pending'], [])

    def test_invalid_update_atomic(self):
        p = empty_packet(); old = copy.deepcopy(p)
        with self.assertRaises(ValueError):
            apply(p, delta(remove_evidence=[0]), ['task'])
        self.assertEqual(p, old)

    def test_future_ref_rejected(self):
        with self.assertRaises(ValueError):
            apply(empty_packet(), delta(add_needs=[{'id': 'N1', 'text': 'State'}],
                add_evidence=[{'need_id': 'N1', 'fact': 'x', 'source_refs': ['e999']}]), ['task'])

    def test_duplicate_subject_need_rejected(self):
        with self.assertRaises(ValueError):
            apply(empty_packet(), delta(add_needs=[{'id': 'N1', 'text': 'Morgan'},
                {'id': 'N1', 'text': 'Avery'}]), ['task'])

    def test_exact_wrapper(self):
        d = delta()
        self.assertEqual(parse(response({'packet_delta': d})), d)
        for value in [d, {'packet': d}, {'packet': {'packet_delta': d}},
                      {'packet_delta': d, 'private_score': 1}]:
            with self.assertRaises(ProtocolError):
                parse(response(value))

    def test_broken_json_never_repaired(self):
        r = response({'packet_delta': delta()})
        r['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"packet_delta": {'
        with self.assertRaisesRegex(ProtocolError, 'MALFORMED_JSON'):
            parse(r)

    def test_duplicate_keys_rejected(self):
        r = response({})
        r['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"packet_delta":{},"packet_delta":{}}'
        with self.assertRaises(ProtocolError):
            parse(r)

    def test_private_field_not_in_schema(self):
        with self.assertRaisesRegex(ProtocolError, 'SCHEMA_INVALID'):
            parse(response({'packet_delta': delta(scorer=1)}))

    def test_init_requires_explicit_readiness(self):
        with self.assertRaises(ValueError):
            apply(empty_packet(), delta(set_ready_to_commit='keep'), ['task'], init=True)

    def test_frozen_answer_projection(self):
        req = {'model': 'fixed', 'tools': ['frozen'], 'messages': [
            {'role': 'system', 'content': 'frozen'}, {'role': 'user', 'content': 'task'},
            {'role': 'user', 'content': '[Retrieved Memory]\nraw'},
            {'role': 'assistant', 'content': 'original public history'}]}
        self.assertEqual(answer_request(req, empty_packet(), 'R'), req)
        p = answer_request(req, empty_packet(), 'P')
        self.assertFalse(any(m.get('content', '').startswith('[Retrieved Memory]') for m in p['messages']))
        self.assertEqual(p['messages'][:2], req['messages'][:2])
        self.assertEqual(p['messages'][2], req['messages'][3])
        self.assertEqual(p['tools'], req['tools'])
        self.assertEqual(answer_request(req, empty_packet(), 'RP')['messages'][:-1], req['messages'])

    def test_no_history_input_slot(self):
        with self.assertRaises(ValueError):
            messages({'mode': 'PACKET', 'public_history': []})

    def test_indices_refer_to_previous_packet(self):
        p = empty_packet(); p['needs'] = [{'id': 'N1', 'text': 'subject'}]
        p['evidence'] = [{'need_id': 'N1', 'fact': str(i), 'source_refs': ['task']} for i in range(3)]
        q = apply(p, delta(remove_evidence=[0, 2]), ['task'])
        self.assertEqual([e['fact'] for e in q['evidence']], ['1'])


if __name__ == '__main__':
    unittest.main()
