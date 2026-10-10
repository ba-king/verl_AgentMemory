"""Only live behavioral change: insert verbatim, already-delivered READ blocks."""
import copy
import hashlib
import json
from .common import read, save, digest
from ..packet_v3_live_data_pilot_v1.execute import Answer as FrozenAnswer
from ....travel_v0.structured_record.record_renderer import evidence_block, render
from examples.memory_agent_grpo.answer_agent.providers.deepseek import DeepSeekAnswerAgentClient


def actual_read(dest, packet_public):
    # Decisions are already persisted by READ's original flush_deliveries.
    # Read no Bank/private/scorer file and do not perform another retrieval.
    outputs, receipts, projected, seen = [], [], [], set()
    for path in sorted((dest / 'decisions').glob('*.json')):
        row = read(path)
        delivered = row.get('answer_memory_evidence_delivered')
        if not delivered:
            continue
        assert row['opportunity_stage'] == 'session_start'
        receipt = row['retrieval_result']
        assert receipt['final_k'] == 5
        assert delivered['hits'] == receipt['hits']
        assert delivered['content'] == evidence_block(receipt['hits'])
        outputs.append({'role': 'user', 'content': delivered['content']})
        receipts.append(copy.deepcopy(receipt))
        for hit in receipt['hits']:
            assert hit['content'] == render(hit['record'])
            version = hashlib.sha256(hit['content'].encode()).hexdigest()
            ref = hit['memory_id'] + '@' + version[:12]
            if ref not in seen:
                seen.add(ref)
                projected.append({'memory_id': hit['memory_id'], 'version_sha256': version,
                                  'source_ref': ref, 'content': hit['content']})
    assert receipts == read(dest / 'packet/retrieval_receipts.json')
    assert projected == packet_public['actual_retrieved_memories']
    return outputs, receipts


def compose(p_messages, raw_output):
    assert not any(m.get('role') == 'user' and str(m.get('content', '')).startswith('[Retrieved Memory]')
                   for m in p_messages)
    visible = copy.deepcopy(p_messages)
    # Original raw-memory runtime delivery position, before within-session history.
    visible[2:2] = copy.deepcopy(raw_output)
    sent = [m for m in visible if m.get('role') == 'user'
            and str(m.get('content', '')).startswith('[Retrieved Memory]')]
    assert sent == raw_output, 'RP_WIRING_FAILURE'
    stripped = visible[:2] + visible[2 + len(raw_output):]
    assert stripped == p_messages, 'NON_INTERFACE_INPUT_CHANGE'
    return visible, sent


class RPAnswer(FrozenAnswer):
    def generate(self, conversation, definitions):
        packet = self.packet.update()  # Unchanged update frequency and actual delta.
        p_messages = DeepSeekAnswerAgentClient._validate_and_serialize_conversation(conversation)
        p_messages.append({'role': 'user', 'content': '[Working Packet]\n' +
                           json.dumps(packet, ensure_ascii=False, separators=(',', ':'))})
        try:
            actual, receipts = actual_read(self.dest, self.packet.public)
            visible, sent = compose(p_messages, actual)
        except Exception as exc:
            self.packet.control.halt('RP_WIRING_FAILURE', error=type(exc).__name__ + ': ' + str(exc))
            raise
        tools = DeepSeekAnswerAgentClient._validate_and_serialize_tools(definitions)
        number = self.client.count + 1
        check = self.dest / 'rp_wiring' / f'answer_{number:03d}.json'
        save(check, {'condition': 'RP', 'actual_read_output': actual,
            'retrieved_memories_sent_to_answer': sent, 'packet_sent_to_answer': packet,
            'actual_read_receipts': receipts,
            'actual_read_output_definition': 'Exact public message blocks from frozen READ flush_deliveries; IDs/records/scores/order are in actual_read_receipts.',
            'same_read_as_packet': True, 'RP_WIRING_VALID': True,
            'p_projection_sha256': digest(p_messages), 'rp_messages_sha256': digest(visible),
            'request_ref': str(self.dest / 'api' / f'answer_{number:03d}' / 'request.json'),
            'packet_event_id': self.packet.rows[-1]['event_id'],
            'extra_retrieval': False, 'post_serialization_checked': False})
        try:
            body, self.last_meta = self.client.call(visible, tools)
        finally:
            record = read(check)
            request = self.dest / 'api' / f'answer_{number:03d}' / 'request.json'
            if request.exists():
                outbound = read(request)['messages']
                actual_sent = [m for m in outbound if m.get('role') == 'user'
                    and str(m.get('content', '')).startswith('[Retrieved Memory]')]
                record['retrieved_memories_sent_to_answer'] = actual_sent
                record['post_serialization_checked'] = True
                record['RP_WIRING_VALID'] = actual_sent == actual and outbound == visible
                save(check, record)
                if not record['RP_WIRING_VALID']:
                    self.packet.control.halt('RP_WIRING_FAILURE', request_ref=str(request))
        save(self.dest / 'answer_exact_conversation.json', visible)
        return DeepSeekAnswerAgentClient._parse_response(json.dumps(body).encode())
