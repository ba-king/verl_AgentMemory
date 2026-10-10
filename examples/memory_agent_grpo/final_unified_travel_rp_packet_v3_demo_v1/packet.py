"""Generic frozen V3 delta execution, Travel public contract and RP delivery."""
import copy
import hashlib
import json
import re
from .common import save, digest, append
from .adapter import build, V3
from teacher_pilot.natural_rollout_v1.shopping_batch_v1.packet_interface_v2_validation import interface as generic
from teacher_pilot.natural_rollout_v1.shopping_batch_v1.packet_interface_v2_validation.transport import PacketClient
from teacher_pilot.natural_rollout_v1.shopping_batch_v1.trusted_packet_scaffold_convergence_v1.prompt import SCAFFOLD_RULE
from teacher_pilot.natural_rollout_v1.shopping_batch_v1.packet_v3_state_maintenance_policy_convergence_v1.collector import replay_metadata
from teacher_pilot.travel_v0.structured_record.record_renderer import evidence_block, render

PROMPT = generic.PROMPT + SCAFFOLD_RULE


class Client(PacketClient):
    def request(self, inp):
        messages = generic.messages(inp)
        messages[0]['content'] = PROMPT
        return {**{k: self.config[k] for k in ('model', 'temperature', 'max_tokens', 'thinking', 'stream')},
                'messages': messages, 'tools': V3.TOOLS,
                'tool_choice': {'type': 'function', 'function': {'name': 'emit_packet_delta'}},
                'parallel_tool_calls': False}


def delivered(memory):
    blocks, receipts, memories, seen = [], [], [], set()
    for row in memory.decisions:
        out = row.get('answer_memory_evidence_delivered')
        if not out:
            continue
        receipt = row['retrieval_result']
        assert receipt['final_k'] == 3
        assert receipt['hits'] == out['hits'] and out['content'] == evidence_block(out['hits'])
        blocks.append({'role': 'user', 'content': out['content']})
        receipts.append(copy.deepcopy(receipt))
        for hit in out['hits']:
            assert hit['content'] == render(hit['record'])
            version = hashlib.sha256(hit['content'].encode()).hexdigest()
            ref = hit['memory_id'] + '@' + version[:12]
            if ref not in seen:
                memories.append({'memory_id': hit['memory_id'], 'source_ref': ref, 'version_sha256': version,
                                 'content': hit['content'], 'record': copy.deepcopy(hit['record'])})
                seen.add(ref)
    return blocks, receipts, memories


class Session:
    def __init__(self, current, memory, budget, dest):
        self.current, self.memory, self.dest = copy.deepcopy(current), memory, dest
        self.scaffold = build(current)
        self.contract = self.scaffold['contract']
        self.packet = copy.deepcopy(self.scaffold['packet'])
        self.events, self.transitions, self.seen_refs = [], [], set()
        self.last_cutoff = None
        self.client = Client(budget)
        budget.wrap(self.client, 'PACKET')
        self.sources = {'task': {'text': current['instruction'], 'subjects': list(self.contract['subject_registry'])},
                        'e000': {'text': json.dumps(current.get('initial_observation', {}), ensure_ascii=False),
                                 'subjects': ['T0'] + ([f"person:{self.contract['base']}"] if 'initial_observation' in current else [])}}
        if 'initial_observation' in current:
            self.sources['e000']['bootstrap'] = copy.deepcopy(current['initial_observation']['base_person'])
        self.subjects = copy.deepcopy(self.contract['subject_registry'])
        save(dest / 'packet/scaffold.json', self.scaffold)

    def event(self, event):
        projected = {'event_id': f'e{len(self.events)+1:03d}',
                     'action': copy.deepcopy(event['action']), 'observation': copy.deepcopy(event['observation'])}
        self.events.append(projected)
        owners = ['T0']
        if projected['action'].get('tool') == 'AccommodationSearch':
            city = projected['action']['arguments']['city']
            owners.append(f'lodgings:{city}')
            # Public table identifies candidate subjects only. It creates no
            # Packet fact and does not select or rank accommodations.
            for line in str(projected['observation']).splitlines()[1:]:
                match = re.match(r'^\s*(.*?)\s+\d+(?:\.\d+)?\s+(?:Private room|Shared room|Entire home/apt)\s+', line)
                if match:
                    ref = f'hotel:{city}:{match[1]}'
                    self.subjects[ref] = {'name': match[1], 'city': city, 'role': 'public candidate'}
                    owners.append(ref)
        self.sources[projected['event_id']] = {'text': str(projected['observation']), 'subjects': owners}
        append(self.dest / 'packet/public_events.jsonl', projected)

    def update(self):
        blocks, receipts, memories = delivered(self.memory)
        cutoff = (len(self.events), tuple(m['source_ref'] for m in memories))
        if self.last_cutoff == cutoff:
            return self.packet
        if len(self.transitions) >= 30:
            raise RuntimeError('PREDECLARED_PACKET_SESSION_CAP')
        new = [m for m in memories if m['source_ref'] not in self.seen_refs]
        for m in memories:
            subject = 'person:' + m['record']['subject']
            self.subjects.setdefault(subject, {'name': m['record']['subject'], 'role': 'actually retrieved record subject'})
            self.sources[m['source_ref']] = {'text': m['content'], 'subjects': [subject], 'record': m['record']}
        event = self.events[-1] if self.events else {'event_id': 'e000', 'action': None,
                                                     'observation': self.current.get('initial_observation', {})}
        inp = {'mode': 'PACKET', 'operation': 'UPDATE_PACKET', 'current_task': self.current['instruction'],
            'previous_packet': copy.deepcopy(self.packet), 'latest_action': event['action'],
            'latest_observation': {'event_id': event['event_id'], 'observation': event['observation'],
                'public_subject_registry': self.subjects,
                'task_contract': {'authority': self.contract['authority'], 'need_kinds': self.contract['need_kinds'],
                    'subject_namespace': 'scope_refs and evidence.subject_ref use exact supplied public subject IDs; source_refs use task, event ID, or versioned Memory ref.',
                    'fact_convention': 'Prefer one exact public field=value claim per evidence; preserve traveler and day binding. Derived claims require cited premises. A subject ID is not a source ID.'}},
            'retrieved_memories': copy.deepcopy(new)}
        folder = self.dest / 'packet' / f'update_{len(self.transitions)+1:03d}'
        row = {'episode_id': self.memory.eid, 'session_id': self.current['session_id'],
               'event_id': f'update_{len(self.transitions)+1:03d}', 'previous_packet': copy.deepcopy(self.packet),
               'parsed_packet_delta': None, 'applied_packet': None, 'schema_valid': False,
               'apply_valid': False, 'replay_equal': False, 'input_ref': str(folder/'model_input.json'),
               'request_ref': str(folder/'api/request.json'), 'response_ref': str(folder/'api/response.json'),
               'audit_ref': str(folder/'audit.json'), 'source_registry_ref': str(folder/'source_registry.json'),
               'manual_repair': False, 'semantic_retries': 0}
        save(folder/'model_input.json', inp)
        save(folder/'source_registry.json', self.sources)
        save(folder/'subject_registry.json', self.subjects)
        save(folder/'actual_read_receipts.json', receipts)
        try:
            response = self.client.call(inp, folder/'api')
            row['usage'] = response.get('usage', {})
            delta = V3.parse(response)
            row.update(schema_valid=True, parsed_packet_delta=delta)
            applied = V3.apply(self.packet, delta, self.sources, init=False)
            row.update(apply_valid=True, applied_packet=applied)
            replay = replay_metadata(self.packet, delta, applied, self.sources)
            row.update(replay_equal=replay['replay_equal'], replay_metadata=replay)
            if not row['replay_equal']:
                raise RuntimeError('PACKET_REPLAY_MISMATCH')
            self.packet = applied
            self.seen_refs.update(m['source_ref'] for m in memories)
        except Exception as exc:
            row['error'] = (type(exc).__name__ + ': ' + str(exc)).replace(self.client.key, '[REDACTED]')
            raise
        finally:
            save(folder/'transition.json', row)
            self.transitions.append(row)
            self.last_cutoff = cutoff
        return self.packet

    def compose(self, conversation, number):
        packet = self.update()
        raw, receipts, memories = delivered(self.memory)
        actual = [m for m in conversation if m.get('role') == 'user' and m.get('content', '').startswith('[Retrieved Memory]')]
        assert actual == raw, 'RP_WIRING_FAILURE'
        assert not any(m.get('content', '').startswith('[Working Packet]') for m in conversation)
        request_messages = copy.deepcopy(conversation) + [{'role': 'user', 'content': '[Working Packet]\n' +
            json.dumps(packet, ensure_ascii=False, separators=(',', ':'))}]
        transition = self.transitions[-1]
        record = {'actual_read_output': raw, 'retrieved_memories_sent_to_answer': actual,
                  'actual_read_receipts': receipts, 'packet_before': transition['previous_packet'],
                  'packet_delta': transition['parsed_packet_delta'], 'packet_after_apply': transition['applied_packet'],
                  'packet_sent_to_answer': copy.deepcopy(packet), 'RP_WIRING_VALID': True,
                  'same_read_as_packet': all(m['source_ref'] in self.seen_refs for m in memories),
                  'answer_request_ref': str(self.dest/'api'/f'answer_{number:03d}'/'request.json'),
                  'answer_response_ref': str(self.dest/'api'/f'answer_{number:03d}'/'response.json'),
                  'packet_transition_ref': str(self.dest/'packet'/transition['event_id']/'transition.json')}
        save(self.dest/'rp_wiring'/f'answer_{number:03d}.json', record)
        return request_messages

    def close(self):
        self.client.transport.close()
