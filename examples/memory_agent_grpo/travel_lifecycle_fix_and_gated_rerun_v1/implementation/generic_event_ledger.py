"""Ordered public delivery with atomic Packet/checkpoint commit, no semantics.

An event ID is immutable. A conflicting duplicate is an error, not an update.
Only a successful schema/apply/replay path calls commit. Persistence uses one
atomic replace for Packet and consumed IDs, so a failed save cannot consume data.
"""
import copy
import json
import os
from pathlib import Path


class EventLedger:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.events = []
        self.consumed = []
        self.packet = None
        self.memory_refs = []
        if self.path and self.path.exists():
            state = json.loads(self.path.read_text())
            self.events, self.consumed = state['events'], state['consumed_event_ids']
            self.packet, self.memory_refs = state['packet'], state['memory_refs']

    def snapshot(self):
        return copy.deepcopy({'events':self.events,'consumed_event_ids':self.consumed,
                              'packet':self.packet,'memory_refs':self.memory_refs})

    def _persist(self, state):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            with temporary.open('w') as stream:
                json.dump(state, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)

    def append(self, event):
        if set(event) != {'event_id','action','observation'} or not event['event_id']:
            raise ValueError('PUBLIC_EVENT_PROJECTION_REQUIRED')
        for old in self.events:
            if old['event_id'] == event['event_id']:
                if {k:old[k] for k in event} != event:
                    raise ValueError('CONFLICTING_DUPLICATE_EVENT_ID')
                return False
        row = copy.deepcopy(event) | {'event_sequence':len(self.events)}
        state = self.snapshot()
        state['events'].append(row)
        self._persist(state)
        self.events = state['events']
        return True

    def unseen(self):
        return copy.deepcopy([e for e in self.events if e['event_id'] not in self.consumed])

    def commit(self, batch, packet, memory_refs=()):
        unseen = self.unseen()
        if batch != unseen[:len(batch)]:
            raise ValueError('CHECKPOINT_MUST_CONSUME_ORDERED_UNSEEN_PREFIX')
        state = self.snapshot()
        state['consumed_event_ids'].extend(e['event_id'] for e in batch)
        state['packet'] = copy.deepcopy(packet)
        state['memory_refs'] = list(memory_refs)
        self._persist(state)
        self.consumed, self.packet, self.memory_refs = (
            state['consumed_event_ids'], state['packet'], state['memory_refs'])


def batch_input(previous_input, batch):
    """Keep frozen seven top-level input fields; batch inside latest_observation.

No task/domain interpretation. Single-event public content and source ID remain
byte-for-byte representable. No consumed history is re-sent.
"""
    result = copy.deepcopy(previous_input)
    observation = result['latest_observation']
    observation.pop('event_id', None)
    observation.pop('observation', None)
    observation['new_events'] = copy.deepcopy(batch)
    result['latest_action'] = copy.deepcopy(batch[-1]['action']) if batch else None
    return result
