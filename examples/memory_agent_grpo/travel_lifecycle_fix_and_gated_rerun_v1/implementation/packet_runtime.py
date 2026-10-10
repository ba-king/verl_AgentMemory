"""Frozen Travel semantics and Packet client, generic unseen-event delivery."""
import copy
from final_unified_travel_rp_packet_v3_demo_v1.packet import (
    Session as PreviousSession, delivered, V3, replay_metadata)
from .generic_event_ledger import EventLedger, batch_input
from ..common import save, digest


class Session(PreviousSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ledger = EventLedger(self.dest/'packet/event_ledger.json')
        self.original_event_ids = {}
        self.ledger.append({'event_id':'e000','action':None,
                            'observation':self.current.get('initial_observation', {})})

    def event(self, event):
        identity = event['event_id']
        public_digest = digest({'action':event['action'],'observation':event['observation']})
        if identity in self.original_event_ids:
            if self.original_event_ids[identity] != public_digest:
                raise ValueError('CONFLICTING_DUPLICATE_EVENT_ID')
            return
        # Existing Travel adapter alone interprets candidate ownership.
        super().event(event)
        self.ledger.append(self.events[-1])
        self.original_event_ids[identity] = public_digest

    def update(self):
        _, receipts, memories = delivered(self.memory)
        cutoff = (len(self.events),tuple(m['source_ref'] for m in memories))
        if self.last_cutoff == cutoff and not self.ledger.unseen():
            return self.packet
        if len(self.transitions) >= 30:
            raise RuntimeError('PREDECLARED_PACKET_SESSION_CAP')
        new = [m for m in memories if m['source_ref'] not in self.seen_refs]
        for m in memories:
            subject = 'person:'+m['record']['subject']
            self.subjects.setdefault(subject, {'name':m['record']['subject'],'role':'actually retrieved record subject'})
            self.sources[m['source_ref']] = {'text':m['content'],'subjects':[subject],'record':m['record']}
        batch = self.ledger.unseen()
        inp = batch_input({'mode':'PACKET','operation':'UPDATE_PACKET',
            'current_task':self.current['instruction'],'previous_packet':copy.deepcopy(self.packet),
            'latest_action':None,'latest_observation':{
                'public_subject_registry':self.subjects,
                'task_contract':{'authority':self.contract['authority'],'need_kinds':self.contract['need_kinds'],
                    'subject_namespace':'scope_refs and evidence.subject_ref use exact supplied public subject IDs; source_refs use task, event ID, or versioned Memory ref.',
                    'fact_convention':'Prefer one exact public field=value claim per evidence; preserve traveler and day binding. Derived claims require cited premises. A subject ID is not a source ID.'}},
            'retrieved_memories':copy.deepcopy(new)}, batch)
        folder = self.dest/'packet'/f'update_{len(self.transitions)+1:03d}'
        row = {'episode_id':self.memory.eid,'session_id':self.current['session_id'],
            'event_id':folder.name,'previous_packet':copy.deepcopy(self.packet),
            'parsed_packet_delta':None,'applied_packet':None,'schema_valid':False,
            'apply_valid':False,'replay_equal':False,'input_ref':str(folder/'model_input.json'),
            'request_ref':str(folder/'api/request.json'),'response_ref':str(folder/'api/response.json'),
            'audit_ref':str(folder/'audit.json'),'source_registry_ref':str(folder/'source_registry.json'),
            'manual_repair':False,'semantic_retries':0,
            'unseen_event_ids':[e['event_id'] for e in batch],
            'delivered_event_ids':[e['event_id'] for e in inp['latest_observation']['new_events']]}
        save(folder/'model_input.json',inp)
        save(folder/'source_registry.json',self.sources)
        save(folder/'subject_registry.json',self.subjects)
        save(folder/'actual_read_receipts.json',receipts)
        save(folder/'ledger_before.json',self.ledger.snapshot())
        try:
            response = self.client.call(inp,folder/'api')
            row['usage'] = response.get('usage',{})
            delta = V3.parse(response)
            row.update(schema_valid=True,parsed_packet_delta=delta)
            applied = V3.apply(self.packet,delta,self.sources,init=False)
            row.update(apply_valid=True,applied_packet=applied)
            replay = replay_metadata(self.packet,delta,applied,self.sources)
            row.update(replay_equal=replay['replay_equal'],replay_metadata=replay)
            if not row['replay_equal']:
                raise RuntimeError('PACKET_REPLAY_MISMATCH')
            self.ledger.commit(batch,applied,[m['source_ref'] for m in memories])
            self.packet = applied
            self.seen_refs.update(m['source_ref'] for m in memories)
            self.last_cutoff = cutoff  # only after successful atomic checkpoint
        except Exception as exc:
            row['error'] = (type(exc).__name__+': '+str(exc)).replace(self.client.key,'[REDACTED]')
            raise
        finally:
            row['consumed_event_ids_after'] = list(self.ledger.consumed)
            save(folder/'transition.json',row)
            save(folder/'ledger_after.json',self.ledger.snapshot())
            self.transitions.append(row)
        return self.packet
