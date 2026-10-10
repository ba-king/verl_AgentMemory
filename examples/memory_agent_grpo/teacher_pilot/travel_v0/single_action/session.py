"""Override only Memory serialization/masking; inherit original scheduling hooks."""
import copy
from contextlib import contextmanager
from .. import provider, rollout
from ..artifacts import save, read, text
from ..memory_runtime import delta
from .protocol import current_state, messages_for, json_request, parse_decision, ProtocolError


@contextmanager
def bindings(module, **values):
    """Scoped injection into the existing synchronous pilot, restored on exit."""
    previous = {k: getattr(module, k) for k in values}
    try:
        for k, v in values.items():
            setattr(module, k, v)
        yield
    finally:
        for k, v in previous.items():
            setattr(module, k, v)


class JsonTeacherClient(provider.FixedClient):
    def call(self, messages, tools=()):
        # Reuse exact HTTP/model/budget/error recorder. Answer calls never enter
        # this scope and continue using the original native-tool request builder.
        with bindings(provider, request_for=json_request):
            return super().call(messages, ())


class SingleActionSession(rollout.SplitSession):
    def __init__(self, *args, **kwargs):
        with bindings(rollout, FixedClient=JsonTeacherClient):
            super().__init__(*args, **kwargs)

    def opportunity(self, stage, defer_delivery=False):
        self.opportunities += 1
        pending_deliveries = []
        while len(self.decisions) < rollout.MEMORY_DECISIONS_PER_SESSION:
            before = self.bank.snapshot()
            state = current_state(self.bank, self.current, self.history,
                rollout.MEMORY_DECISIONS_PER_SESSION - len(self.decisions),
                {'index': self.opportunities, 'stage': stage})
            messages = messages_for(state)
            did = f"{self.current['session_id']}_d{len(self.decisions) + 1:03d}"
            row = {'run_id': self.run_id, 'episode_id': self.eid, 'session_id': self.current['session_id'],
                'decision_id': did, 'public_task': self.current, 'answer_public_history': copy.deepcopy(self.history),
                'memory_teacher_messages': messages, 'available_memory_tools': [],
                'available_actions': state['available_actions'], 'memory_action_schema': state['decision_schema'],
                'visible_memory_ids': state['visible_memory_ids'],
                'visible_memories': copy.deepcopy(self.bank.visible), 'bank_metadata': state['bank_metadata'],
                'student_input': state, 'raw_teacher_response': None, 'parsed_memory_action': None,
                'memory_action_result': None, 'retrieval_result': None, 'format_normalization': None,
                'memory_bank_before_ref': self.snapshot_ref(before), 'memory_bank_after_ref': None,
                'memory_delta': None, 'answer_memory_evidence_delivered': None, 'next_answer_action': None,
                'environment_observation': None, 'native_result': None, 'strict_result': None,
                'termination_reason': None, 'model_config': self.client.config, 'token_usage': {}, 'latency': None,
                'source_event_refs': [f'public:{self.current["session_id"]}',
                    *[h['event_id'] for h in self.history], *[r['decision_id'] for r in self.decisions]],
                'opportunity_stage': stage, 'opportunity_index': self.opportunities}
            self.decisions.append(row)
            self.pending.append(row)
            self.persist(row)
            try:
                body, meta = self.client.call(messages)
                row.update(meta, raw_teacher_response=body)
                action, normalization = parse_decision(body, state['decision_schema'])
                row.update(parsed_memory_action=action, format_normalization=normalization)
                result, hits = self.bank.execute(action)
                row['memory_action_result'] = result
                if hits is not None:
                    row['retrieval_result'] = {'query': action['arguments']['query'], 'hits': hits,
                                              'final_k': self.bank.retriever.backend.config['top_k']}
                    save(self.directory / 'private/retrieval' / (did + '.json'), self.bank.retriever.last_rankings)
                    pending_deliveries.append((row, hits))
                after = self.bank.snapshot()
                row['memory_bank_after_ref'] = self.snapshot_ref(after)
                row['memory_delta'] = delta(before, after)
                self.persist(row)
                if action['action'] == 'YIELD':
                    self.buffered_deliveries.extend((source, hits, did) for source, hits in pending_deliveries)
                    if not defer_delivery:
                        self.flush_deliveries()
                    return
            except Exception as error:
                row['termination_reason'] = 'memory_decision_error'
                row['error'] = type(error).__name__ + ': ' + str(error)
                row['protocol_error_code'] = error.code if isinstance(error, ProtocolError) else None
                # Even rejected parsing has a real before/after snapshot. No
                # executor call occurs before all dynamic schema checks succeed.
                after = self.bank.snapshot()
                row['memory_bank_after_ref'] = self.snapshot_ref(after)
                row['memory_delta'] = delta(before, after)
                if isinstance(error, ProtocolError):
                    row['memory_action_result'] = {'status': 'REJECTED', 'code': error.code, 'executor_entered': False}
                if row['raw_teacher_response'] is None:
                    response = self.directory / 'api' / f'memory_teacher_{self.client.count:03d}' / 'response.json'
                    if response.exists():
                        row['raw_teacher_response'] = read(response)
                        row['response_ref'] = str(response)
                        row['token_usage'] = row['raw_teacher_response'].get('usage', {})
                self.persist(row)
                raise
        raise RuntimeError('MEMORY_DECISION_BUDGET_EXHAUSTED_WITHOUT_YIELD')
