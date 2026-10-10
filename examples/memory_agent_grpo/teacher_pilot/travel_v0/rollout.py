"""Three-arm controller; opportunities grant choice, never mandate a memory action."""
import copy
import time
from pathlib import Path
from .artifacts import *
from .memory_runtime import BankBranch, delta
from .prompts import (answer_messages, teacher_input, teacher_messages, memory_tools,
                      parse_memory_response, evidence_block, strict_json, teacher_public_event)
from .provider import FixedClient

MEMORY_DECISIONS_PER_SESSION = 32
ANSWER_RESPONSES_PER_SESSION = 30


class SplitSession:
    def __init__(self, run_id, eid, current, directory, bank, budget, answer_conversation):
        self.run_id, self.eid, self.current = run_id, eid, current
        self.directory, self.bank, self.budget = Path(directory), bank, budget
        self.answer_conversation = answer_conversation
        self.client = FixedClient('memory_teacher', self.directory / 'api', budget)
        self.history, self.decisions, self.pending, self.deliveries = [], [], [], []
        self.buffered_deliveries = []
        self.opportunities = 0
        self.bank.begin_session()

    def snapshot_ref(self, value):
        path = self.directory / 'private/bank' / (digest(value) + '.json')
        save(path, value)
        return str(path)

    def persist(self, row):
        save(self.directory / 'decisions' / (row['decision_id'] + '.json'), row)

    def opportunity(self, stage, defer_delivery=False):
        self.opportunities += 1
        pending_deliveries = []
        while len(self.decisions) < MEMORY_DECISIONS_PER_SESSION:
            before = self.bank.snapshot()
            state = teacher_input(self.current, self.history, self.bank.visible, self.bank.receipts,
                self.bank.actions, len(before['records']), MEMORY_DECISIONS_PER_SESSION - len(self.decisions),
                {'index': self.opportunities, 'stage': stage})
            messages = teacher_messages(state)
            did = f"{self.current['session_id']}_d{len(self.decisions) + 1:03d}"
            row = {'run_id': self.run_id, 'episode_id': self.eid, 'session_id': self.current['session_id'],
                'decision_id': did, 'public_task': self.current, 'answer_public_history': copy.deepcopy(self.history),
                'memory_teacher_messages': messages, 'available_memory_tools': memory_tools(),
                'visible_memories': copy.deepcopy(self.bank.visible), 'bank_metadata': state['bank_metadata'],
                'student_input': state, 'raw_teacher_response': None, 'parsed_memory_action': None,
                'memory_action_result': None, 'retrieval_result': None,
                'memory_bank_before_ref': self.snapshot_ref(before), 'memory_bank_after_ref': None,
                'memory_delta': None, 'answer_memory_evidence_delivered': None, 'next_answer_action': None,
                'environment_observation': None, 'native_result': None, 'strict_result': None,
                'termination_reason': None, 'model_config': self.client.config, 'token_usage': {}, 'latency': None,
                'source_event_refs': [f'public:{self.current["session_id"]}',
                    *[h['event_id'] for h in self.history],
                    *[r['decision_id'] for r in self.decisions]], 'opportunity_stage': stage,
                'opportunity_index': self.opportunities}
            self.decisions.append(row)
            self.pending.append(row)
            self.persist(row)
            try:
                body, meta = self.client.call(messages, memory_tools())
                row.update(meta, raw_teacher_response=body)
                action = parse_memory_response(body)
                row['parsed_memory_action'] = action
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
                    # Execution resumes only on an actual YIELD, never budget exhaustion.
                    self.buffered_deliveries.extend((source, delivered, did) for source, delivered in pending_deliveries)
                    if not defer_delivery:
                        self.flush_deliveries()
                    return
            except Exception as error:
                row['termination_reason'] = 'memory_decision_error'
                row['error'] = type(error).__name__ + ': ' + str(error)
                self.persist(row)
                raise
        raise RuntimeError('MEMORY_DECISION_BUDGET_EXHAUSTED_WITHOUT_YIELD')

    def flush_deliveries(self):
        for source, delivered, did in self.buffered_deliveries:
            block = evidence_block(delivered)
            self.answer_conversation.append({'role': 'user', 'content': block})
            source['answer_memory_evidence_delivered'] = {'content': block,
                'hits': copy.deepcopy(delivered), 'after_yield_decision': did}
            self.deliveries.append(copy.deepcopy(source['answer_memory_evidence_delivered']))
            self.persist(source)
        self.buffered_deliveries.clear()

    def public_event(self, event):
        self.history.append(teacher_public_event(event))
        for row in self.pending:
            row['next_answer_action'] = copy.deepcopy(event['action'])
            row['environment_observation'] = copy.deepcopy(event['observation'])
            row['next_answer_event_ref'] = event['event_id']
            row['next_action_precommitted_in_tool_batch'] = event.get('tool_index', 0) > 0
            self.persist(row)
        self.pending.clear()

    def finish(self, score):
        for row in self.decisions:
            row['native_result'] = score.get('native_score')
            row['strict_result'] = score.get('strict_validation')
            row['termination_reason'] = row['termination_reason'] or score['termination_reason']
            self.persist(row)


def run_episode(arm, public, retriever, budget, run_id):
    # Memora reuses the Bank/parser classes without running a Travel environment.
    # Load the unchanged native environment only when a Travel episode starts.
    from .environment import TravelBoundary, TOOLS

    assert arm in ARMS
    eid = public['episode_id']
    directory = HERE / 'raw_trajectories' / arm / eid
    if directory.exists():
        raise RuntimeError('refuse_to_overwrite_previous_episode_attempt')
    directory.mkdir(parents=True)
    boundary = TravelBoundary(public)
    bank = BankBranch(retriever) if arm == 'split_teacher' else None
    summary = {'episode_id': eid, 'arm': arm, 'ORACLE_MEMORY': arm == 'oracle_memory',
        'sessions': [], 'strict_episode_success': False, 'planned_sessions': 5,
        'gold_replacement_performed': False, 'run_id': run_id}
    save(directory / 'private/reference.json', boundary.hidden)
    start = time.monotonic()
    try:
        for index in range(5):
            if boundary.stopped:
                break
            current, private_reset = boundary.reset(index)
            dest = directory / current['session_id']
            save(dest / 'public_input.json', current)
            save(dest / 'private/native_reset.json', private_reset)
            facts = boundary.required_facts(index)  # Private diagnostic only; never enters C inputs.
            save(dest / 'private/required_history_facts.json', facts)
            conversation = answer_messages(current)
            initial_bank_count = len(bank.snapshot()['records']) if bank else 0
            save(dest / 'reset_check.json', {'answer_prior_messages': 0, 'teacher_prior_messages': 0,
                'retrieved_memory_count': 0, 'initial_bank_record_count': initial_bank_count,
                'bootstrap_present': 'initial_observation' in current,
                'no_automatic_history_channel': True, 'native_history_empty': boundary.env.history == []})
            if arm == 'oracle_memory':
                oracle = boundary.oracle(index)
                save(dest / 'private/oracle_delivery.json', {'ORACLE_MEMORY': True,
                    'source': 'dependency graph + public bootstrap + validated actual prefix + public DB',
                    'facts': facts, 'hits': oracle, 'current_target_injected': False})
                if oracle:
                    conversation.append({'role': 'user', 'content': evidence_block(oracle)})
            answer = FixedClient('answer', dest / 'api', budget)
            split = SplitSession(run_id, eid, current, dest, bank, budget, conversation) if bank else None
            session = {'session_id': current['session_id'], 'native_score': None,
                'strict_validation': {'passed': False, 'reason': 'not_submitted'},
                'termination_reason': 'not_started', 'tool_calls': 0, 'answer_calls': 0,
                'memory_decisions': 0, 'decision_opportunities': 0}
            try:
                if split:
                    split.opportunity('session_start')
                for step in range(ANSWER_RESPONSES_PER_SESSION):
                    body, meta = answer.call(conversation, TOOLS)
                    session['answer_calls'] += 1
                    choice = body['choices'][0]
                    message = choice['message']
                    if message.get('reasoning_content'):
                        raise RuntimeError('unexpected_answer_reasoning_in_disabled_mode')
                    calls = message.get('tool_calls') or []
                    clean = {k: v for k, v in message.items() if k in {'role', 'content', 'tool_calls'}}
                    conversation.append(clean)
                    event = {'event_id': f'{eid}:{arm}:{current["session_id"]}:answer:{step + 1}',
                             'action': None, 'observation': None, 'request_ref': meta['request_ref'],
                             'response_ref': meta['response_ref'], 'response_index': step}
                    if calls:
                        for tool_index, call in enumerate(calls):
                            f = call['function']
                            args = strict_json(f['arguments'])
                            observation = boundary.tool(f['name'], args)
                            tool_event = {**event, 'event_id': event['event_id'] + f':tool:{tool_index}',
                                'tool_index': tool_index, 'action': {'tool': f['name'], 'arguments': args},
                                'observation': observation}
                            conversation.append({'role': 'tool', 'tool_call_id': call['id'], 'content': observation})
                            session['tool_calls'] += 1
                            append(dest / 'answer_public_events.jsonl', tool_event)
                            if split:
                                split.public_event(tool_event)
                                split.opportunity('after_travel_tool', defer_delivery=True)
                        if split:
                            # Finish the API's pending tool-result batch before
                            # injecting raw evidence for the next Answer decision.
                            split.flush_deliveries()
                    else:
                        content = message.get('content')
                        if not content:
                            raise RuntimeError('empty_answer_final')
                        visible, score = boundary.submit(content, index)
                        event.update(action={'final_itinerary': content}, observation=visible)
                        append(dest / 'answer_public_events.jsonl', event)
                        save(dest / 'private/submission_score.json', score)
                        session.update(score)
                        if split:
                            split.public_event(event)
                            split.opportunity('after_final_itinerary')
                        break
                else:
                    raise RuntimeError('ANSWER_RESPONSE_BUDGET_EXHAUSTED')
            except Exception as error:
                session['error'] = type(error).__name__ + ': ' + str(error)
                session['termination_reason'] = 'runtime_error'
                boundary.stopped = True
            finally:
                if split:
                    session['memory_decisions'] = len(split.decisions)
                    session['decision_opportunities'] = split.opportunities
                    split.finish(session)
                save(dest / 'answer_exact_conversation.json', conversation)
                save(dest / 'private/score.json', session)
                summary['sessions'].append(session)
                save(directory / 'episode_summary.json', summary)
                print(text({'arm': arm, 'episode_id': eid, 'session_id': current['session_id'],
                    'strict': session['strict_validation']['passed'], 'termination': session['termination_reason'],
                    'answer_calls': session['answer_calls'], 'memory_decisions': session['memory_decisions']}), flush=True)
    finally:
        summary['strict_episode_success'] = len(summary['sessions']) == 5 and all(
            s['strict_validation']['passed'] and s['termination_reason'] == 'submitted' for s in summary['sessions'])
        summary['stopped_after_failure'] = boundary.stopped
        summary['actual_history_ref'] = str(directory / 'private/actual_history.json')
        summary['latency_seconds'] = time.monotonic() - start
        save(directory / 'private/actual_history.json', boundary.actual_history)
        save(directory / 'episode_summary.json', summary)
        boundary.close()
    return summary
