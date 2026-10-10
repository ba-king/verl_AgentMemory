"""Consolidation-only wrapper; READ and all original action schemas unchanged."""
import copy
from teacher_pilot.travel_v0.structured_record import policy
from teacher_pilot.travel_v0.single_action import session
from teacher_pilot.travel_v0.single_action.session import bindings
from .adapter_durable_obligations import get_durable_obligations
from .consolidation_coverage_guard import CoverageGuard, coverage
from ..common import save


class GuardedSession(policy.StructuredSession):
    def __init__(self, *args, future_public_sessions=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.future_public_sessions = copy.deepcopy(future_public_sessions)
        self.durable_guard = None

    def opportunity(self, stage, defer_delivery=False):
        if stage != 'after_final_itinerary':
            return super().opportunity(stage, defer_delivery=defer_delivery)
        if self.durable_guard is not None:
            raise RuntimeError('SECOND_CONSOLIDATION_FORBIDDEN')
        contract = get_durable_obligations(self.current, self.history, self.future_public_sessions)
        guard = self.durable_guard = CoverageGuard(contract)
        save(self.directory/'durable_contract.json', contract)
        execute = self.bank.execute

        def guarded_execute(action):
            if action['action'] == 'YIELD':
                checked = guard.check_yield(self.bank.snapshot())
                if not checked['satisfied']:
                    return {'status':'REJECTED','code':checked['code'],
                            'coverage':checked,'executor_entered':False}, None
            return execute(action)

        def guarded_state(bank, current, history, remaining, opportunity):
            state = policy.current_state(bank, current, history, remaining, opportunity)
            state['durable_obligations'] = copy.deepcopy(contract['obligations'])
            state['durable_coverage'] = coverage(contract, bank.snapshot())
            state['consolidation_guard'] = {
                'authority':'PUBLIC_ADAPTER_CONTRACT',
                'meaning':'YIELD is accepted only when these subject-specific public states are independently recoverable from Bank. Values must come from the supplied actual public evidence; no automatic writes occur.',
                'bounded_continuations_used':guard.continuations,
                'maximum_bounded_continuations':1,
                'previous_yield_check':guard.yield_checks[-1] if guard.yield_checks else None}
            return state

        try:
            # Original parser, system prompt, executor and single-action loop.
            # A rejected YIELD ends that loop; exactly one explicit continuation
            # opportunity is available, still within the original session budget.
            with bindings(self.bank, execute=guarded_execute), bindings(session,
                    current_state=guarded_state, messages_for=policy.messages_for,
                    parse_decision=policy.parse_decision):
                session.SingleActionSession.opportunity(self, stage, defer_delivery)
                if guard.allow_continuation(self.bank.snapshot()):
                    session.SingleActionSession.opportunity(self, stage, defer_delivery)
        finally:
            save(self.directory/'durable_coverage.json', {
                'final':coverage(contract,self.bank.snapshot()),
                'yield_checks':guard.yield_checks,'bounded_continuations':guard.continuations,
                'automatic_bank_writes':0,'system_prompt_changed':False})
