"""Shopping public submission projection into the frozen structured Memory loop."""
import copy
from ....travel_v0.single_action import protocol, session
from ....travel_v0.single_action.session import bindings
from ....travel_v0.structured_record import policy


def current_state(bank,current,history,remaining,opportunity):
    with bindings(protocol,action_schema=policy.action_schema):
        state=protocol.current_state(bank,current,history,remaining,opportunity)
    mode='READ' if opportunity['stage']=='session_start' else 'CONSOLIDATE'
    assert opportunity['stage'] in {'session_start','after_purchase','session_ended_without_purchase'}
    allowed=state['available_actions'] if mode=='CONSOLIDATE' else [a for a in state['available_actions'] if a in {'RETRIEVE','YIELD'}]
    state.update(decision_mode=mode,available_actions=allowed,
        decision_schema=policy.action_schema(allowed,state['visible_memory_ids']))
    if mode=='CONSOLIDATE':
        state['session_public_history']=state.pop('answer_public_history')
        last=history[-1] if history else None
        # Literal actual public terminal event only; no target, label lookup or repaired fields.
        state['submitted_result']={'source_event_id':last['event_id'] if last else None,
            'source':'actual_public_submission',
            'public_text':copy.deepcopy(last['observation']) if last else '',
            'purchase_submitted':opportunity['stage']=='after_purchase'}
        state['successful_memory_action_receipts']=state.pop('successful_write_receipts')
    return state


class ShoppingMemorySession(policy.StructuredSession):
    def opportunity(self,stage,defer_delivery=False):
        with bindings(session,current_state=current_state,messages_for=policy.messages_for,
                      parse_decision=policy.parse_decision):
            return session.SingleActionSession.opportunity(self,stage,defer_delivery)
