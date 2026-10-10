"""Only append current-task grounded retention guidance during CONSOLIDATE."""
from ..pilot4_v1.memory import current_state
from ....travel_v0.single_action import session
from ....travel_v0.single_action.session import bindings
from ....travel_v0.structured_record import policy as frozen

RETENTION_GUIDANCE = '''
Task-relevant grounded retention:
After a completed purchase or selection, prioritize the chosen subject's identity,
paid price, explicit attributes materially relevant to the current task's
selection or validation, and necessary supported cumulative state such as budget.
When an attribute is explicitly declared in the current public observations
(for example Material: X, Color: Y, or Size: Z), preserve its attribute role and
observed value; a word in the product name alone is not an attribute declaration.
Select attributes constrained or actually used in the current task whose facts
remain meaningful after completion; do not copy every attribute or whole pages.
Use only current public observations and legitimately grounded visible memory;
do not infer attributes from names, predict later tasks, or use hidden targets.
'''


def messages_for(state):
    result = frozen.messages_for(state)
    if state['decision_mode'] == 'CONSOLIDATE':
        # Insertion occurs before unchanged dynamic action descriptions.
        prefix = frozen.CONSOLIDATE_PROMPT
        assert result[0]['content'].startswith(prefix)
        result[0]['content'] = prefix + RETENTION_GUIDANCE + result[0]['content'][len(prefix):]
    return result


class ShoppingMemorySessionV2(frozen.StructuredSession):
    def __init__(self, run_id, *args, **kwargs):
        super().__init__('memory_policy_and_labeler_v2_regression', *args, **kwargs)

    def opportunity(self, stage, defer_delivery=False):
        with bindings(session, current_state=current_state, messages_for=messages_for,
                      parse_decision=frozen.parse_decision):
            return session.SingleActionSession.opportunity(self, stage, defer_delivery)
