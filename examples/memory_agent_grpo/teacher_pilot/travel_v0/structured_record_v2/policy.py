"""Versioned parser/session integration for local UPDATE; no new action name."""
from ....memory_agent.action.structured_update import update_arguments_schema, validate_update_arguments
from ..structured_record import policy as previous
from ..read_consolidate import modes
from ..single_action import protocol, session
from ..single_action.session import bindings
from ..artifacts import text
from .bank import StructuredBank

READ_PROMPT = previous.READ_PROMPT
CONSOLIDATE_PROMPT = previous.CONSOLIDATE_PROMPT.replace(
    'ADD and UPDATE now take a structured record with subject,\nscope, and nonempty fields of generic dot-path/string-value pairs.',
    'ADD takes a structured record with subject, scope, and nonempty fields of '
    'generic dot-path/string-value pairs. UPDATE takes memory_id, set_fields, and remove_fields. '
    'Supply only changed/new fields in set_fields and exact field paths to delete in remove_fields. '
    'Fields not mentioned are preserved exactly; do not copy the full record. '
    'subject and scope stay unchanged. Paths identify whole existing field slots, not list indices '
    'inside a field value. To remove a list member, set that field to the revised list value. '
    'Invalid edits are rejected atomically; use DELETE to remove an entire record.')
DESCRIPTIONS = {**previous.DESCRIPTIONS,
    'UPDATE': 'Edit one visible record; arguments: memory_id, set_fields [{path,value}], remove_fields [path]. Unmentioned fields are preserved.'}


def action_schema(allowed, visible_ids):
    schema = previous.action_schema(allowed, visible_ids)
    for branch in schema['oneOf']:
        if branch['properties']['action']['const'] == 'UPDATE':
            branch['properties']['arguments'] = update_arguments_schema(visible_ids)
    return schema


def validate_arguments(value, schema):
    if value['action'] != 'UPDATE':
        return previous.validate_arguments(value, schema)
    branch = next(b for b in schema['oneOf'] if b['properties']['action']['const'] == 'UPDATE')
    ids = branch['properties']['arguments']['properties']['memory_id'].get('enum', [])
    try:
        validate_update_arguments(value['arguments'], visible_ids=ids)
    except (ValueError, TypeError) as error:
        raise protocol.ProtocolError('DYNAMIC_SCHEMA_REJECTED', str(error)) from error


def parse_decision(body, schema):
    with bindings(protocol, validate_arguments=validate_arguments):
        return protocol.parse_decision(body, schema)


def current_state(bank, current, history, remaining, opportunity):
    with bindings(protocol, action_schema=action_schema):
        return modes.current_state(bank, current, history, remaining, opportunity)


def messages_for(state):
    prompt = READ_PROMPT if state['decision_mode'] == 'READ' else CONSOLIDATE_PROMPT
    descriptions = '\n'.join(f'- {a}: {DESCRIPTIONS[a]}' for a in state['available_actions'])
    return [{'role': 'system', 'content': prompt + '\nCurrent available actions:\n' + descriptions},
            {'role': 'user', 'content': text(state)}]


class StructuredSession(previous.StructuredSession):
    def opportunity(self, stage, defer_delivery=False):
        if not isinstance(self.bank, StructuredBank):
            raise TypeError('structured_record_v2_requires_matching_bank')
        with bindings(session, current_state=current_state, messages_for=messages_for, parse_decision=parse_decision):
            return session.SingleActionSession.opportunity(self, stage, defer_delivery)
