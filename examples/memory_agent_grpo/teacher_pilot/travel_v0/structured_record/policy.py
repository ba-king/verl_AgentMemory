"""Only ADD/UPDATE payload and CONSOLIDATE record policy differ from v3."""
import copy
from ..read_consolidate import modes
from ..single_action import protocol, session
from ..single_action.session import bindings
from ..artifacts import text
from .record_renderer import record_schema, validate_record

READ_PROMPT = modes.READ_PROMPT
CONSOLIDATE_PROMPT = modes.CONSOLIDATE_PROMPT + '''
In CONSOLIDATE mode, preserve durable public state that would otherwise be lost
after session reset. ADD and UPDATE now take a structured record with subject,
scope, and nonempty fields of generic dot-path/string-value pairs.
A persistent record must make clear who or what the facts belong to, what
event/state/time scope they describe, and what exact structured values were
publicly established. Scope must distinguish this subject's different events.
If the current session contains an important public base state and also
establishes a new subject-specific result, do not silently collapse the two
subjects into one ambiguous summary. Prefer separate self-contained records
when different subjects represent distinct persistent states. Do not infer an
inheritance relation unless it is supported by the public task/result.
For structured public fields, preserve exact strings whenever later
reconstruction could depend on them. Do not shorten, normalize or paraphrase
an available exact structured value. One coherent state is one record, not one
record per field. Do not put system metadata or source IDs in the record schema;
the harness records provenance. Do not store chain-of-thought, rejected candidates,
speculative future use, private scorer information, or hidden dependencies.
This does not mean copy everything. Store durable task state, not the reasoning
trajectory. Full states, compact states, and multiple subject records are allowed;
claims and their scope must be reconstructable from legitimately persisted state.
'''

DESCRIPTIONS = {**protocol.DESCRIPTIONS,
    'ADD': 'Store one structured persistent state; arguments: record (subject, scope, fields).',
    'UPDATE': 'Replace one visible active record; arguments: memory_id, record (subject, scope, fields).'}
_old_schema = protocol.action_schema
_old_validator = protocol.validate_arguments


def action_schema(allowed, visible_ids):
    schema = _old_schema(allowed, visible_ids)
    for branch in schema['oneOf']:
        if branch['properties']['action']['const'] not in {'ADD', 'UPDATE'}:
            continue
        args = branch['properties']['arguments']
        args['properties'].pop('content')
        args['properties']['record'] = record_schema()
        args['required'] = ['record' if key == 'content' else key for key in args['required']]
    return schema


def validate_arguments(value, schema):
    if value['action'] not in {'ADD', 'UPDATE'}:
        return _old_validator(value, schema)
    name, args = value['action'], value['arguments']
    expected = {'record'} if name == 'ADD' else {'memory_id', 'record'}
    try:
        if not isinstance(args, dict) or set(args) != expected:
            raise ValueError('wrong_structured_action_arguments')
        validate_record(args['record'])
        if name == 'UPDATE':
            branch = next(b for b in schema['oneOf'] if b['properties']['action']['const'] == name)
            if args['memory_id'] not in branch['properties']['arguments']['properties']['memory_id']['enum']:
                raise ValueError('update_id_not_visible')
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
    return [{'role': 'system', 'content': prompt+'\nCurrent available actions:\n'+descriptions},
            {'role': 'user', 'content': text(state)}]


class StructuredSession(session.SingleActionSession):
    def opportunity(self, stage, defer_delivery=False):
        with bindings(session, current_state=current_state, messages_for=messages_for, parse_decision=parse_decision):
            return super().opportunity(stage, defer_delivery=defer_delivery)

    def flush_deliveries(self):
        from .. import rollout
        from .record_renderer import evidence_block
        with bindings(rollout, evidence_block=evidence_block):
            return super().flush_deliveries()

    def persist(self, row):
        row['decision_mode'] = row['student_input']['decision_mode']
        super().persist(row)
