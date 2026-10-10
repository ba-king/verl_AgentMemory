"""One action, dynamic availability, and bounded syntax-only normalization."""
import copy
import re
from ..artifacts import text
from ..prompts import strict_json, teacher_input

BASE_PROMPT = """You are the Memory Teacher, a separate agent responsible only for persistent
memory. Your input is exactly the public information available to a future student.

You must choose exactly ONE next memory action.
If you want to perform multiple memory operations, output only the first next action.
The system will execute it and return the real result to you.
You may then decide the next action in a new turn.
Never plan or emit multiple memory actions in one response.

Return exactly one JSON object with keys "action" and "arguments", conforming to
the current decision_schema. No arrays, multiple objects, wrapper keys, native
function calls, Markdown, commentary, or reasoning outside or inside this object.
Only actions and IDs in the CURRENT schema are available. Availability is recomputed
after every executed action. Unavailable actions are rejected without execution.
YIELD hands control back to the Answer Agent. There is no mandatory read or write
at any time. You can perform several operations serially, including during a session.
Use the remaining decision budget.

Only successful retrieval results are delivered verbatim to the Answer Agent.
Writes do not themselves deliver evidence. There is no packet or organizer.
Use only the current public task/bootstrap, actual current-session public tool
calls and observations, legitimately retrieved records, and your successful write
receipts. The metadata is a count, not access to the full Bank. At session boundaries
all short-term context, including retrieved records, is cleared. Only the Bank persists.
Write self-contained factual concise records with the relevant traveler, day/slot,
object, date/city, and public attributes when known. Preserve the distinction
between a listed option, a requested change, and an actually submitted itinerary.
Keep evidence useful for possible later requests without guessing future tasks.
Do not write reasoning, vague advice, or unverified attempts as successful facts.
Submission confirms only the submitted text; correctness is not disclosed to you.
You may YIELD immediately. Never plan the current trip or select its answer
on behalf of the Answer Agent.
"""

DESCRIPTIONS = {
    'RETRIEVE': 'Read raw memories with the frozen Hybrid Retriever; arguments: query.',
    'ADD': 'Store one self-contained factual memory; arguments: content.',
    'UPDATE': 'Replace the content of one currently visible active memory ID; arguments: memory_id, content.',
    'DELETE': 'Delete one currently visible active memory ID; arguments: memory_id.',
    'YIELD': 'Return control without more operations at this opportunity; arguments: {}.',
}
ARGUMENTS = {'RETRIEVE': ['query'], 'ADD': ['content'], 'UPDATE': ['memory_id', 'content'],
             'DELETE': ['memory_id'], 'YIELD': []}


def availability(bank):
    active_ids = {r['memory_id'] for r in bank.snapshot()['records']}
    visible_ids = sorted(active_ids & bank.visible_ids)
    if not active_ids:
        return ['ADD', 'YIELD'], []
    if not visible_ids:
        return ['RETRIEVE', 'ADD', 'YIELD'], []
    return ['RETRIEVE', 'ADD', 'UPDATE', 'DELETE', 'YIELD'], visible_ids


def action_schema(allowed, visible_ids):
    branches = []
    for name in allowed:
        args = ARGUMENTS[name]
        properties = {k: ({'type': 'string', 'enum': list(visible_ids)} if k == 'memory_id' else
                         {'type': 'string', 'minLength': 1, 'pattern': r'\S'}) for k in args}
        branches.append({'type': 'object', 'properties': {
            'action': {'const': name},
            'arguments': {'type': 'object', 'properties': properties, 'required': args, 'additionalProperties': False}},
            'required': ['action', 'arguments'], 'additionalProperties': False})
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'object',
        'properties': {'action': {'type': 'string', 'enum': list(allowed)}, 'arguments': {'type': 'object'}},
        'required': ['action', 'arguments'], 'additionalProperties': False, 'oneOf': branches}


def current_state(bank, current, history, remaining, opportunity):
    state = teacher_input(current, history, bank.visible, bank.receipts, bank.actions,
                          len(bank.snapshot()['records']), remaining, opportunity)
    allowed, ids = availability(bank)
    state.update(available_actions=allowed, visible_memory_ids=ids, decision_schema=action_schema(allowed, ids))
    return state


def messages_for(state):
    descriptions = '\n'.join(f'- {name}: {DESCRIPTIONS[name]}' for name in state['available_actions'])
    system = BASE_PROMPT + '\nCurrent available actions:\n' + descriptions
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': text(state)}]


def json_request(messages, _tools, role):
    from ..provider import model_config
    if role != 'memory_teacher':
        raise RuntimeError('JSON_MEMORY_PROTOCOL_MUST_NOT_HANDLE_ANSWER')
    cfg = model_config()
    return {**{k: cfg[k] for k in ['model', 'temperature', 'max_tokens', 'thinking', 'stream']},
            'messages': copy.deepcopy(messages), 'response_format': {'type': 'json_object'}}


class ProtocolError(ValueError):
    def __init__(self, code, detail=''):
        self.code, self.detail = code, detail
        super().__init__(code + (': ' + detail if detail else ''))


def validate_arguments(value, schema):
    """Validate our closed generated schema, not arbitrary third-party JSON Schema.

    The only emitted argument constraints are exact keys, string type,
    non-whitespace text, and visible-ID enums. Keep this validator dependency-free.
    Root/action constraints are checked by parse_decision before reaching here.
    """
    branches = [b for b in schema['oneOf'] if b['properties']['action']['const'] == value['action']]
    if len(branches) != 1:
        raise ProtocolError('DYNAMIC_SCHEMA_REJECTED', 'action must match exactly one branch')
    spec = branches[0]['properties']['arguments']
    args = value['arguments']
    if not isinstance(args, dict) or set(args) != set(spec['required']):
        raise ProtocolError('DYNAMIC_SCHEMA_REJECTED', 'arguments must have exactly the required keys')
    for key, item in args.items():
        constraint = spec['properties'][key]
        if not isinstance(item, str):
            raise ProtocolError('DYNAMIC_SCHEMA_REJECTED', key + ' must be a string')
        if 'enum' in constraint and item not in constraint['enum']:
            raise ProtocolError('DYNAMIC_SCHEMA_REJECTED', key + ' is not currently visible and active')
        if len(item) < constraint.get('minLength', 0) or ('pattern' in constraint and not re.search(constraint['pattern'], item)):
            raise ProtocolError('DYNAMIC_SCHEMA_REJECTED', key + ' must be nonempty text')


def parse_decision(body, schema):
    choices = body.get('choices', [])
    if len(choices) != 1:
        raise ProtocolError('CHOICE_COUNT')
    choice = choices[0]
    message = choice.get('message', {})
    if message.get('tool_calls') or message.get('function_call') or choice.get('finish_reason') == 'tool_calls':
        raise ProtocolError('NATIVE_FUNCTION_CALLS_REJECTED')
    if choice.get('finish_reason') != 'stop' or message.get('reasoning_content'):
        raise ProtocolError('INCOMPLETE_OR_REASONING_OUTPUT')
    raw = message.get('content')
    if not isinstance(raw, str) or not raw.strip():
        raise ProtocolError('EMPTY_CONTENT')
    normalized = raw.strip()
    repairs = ['strip_outer_whitespace'] if normalized != raw else []
    fence = re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```', normalized, re.IGNORECASE)
    if fence:
        normalized = fence.group(1).strip()
        repairs.append('remove_one_complete_code_fence')
    try:
        value = strict_json(normalized)
    except (ValueError, TypeError) as error:
        raise ProtocolError('INVALID_SINGLE_JSON', str(error)) from error
    # A single legacy outer object can be unwrapped, never an array/list of choices.
    if isinstance(value, dict) and len(value) == 1 and next(iter(value)) in {'decision', 'memory_action'}:
        key = next(iter(value))
        value = value[key]
        repairs.append('unwrap_one_' + key + '_object')
    if not isinstance(value, dict):
        raise ProtocolError('TOP_LEVEL_NOT_SINGLE_DECISION_OBJECT')
    if any(k in value for k in ['actions', 'multiple_actions', 'tool_calls']):
        raise ProtocolError('MULTI_ACTION_ENVELOPE_REJECTED')
    if set(value) != {'action', 'arguments'}:
        raise ProtocolError('DECISION_KEYS_INVALID')
    if not isinstance(value['action'], str) or value['action'] not in schema['properties']['action']['enum']:
        raise ProtocolError('ACTION_NOT_AVAILABLE')
    validate_arguments(value, schema)
    return value, {'applied': repairs, 'semantic_action_selection_or_splitting': False}
