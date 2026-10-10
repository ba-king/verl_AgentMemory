"""Native tool transport around the existing structured parser and executor."""
import copy
import json

from ...memory_agent.bank.bank_state import MemoryBankState, MemoryRecord
from ...memory_agent.bank.store import MemoryBankStore
from ...memory_agent.executor.engine import MemoryActionExecutor
from ...teacher_pilot.travel_v0.structured_record_v2.bank import StructuredBank
from ...teacher_pilot.travel_v0.structured_record_v2.policy import action_schema, parse_decision


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('DUPLICATE_JSON_KEY')
            result[key] = value
        return result

    def constant(value):
        raise ValueError('NON_FINITE_JSON')

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def snapshot(bank):
    return strict_json(canonical(bank.snapshot()))


def restore(before):
    bank = StructuredBank(None)
    bank.records = {r['memory_id']: copy.deepcopy(r['record']) for r in before['records']}
    if len(bank.records) != len(before['records']):
        raise ValueError('DUPLICATE_MEMORY_ID')
    bank.executor = MemoryActionExecutor(store=MemoryBankStore(MemoryBankState(
        tuple(MemoryRecord(r['memory_id'], r['content']) for r in before['records']), before['next_id'])))
    bank.visible_ids = set(bank.records)
    bank.visible = copy.deepcopy(before['records'])
    if snapshot(bank) != before:
        raise ValueError('BANK_RESTORE_MISMATCH')
    return bank


DESCRIPTIONS = {
    'ADD': 'Create one independent durable record supported by the visible dialogue.',
    'UPDATE': 'Edit changed fields of one visible record. Unmentioned fields stay unchanged.',
    'DELETE': 'Remove one entire visible record; use UPDATE for partial field removal.',
    'YIELD': 'Finish this maintenance episode when no further supported durable change remains.',
}


def tools_for(bank):
    ids = sorted(bank.visible_ids & set(bank.records))
    schema = action_schema(['ADD', 'UPDATE', 'DELETE', 'YIELD'] if ids else ['ADD', 'YIELD'], ids)
    tools = []
    for branch in schema['oneOf']:
        name = branch['properties']['action']['const']
        tools.append({'type': 'function', 'function': {'name': name,
            'description': DESCRIPTIONS[name],
            'parameters': copy.deepcopy(branch['properties']['arguments'])}})
    return tools, schema


def native_prompt(existing):
    old = 'Return exactly one JSON action with keys action and arguments, matching decision_schema.'
    new = ('Choose exactly one next action using the provided native function tools. '
           'The executor will return its actual result before you decide the next action.')
    if existing.count(old) != 1:
        raise ValueError('UNEXPECTED_SOURCE_PROMPT')
    return existing.replace(old, new).replace(
        'Return only the action JSON, without reasoning or Markdown.',
        'Emit exactly one native tool call per turn. Do not simulate tool results. '
        'Call YIELD to finish; do not write an action JSON object in ordinary text.')


def execute_message(body, bank):
    choices = body.get('choices', [])
    if len(choices) != 1 or choices[0].get('finish_reason') != 'tool_calls':
        raise ValueError('ONE_COMPLETE_NATIVE_TOOL_TURN_REQUIRED')
    message = choices[0]['message']
    calls = message.get('tool_calls', [])
    if len(calls) != 1:
        raise ValueError('EXACTLY_ONE_TOOL_CALL_REQUIRED')
    call = calls[0]
    if call.get('type') != 'function' or not isinstance(call.get('id'), str) or not call['id']:
        raise ValueError('INVALID_TOOL_CALL_ID_OR_TYPE')
    function = call['function']
    action = {'action': function['name'], 'arguments': strict_json(function['arguments'])}
    _, schema = tools_for(bank)
    # Only the transport envelope changes; frozen argument validation is reused.
    parsed, repairs = parse_decision({'choices': [{'finish_reason': 'stop',
        'message': {'content': canonical(action)}}]}, schema)
    if parsed != action or repairs['applied']:
        raise ValueError('ACTION_MODIFIED_DURING_VALIDATION')
    before = snapshot(bank)
    receipt, hits = bank.execute(action)
    if hits is not None:
        raise ValueError('UNEXPECTED_RETRIEVAL')
    after = snapshot(bank)
    if receipt['status'] == 'REJECTED' and after != before:
        raise ValueError('REJECTION_MUTATED_BANK')
    assistant = {k: copy.deepcopy(message[k]) for k in ('role', 'content', 'tool_calls', 'reasoning_content') if k in message}
    tool_result = {'role': 'tool', 'tool_call_id': call['id'], 'content': canonical(receipt)}
    return {'assistant_message': assistant, 'tool_message': tool_result,
            'action': action, 'receipt': receipt, 'bank_before': before, 'bank_after': after}
