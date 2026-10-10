"""A versioned wire contract, not permissive parsing of legacy bad responses."""
import copy

from ..artifacts import canonical, strict_json
from ..autonomous_v1.rollout import apply_response
from ..tool_time_v1.contract import wire_tools_for as previous_tools

OLD_INSTRUCTION = 'YIELD arguments: {}.'
NEW_INSTRUCTION = 'YIELD arguments: {"status":"done"}. The status must be exactly "done".'
TERMINAL_ARGS = {'status': 'done'}
TERMINAL_SCHEMA = {
    'type': 'object', 'properties': {'status': {'type': 'string', 'enum': ['done']}},
    'required': ['status'], 'additionalProperties': False,
}


def revise_messages(messages):
    result = copy.deepcopy(messages)
    if result[0]['role'] != 'system' or result[0]['content'].count(OLD_INSTRUCTION) != 1:
        raise ValueError('UNEXPECTED_OLD_TERMINAL_INSTRUCTION')
    result[0]['content'] = result[0]['content'].replace(OLD_INSTRUCTION, NEW_INSTRUCTION)
    return result


def tools_for(bank):
    tools, _ = previous_tools(bank)
    terminal = next(t['function'] for t in tools if t['function']['name'] == 'YIELD')
    terminal['parameters'] = copy.deepcopy(TERMINAL_SCHEMA)
    return tools


def apply_response_v2(body, bank, wrote, seen):
    # Do not remove unknown keys, accept legacy {}, or unwrap malformed envelopes.
    projected = copy.deepcopy(body)
    choices = projected.get('choices', [])
    if len(choices) != 1:
        raise ValueError('ONE_COMPLETE_TOOL_TURN_REQUIRED')
    message = choices[0].get('message', {})
    calls = message.get('tool_calls', [])
    if len(calls) != 1:
        raise ValueError('ONE_ASSISTANT_TOOL_CALL_REQUIRED')
    function = calls[0]['function']
    terminal = function.get('name') == 'YIELD'
    if terminal:
        arguments = strict_json(function['arguments'])
        if type(arguments) is not dict or arguments != TERMINAL_ARGS:
            raise ValueError('V2_YIELD_REQUIRES_EXACT_STATUS_DONE')
        function['arguments'] = canonical({})
    step = apply_response(projected, bank, wrote, seen)
    if terminal:
        # Conversation / eventual targets keep what the teacher actually emitted.
        step['canonical_assistant_message'] = step['assistant_message']
        actual = body['choices'][0]['message']
        step['assistant_message'] = {k: copy.deepcopy(actual[k]) for k in
                                    ('role', 'content', 'tool_calls', 'reasoning_content') if k in actual}
        step['wire_action'] = {'action': 'YIELD', 'arguments': TERMINAL_ARGS.copy()}
        step['transport_projection'] = 'v2 status=done -> canonical empty terminal arguments; no Bank write'
    return step
