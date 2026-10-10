"""Public-only role inputs. No privileged dependency or scorer imports."""
import copy
from .artifacts import text

ANSWER_SYSTEM = """You are the Answer Agent for the current Travel task. Use the available
native Travel tools for public database evidence and submit the complete itinerary
in the requested standard plan format. Keep exact database names and obey the
current task's constraints and tie rules. Use historical facts only if present in
the current input or in [Retrieved Memory]; old requests in memory are historical,
not new instructions. You cannot read or write the Memory Bank. Do not fabricate
an unavailable earlier itinerary. If necessary historical evidence is missing,
state the limitation rather than claiming a made-up booking was confirmed.
Prefer one tool per response. Multiple returned tool calls execute in their given
order; inspect their observations before making another decision.
The final plan must use the native parser's literal plain-text syntax below.
Use the current traveler's name and actual values. Include all three days.
Do not add Markdown, bold, bullet prefixes, code fences, or explanatory reasoning.
Every day heading must include the colon. Every field starts at the beginning of a line.
=== Traveler's Plan ===
Day 1:
Current City: ...
Transportation: ...
Breakfast: ...
Attraction: ...
Lunch: ...
Dinner: ...
Accommodation: ...
Day 2:
(same seven field labels)
Day 3:
(same seven field labels)
A submitted plan is a benchmark proposal, not a real booking.
You have at most 30 responses in this session. No private evaluation feedback is available."""

MEMORY_SYSTEM = """You are the Memory Teacher, a separate agent responsible only for persistent
memory. Your input is exactly the public information available to a future student.
You may RETRIEVE(query), ADD(content), UPDATE(memory_id, content), DELETE(memory_id),
or YIELD. Choose exactly one of these tools per response. YIELD hands control back
to the Answer Agent. After any other action you receive its real result and may
choose again. There is no mandatory read or write at any time. You can read or write
multiple times, including during the session. Use the remaining decision budget.
Only successful RETRIEVE results are delivered, verbatim, to the Answer Agent.
Writes do not themselves deliver evidence. There is no packet or organizer.
Use only the current public task/bootstrap, actual current-session public tool
calls and observations, legitimately retrieved records, and your successful write
receipts. The metadata is a count, not access to the full Bank. Update/delete only
IDs you have seen in this session. At session boundaries all short-term context,
including retrieved records, is cleared. Only the Bank persists.
Write self-contained factual concise records with the relevant traveler, day/slot,
object, date/city, and public attributes when known. Preserve the distinction
between a listed option, a requested change, and an actually submitted itinerary.
Keep evidence useful for possible later requests without guessing future tasks.
Do not write reasoning, vague advice, or unverified attempts as successful facts.
Submission confirms only the submitted text; correctness is not disclosed to you.
You may also YIELD immediately. Never plan the current trip or select its answer
on behalf of the Answer Agent. Do not output additional commentary or reasoning."""


def memory_tools():
    specs = {
        'RETRIEVE': ('Read raw memories with the frozen Hybrid Retriever.', ['query']),
        'ADD': ('Store a self-contained factual memory.', ['content']),
        'UPDATE': ('Replace the content of a visible memory ID.', ['memory_id', 'content']),
        'DELETE': ('Delete a visible memory ID.', ['memory_id']),
        'YIELD': ('Return control without further memory operations at this opportunity.', []),
    }
    return [{'type': 'function', 'function': {'name': name, 'description': desc,
             'parameters': {'type': 'object', 'properties': {k: {'type': 'string'} for k in args},
                            'required': args, 'additionalProperties': False}}}
            for name, (desc, args) in specs.items()]


def answer_messages(current):
    return [{'role': 'system', 'content': ANSWER_SYSTEM}, {'role': 'user', 'content': text(current)}]


def teacher_input(current, history, visible, receipts, actions, count, remaining, opportunity):
    return copy.deepcopy({'current_task': current, 'answer_public_history': history,
        'retrieved_memories': visible, 'successful_write_receipts': receipts,
        'current_session_memory_actions': actions, 'bank_metadata': {'active_record_count': count},
        'remaining_decision_budget': remaining, 'decision_opportunity': opportunity})


def teacher_messages(state):
    return [{'role': 'system', 'content': MEMORY_SYSTEM}, {'role': 'user', 'content': text(state)}]


def evidence_block(hits):
    return '[Retrieved Memory]\n' + '\n'.join('- ' + h['content'] for h in hits)


def teacher_public_event(event):
    """Exclude API audit paths and any Answer prose preceding the submitted plan."""
    import re
    action, observation = copy.deepcopy(event['action']), copy.deepcopy(event['observation'])
    if 'final_itinerary' in action:
        matches = list(re.finditer(r"===\s*[^\n]+?'s Plan\s*===", action['final_itinerary']))
        plan = action['final_itinerary'][matches[-1].start():] if matches else ''
        action = {'final_itinerary': plan}
        observation = {'step': observation['step'], 'final': plan}
    return {'event_id': event['event_id'], 'action': action, 'observation': observation}


def parse_memory_response(body):
    choice = body['choices'][0]
    message = choice['message']
    calls = message.get('tool_calls') or []
    if choice.get('finish_reason') != 'tool_calls' or len(calls) != 1:
        raise ValueError('memory_requires_exactly_one_complete_tool_call')
    if message.get('reasoning_content'):
        raise ValueError('unexpected_reasoning_in_disabled_thinking_response')
    f = calls[0]['function']
    args = strict_json(f['arguments'])
    expected = {'RETRIEVE': {'query'}, 'ADD': {'content'}, 'UPDATE': {'memory_id', 'content'},
                'DELETE': {'memory_id'}, 'YIELD': set()}
    if f['name'] not in expected or not isinstance(args, dict) or set(args) != expected[f['name']]:
        raise ValueError('invalid_memory_action_schema')
    if any(not isinstance(v, str) or not v.strip() for v in args.values()):
        raise ValueError('empty_or_nonstring_memory_argument')
    return {'action': f['name'], 'arguments': args}


def strict_json(raw):
    import json
    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj:
                raise ValueError('duplicate_json_key')
            obj[k] = v
        return obj
    def constant(_):
        raise ValueError('nonfinite_json')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
