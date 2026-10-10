"""Only two structural additions. Delegate all delta operations to frozen V2.

Stripping the two fields before V2 apply and restoring their corresponding
records afterwards preserves removal indices, ordering, progress and enum
semantics. No prompt, provider, Answer interface or production import is added.
"""
import copy
import json
from jsonschema import Draft202012Validator
from ..packet_interface_v2_validation import interface as v2

NEED = copy.deepcopy(v2.NEED)
NEED['properties']['scope_refs'] = copy.deepcopy(v2.STRINGS)
NEED['required'].append('scope_refs')
EVIDENCE = copy.deepcopy(v2.EVIDENCE)
EVIDENCE['properties']['subject_ref'] = copy.deepcopy(v2.STR)
EVIDENCE['required'].append('subject_ref')
PACKET_SCHEMA = copy.deepcopy(v2.PACKET_SCHEMA)
PACKET_SCHEMA['properties']['needs']['items'] = NEED
PACKET_SCHEMA['properties']['evidence']['items'] = EVIDENCE
DELTA_SCHEMA = copy.deepcopy(v2.DELTA_SCHEMA)
DELTA_SCHEMA['properties']['add_needs']['items'] = NEED
DELTA_SCHEMA['properties']['add_evidence']['items'] = EVIDENCE
ENVELOPE_SCHEMA = v2.obj({'packet_delta': DELTA_SCHEMA})
TOOLS = copy.deepcopy(v2.TOOLS)
TOOLS[0]['function']['parameters'] = ENVELOPE_SCHEMA
empty_packet = v2.empty_packet
ProtocolError = v2.ProtocolError


def project_v2(value, delta=False):
    out = copy.deepcopy(value)
    for need in out['add_needs' if delta else 'needs']:
        need.pop('scope_refs', None)
    for evidence in out['add_evidence' if delta else 'evidence']:
        evidence.pop('subject_ref', None)
    return out


def parse(response):
    # Retain V2's envelope/duplicate-key checks, using the local V3 schema.
    # No monkey-patching of the frozen module and no wrapper repair.
    choices = response.get('choices', [])
    if len(choices) != 1:
        raise ProtocolError('CHOICE_COUNT')
    choice = choices[0]
    if choice.get('finish_reason') != 'tool_calls':
        raise ProtocolError('NOT_COMPLETE_TOOL_CALL')
    calls = choice.get('message', {}).get('tool_calls', [])
    if len(calls) != 1 or calls[0].get('type') != 'function':
        raise ProtocolError('SINGLE_TOOL_REQUIRED')
    function = calls[0].get('function', {})
    if function.get('name') != 'emit_packet_delta':
        raise ProtocolError('UNEXPECTED_TOOL')
    raw = function.get('arguments')
    if not isinstance(raw, str):
        raise ProtocolError('ARGUMENTS_NOT_JSON_STRING')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError('DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique,
            parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    except (ValueError, json.JSONDecodeError) as error:
        raise ProtocolError('MALFORMED_JSON: ' + str(error)) from error
    if not isinstance(value, dict) or set(value) != {'packet_delta'}:
        raise ProtocolError('UNEXPECTED_WRAPPER')
    errors = list(Draft202012Validator(ENVELOPE_SCHEMA).iter_errors(value))
    if errors:
        raise ProtocolError('SCHEMA_INVALID: ' + errors[0].message)
    return value['packet_delta']


def apply(previous, delta, source_refs, init=False):
    Draft202012Validator(PACKET_SCHEMA).validate(previous)
    Draft202012Validator(DELTA_SCHEMA).validate(delta)
    old_result = v2.apply(project_v2(previous), project_v2(delta, True), source_refs, init=init)
    needs = copy.deepcopy(previous['needs'] + delta['add_needs'])
    evidence = [copy.deepcopy(e) for i, e in enumerate(previous['evidence'])
                if i not in delta['remove_evidence']] + copy.deepcopy(delta['add_evidence'])
    result = copy.deepcopy(old_result)
    result['needs'] = needs
    result['evidence'] = evidence
    Draft202012Validator(PACKET_SCHEMA).validate(result)
    assert project_v2(result) == old_result
    return result
