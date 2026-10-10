"""Provider-only schema projection; the original local executor stays strict.

DeepSeek beta requires all tools to be strict and cannot enforce array length.
The wire schema is a superset of our local schema. We remove only the listed
unsupported constraints, never repair a model response or add a Bank field.
"""
import copy
from pathlib import Path

from ...memora_sft_dev.teacher_rollout_v1.interface import tools_for

PACKAGE = Path(__file__).resolve().parent
UPDATE_NONEMPTY = [
    {'properties': {'set_fields': {'minItems': 1}}},
    {'properties': {'remove_fields': {'minItems': 1}}},
]
LOCAL_ONLY = {'minLength', 'maxLength', 'minItems', 'maxItems', 'uniqueItems'}
SUPPORTED = {'type', 'properties', 'required', 'additionalProperties',
             'items', 'pattern', 'enum'}


def instruction():
    return '\n\n' + (PACKAGE/'time_instruction.txt').read_text().strip() + '\n'


def _wire_schema(schema):
    node = copy.deepcopy(schema)
    if 'anyOf' in node:
        if node['anyOf'] != UPDATE_NONEMPTY:
            raise ValueError('UNREVIEWED_ANYOF_SCHEMA')
        # Nonempty UPDATE is still enforced by the frozen parser/executor.
        del node['anyOf']
    for key in LOCAL_ONLY:
        node.pop(key, None)
    if set(node) - SUPPORTED:
        raise ValueError('UNREVIEWED_SCHEMA_KEYWORD')
    if node.get('type') == 'object':
        if (node.get('additionalProperties') is not False
                or set(node.get('required', [])) != set(node.get('properties', {}))):
            raise ValueError('STRICT_OBJECT_CONTRACT_REQUIRED')
        node['properties'] = {k: _wire_schema(v) for k, v in node['properties'].items()}
    elif node.get('type') == 'array':
        node['items'] = _wire_schema(node['items'])
    elif node.get('type') != 'string':
        raise ValueError('UNREVIEWED_SCHEMA_TYPE')
    return node


def strict_tools(original_tools):
    tools = copy.deepcopy(original_tools)
    for tool in tools:
        function = tool['function']
        function['parameters'] = _wire_schema(function['parameters'])
        function['strict'] = True
    return tools


def wire_tools_for(bank):
    tools, schema = tools_for(bank)
    return strict_tools(tools), schema


def revised_context(old):
    # Keep the local tool schema untouched in the source capsule. Projection
    # happens only when constructing the provider request.
    new = copy.deepcopy(old)
    new['messages'][0]['content'] += instruction()
    return new


def assert_request_change(old, new):
    expected = revised_context(old)
    expected['tools'] = strict_tools(old['tools'])
    if new != expected:
        raise ValueError('UNEXPECTED_REQUEST_CHANGE')
