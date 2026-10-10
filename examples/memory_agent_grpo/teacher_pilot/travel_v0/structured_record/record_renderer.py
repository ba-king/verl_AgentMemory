"""Generic structured record validation and deterministic, reversible text view."""
import json
import re

PATH_PATTERN = r'^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*$'


def record_schema():
    nonempty = {'type': 'string', 'minLength': 1, 'pattern': r'\S'}
    return {'type': 'object', 'additionalProperties': False,
        'properties': {'subject': dict(nonempty), 'scope': dict(nonempty),
            'fields': {'type': 'array', 'minItems': 1, 'items': {
                'type': 'object', 'additionalProperties': False,
                'properties': {'path': {'type': 'string', 'pattern': PATH_PATTERN},
                               'value': {'type': 'string'}}, 'required': ['path', 'value']}}},
        'required': ['subject', 'scope', 'fields']}


def validate_record(record):
    if not isinstance(record, dict) or set(record) != {'subject', 'scope', 'fields'}:
        raise ValueError('record_requires_only_subject_scope_fields')
    for key in ('subject', 'scope'):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError('record_empty_or_nonstring_' + key)
    if not isinstance(record['fields'], list) or not record['fields']:
        raise ValueError('record_requires_nonempty_fields')
    paths = set()
    for field in record['fields']:
        if not isinstance(field, dict) or set(field) != {'path', 'value'}:
            raise ValueError('record_field_requires_only_path_value')
        if not isinstance(field['path'], str) or re.fullmatch(PATH_PATTERN, field['path']) is None:
            raise ValueError('invalid_generic_dot_path')
        if field['path'] in paths:
            raise ValueError('duplicate_record_path')
        paths.add(field['path'])
        if not isinstance(field['value'], str):
            raise ValueError('record_value_must_be_string')
    return record


def render(record):
    validate_record(record)
    quote = lambda s: json.dumps(s, ensure_ascii=False)
    return '\n'.join([f"Subject: {quote(record['subject'])}", f"Scope: {quote(record['scope'])}", 'Fields:',
        *[f"- {f['path']} = {quote(f['value'])}" for f in record['fields']]])


def evidence_block(hits):
    return '[Retrieved Memory]\n\n' + '\n\n'.join(
        'Memory ' + h['memory_id'] + '\n' + render(h['record']) for h in hits)
