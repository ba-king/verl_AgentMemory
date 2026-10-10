"""Atomic field-level UPDATE for subject/scope structured memory records.

Paths name entries in ``record['fields']``; they do not traverse a nested object.
Values stay exact strings, including any serialized list value. This module
validates and applies an explicit edit, but never selects facts or writes a Bank.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Iterable


# Matches the frozen structured_record_v1 path syntax without depending on pilots.
PATH_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*$"
_ARGUMENT_KEYS = {"memory_id", "set_fields", "remove_fields"}


def update_arguments_schema(visible_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Return the UPDATE arguments JSON schema (not an action envelope).

    ``None`` means visibility is checked elsewhere. An empty collection permits
    no memory IDs. Requiring an edit does not require a changed value: explicit
    same-value sets are valid and may produce the Bank's usual NO_CHANGE result.
    """
    memory_id: dict[str, Any] = {"type": "string", "minLength": 1, "pattern": r"\S"}
    if visible_ids is not None:
        # Keep an empty visibility restriction valid under standard JSON Schema.
        ids = sorted(set(visible_ids))
        if ids:
            memory_id["enum"] = ids
        else:
            memory_id["not"] = {}
    path = {"type": "string", "pattern": PATH_PATTERN}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "memory_id": memory_id,
            "set_fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"path": dict(path), "value": {"type": "string"}},
                    "required": ["path", "value"],
                },
            },
            "remove_fields": {"type": "array", "uniqueItems": True, "items": dict(path)},
        },
        "required": ["memory_id", "set_fields", "remove_fields"],
        "anyOf": [
            {"properties": {"set_fields": {"minItems": 1}}},
            {"properties": {"remove_fields": {"minItems": 1}}},
        ],
    }


def _valid_path(path: Any) -> bool:
    return isinstance(path, str) and re.fullmatch(PATH_PATTERN, path) is not None


def validate_update_arguments(
    args: Any, visible_ids: Iterable[str] | None = None
) -> dict[str, Any]:
    """Validate only the proposed edit; return it unchanged or raise ValueError.

    Record-dependent checks (missing removals, empty result, conflicting field
    hierarchy) happen in :func:`apply_record_update`. No trimming or coercion is
    performed. Error messages are stable diagnostic identifiers.
    """
    if not isinstance(args, dict) or set(args) != _ARGUMENT_KEYS:
        raise ValueError("update_requires_only_memory_id_set_fields_remove_fields")
    if not isinstance(args["memory_id"], str) or not args["memory_id"].strip():
        raise ValueError("update_memory_id_must_be_nonempty_string")
    if visible_ids is not None and args["memory_id"] not in set(visible_ids):
        raise ValueError("update_id_not_visible")
    if not isinstance(args["set_fields"], list) or not isinstance(args["remove_fields"], list):
        raise ValueError("update_edits_must_be_arrays")
    if not args["set_fields"] and not args["remove_fields"]:
        raise ValueError("update_requires_at_least_one_edit")

    set_paths: set[str] = set()
    for field in args["set_fields"]:
        if not isinstance(field, dict) or set(field) != {"path", "value"}:
            raise ValueError("update_set_requires_only_path_value")
        if not _valid_path(field["path"]):
            raise ValueError("invalid_generic_dot_path")
        if not isinstance(field["value"], str):
            raise ValueError("record_value_must_be_string")
        if field["path"] in set_paths:
            raise ValueError("duplicate_update_set_path")
        set_paths.add(field["path"])

    remove_paths: set[str] = set()
    for path in args["remove_fields"]:
        if not _valid_path(path):
            raise ValueError("invalid_generic_dot_path")
        if path in remove_paths:
            raise ValueError("duplicate_update_remove_path")
        remove_paths.add(path)
    if set_paths & remove_paths:
        raise ValueError("update_set_remove_overlap")
    return args


def _validate_record(record: Any) -> None:
    """Validate the frozen record shape without importing experimental runtime."""
    if not isinstance(record, dict) or set(record) != {"subject", "scope", "fields"}:
        raise ValueError("record_requires_only_subject_scope_fields")
    for key in ("subject", "scope"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError("record_empty_or_nonstring_" + key)
    if not isinstance(record["fields"], list) or not record["fields"]:
        raise ValueError("record_requires_nonempty_fields")
    paths: set[str] = set()
    for field in record["fields"]:
        if not isinstance(field, dict) or set(field) != {"path", "value"}:
            raise ValueError("record_field_requires_only_path_value")
        if not _valid_path(field["path"]):
            raise ValueError("invalid_generic_dot_path")
        if field["path"] in paths:
            raise ValueError("duplicate_record_path")
        paths.add(field["path"])
        if not isinstance(field["value"], str):
            raise ValueError("record_value_must_be_string")


def apply_record_update(record: Any, args: Any) -> dict[str, Any]:
    """Apply all edits to a deep copy, or reject everything without mutation.

    Existing fields retain their order; new paths append in ``set_fields`` order.
    Subject and scope are copied exactly. The caller must bind ``memory_id`` to
    this record and enforce ID visibility. Setting a field to its current value
    is accepted and returns an equal (but independent) record.
    """
    validate_update_arguments(args)
    _validate_record(record)
    old_paths = {field["path"] for field in record["fields"]}
    removed = set(args["remove_fields"])
    if removed - old_paths:
        raise ValueError("update_remove_path_not_found")

    sets = {field["path"]: field["value"] for field in args["set_fields"]}
    final_paths = (old_paths - removed) | sets.keys()
    if not final_paths:
        raise ValueError("update_cannot_remove_all_fields")
    for path in final_paths:
        parts = path.split(".")
        if any(".".join(parts[:i]) in final_paths for i in range(1, len(parts))):
            raise ValueError("record_path_ancestor_conflict")

    updated = copy.deepcopy(record)
    updated["fields"] = [field for field in updated["fields"] if field["path"] not in removed]
    for field in updated["fields"]:
        if field["path"] in sets:
            field["value"] = sets[field["path"]]
    updated["fields"].extend(copy.deepcopy(field) for field in args["set_fields"] if field["path"] not in old_paths)
    return updated
