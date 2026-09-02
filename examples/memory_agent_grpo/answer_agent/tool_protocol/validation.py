"""Structural validation for the dataset-neutral task-tool protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from .definition import ToolDefinition
from .observation import ToolObservation, ToolObservationStatus


class ToolDefinitionErrorCode(str, Enum):
    DEFINITION_NOT_TYPED = "DEFINITION_NOT_TYPED"
    NAME_NOT_STRING = "NAME_NOT_STRING"
    EMPTY_NAME = "EMPTY_NAME"
    DESCRIPTION_NOT_STRING = "DESCRIPTION_NOT_STRING"
    EMPTY_DESCRIPTION = "EMPTY_DESCRIPTION"
    PARAMETERS_SCHEMA_NOT_STRING = "PARAMETERS_SCHEMA_NOT_STRING"
    EMPTY_PARAMETERS_SCHEMA = "EMPTY_PARAMETERS_SCHEMA"
    INVALID_PARAMETERS_SCHEMA_JSON = "INVALID_PARAMETERS_SCHEMA_JSON"
    PARAMETERS_SCHEMA_NOT_OBJECT = "PARAMETERS_SCHEMA_NOT_OBJECT"
    PARAMETERS_SCHEMA_TYPE_NOT_OBJECT = "PARAMETERS_SCHEMA_TYPE_NOT_OBJECT"


@dataclass(frozen=True)
class ToolDefinitionValidationResult:
    error_code: ToolDefinitionErrorCode | None = None
    error_message: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.error_code is None


class ToolObservationErrorCode(str, Enum):
    OBSERVATION_NOT_TYPED = "OBSERVATION_NOT_TYPED"
    TOOL_CALL_ID_NOT_STRING = "TOOL_CALL_ID_NOT_STRING"
    EMPTY_TOOL_CALL_ID = "EMPTY_TOOL_CALL_ID"
    TOOL_NAME_NOT_STRING = "TOOL_NAME_NOT_STRING"
    EMPTY_TOOL_NAME = "EMPTY_TOOL_NAME"
    STATUS_NOT_TYPED = "STATUS_NOT_TYPED"
    CONTENT_NOT_STRING = "CONTENT_NOT_STRING"
    EMPTY_CONTENT = "EMPTY_CONTENT"


@dataclass(frozen=True)
class ToolObservationValidationResult:
    error_code: ToolObservationErrorCode | None = None
    error_message: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.error_code is None


def _invalid_definition(
    code: ToolDefinitionErrorCode,
    message: str,
) -> ToolDefinitionValidationResult:
    return ToolDefinitionValidationResult(error_code=code, error_message=message)


def _invalid_observation(
    code: ToolObservationErrorCode,
    message: str,
) -> ToolObservationValidationResult:
    return ToolObservationValidationResult(error_code=code, error_message=message)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-standard JSON constant {value!r} is not allowed.")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r} is not allowed.")
        result[key] = value
    return result


def validate_tool_definition(value: object) -> ToolDefinitionValidationResult:
    """Validate one adapter-supplied, model-facing tool description.

    This validates the transport contract without claiming to implement a
    complete JSON Schema metaschema. Argument-instance validation belongs at
    the later ToolCall Gateway boundary.
    """

    if not isinstance(value, ToolDefinition):
        return _invalid_definition(
            ToolDefinitionErrorCode.DEFINITION_NOT_TYPED,
            "definition must be a ToolDefinition.",
        )
    if not isinstance(value.name, str):
        return _invalid_definition(
            ToolDefinitionErrorCode.NAME_NOT_STRING,
            "name must be a string.",
        )
    if not value.name.strip():
        return _invalid_definition(
            ToolDefinitionErrorCode.EMPTY_NAME,
            "name must not be empty.",
        )
    if not isinstance(value.description, str):
        return _invalid_definition(
            ToolDefinitionErrorCode.DESCRIPTION_NOT_STRING,
            "description must be a string.",
        )
    if not value.description.strip():
        return _invalid_definition(
            ToolDefinitionErrorCode.EMPTY_DESCRIPTION,
            "description must not be empty.",
        )
    if not isinstance(value.parameters_json_schema, str):
        return _invalid_definition(
            ToolDefinitionErrorCode.PARAMETERS_SCHEMA_NOT_STRING,
            "parameters_json_schema must be a string.",
        )
    if not value.parameters_json_schema.strip():
        return _invalid_definition(
            ToolDefinitionErrorCode.EMPTY_PARAMETERS_SCHEMA,
            "parameters_json_schema must not be empty.",
        )

    try:
        schema = json.loads(
            value.parameters_json_schema,
            parse_float=str,
            parse_int=str,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        return _invalid_definition(
            ToolDefinitionErrorCode.INVALID_PARAMETERS_SCHEMA_JSON,
            "parameters_json_schema must be strict JSON without duplicate keys.",
        )
    if not isinstance(schema, dict):
        return _invalid_definition(
            ToolDefinitionErrorCode.PARAMETERS_SCHEMA_NOT_OBJECT,
            "parameters_json_schema must encode one JSON object.",
        )
    if schema.get("type") != "object":
        return _invalid_definition(
            ToolDefinitionErrorCode.PARAMETERS_SCHEMA_TYPE_NOT_OBJECT,
            "parameters_json_schema must explicitly declare top-level type 'object'.",
        )
    return ToolDefinitionValidationResult()


def validate_tool_observation(value: object) -> ToolObservationValidationResult:
    """Validate one adapter-filtered result before it enters model messages."""

    if not isinstance(value, ToolObservation):
        return _invalid_observation(
            ToolObservationErrorCode.OBSERVATION_NOT_TYPED,
            "observation must be a ToolObservation.",
        )
    if not isinstance(value.tool_call_id, str):
        return _invalid_observation(
            ToolObservationErrorCode.TOOL_CALL_ID_NOT_STRING,
            "tool_call_id must be a string.",
        )
    if not value.tool_call_id.strip():
        return _invalid_observation(
            ToolObservationErrorCode.EMPTY_TOOL_CALL_ID,
            "tool_call_id must not be empty.",
        )
    if not isinstance(value.tool_name, str):
        return _invalid_observation(
            ToolObservationErrorCode.TOOL_NAME_NOT_STRING,
            "tool_name must be a string.",
        )
    if not value.tool_name.strip():
        return _invalid_observation(
            ToolObservationErrorCode.EMPTY_TOOL_NAME,
            "tool_name must not be empty.",
        )
    if not isinstance(value.status, ToolObservationStatus):
        return _invalid_observation(
            ToolObservationErrorCode.STATUS_NOT_TYPED,
            "status must be a ToolObservationStatus.",
        )
    if not isinstance(value.content, str):
        return _invalid_observation(
            ToolObservationErrorCode.CONTENT_NOT_STRING,
            "content must be a string.",
        )
    if not value.content.strip():
        return _invalid_observation(
            ToolObservationErrorCode.EMPTY_CONTENT,
            "content must not be empty.",
        )
    return ToolObservationValidationResult()


__all__ = [
    "ToolDefinitionErrorCode",
    "ToolDefinitionValidationResult",
    "ToolObservationErrorCode",
    "ToolObservationValidationResult",
    "validate_tool_definition",
    "validate_tool_observation",
]
