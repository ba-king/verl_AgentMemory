"""Fail-closed validation of one Answer Agent task-tool call."""

from __future__ import annotations

import json
import math

from ..response import AnswerAgentResponse, AnswerAgentToolCall
from ..tool_protocol.definition import ToolDefinition
from ..tool_protocol.validation import validate_tool_definition
from ..validation import validate_answer_agent_response
from .result import ToolCallGatewayErrorCode, ToolCallGatewayResult
from .validated_call import ValidatedToolCall

_ROOT_SCHEMA_KEYS = {
    "$id",
    "$schema",
    "additionalProperties",
    "default",
    "description",
    "examples",
    "properties",
    "required",
    "title",
    "type",
}
_PROPERTY_SCHEMA_KEYS = {
    "default",
    "description",
    "enum",
    "examples",
    "title",
    "type",
}
_SUPPORTED_PROPERTY_TYPES = {"boolean", "integer", "null", "number", "string"}


def _invalid(
    code: ToolCallGatewayErrorCode,
    message: str,
) -> ToolCallGatewayResult:
    return ToolCallGatewayResult(error_code=code, error_message=message)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-standard JSON constant {value!r} is not allowed.")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r} is not allowed.")
        result[key] = value
    return result


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number must be finite.")
    return parsed


def _parse_json_object(value: str) -> dict[str, object]:
    parsed = json.loads(
        value,
        parse_float=_parse_finite_float,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(parsed, dict):
        raise ValueError("JSON value must be an object.")
    return parsed


def _validate_definitions(
    definitions: object,
) -> tuple[dict[str, ToolDefinition] | None, ToolCallGatewayResult | None]:
    if type(definitions) is not tuple:
        return None, _invalid(
            ToolCallGatewayErrorCode.TOOL_DEFINITIONS_NOT_TUPLE,
            "tool_definitions must be a tuple.",
        )

    by_name: dict[str, ToolDefinition] = {}
    for position, definition in enumerate(definitions):
        validation = validate_tool_definition(definition)
        if not validation.is_valid:
            detail = "UNKNOWN" if validation.error_code is None else validation.error_code.value
            return None, _invalid(
                ToolCallGatewayErrorCode.INVALID_TOOL_DEFINITION,
                f"tool_definitions[{position}] is invalid ({detail}).",
            )
        assert isinstance(definition, ToolDefinition)
        if definition.name in by_name:
            return None, _invalid(
                ToolCallGatewayErrorCode.DUPLICATE_TOOL_NAME,
                f"tool_definitions[{position}] duplicates an earlier name.",
            )
        by_name[definition.name] = definition
    return by_name, None


def _type_matches(value: object, expected_type: str) -> bool:
    if expected_type == "string":
        return type(value) is str
    if expected_type == "boolean":
        return type(value) is bool
    if expected_type == "integer":
        return type(value) is int
    if expected_type == "number":
        return type(value) in (int, float)
    if expected_type == "null":
        return value is None
    return False


def _json_values_equal(left: object, right: object) -> bool:
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(  # type: ignore[arg-type]
                left,
                right,
                strict=True,
            )
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(  # type: ignore[union-attr]
            _json_values_equal(left[key], right[key])  # type: ignore[index]
            for key in left
        )
    return left == right


def _validate_arguments_against_schema(
    arguments: dict[str, object],
    schema: dict[str, object],
) -> ToolCallGatewayResult | None:
    unsupported_root_keys = set(schema) - _ROOT_SCHEMA_KEYS
    if unsupported_root_keys:
        return _invalid(
            ToolCallGatewayErrorCode.UNSUPPORTED_SCHEMA,
            "tool schema contains unsupported root validation keywords.",
        )
    if schema.get("type") != "object":
        return _invalid(
            ToolCallGatewayErrorCode.INVALID_SCHEMA,
            "tool schema must declare root type 'object'.",
        )

    properties = schema.get("properties", {})
    if type(properties) is not dict:
        return _invalid(
            ToolCallGatewayErrorCode.INVALID_SCHEMA,
            "tool schema properties must be an object.",
        )
    if any(type(property_name) is not str for property_name in properties):
        return _invalid(
            ToolCallGatewayErrorCode.INVALID_SCHEMA,
            "tool schema property names must be strings.",
        )

    required = schema.get("required", [])
    if (
        type(required) is not list
        or any(type(property_name) is not str for property_name in required)
        or len(set(required)) != len(required)
    ):
        return _invalid(
            ToolCallGatewayErrorCode.INVALID_SCHEMA,
            "tool schema required must contain unique string names.",
        )

    additional_properties = schema.get("additionalProperties", True)
    if type(additional_properties) is not bool:
        return _invalid(
            ToolCallGatewayErrorCode.UNSUPPORTED_SCHEMA,
            "only boolean additionalProperties is supported.",
        )

    missing = [property_name for property_name in required if property_name not in arguments]
    if missing:
        return _invalid(
            ToolCallGatewayErrorCode.ARGUMENTS_DO_NOT_MATCH_SCHEMA,
            f"tool arguments are missing required property {missing[0]!r}.",
        )

    if not additional_properties:
        unexpected = [name for name in arguments if name not in properties]
        if unexpected:
            return _invalid(
                ToolCallGatewayErrorCode.ARGUMENTS_DO_NOT_MATCH_SCHEMA,
                f"tool arguments contain unexpected property {unexpected[0]!r}.",
            )

    for property_name, property_schema in properties.items():
        if type(property_schema) is not dict:
            return _invalid(
                ToolCallGatewayErrorCode.INVALID_SCHEMA,
                f"schema for property {property_name!r} must be an object.",
            )
        unsupported_property_keys = set(property_schema) - _PROPERTY_SCHEMA_KEYS
        if unsupported_property_keys:
            return _invalid(
                ToolCallGatewayErrorCode.UNSUPPORTED_SCHEMA,
                f"schema for property {property_name!r} uses unsupported keywords.",
            )

        expected_type = property_schema.get("type")
        if type(expected_type) is not str or expected_type not in _SUPPORTED_PROPERTY_TYPES:
            return _invalid(
                ToolCallGatewayErrorCode.UNSUPPORTED_SCHEMA,
                f"schema for property {property_name!r} has an unsupported type.",
            )
        enum = property_schema.get("enum")
        if enum is not None and (type(enum) is not list or not enum):
            return _invalid(
                ToolCallGatewayErrorCode.INVALID_SCHEMA,
                f"schema enum for property {property_name!r} must be a non-empty list.",
            )

        if property_name not in arguments:
            continue
        argument_value = arguments[property_name]
        assert isinstance(expected_type, str)
        if not _type_matches(argument_value, expected_type):
            return _invalid(
                ToolCallGatewayErrorCode.ARGUMENTS_DO_NOT_MATCH_SCHEMA,
                f"tool argument {property_name!r} has the wrong JSON type.",
            )
        if enum is not None and not any(_json_values_equal(argument_value, enum_value) for enum_value in enum):
            return _invalid(
                ToolCallGatewayErrorCode.ARGUMENTS_DO_NOT_MATCH_SCHEMA,
                f"tool argument {property_name!r} is not an allowed enum value.",
            )
    return None


def validate_tool_call_for_execution(
    tool_call: object,
    *,
    tool_definitions: object,
) -> ToolCallGatewayResult:
    """Validate one call without invoking an adapter or task tool."""

    definitions_by_name, definition_error = _validate_definitions(tool_definitions)
    if definition_error is not None:
        return definition_error
    assert definitions_by_name is not None

    response_validation = validate_answer_agent_response(
        AnswerAgentResponse(tool_calls=(tool_call,))  # type: ignore[arg-type]
    )
    if not response_validation.is_valid:
        detail = "UNKNOWN" if response_validation.error_code is None else response_validation.error_code.value
        return _invalid(
            ToolCallGatewayErrorCode.INVALID_TOOL_CALL,
            f"tool call failed provider-neutral validation ({detail}).",
        )
    assert isinstance(tool_call, AnswerAgentToolCall)

    definition = definitions_by_name.get(tool_call.name)
    if definition is None:
        return _invalid(
            ToolCallGatewayErrorCode.UNKNOWN_TOOL,
            "tool call name is not present in the current tool definitions.",
        )

    try:
        schema = _parse_json_object(definition.parameters_json_schema)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return _invalid(
            ToolCallGatewayErrorCode.INVALID_SCHEMA,
            "selected tool schema is not strict finite JSON.",
        )
    try:
        arguments = _parse_json_object(tool_call.arguments_json)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return _invalid(
            ToolCallGatewayErrorCode.INVALID_TOOL_CALL,
            "tool arguments are not strict finite JSON.",
        )

    schema_error = _validate_arguments_against_schema(arguments, schema)
    if schema_error is not None:
        return schema_error

    try:
        canonical_arguments = json.dumps(
            arguments,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        canonical_arguments.encode("utf-8")
    except (UnicodeEncodeError, TypeError, ValueError, OverflowError, RecursionError):
        return _invalid(
            ToolCallGatewayErrorCode.CANONICALIZATION_FAILED,
            "validated tool arguments cannot be canonicalized as UTF-8 JSON.",
        )

    return ToolCallGatewayResult(
        validated_call=ValidatedToolCall(
            tool_call_id=tool_call.tool_call_id,
            name=tool_call.name,
            arguments_json=canonical_arguments,
        )
    )


__all__ = ["validate_tool_call_for_execution"]
