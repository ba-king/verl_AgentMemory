import json
from dataclasses import FrozenInstanceError

import pytest

from examples.memory_agent_grpo.answer_agent.response import AnswerAgentToolCall
from examples.memory_agent_grpo.answer_agent.tool_call_gateway import (
    ToolCallGatewayErrorCode,
    ToolCallGatewayResult,
    ValidatedToolCall,
    validate_tool_call_for_execution,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.definition import (
    ToolDefinition,
)


def _definition(
    *,
    name: str = "FlightSearch",
    schema: object = None,
) -> ToolDefinition:
    if schema is None:
        schema = {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "mode": {"type": "string", "enum": ["driving", "taxi"]},
            },
            "required": ["origin", "destination", "mode"],
            "additionalProperties": False,
        }
    return ToolDefinition(
        name=name,
        description="Test task tool.",
        parameters_json_schema=json.dumps(schema, separators=(",", ":")),
    )


def _call(
    *,
    name: str = "FlightSearch",
    arguments_json: str = ('{"mode":"driving","destination":"Boston","origin":"Ithaca"}'),
) -> AnswerAgentToolCall:
    return AnswerAgentToolCall(
        tool_call_id="call-1",
        name=name,
        arguments_json=arguments_json,
    )


def test_valid_call_returns_immutable_canonical_boundary() -> None:
    result = validate_tool_call_for_execution(
        _call(),
        tool_definitions=(_definition(),),
    )

    assert result.is_valid
    assert result.error_code is None
    assert result.validated_call is not None
    assert result.validated_call.to_dict() == {
        "tool_call_id": "call-1",
        "name": "FlightSearch",
        "arguments_json": ('{"destination":"Boston","mode":"driving","origin":"Ithaca"}'),
    }
    with pytest.raises(FrozenInstanceError):
        result.validated_call.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("arguments_json", "expected_message_part"),
    [
        (
            '{"destination":"Boston","origin":"Ithaca"}',
            "missing required property 'mode'",
        ),
        (
            '{"destination":"Boston","mode":"walking","origin":"Ithaca"}',
            "not an allowed enum value",
        ),
        (
            '{"destination":"Boston","mode":"driving","origin":7}',
            "wrong JSON type",
        ),
        (
            '{"destination":"Boston","extra":true,"mode":"driving","origin":"Ithaca"}',
            "unexpected property 'extra'",
        ),
    ],
)
def test_schema_mismatches_are_rejected_without_a_validated_call(
    arguments_json: str,
    expected_message_part: str,
) -> None:
    result = validate_tool_call_for_execution(
        _call(arguments_json=arguments_json),
        tool_definitions=(_definition(),),
    )

    assert not result.is_valid
    assert result.validated_call is None
    assert result.error_code is ToolCallGatewayErrorCode.ARGUMENTS_DO_NOT_MATCH_SCHEMA
    assert expected_message_part in (result.error_message or "")


def test_schema_default_allows_additional_properties() -> None:
    definition = _definition(
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    )
    result = validate_tool_call_for_execution(
        _call(
            name=definition.name,
            arguments_json='{"extra":true,"query":"public data"}',
        ),
        tool_definitions=(definition,),
    )

    assert result.is_valid


def test_unknown_tool_is_rejected() -> None:
    result = validate_tool_call_for_execution(
        _call(name="UnknownTool"),
        tool_definitions=(_definition(),),
    )

    assert result.error_code is ToolCallGatewayErrorCode.UNKNOWN_TOOL
    assert result.validated_call is None


@pytest.mark.parametrize("definitions", [[], None, {"tool": "definition"}])
def test_definitions_must_be_a_tuple(definitions: object) -> None:
    result = validate_tool_call_for_execution(
        _call(),
        tool_definitions=definitions,
    )

    assert result.error_code is ToolCallGatewayErrorCode.TOOL_DEFINITIONS_NOT_TUPLE


def test_duplicate_definition_names_are_rejected() -> None:
    result = validate_tool_call_for_execution(
        _call(),
        tool_definitions=(_definition(), _definition()),
    )

    assert result.error_code is ToolCallGatewayErrorCode.DUPLICATE_TOOL_NAME


def test_invalid_definition_is_rejected_before_call_interpretation() -> None:
    result = validate_tool_call_for_execution(
        _call(),
        tool_definitions=(object(),),
    )

    assert result.error_code is ToolCallGatewayErrorCode.INVALID_TOOL_DEFINITION
    assert result.validated_call is None


def test_invalid_tool_call_contract_is_rejected() -> None:
    result = validate_tool_call_for_execution(
        _call(arguments_json='{"query":NaN}'),
        tool_definitions=(_definition(),),
    )

    assert result.error_code is ToolCallGatewayErrorCode.INVALID_TOOL_CALL
    assert result.validated_call is None


@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "object",
            "properties": {"query": {"type": "array"}},
        },
        {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 2}},
        },
        {
            "type": "object",
            "properties": {"query": {"type": []}},
        },
        {
            "type": "object",
            "properties": {"query": {"type": {"unexpected": "object"}}},
        },
        {
            "type": "object",
            "properties": {},
            "additionalProperties": {"type": "string"},
        },
    ],
)
def test_unsupported_schema_features_fail_closed(schema: object) -> None:
    result = validate_tool_call_for_execution(
        _call(arguments_json='{"query":"value"}'),
        tool_definitions=(_definition(schema=schema),),
    )

    assert result.error_code is ToolCallGatewayErrorCode.UNSUPPORTED_SCHEMA
    assert result.validated_call is None


def test_json_number_is_not_mistaken_for_a_string() -> None:
    definition = _definition(
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    )

    result = validate_tool_call_for_execution(
        _call(arguments_json='{"query":123}'),
        tool_definitions=(definition,),
    )

    assert result.error_code is ToolCallGatewayErrorCode.ARGUMENTS_DO_NOT_MATCH_SCHEMA


def test_result_serialization_never_claims_an_invalid_call_is_executable() -> None:
    result = validate_tool_call_for_execution(
        _call(name="unknown"),
        tool_definitions=(_definition(),),
    )

    assert result.to_dict() == {
        "validated_call": None,
        "error_code": "UNKNOWN_TOOL",
        "error_message": "tool call name is not present in the current tool definitions.",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {
            "validated_call": ValidatedToolCall("id", "tool", "{}"),
            "error_message": "failure",
        },
        {
            "validated_call": ValidatedToolCall("id", "tool", "{}"),
            "error_code": ToolCallGatewayErrorCode.UNKNOWN_TOOL,
            "error_message": "failure",
        },
        {"error_code": ToolCallGatewayErrorCode.UNKNOWN_TOOL},
        {"error_message": "failure"},
        {
            "error_code": ToolCallGatewayErrorCode.UNKNOWN_TOOL,
            "error_message": " ",
        },
    ],
)
def test_result_rejects_ambiguous_or_incomplete_states(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ToolCallGatewayResult(**changes)  # type: ignore[arg-type]
