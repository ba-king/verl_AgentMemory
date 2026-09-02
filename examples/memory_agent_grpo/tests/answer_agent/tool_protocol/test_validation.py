import pytest

from examples.memory_agent_grpo.answer_agent.tool_protocol.definition import ToolDefinition
from examples.memory_agent_grpo.answer_agent.tool_protocol.observation import (
    ToolObservation,
    ToolObservationStatus,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.validation import (
    ToolDefinitionErrorCode,
    ToolObservationErrorCode,
    validate_tool_definition,
    validate_tool_observation,
)


def _definition(**changes: object) -> ToolDefinition:
    values: dict[str, object] = {
        "name": "lookup",
        "description": "Look up public task information.",
        "parameters_json_schema": '{"type":"object","properties":{}}',
    }
    values.update(changes)
    return ToolDefinition(**values)  # type: ignore[arg-type]


def _observation(**changes: object) -> ToolObservation:
    values: dict[str, object] = {
        "tool_call_id": "call-1",
        "tool_name": "lookup",
        "status": ToolObservationStatus.SUCCESS,
        "content": "Public adapter-filtered result.",
    }
    values.update(changes)
    return ToolObservation(**values)  # type: ignore[arg-type]


def test_valid_tool_definition_transport_contract() -> None:
    assert validate_tool_definition(_definition()).is_valid
    assert validate_tool_definition(
        _definition(
            parameters_json_schema=(
                '{"type":"object","properties":{"filters":{"type":"object"}},"additionalProperties":false}'
            )
        )
    ).is_valid


def test_extreme_strict_json_number_cannot_escape_definition_validation() -> None:
    definition = _definition(
        parameters_json_schema=('{"type":"object","properties":{},"maximum":1e9999999999999999999}')
    )

    assert validate_tool_definition(definition).is_valid


@pytest.mark.parametrize(
    ("value", "expected_code"),
    [
        ({"name": "lookup"}, ToolDefinitionErrorCode.DEFINITION_NOT_TYPED),
        (_definition(name=1), ToolDefinitionErrorCode.NAME_NOT_STRING),
        (_definition(name=" \n"), ToolDefinitionErrorCode.EMPTY_NAME),
        (_definition(description=1), ToolDefinitionErrorCode.DESCRIPTION_NOT_STRING),
        (_definition(description=""), ToolDefinitionErrorCode.EMPTY_DESCRIPTION),
        (
            _definition(parameters_json_schema=1),
            ToolDefinitionErrorCode.PARAMETERS_SCHEMA_NOT_STRING,
        ),
        (
            _definition(parameters_json_schema=" "),
            ToolDefinitionErrorCode.EMPTY_PARAMETERS_SCHEMA,
        ),
        (
            _definition(parameters_json_schema="not json"),
            ToolDefinitionErrorCode.INVALID_PARAMETERS_SCHEMA_JSON,
        ),
        (
            _definition(parameters_json_schema='{"type":"object","type":"array"}'),
            ToolDefinitionErrorCode.INVALID_PARAMETERS_SCHEMA_JSON,
        ),
        (
            _definition(parameters_json_schema='{"type":"object","limit":NaN}'),
            ToolDefinitionErrorCode.INVALID_PARAMETERS_SCHEMA_JSON,
        ),
        (
            _definition(parameters_json_schema="[]"),
            ToolDefinitionErrorCode.PARAMETERS_SCHEMA_NOT_OBJECT,
        ),
        (
            _definition(parameters_json_schema='{"properties":{}}'),
            ToolDefinitionErrorCode.PARAMETERS_SCHEMA_TYPE_NOT_OBJECT,
        ),
        (
            _definition(parameters_json_schema='{"type":"array"}'),
            ToolDefinitionErrorCode.PARAMETERS_SCHEMA_TYPE_NOT_OBJECT,
        ),
    ],
)
def test_invalid_tool_definition_reports_stable_error(
    value: object,
    expected_code: ToolDefinitionErrorCode,
) -> None:
    result = validate_tool_definition(value)

    assert not result.is_valid
    assert result.error_code is expected_code
    assert result.error_message


@pytest.mark.parametrize(
    "status",
    [ToolObservationStatus.SUCCESS, ToolObservationStatus.ERROR],
)
def test_valid_tool_observation(status: ToolObservationStatus) -> None:
    assert validate_tool_observation(_observation(status=status)).is_valid


@pytest.mark.parametrize(
    ("value", "expected_code"),
    [
        ({"content": "result"}, ToolObservationErrorCode.OBSERVATION_NOT_TYPED),
        (_observation(tool_call_id=1), ToolObservationErrorCode.TOOL_CALL_ID_NOT_STRING),
        (_observation(tool_call_id=" "), ToolObservationErrorCode.EMPTY_TOOL_CALL_ID),
        (_observation(tool_name=1), ToolObservationErrorCode.TOOL_NAME_NOT_STRING),
        (_observation(tool_name=""), ToolObservationErrorCode.EMPTY_TOOL_NAME),
        (_observation(status="success"), ToolObservationErrorCode.STATUS_NOT_TYPED),
        (_observation(content=1), ToolObservationErrorCode.CONTENT_NOT_STRING),
        (_observation(content="\n"), ToolObservationErrorCode.EMPTY_CONTENT),
    ],
)
def test_invalid_tool_observation_reports_stable_error(
    value: object,
    expected_code: ToolObservationErrorCode,
) -> None:
    result = validate_tool_observation(value)

    assert not result.is_valid
    assert result.error_code is expected_code
    assert result.error_message
