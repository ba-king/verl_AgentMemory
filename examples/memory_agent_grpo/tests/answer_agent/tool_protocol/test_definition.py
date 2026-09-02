from dataclasses import FrozenInstanceError

import pytest

from examples.memory_agent_grpo.answer_agent.tool_protocol.definition import ToolDefinition


def _definition() -> ToolDefinition:
    return ToolDefinition(
        name="lookup",
        description="Look up public task information.",
        parameters_json_schema=(
            '{"type":"object","properties":{"query":{"type":"string"}},'
            '"required":["query"],"additionalProperties":false}'
        ),
    )


def test_tool_definition_is_a_data_only_serializable_contract() -> None:
    definition = _definition()

    assert definition.to_dict() == {
        "name": "lookup",
        "description": "Look up public task information.",
        "parameters_json_schema": (
            '{"type":"object","properties":{"query":{"type":"string"}},'
            '"required":["query"],"additionalProperties":false}'
        ),
    }


def test_tool_definition_is_immutable() -> None:
    definition = _definition()

    with pytest.raises(FrozenInstanceError):
        definition.name = "changed"  # type: ignore[misc]
