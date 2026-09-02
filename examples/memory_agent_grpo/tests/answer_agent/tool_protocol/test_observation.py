from dataclasses import FrozenInstanceError

import pytest

from examples.memory_agent_grpo.answer_agent.tool_protocol.observation import (
    ToolObservation,
    ToolObservationStatus,
)


@pytest.mark.parametrize(
    "status",
    [ToolObservationStatus.SUCCESS, ToolObservationStatus.ERROR],
)
def test_tool_observation_serializes_public_result(status: ToolObservationStatus) -> None:
    observation = ToolObservation(
        tool_call_id="call-1",
        tool_name="lookup",
        status=status,
        content="Public adapter-filtered result.",
    )

    assert observation.to_dict() == {
        "tool_call_id": "call-1",
        "tool_name": "lookup",
        "status": status.value,
        "content": "Public adapter-filtered result.",
    }


def test_tool_observation_is_immutable() -> None:
    observation = ToolObservation(
        tool_call_id="call-1",
        tool_name="lookup",
        status=ToolObservationStatus.SUCCESS,
        content="result",
    )

    with pytest.raises(FrozenInstanceError):
        observation.content = "changed"  # type: ignore[misc]
