from dataclasses import FrozenInstanceError

import pytest

from examples.memory_agent_grpo.answer_agent.response import (
    AnswerAgentResponse,
    AnswerAgentToolCall,
)


def _tool_call() -> AnswerAgentToolCall:
    return AnswerAgentToolCall(
        tool_call_id="call-1",
        name="Search",
        arguments_json='{"query":"quiet hotel"}',
    )


def test_text_only_response_serializes_without_interpretation() -> None:
    response = AnswerAgentResponse(text="final answer")

    assert response.has_tool_calls is False
    assert response.to_dict() == {"text": "final answer", "tool_calls": []}


def test_tool_response_preserves_text_and_structured_calls() -> None:
    tool_call = _tool_call()
    response = AnswerAgentResponse(text="checking", tool_calls=(tool_call,))

    assert response.has_tool_calls is True
    assert response.to_dict() == {
        "text": "checking",
        "tool_calls": [
            {
                "tool_call_id": "call-1",
                "name": "Search",
                "arguments_json": '{"query":"quiet hotel"}',
            }
        ],
    }


def test_multiple_tool_calls_preserve_model_order() -> None:
    first = _tool_call()
    second = AnswerAgentToolCall(
        tool_call_id="call-2",
        name="Lookup",
        arguments_json='{"filters":{"city":"北京"}}',
    )
    response = AnswerAgentResponse(tool_calls=(first, second))

    assert [call["tool_call_id"] for call in response.to_dict()["tool_calls"]] == [
        "call-1",
        "call-2",
    ]


def test_response_types_are_immutable() -> None:
    tool_call = _tool_call()
    response = AnswerAgentResponse(tool_calls=(tool_call,))

    with pytest.raises(FrozenInstanceError):
        tool_call.name = "Changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        response.text = "changed"  # type: ignore[misc]
