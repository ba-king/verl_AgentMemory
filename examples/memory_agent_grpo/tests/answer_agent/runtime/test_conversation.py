from dataclasses import FrozenInstanceError

import pytest

from examples.memory_agent_grpo.answer_agent.input import AnswerAgentInitialMessage
from examples.memory_agent_grpo.answer_agent.response import (
    AnswerAgentResponse,
    AnswerAgentToolCall,
)
from examples.memory_agent_grpo.answer_agent.runtime import (
    AnswerAgentConversation,
    AnswerAgentToolExchange,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.observation import (
    ToolObservation,
    ToolObservationStatus,
)


def _initial_messages() -> tuple[
    AnswerAgentInitialMessage,
    AnswerAgentInitialMessage,
]:
    return (
        AnswerAgentInitialMessage(role="system", content="system rules"),
        AnswerAgentInitialMessage(role="user", content="current request"),
    )


def _call(
    call_id: str,
    *,
    name: str = "CitySearch",
    arguments_json: str = '{"state":"New York"}',
) -> AnswerAgentToolCall:
    return AnswerAgentToolCall(
        tool_call_id=call_id,
        name=name,
        arguments_json=arguments_json,
    )


def _response(
    *call_ids: str,
    text: str | None = None,
) -> AnswerAgentResponse:
    return AnswerAgentResponse(
        text=text,
        tool_calls=tuple(_call(call_id) for call_id in call_ids),
    )


def _observation(
    call_id: str,
    *,
    name: str = "CitySearch",
    content: str | None = None,
    status: ToolObservationStatus = ToolObservationStatus.SUCCESS,
) -> ToolObservation:
    return ToolObservation(
        tool_call_id=call_id,
        tool_name=name,
        status=status,
        content=(f"public result for {call_id}" if content is None else content),
    )


def _exchange(
    *call_ids: str,
    text: str | None = None,
) -> AnswerAgentToolExchange:
    return AnswerAgentToolExchange(
        assistant_response=_response(*call_ids, text=text),
        tool_observations=tuple(_observation(call_id) for call_id in call_ids),
    )


def test_empty_conversation_contains_only_initial_pair() -> None:
    conversation = AnswerAgentConversation(_initial_messages())

    assert conversation.model_message_count == 2
    assert conversation.tool_exchanges == ()


def test_append_returns_new_conversation_with_one_batched_exchange() -> None:
    initial = AnswerAgentConversation(_initial_messages())
    exchange = _exchange("call-1", "call-2")

    updated = initial.append_tool_exchange(exchange)

    assert initial.tool_exchanges == ()
    assert updated.tool_exchanges == (exchange,)
    assert updated.model_message_count == 5
    assert updated.to_dict()["tool_exchanges"] == [exchange.to_dict()]
    assert exchange.to_dict()["tool_observations"] == [
        _observation("call-1").to_dict(),
        _observation("call-2").to_dict(),
    ]


def test_assistant_text_is_stored_once_for_the_complete_batch() -> None:
    exchange = _exchange(
        "call-1",
        "call-2",
        text="I will check both independent cities.",
    )

    assert exchange.assistant_response.text == ("I will check both independent cities.")
    assert len(exchange.tool_observations) == 2


def test_model_message_count_uses_one_assistant_plus_every_tool_observation() -> None:
    conversation = AnswerAgentConversation(
        _initial_messages(),
        (
            _exchange("batch-one-a", "batch-one-b"),
            _exchange("batch-two-a"),
        ),
    )

    assert conversation.model_message_count == 7


def test_exchange_requires_at_least_one_tool_call() -> None:
    with pytest.raises(ValueError, match="at least one tool call"):
        AnswerAgentToolExchange(
            assistant_response=AnswerAgentResponse(text="final answer"),
            tool_observations=(),
        )


@pytest.mark.parametrize(
    "observations",
    [
        [],
        (),
        (_observation("call-1"),),
        (
            _observation("call-1"),
            _observation("call-2"),
            _observation("call-3"),
        ),
    ],
)
def test_exchange_requires_one_observation_per_tool_call(
    observations: object,
) -> None:
    with pytest.raises(ValueError, match="tool_observations"):
        AnswerAgentToolExchange(
            assistant_response=_response("call-1", "call-2"),
            tool_observations=observations,  # type: ignore[arg-type]
        )


def test_swapped_observations_are_rejected_even_when_the_set_matches() -> None:
    with pytest.raises(ValueError, match="IDs must match.*in order"):
        AnswerAgentToolExchange(
            assistant_response=_response("call-1", "call-2"),
            tool_observations=(
                _observation("call-2"),
                _observation("call-1"),
            ),
        )


def test_observation_tool_name_must_match_at_the_same_position() -> None:
    with pytest.raises(ValueError, match="names must match.*in order"):
        AnswerAgentToolExchange(
            assistant_response=_response("call-1", "call-2"),
            tool_observations=(
                _observation("call-1"),
                _observation("call-2", name="WrongTool"),
            ),
        )


def test_duplicate_call_ids_are_forbidden_across_every_batch_position() -> None:
    with pytest.raises(ValueError, match="unique"):
        AnswerAgentConversation(
            initial_messages=_initial_messages(),
            tool_exchanges=(
                _exchange("first-a", "shared"),
                _exchange("second-a", "shared"),
            ),
        )


@pytest.mark.parametrize(
    "initial_messages",
    [
        (),
        (
            AnswerAgentInitialMessage(role="user", content="wrong order"),
            AnswerAgentInitialMessage(role="system", content="wrong order"),
        ),
        (
            AnswerAgentInitialMessage(role="system", content="system"),
            AnswerAgentInitialMessage(role="user", content="bad\ud800"),
        ),
    ],
)
def test_invalid_initial_history_fails_closed(initial_messages: object) -> None:
    with pytest.raises(ValueError):
        AnswerAgentConversation(initial_messages)  # type: ignore[arg-type]


def test_invalid_observation_unicode_fails_before_provider() -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        AnswerAgentToolExchange(
            assistant_response=_response("call-1", "call-2"),
            tool_observations=(
                _observation("call-1"),
                _observation("call-2", content="bad\ud800"),
            ),
        )


def test_conversation_exchange_and_observation_tuple_are_immutable() -> None:
    conversation = AnswerAgentConversation(
        _initial_messages(),
        (_exchange("call-1", "call-2"),),
    )

    with pytest.raises(FrozenInstanceError):
        conversation.tool_exchanges = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        conversation.tool_exchanges[0].tool_observations = ()  # type: ignore[misc]
