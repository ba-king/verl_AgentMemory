import json

import pytest

from examples.memory_agent_grpo.answer_agent.model_client import (
    AnswerAgentClientError,
    AnswerAgentClientErrorCode,
)
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

from ._support import (
    RecordingTransport,
    client,
    config,
    definition,
    messages,
    response_body,
)


def _exchange(
    *,
    call_id: str = "call-1",
    name: str = "lookup",
    arguments_json: str = '{"query": "hotel", "city": "Boston"}',
    assistant_text: str | None = "I will look that up.",
    observation_content: str = "Public hotel result",
    observation_status: ToolObservationStatus = ToolObservationStatus.SUCCESS,
) -> AnswerAgentToolExchange:
    return AnswerAgentToolExchange(
        assistant_response=AnswerAgentResponse(
            text=assistant_text,
            tool_calls=(
                AnswerAgentToolCall(
                    tool_call_id=call_id,
                    name=name,
                    arguments_json=arguments_json,
                ),
            ),
        ),
        tool_observations=(
            ToolObservation(
                tool_call_id=call_id,
                tool_name=name,
                status=observation_status,
                content=observation_content,
            ),
        ),
    )


def test_continuation_serializes_one_assistant_batch_then_ordered_tool_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_raw_arguments = '{"query": "hotel", "city": "Boston"}'
    second_raw_arguments = '{ "query" : "restaurant", "city" : "Cambridge" }'
    exchange = AnswerAgentToolExchange(
        assistant_response=AnswerAgentResponse(
            text="I will look up both options.",
            tool_calls=(
                AnswerAgentToolCall(
                    tool_call_id="call-1",
                    name="lookup",
                    arguments_json=first_raw_arguments,
                ),
                AnswerAgentToolCall(
                    tool_call_id="call-2",
                    name="lookup_two",
                    arguments_json=second_raw_arguments,
                ),
            ),
        ),
        tool_observations=(
            ToolObservation(
                tool_call_id="call-1",
                tool_name="lookup",
                status=ToolObservationStatus.SUCCESS,
                content="Public hotel result",
            ),
            ToolObservation(
                tool_call_id="call-2",
                tool_name="lookup_two",
                status=ToolObservationStatus.ERROR,
                content="Public restaurant error",
            ),
        ),
    )
    conversation = AnswerAgentConversation(
        initial_messages=messages(),
        tool_exchanges=(exchange,),
    )
    definitions = (definition(), definition("lookup_two"))
    provider, transport, counter = client(
        monkeypatch,
        body=response_body(content="final plan"),
    )

    result = provider.generate(conversation, definitions)

    assert result.text == "final plan"
    assert exchange.assistant_response.text == "I will look up both options."
    payload = json.loads(transport.calls[0]["body"])
    assert payload["messages"] == [message.to_dict() for message in conversation.initial_messages] + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": first_raw_arguments,
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "lookup_two",
                        "arguments": second_raw_arguments,
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "Public hotel result",
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": "Public restaurant error",
        },
    ]
    assert counter.calls == []
    assert counter.conversation_calls == [(conversation, definitions, "deepseek-test-model")]


def test_multiple_exchanges_keep_strict_pair_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation = AnswerAgentConversation(
        initial_messages=messages(),
        tool_exchanges=(
            _exchange(call_id="call-1"),
            _exchange(
                call_id="call-2",
                name="lookup_two",
                arguments_json='{"query":"restaurant"}',
                assistant_text=None,
                observation_content="Second result",
                observation_status=ToolObservationStatus.ERROR,
            ),
        ),
    )
    provider, transport, _ = client(monkeypatch)

    provider.generate(
        conversation,
        (definition(), definition("lookup_two")),
    )

    payload = json.loads(transport.calls[0]["body"])
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assistant_messages = [message for message in payload["messages"] if message["role"] == "assistant"]
    assert [message["content"] for message in assistant_messages] == [None, None]
    assert conversation.tool_exchanges[0].assistant_response.text == ("I will look that up.")
    assert payload["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-2",
        "content": "Second result",
    }


def test_empty_conversation_uses_generalized_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation = AnswerAgentConversation(messages())
    provider, transport, counter = client(monkeypatch)

    provider.generate(conversation)

    payload = json.loads(transport.calls[0]["body"])
    assert payload["messages"] == [message.to_dict() for message in conversation.initial_messages]
    assert counter.calls == []
    assert counter.conversation_calls == [(conversation, (), "deepseek-test-model")]


def test_invalid_conversation_fails_before_counter_key_or_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, transport, counter = client(monkeypatch, set_key=False)

    with pytest.raises(AnswerAgentClientError) as caught:
        provider.generate(object())  # type: ignore[arg-type]

    assert caught.value.code is AnswerAgentClientErrorCode.INVALID_CONVERSATION
    assert counter.conversation_calls == []
    assert transport.calls == []


def test_tampered_exchange_is_revalidated_by_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _exchange()
    object.__setattr__(
        exchange,
        "tool_observations",
        (
            ToolObservation(
                tool_call_id="mismatched",
                tool_name="lookup",
                status=ToolObservationStatus.SUCCESS,
                content="result",
            ),
        ),
    )
    conversation = AnswerAgentConversation(messages())
    object.__setattr__(conversation, "tool_exchanges", (exchange,))
    provider, transport, counter = client(monkeypatch, set_key=False)

    with pytest.raises(AnswerAgentClientError) as caught:
        provider.generate(conversation, (definition(),))

    assert caught.value.code is AnswerAgentClientErrorCode.INVALID_CONVERSATION
    assert counter.conversation_calls == []
    assert transport.calls == []


def test_conversation_token_counter_failure_is_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, transport, counter = client(monkeypatch, set_key=False)
    counter.result = RuntimeError("private counter detail")

    with pytest.raises(AnswerAgentClientError) as caught:
        provider.generate(AnswerAgentConversation(messages()), (definition(),))

    assert caught.value.code is AnswerAgentClientErrorCode.TOKEN_COUNT_ERROR
    assert "private counter detail" not in str(caught.value)
    assert transport.calls == []


def test_initial_only_counter_remains_compatible_until_continuation_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _InitialOnlyCounter:
        def count_initial_input_tokens(
            self,
            initial_messages: object,
            tool_definitions: object,
            *,
            model: str,
        ) -> int:
            return 10

    from examples.memory_agent_grpo.answer_agent.providers.deepseek import (
        DeepSeekAnswerAgentClient,
    )

    monkeypatch.setenv("TEST_ANSWER_AGENT_API_KEY", "test-answer-agent-secret")
    transport = RecordingTransport((200, response_body()))
    provider = DeepSeekAnswerAgentClient(
        config(),
        token_counter=_InitialOnlyCounter(),
        transport=transport,  # type: ignore[arg-type]
    )

    initial_result = provider.generate_initial(messages())

    assert initial_result.text == "final answer"
    assert len(transport.calls) == 1
    with pytest.raises(AnswerAgentClientError) as caught:
        provider.generate(AnswerAgentConversation(messages()))
    assert caught.value.code is AnswerAgentClientErrorCode.INVALID_CONFIG
    assert len(transport.calls) == 1
