from dataclasses import FrozenInstanceError

import pytest

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
from examples.memory_agent_grpo.smoke.deepseek_answer_agent_initial_smoke import (
    DEEPSEEK_MODEL,
    SMOKE_TOOL_NAME,
    DeepSeekAnswerConservativeTokenCounter,
    build_deepseek_answer_smoke_client,
    build_deepseek_answer_smoke_input,
    build_deepseek_answer_smoke_tool,
    run_deepseek_answer_initial_smoke,
)


class _FakeClient:
    def __init__(self, response: AnswerAgentResponse) -> None:
        self.response = response
        self.calls: list[tuple[object, object]] = []

    def generate_initial(self, messages, tool_definitions=()) -> AnswerAgentResponse:
        self.calls.append((messages, tool_definitions))
        return self.response


def _tool_response(*, count: int = 1) -> AnswerAgentResponse:
    return AnswerAgentResponse(
        tool_calls=tuple(
            AnswerAgentToolCall(
                tool_call_id=f"call-{index}",
                name=SMOKE_TOOL_NAME,
                arguments_json='{"origin":"ALPHA","destination":"BETA"}',
            )
            for index in range(count)
        )
    )


def test_smoke_calls_once_and_does_not_execute_tool() -> None:
    client = _FakeClient(_tool_response())

    result = run_deepseek_answer_initial_smoke(client)

    assert result.emitted_one_expected_tool_call
    assert result.to_dict()["tool_was_executed"] is False
    assert len(client.calls) == 1
    _, definitions = client.calls[0]
    assert len(definitions) == 1
    assert definitions[0].name == SMOKE_TOOL_NAME


@pytest.mark.parametrize(
    "response",
    [
        AnswerAgentResponse(text="guessed answer"),
        _tool_response(count=2),
        AnswerAgentResponse(
            tool_calls=(
                AnswerAgentToolCall(
                    tool_call_id="call-1",
                    name="different_tool",
                    arguments_json="{}",
                ),
            )
        ),
    ],
)
def test_smoke_requires_exactly_one_expected_tool_call(
    response: AnswerAgentResponse,
) -> None:
    result = run_deepseek_answer_initial_smoke(_FakeClient(response))

    assert not result.emitted_one_expected_tool_call


def test_smoke_counter_covers_messages_and_tool_definition() -> None:
    from examples.memory_agent_grpo.answer_agent.prompts.builder import (
        build_answer_agent_initial_messages,
    )

    messages = build_answer_agent_initial_messages(build_deepseek_answer_smoke_input())
    definition = build_deepseek_answer_smoke_tool()
    counter = DeepSeekAnswerConservativeTokenCounter(
        tokens_per_utf8_byte=2,
        chat_and_tool_template_reserve=10,
    )
    expected_bytes = sum(len(message.content.encode("utf-8")) for message in messages)
    expected_bytes += len(definition.name.encode("utf-8"))
    expected_bytes += len(definition.description.encode("utf-8"))
    expected_bytes += len(definition.parameters_json_schema.encode("utf-8"))

    assert (
        counter.count_initial_input_tokens(
            messages,
            (definition,),
            model=DEEPSEEK_MODEL,
        )
        == 10 + 2 * expected_bytes
    )


def test_conversation_counter_counts_every_batch_item_and_assistant_text_once() -> None:
    from examples.memory_agent_grpo.answer_agent.prompts.builder import (
        build_answer_agent_initial_messages,
    )

    messages = build_answer_agent_initial_messages(build_deepseek_answer_smoke_input())
    definition = build_deepseek_answer_smoke_tool()
    assistant_text = "One assistant batch message."
    first_call = AnswerAgentToolCall(
        tool_call_id="call-1",
        name=SMOKE_TOOL_NAME,
        arguments_json='{ "origin" : "ALPHA", "destination" : "BETA" }',
    )
    second_call = AnswerAgentToolCall(
        tool_call_id="call-2",
        name=SMOKE_TOOL_NAME,
        arguments_json='{"origin":"GAMMA","destination":"DELTA"}',
    )
    observations = (
        ToolObservation(
            tool_call_id="call-1",
            tool_name=SMOKE_TOOL_NAME,
            status=ToolObservationStatus.SUCCESS,
            content="first public observation",
        ),
        ToolObservation(
            tool_call_id="call-2",
            tool_name=SMOKE_TOOL_NAME,
            status=ToolObservationStatus.ERROR,
            content="second public observation",
        ),
    )
    conversation = AnswerAgentConversation(
        initial_messages=messages,
        tool_exchanges=(
            AnswerAgentToolExchange(
                assistant_response=AnswerAgentResponse(
                    text=assistant_text,
                    tool_calls=(first_call, second_call),
                ),
                tool_observations=observations,
            ),
        ),
    )
    counter = DeepSeekAnswerConservativeTokenCounter(
        tokens_per_utf8_byte=2,
        chat_and_tool_template_reserve=10,
    )
    initial_count = counter.count_initial_input_tokens(
        messages,
        (definition,),
        model=DEEPSEEK_MODEL,
    )
    exchange_bytes = len(assistant_text.encode("utf-8"))
    exchange_bytes += sum(
        len(tool_call.tool_call_id.encode("utf-8"))
        + len(tool_call.name.encode("utf-8"))
        + len(tool_call.arguments_json.encode("utf-8"))
        for tool_call in (first_call, second_call)
    )
    exchange_bytes += sum(
        len(observation.tool_call_id.encode("utf-8"))
        + len(observation.content.encode("utf-8"))
        for observation in observations
    )

    assert (
        counter.count_conversation_input_tokens(
            conversation,
            (definition,),
            model=DEEPSEEK_MODEL,
        )
        == initial_count + 2 * exchange_bytes
    )


@pytest.mark.parametrize(
    "counter",
    [
        DeepSeekAnswerConservativeTokenCounter,
    ],
)
def test_smoke_counter_is_immutable(counter) -> None:
    value = counter()

    with pytest.raises(FrozenInstanceError):
        value.tokens_per_utf8_byte = 1  # type: ignore[misc]


def test_smoke_client_builder_does_not_read_key(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_SMOKE_KEY", raising=False)

    client = build_deepseek_answer_smoke_client(api_key_env="MISSING_SMOKE_KEY")

    assert client.config.api_key_env == "MISSING_SMOKE_KEY"
