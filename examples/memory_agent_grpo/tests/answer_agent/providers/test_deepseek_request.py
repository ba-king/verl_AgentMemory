import json
import os

import pytest

from examples.memory_agent_grpo.answer_agent.model_client import (
    AnswerAgentClientError,
    AnswerAgentClientErrorCode,
)
from examples.memory_agent_grpo.answer_agent.providers.deepseek import (
    DEEPSEEK_CHAT_COMPLETIONS_URL,
    DeepSeekToolChoice,
)

from ._support import (
    KEY_ENV,
    SECRET,
    client,
    config,
    definition,
    messages,
    response_body,
    tool_call,
)


def test_request_preserves_initial_messages_and_renders_tools(
    monkeypatch,
) -> None:
    initial_messages = messages()
    definitions = (
        definition("lookup_one"),
        definition(
            "lookup-two",
            description="Second public lookup.  ",
            parameters_json_schema=('{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}'),
        ),
    )
    provider, transport, counter = client(
        monkeypatch,
        body=response_body(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[tool_call(name="lookup_one")],
        ),
    )

    result = provider.generate_initial(initial_messages, definitions)

    assert result.tool_calls[0].name == "lookup_one"
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"] == DEEPSEEK_CHAT_COMPLETIONS_URL
    assert call["timeout_seconds"] == 12.5
    assert call["headers"] == {
        "Authorization": f"Bearer {SECRET}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    assert SECRET.encode() not in call["body"]
    payload = json.loads(call["body"])
    assert payload == {
        "model": "deepseek-test-model",
        "messages": [message.to_dict() for message in initial_messages],
        "thinking": {"type": "disabled"},
        "temperature": 0.0,
        "max_tokens": 321,
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_one",
                    "description": "Look up public task information.",
                    "parameters": json.loads(definitions[0].parameters_json_schema),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup-two",
                    "description": "Second public lookup.  ",
                    "parameters": json.loads(definitions[1].parameters_json_schema),
                },
            },
        ],
        "tool_choice": "auto",
    }
    assert counter.calls == [(initial_messages, definitions, "deepseek-test-model")]


def test_request_without_tools_omits_tool_fields(monkeypatch) -> None:
    initial_messages = messages()
    provider, transport, counter = client(monkeypatch)

    result = provider.generate_initial(initial_messages)

    assert result.text == "final answer"
    payload = json.loads(transport.calls[0]["body"])
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert payload["thinking"] == {"type": "disabled"}
    assert counter.calls == [(initial_messages, (), "deepseek-test-model")]


def test_required_tool_choice_is_rendered_when_tools_are_present(
    monkeypatch,
) -> None:
    provider, transport, _ = client(
        monkeypatch,
        body=response_body(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[tool_call()],
        ),
        client_config=config(tool_choice=DeepSeekToolChoice.REQUIRED),
    )

    provider.generate_initial(messages(), (definition(),))

    payload = json.loads(transport.calls[0]["body"])
    assert payload["tool_choice"] == "required"


def test_required_tool_choice_without_tools_fails_before_key_or_transport(
    monkeypatch,
) -> None:
    provider, transport, _ = client(
        monkeypatch,
        client_config=config(tool_choice=DeepSeekToolChoice.REQUIRED),
        set_key=False,
    )

    with pytest.raises(AnswerAgentClientError) as caught:
        provider.generate_initial(messages())

    assert caught.value.code is AnswerAgentClientErrorCode.INVALID_TOOL_DEFINITIONS
    assert KEY_ENV not in os.environ
    assert transport.calls == []


def test_multiple_provider_calls_are_not_hidden_retries(monkeypatch) -> None:
    provider, transport, _ = client(monkeypatch)

    provider.generate_initial(messages())
    provider.generate_initial(messages("system two", "user two"))

    assert len(transport.calls) == 2
