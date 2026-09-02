from __future__ import annotations

import json
from dataclasses import replace
from typing import Mapping

import pytest

from examples.memory_agent_grpo.answer_agent.input import (
    AnswerAgentInitialMessage,
    AnswerAgentInitialMessages,
)
from examples.memory_agent_grpo.answer_agent.providers.deepseek import (
    DeepSeekAnswerAgentClient,
    DeepSeekAnswerAgentClientConfig,
)
from examples.memory_agent_grpo.answer_agent.runtime.conversation import (
    AnswerAgentConversation,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.definition import ToolDefinition

KEY_ENV = "TEST_ANSWER_AGENT_API_KEY"
SECRET = "test-answer-agent-secret"
_MISSING = object()


def config(**changes: object) -> DeepSeekAnswerAgentClientConfig:
    value = DeepSeekAnswerAgentClientConfig(
        model="deepseek-test-model",
        api_key_env=KEY_ENV,
        context_window_tokens=4096,
        timeout_seconds=12.5,
        temperature=0.0,
        max_tokens=321,
    )
    return replace(value, **changes)


def messages(
    system_content: str = "System rules with café spacing.  ",
    user_content: str = "Current request 北京 🚆\n",
) -> AnswerAgentInitialMessages:
    return (
        AnswerAgentInitialMessage(role="system", content=system_content),
        AnswerAgentInitialMessage(role="user", content=user_content),
    )


def definition(
    name: str = "lookup",
    *,
    description: str = "Look up public task information.",
    parameters_json_schema: str = (
        '{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}'
    ),
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        parameters_json_schema=parameters_json_schema,
    )


def tool_call(
    tool_call_id: object = "call-1",
    *,
    name: object = "lookup",
    arguments: object = '{"query":"hotel"}',
    call_type: object = "function",
) -> dict[str, object]:
    return {
        "id": tool_call_id,
        "type": call_type,
        "function": {"name": name, "arguments": arguments},
    }


def response_body(
    *,
    content: object = "final answer",
    finish_reason: object = "stop",
    tool_calls: object = _MISSING,
    reasoning_content: object = _MISSING,
    role: object = "assistant",
) -> bytes:
    message: dict[str, object] = {"role": role, "content": content}
    if tool_calls is not _MISSING:
        message["tool_calls"] = tool_calls
    if reasoning_content is not _MISSING:
        message["reasoning_content"] = reasoning_content
    return json.dumps(
        {
            "id": "completion-1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": message,
                }
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")


class RecordingTransport:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def post(
        self,
        *,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> object:
        self.calls.append(
            {
                "url": url,
                "body": body,
                "headers": dict(headers),
                "timeout_seconds": timeout_seconds,
                "max_response_bytes": max_response_bytes,
            }
        )
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FixedTokenCounter:
    def __init__(self, result: object = 100) -> None:
        self.result = result
        self.calls: list[tuple[AnswerAgentInitialMessages, tuple[ToolDefinition, ...], str]] = []
        self.conversation_calls: list[tuple[AnswerAgentConversation, tuple[ToolDefinition, ...], str]] = []

    def count_initial_input_tokens(
        self,
        initial_messages: AnswerAgentInitialMessages,
        tool_definitions: tuple[ToolDefinition, ...],
        *,
        model: str,
    ) -> object:
        self.calls.append((initial_messages, tool_definitions, model))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def count_conversation_input_tokens(
        self,
        conversation: AnswerAgentConversation,
        tool_definitions: tuple[ToolDefinition, ...],
        *,
        model: str,
    ) -> object:
        self.conversation_calls.append((conversation, tool_definitions, model))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    body: bytes | None = None,
    status_code: int = 200,
    client_config: DeepSeekAnswerAgentClientConfig | None = None,
    token_counter: FixedTokenCounter | None = None,
    set_key: bool = True,
) -> tuple[DeepSeekAnswerAgentClient, RecordingTransport, FixedTokenCounter]:
    if set_key:
        monkeypatch.setenv(KEY_ENV, SECRET)
    counter = FixedTokenCounter() if token_counter is None else token_counter
    transport = RecordingTransport((status_code, response_body() if body is None else body))
    return (
        DeepSeekAnswerAgentClient(
            config() if client_config is None else client_config,
            token_counter=counter,  # type: ignore[arg-type]
            transport=transport,  # type: ignore[arg-type]
        ),
        transport,
        counter,
    )
