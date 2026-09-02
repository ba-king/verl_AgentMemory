"""Provider-neutral output returned by one Answer Agent model call."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerAgentToolCall:
    """One tool request with provider-independent identity and JSON arguments."""

    tool_call_id: str
    name: str
    arguments_json: str

    def to_dict(self) -> dict[str, str]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "arguments_json": self.arguments_json,
        }


@dataclass(frozen=True)
class AnswerAgentResponse:
    """Text and structured tool calls from one Answer Agent inference.

    Text may accompany tool calls but is not automatically a final answer.
    The task adapter, not this dataset-neutral type, interprets the response.
    """

    text: str | None = None
    tool_calls: tuple[AnswerAgentToolCall, ...] = ()

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_dict(self) -> dict[str, object | None]:
        return {
            "text": self.text,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
        }


__all__ = ["AnswerAgentResponse", "AnswerAgentToolCall"]
