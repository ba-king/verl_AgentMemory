"""Immutable provider-neutral history for one active Answer Agent subtask."""

from __future__ import annotations

from dataclasses import dataclass

from ..input import AnswerAgentInitialMessage, AnswerAgentInitialMessages
from ..response import AnswerAgentResponse
from ..tool_protocol.observation import ToolObservation
from ..tool_protocol.validation import validate_tool_observation
from ..validation import validate_answer_agent_response


def _require_utf8(value: str, *, field_name: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{field_name} must be valid UTF-8 text.") from None


def _validate_initial_messages(messages: object) -> None:
    if type(messages) is not tuple or len(messages) != 2:
        raise ValueError("initial_messages must contain exactly one system and one user message.")
    for position, (message, expected_role) in enumerate(zip(messages, ("system", "user"), strict=True)):
        if type(message) is not AnswerAgentInitialMessage:
            raise ValueError(f"initial_messages[{position}] must be an AnswerAgentInitialMessage.")
        if message.role != expected_role:
            raise ValueError(f"initial_messages[{position}] must have role {expected_role!r}.")
        if type(message.content) is not str or not message.content.strip():
            raise ValueError(f"initial_messages[{position}].content must be a non-empty string.")
        _require_utf8(
            message.content,
            field_name=f"initial_messages[{position}].content",
        )


@dataclass(frozen=True)
class AnswerAgentToolExchange:
    """One assistant tool-call batch paired with all public observations."""

    assistant_response: AnswerAgentResponse
    tool_observations: tuple[ToolObservation, ...]

    def __post_init__(self) -> None:
        if type(self.assistant_response) is not AnswerAgentResponse:
            raise ValueError("assistant_response must be an AnswerAgentResponse.")
        response_validation = validate_answer_agent_response(self.assistant_response)
        if not response_validation.is_valid:
            code = "UNKNOWN" if response_validation.error_code is None else response_validation.error_code.value
            raise ValueError(f"assistant_response is invalid ({code}).")
        if not self.assistant_response.tool_calls:
            raise ValueError("assistant_response must contain at least one tool call.")
        if type(self.tool_observations) is not tuple:
            raise ValueError("tool_observations must be a tuple.")
        if len(self.tool_observations) != len(self.assistant_response.tool_calls):
            raise ValueError("tool_observations must match every assistant tool call.")

        for position, (tool_call, observation) in enumerate(
            zip(
                self.assistant_response.tool_calls,
                self.tool_observations,
                strict=True,
            )
        ):
            if type(observation) is not ToolObservation:
                raise ValueError(f"tool_observations[{position}] must be a ToolObservation.")
            observation_validation = validate_tool_observation(observation)
            if not observation_validation.is_valid:
                code = (
                    "UNKNOWN" if observation_validation.error_code is None else observation_validation.error_code.value
                )
                raise ValueError(f"tool_observations[{position}] is invalid ({code}).")
            for field_name, value in (
                ("tool_call_id", observation.tool_call_id),
                ("tool_name", observation.tool_name),
                ("content", observation.content),
            ):
                _require_utf8(
                    value,
                    field_name=(f"tool_observations[{position}].{field_name}"),
                )
            if observation.tool_call_id != tool_call.tool_call_id:
                raise ValueError("tool observation IDs must match assistant calls in order.")
            if observation.tool_name != tool_call.name:
                raise ValueError("tool observation names must match assistant calls in order.")

    def to_dict(self) -> dict[str, object]:
        return {
            "assistant_response": self.assistant_response.to_dict(),
            "tool_observations": [observation.to_dict() for observation in self.tool_observations],
        }


@dataclass(frozen=True)
class AnswerAgentConversation:
    """Initial prompt plus completed tool exchanges for one current subtask."""

    initial_messages: AnswerAgentInitialMessages
    tool_exchanges: tuple[AnswerAgentToolExchange, ...] = ()

    def __post_init__(self) -> None:
        _validate_initial_messages(self.initial_messages)
        if type(self.tool_exchanges) is not tuple:
            raise ValueError("tool_exchanges must be a tuple.")

        seen_call_ids: set[str] = set()
        for position, exchange in enumerate(self.tool_exchanges):
            if type(exchange) is not AnswerAgentToolExchange:
                raise ValueError(f"tool_exchanges[{position}] must be an AnswerAgentToolExchange.")
            for tool_call in exchange.assistant_response.tool_calls:
                if tool_call.tool_call_id in seen_call_ids:
                    raise ValueError("tool call IDs must be unique within one subtask.")
                seen_call_ids.add(tool_call.tool_call_id)

    @property
    def model_message_count(self) -> int:
        return 2 + sum(1 + len(exchange.tool_observations) for exchange in self.tool_exchanges)

    def append_tool_exchange(
        self,
        exchange: AnswerAgentToolExchange,
    ) -> AnswerAgentConversation:
        if type(exchange) is not AnswerAgentToolExchange:
            raise ValueError("exchange must be an AnswerAgentToolExchange.")
        return AnswerAgentConversation(
            initial_messages=self.initial_messages,
            tool_exchanges=self.tool_exchanges + (exchange,),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_messages": [message.to_dict() for message in self.initial_messages],
            "tool_exchanges": [exchange.to_dict() for exchange in self.tool_exchanges],
        }


__all__ = ["AnswerAgentConversation", "AnswerAgentToolExchange"]
