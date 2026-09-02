"""Public model-call boundary for the fixed Answer Agent."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol

from .input import AnswerAgentInitialMessages
from .response import AnswerAgentResponse
from .tool_protocol.definition import ToolDefinition

if TYPE_CHECKING:
    from .runtime.conversation import AnswerAgentConversation


class AnswerAgentClient(Protocol):
    """One initial, provider-neutral Answer Agent model call."""

    def generate_initial(
        self,
        messages: AnswerAgentInitialMessages,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> AnswerAgentResponse:
        """Generate once from the initial system/user pair and public tools."""

        ...


class AnswerAgentConversationClient(Protocol):
    """Provider-neutral model call over one complete current-subtask history."""

    def generate(
        self,
        conversation: AnswerAgentConversation,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> AnswerAgentResponse:
        """Generate the next response from the complete active conversation."""

        ...


class AnswerAgentInitialTokenCounter(Protocol):
    """Count the compatibility initial-message request."""

    def count_initial_input_tokens(
        self,
        messages: AnswerAgentInitialMessages,
        tool_definitions: tuple[ToolDefinition, ...],
        *,
        model: str,
    ) -> int:
        """Return a count that never underestimates the final model input."""

        ...


class AnswerAgentTokenCounter(AnswerAgentInitialTokenCounter, Protocol):
    """Count a complete current-subtask history and rendered task tools."""

    def count_conversation_input_tokens(
        self,
        conversation: AnswerAgentConversation,
        tool_definitions: tuple[ToolDefinition, ...],
        *,
        model: str,
    ) -> int:
        """Count the complete current-subtask history and rendered tools."""

        ...


class AnswerAgentClientErrorCode(str, Enum):
    INVALID_CONFIG = "INVALID_CONFIG"
    INVALID_INITIAL_MESSAGES = "INVALID_INITIAL_MESSAGES"
    INVALID_CONVERSATION = "INVALID_CONVERSATION"
    INVALID_TOOL_DEFINITIONS = "INVALID_TOOL_DEFINITIONS"
    MISSING_API_KEY = "MISSING_API_KEY"
    INVALID_API_KEY = "INVALID_API_KEY"
    TOKEN_COUNT_ERROR = "TOKEN_COUNT_ERROR"
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"
    RESPONSE_TOO_LARGE = "RESPONSE_TOO_LARGE"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    TRANSPORT_ERROR = "TRANSPORT_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    INVALID_RESPONSE_ENCODING = "INVALID_RESPONSE_ENCODING"
    INVALID_RESPONSE_JSON = "INVALID_RESPONSE_JSON"
    INVALID_RESPONSE_SCHEMA = "INVALID_RESPONSE_SCHEMA"
    INCOMPLETE_COMPLETION = "INCOMPLETE_COMPLETION"
    OUTPUT_TOKEN_LIMIT = "OUTPUT_TOKEN_LIMIT"
    CONTENT_FILTERED = "CONTENT_FILTERED"
    INSUFFICIENT_SYSTEM_RESOURCE = "INSUFFICIENT_SYSTEM_RESOURCE"


class AnswerAgentClientError(RuntimeError):
    """A stable, secret-free failure at the Answer Agent provider boundary."""

    def __init__(self, code: AnswerAgentClientErrorCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


__all__ = [
    "AnswerAgentClient",
    "AnswerAgentClientError",
    "AnswerAgentClientErrorCode",
    "AnswerAgentConversationClient",
    "AnswerAgentInitialTokenCounter",
    "AnswerAgentTokenCounter",
]
