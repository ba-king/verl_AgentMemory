"""One initial Answer Agent call through DeepSeek Chat Completions."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from enum import Enum

from examples.memory_agent_grpo.answer_agent.input import (
    AnswerAgentInitialMessage,
    AnswerAgentInitialMessages,
)
from examples.memory_agent_grpo.answer_agent.model_client import (
    AnswerAgentClientError,
    AnswerAgentClientErrorCode,
    AnswerAgentInitialTokenCounter,
)
from examples.memory_agent_grpo.answer_agent.response import (
    AnswerAgentResponse,
    AnswerAgentToolCall,
)
from examples.memory_agent_grpo.answer_agent.runtime.conversation import (
    AnswerAgentConversation,
    AnswerAgentToolExchange,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.definition import (
    ToolDefinition,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.validation import (
    validate_tool_definition,
)
from examples.memory_agent_grpo.answer_agent.validation import (
    validate_answer_agent_response,
)
from examples.memory_agent_grpo.model_io.http_transport import (
    HttpTransport,
    HttpTransportError,
    HttpTransportErrorCode,
    UrllibHttpTransport,
)

DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MAX_TOOL_COUNT = 128

_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DEEPSEEK_TOOL_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


class DeepSeekToolChoice(str, Enum):
    """Provider-level choice between optional and mandatory tool use."""

    AUTO = "auto"
    REQUIRED = "required"


@dataclass(frozen=True)
class DeepSeekAnswerAgentClientConfig:
    """Non-secret configuration for one initial DeepSeek model call."""

    model: str
    api_key_env: str
    context_window_tokens: int
    timeout_seconds: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 2048
    max_request_bytes: int = 2_097_152
    max_response_bytes: int = 4_194_304
    tool_choice: DeepSeekToolChoice = DeepSeekToolChoice.AUTO


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-standard JSON constant {value!r} is not allowed.")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r} is not allowed.")
        result[key] = value
    return result


class DeepSeekAnswerAgentClient:
    """Synchronous, non-streaming and explicitly non-thinking DeepSeek client.

    Each call receives the complete provider-neutral current-subtask
    conversation. The client serializes history but never executes tools,
    enforces runner policy, retries, or stores conversation state.
    """

    def __init__(
        self,
        config: DeepSeekAnswerAgentClientConfig,
        *,
        token_counter: AnswerAgentInitialTokenCounter,
        transport: HttpTransport | None = None,
    ) -> None:
        self._validate_config(config)
        if not callable(getattr(token_counter, "count_initial_input_tokens", None)):
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_CONFIG,
                "token_counter must provide count_initial_input_tokens(...).",
            )
        self._config = config
        self._token_counter = token_counter
        self._transport = UrllibHttpTransport() if transport is None else transport

    @property
    def config(self) -> DeepSeekAnswerAgentClientConfig:
        return self._config

    def generate_initial(
        self,
        messages: AnswerAgentInitialMessages,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> AnswerAgentResponse:
        serialized_messages = self._validate_and_serialize_messages(messages)
        serialized_tools = self._validate_and_serialize_tools(tool_definitions)
        self._check_initial_token_budget(messages, tool_definitions)

        return self._send_request(
            serialized_messages=serialized_messages,
            serialized_tools=serialized_tools,
            invalid_request_code=AnswerAgentClientErrorCode.INVALID_TOOL_DEFINITIONS,
        )

    def generate(
        self,
        conversation: AnswerAgentConversation,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> AnswerAgentResponse:
        """Generate the next response from a complete current-subtask history."""

        serialized_messages = self._validate_and_serialize_conversation(conversation)
        serialized_tools = self._validate_and_serialize_tools(tool_definitions)
        self._check_conversation_token_budget(conversation, tool_definitions)

        return self._send_request(
            serialized_messages=serialized_messages,
            serialized_tools=serialized_tools,
            invalid_request_code=AnswerAgentClientErrorCode.INVALID_CONVERSATION,
        )

    def _send_request(
        self,
        *,
        serialized_messages: list[dict[str, object]],
        serialized_tools: list[dict[str, object]],
        invalid_request_code: AnswerAgentClientErrorCode,
    ) -> AnswerAgentResponse:

        payload: dict[str, object] = {
            "model": self._config.model,
            "messages": serialized_messages,
            "thinking": {"type": "disabled"},
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": False,
        }
        if serialized_tools:
            payload["tools"] = serialized_tools
            payload["tool_choice"] = self._config.tool_choice.value
        elif self._config.tool_choice is DeepSeekToolChoice.REQUIRED:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_TOOL_DEFINITIONS,
                "Required DeepSeek tool choice needs at least one tool definition.",
            )

        try:
            request_body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (UnicodeEncodeError, ValueError, OverflowError, RecursionError):
            raise AnswerAgentClientError(
                invalid_request_code,
                "Answer Agent request cannot be serialized as finite UTF-8 JSON.",
            ) from None
        if len(request_body) > self._config.max_request_bytes:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.REQUEST_TOO_LARGE,
                f"DeepSeek request exceeds {self._config.max_request_bytes} bytes.",
            )

        api_key = self._read_api_key()
        try:
            transport_result = self._transport.post(
                url=DEEPSEEK_CHAT_COMPLETIONS_URL,
                body=request_body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout_seconds=self._config.timeout_seconds,
                max_response_bytes=self._config.max_response_bytes,
            )
        except HttpTransportError as exc:
            if exc.code is HttpTransportErrorCode.RESPONSE_TOO_LARGE:
                error_code = AnswerAgentClientErrorCode.RESPONSE_TOO_LARGE
            elif exc.code is HttpTransportErrorCode.REQUEST_TIMEOUT:
                error_code = AnswerAgentClientErrorCode.REQUEST_TIMEOUT
            else:
                error_code = AnswerAgentClientErrorCode.TRANSPORT_ERROR
            raise AnswerAgentClientError(error_code, "DeepSeek transport failed.") from None
        except Exception:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.TRANSPORT_ERROR,
                "DeepSeek transport failed.",
            ) from None

        status_code, response_body = self._validate_transport_result(transport_result)
        if not 200 <= status_code < 300:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.HTTP_ERROR,
                f"DeepSeek returned HTTP {status_code}.",
            )
        if len(response_body) > self._config.max_response_bytes:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.RESPONSE_TOO_LARGE,
                f"DeepSeek response exceeds {self._config.max_response_bytes} bytes.",
            )
        return self._parse_response(response_body)

    def _check_initial_token_budget(
        self,
        messages: AnswerAgentInitialMessages,
        tool_definitions: tuple[ToolDefinition, ...],
    ) -> None:
        try:
            token_count = self._token_counter.count_initial_input_tokens(
                messages,
                tool_definitions,
                model=self._config.model,
            )
        except Exception:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.TOKEN_COUNT_ERROR,
                "Answer Agent input token counting failed.",
            ) from None
        if type(token_count) is not int or token_count < 0:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.TOKEN_COUNT_ERROR,
                "Answer Agent token counter returned an invalid count.",
            )
        if token_count + self._config.max_tokens > self._config.context_window_tokens:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.REQUEST_TOO_LARGE,
                "Answer Agent input plus output allowance exceeds the configured context window.",
            )

    def _check_conversation_token_budget(
        self,
        conversation: AnswerAgentConversation,
        tool_definitions: tuple[ToolDefinition, ...],
    ) -> None:
        counter = getattr(
            self._token_counter,
            "count_conversation_input_tokens",
            None,
        )
        if not callable(counter):
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_CONFIG,
                "Continuation requires count_conversation_input_tokens(...).",
            )
        try:
            token_count = counter(
                conversation,
                tool_definitions,
                model=self._config.model,
            )
        except Exception:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.TOKEN_COUNT_ERROR,
                "Answer Agent input token counting failed.",
            ) from None
        if type(token_count) is not int or token_count < 0:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.TOKEN_COUNT_ERROR,
                "Answer Agent token counter returned an invalid count.",
            )
        if token_count + self._config.max_tokens > self._config.context_window_tokens:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.REQUEST_TOO_LARGE,
                "Answer Agent input plus output allowance exceeds the configured context window.",
            )

    @staticmethod
    def _validate_config(config: object) -> None:
        if not isinstance(config, DeepSeekAnswerAgentClientConfig):
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_CONFIG,
                "config must be a DeepSeekAnswerAgentClientConfig.",
            )
        if not isinstance(config.model, str) or not config.model.strip():
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_CONFIG,
                "model must be a non-empty string.",
            )
        if config.model != config.model.strip() or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in config.model
        ):
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_CONFIG,
                "model must not contain surrounding whitespace or control characters.",
            )
        try:
            config.model.encode("utf-8")
        except UnicodeEncodeError:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_CONFIG,
                "model must be valid UTF-8 text.",
            ) from None
        if not isinstance(config.api_key_env, str) or _ENV_NAME_PATTERN.fullmatch(config.api_key_env) is None:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_CONFIG,
                "api_key_env must be a valid environment-variable name.",
            )
        timeout_seconds = _finite_number(config.timeout_seconds)
        if timeout_seconds is None or not 0 < timeout_seconds <= 3600:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_CONFIG,
                "timeout_seconds must be a finite number between 0 and 3600.",
            )
        temperature = _finite_number(config.temperature)
        if temperature is None or not 0.0 <= temperature <= 2.0:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_CONFIG,
                "temperature must be a finite number between 0 and 2.",
            )
        if isinstance(config.max_tokens, bool) or not isinstance(config.max_tokens, int) or config.max_tokens < 1:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_CONFIG,
                "max_tokens must be a positive integer.",
            )
        if (
            isinstance(config.context_window_tokens, bool)
            or not isinstance(config.context_window_tokens, int)
            or config.context_window_tokens < config.max_tokens
        ):
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_CONFIG,
                "context_window_tokens must be an integer at least as large as max_tokens.",
            )
        if type(config.tool_choice) is not DeepSeekToolChoice:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_CONFIG,
                "tool_choice must be a DeepSeekToolChoice.",
            )
        for field_name, value in (
            ("max_request_bytes", config.max_request_bytes),
            ("max_response_bytes", config.max_response_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AnswerAgentClientError(
                    AnswerAgentClientErrorCode.INVALID_CONFIG,
                    f"{field_name} must be a positive integer.",
                )

    @staticmethod
    def _validate_and_serialize_messages(
        messages: object,
        *,
        error_code: AnswerAgentClientErrorCode = (AnswerAgentClientErrorCode.INVALID_INITIAL_MESSAGES),
    ) -> list[dict[str, object]]:
        if not isinstance(messages, tuple) or len(messages) != 2:
            raise AnswerAgentClientError(
                error_code,
                "messages must contain exactly one system message and one user message.",
            )
        serialized: list[dict[str, object]] = []
        for position, (message, expected_role) in enumerate(zip(messages, ("system", "user"), strict=True)):
            if not isinstance(message, AnswerAgentInitialMessage):
                raise AnswerAgentClientError(
                    error_code,
                    f"messages[{position}] must be an AnswerAgentInitialMessage.",
                )
            if message.role != expected_role:
                raise AnswerAgentClientError(
                    error_code,
                    f"messages[{position}] must have role {expected_role!r}.",
                )
            if not isinstance(message.content, str) or not message.content.strip():
                raise AnswerAgentClientError(
                    error_code,
                    f"messages[{position}].content must be a non-empty string.",
                )
            try:
                message.content.encode("utf-8")
            except UnicodeEncodeError:
                raise AnswerAgentClientError(
                    error_code,
                    f"messages[{position}].content must be valid UTF-8 text.",
                ) from None
            serialized.append(message.to_dict())
        return serialized

    @classmethod
    def _validate_and_serialize_conversation(
        cls,
        conversation: object,
    ) -> list[dict[str, object]]:
        if type(conversation) is not AnswerAgentConversation:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_CONVERSATION,
                "conversation must be an AnswerAgentConversation.",
            )
        try:
            verified_exchanges = tuple(
                AnswerAgentToolExchange(
                    assistant_response=exchange.assistant_response,
                    tool_observations=exchange.tool_observations,
                )
                for exchange in conversation.tool_exchanges
            )
            verified = AnswerAgentConversation(
                initial_messages=conversation.initial_messages,
                tool_exchanges=verified_exchanges,
            )
        except (AttributeError, TypeError, ValueError):
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_CONVERSATION,
                "conversation failed provider-neutral validation.",
            ) from None

        serialized = cls._validate_and_serialize_messages(
            verified.initial_messages,
            error_code=AnswerAgentClientErrorCode.INVALID_CONVERSATION,
        )
        for exchange in verified.tool_exchanges:
            response = exchange.assistant_response
            serialized.append(
                {
                    "role": "assistant",
                    # Match MemoryArena's continuation semantics: text emitted
                    # alongside tool calls remains available in the internal
                    # trace, but only the structured calls are replayed to the
                    # model. Replaying model-authored scratch text can amplify
                    # it across tool turns and cause runaway final generation.
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments_json,
                            },
                        }
                        for tool_call in response.tool_calls
                    ],
                }
            )
            serialized.extend(
                {
                    "role": "tool",
                    "tool_call_id": observation.tool_call_id,
                    "content": observation.content,
                }
                for observation in exchange.tool_observations
            )
        return serialized

    @staticmethod
    def _validate_and_serialize_tools(
        definitions: object,
    ) -> list[dict[str, object]]:
        if not isinstance(definitions, tuple):
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_TOOL_DEFINITIONS,
                "tool_definitions must be a tuple.",
            )
        if len(definitions) > DEEPSEEK_MAX_TOOL_COUNT:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_TOOL_DEFINITIONS,
                f"DeepSeek accepts at most {DEEPSEEK_MAX_TOOL_COUNT} tools.",
            )

        seen_names: set[str] = set()
        serialized: list[dict[str, object]] = []
        for position, definition in enumerate(definitions):
            validation = validate_tool_definition(definition)
            if not validation.is_valid:
                code = "UNKNOWN" if validation.error_code is None else validation.error_code.value
                raise AnswerAgentClientError(
                    AnswerAgentClientErrorCode.INVALID_TOOL_DEFINITIONS,
                    f"tool_definitions[{position}] is invalid ({code}).",
                )
            assert isinstance(definition, ToolDefinition)
            if _DEEPSEEK_TOOL_NAME_PATTERN.fullmatch(definition.name) is None:
                raise AnswerAgentClientError(
                    AnswerAgentClientErrorCode.INVALID_TOOL_DEFINITIONS,
                    f"tool_definitions[{position}].name is not supported by DeepSeek.",
                )
            if definition.name in seen_names:
                raise AnswerAgentClientError(
                    AnswerAgentClientErrorCode.INVALID_TOOL_DEFINITIONS,
                    f"Tool name {definition.name!r} appears more than once.",
                )
            seen_names.add(definition.name)
            try:
                parameters = json.loads(
                    definition.parameters_json_schema,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
                json.dumps(
                    parameters,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            except (
                UnicodeEncodeError,
                TypeError,
                ValueError,
                OverflowError,
                RecursionError,
            ):
                raise AnswerAgentClientError(
                    AnswerAgentClientErrorCode.INVALID_TOOL_DEFINITIONS,
                    f"tool_definitions[{position}] cannot be rendered for DeepSeek.",
                ) from None
            if not isinstance(parameters, dict):
                raise AnswerAgentClientError(
                    AnswerAgentClientErrorCode.INVALID_TOOL_DEFINITIONS,
                    f"tool_definitions[{position}] parameters must be an object.",
                )
            serialized.append(
                {
                    "type": "function",
                    "function": {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": parameters,
                    },
                }
            )
        return serialized

    def _read_api_key(self) -> str:
        api_key = os.environ.get(self._config.api_key_env)
        if api_key is None or not api_key:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.MISSING_API_KEY,
                f"Environment variable {self._config.api_key_env} is not set.",
            )
        if any(ord(character) < 0x21 or ord(character) > 0x7E for character in api_key):
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_API_KEY,
                f"Environment variable {self._config.api_key_env} is not header-safe.",
            )
        return api_key

    @staticmethod
    def _validate_transport_result(result: object) -> tuple[int, bytes]:
        if type(result) is not tuple or len(result) != 2:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.TRANSPORT_ERROR,
                "DeepSeek transport returned an invalid result.",
            )
        status_code, response_body = result
        if type(status_code) is not int or not 100 <= status_code <= 599:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.TRANSPORT_ERROR,
                "DeepSeek transport returned an invalid HTTP status.",
            )
        if type(response_body) is not bytes:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "DeepSeek response body must be bytes.",
            )
        return status_code, response_body

    @staticmethod
    def _parse_response(response_body: bytes) -> AnswerAgentResponse:
        try:
            response_text = response_body.decode("utf-8")
        except UnicodeDecodeError:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_RESPONSE_ENCODING,
                "DeepSeek response is not valid UTF-8.",
            ) from None
        try:
            payload = json.loads(
                response_text,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (json.JSONDecodeError, ValueError, OverflowError, RecursionError):
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_RESPONSE_JSON,
                "DeepSeek response is not strict JSON.",
            ) from None
        if not isinstance(payload, dict):
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "DeepSeek response must be one JSON object.",
            )
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "DeepSeek response must contain exactly one choice object.",
            )
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str):
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "DeepSeek choice must contain a string finish_reason.",
            )
        incomplete_error_code = {
            "length": AnswerAgentClientErrorCode.OUTPUT_TOKEN_LIMIT,
            "content_filter": AnswerAgentClientErrorCode.CONTENT_FILTERED,
            "insufficient_system_resource": (AnswerAgentClientErrorCode.INSUFFICIENT_SYSTEM_RESOURCE),
        }.get(finish_reason)
        if incomplete_error_code is not None:
            raise AnswerAgentClientError(
                incomplete_error_code,
                "DeepSeek did not complete the assistant turn normally.",
            )
        if finish_reason not in {"stop", "tool_calls"}:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "DeepSeek choice has an unsupported finish_reason.",
            )

        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "DeepSeek choice must contain one assistant message object.",
            )
        reasoning_content = message.get("reasoning_content")
        if reasoning_content is not None and reasoning_content != "":
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "Non-thinking DeepSeek response unexpectedly contains reasoning_content.",
            )

        raw_content = message.get("content")
        if raw_content is not None and not isinstance(raw_content, str):
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "DeepSeek assistant content must be a string or null.",
            )
        text = None if raw_content in {None, ""} else raw_content

        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls is None:
            raw_tool_calls = []
        if not isinstance(raw_tool_calls, list):
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "DeepSeek assistant tool_calls must be a list or null.",
            )
        tool_calls: list[AnswerAgentToolCall] = []
        for position, raw_tool_call in enumerate(raw_tool_calls):
            if not isinstance(raw_tool_call, dict) or raw_tool_call.get("type") != "function":
                raise AnswerAgentClientError(
                    AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                    f"DeepSeek tool_calls[{position}] must be a function call object.",
                )
            raw_function = raw_tool_call.get("function")
            if not isinstance(raw_function, dict):
                raise AnswerAgentClientError(
                    AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                    f"DeepSeek tool_calls[{position}].function must be an object.",
                )
            tool_call_id = raw_tool_call.get("id")
            name = raw_function.get("name")
            arguments = raw_function.get("arguments")
            if not isinstance(tool_call_id, str) or not isinstance(name, str) or not isinstance(arguments, str):
                raise AnswerAgentClientError(
                    AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                    f"DeepSeek tool_calls[{position}] has invalid id, name, or arguments.",
                )
            tool_calls.append(
                AnswerAgentToolCall(
                    tool_call_id=tool_call_id,
                    name=name,
                    arguments_json=arguments,
                )
            )

        if finish_reason == "stop" and tool_calls:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "DeepSeek stop completion must not contain tool calls.",
            )
        if finish_reason == "tool_calls" and not tool_calls:
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "DeepSeek tool_calls completion must contain at least one tool call.",
            )

        response = AnswerAgentResponse(text=text, tool_calls=tuple(tool_calls))
        validation = validate_answer_agent_response(response)
        if not validation.is_valid:
            code = "UNKNOWN" if validation.error_code is None else validation.error_code.value
            raise AnswerAgentClientError(
                AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                f"Normalized Answer Agent response is invalid ({code}).",
            )
        return response


__all__ = [
    "DEEPSEEK_CHAT_COMPLETIONS_URL",
    "DEEPSEEK_MAX_TOOL_COUNT",
    "DeepSeekAnswerAgentClient",
    "DeepSeekAnswerAgentClientConfig",
    "DeepSeekToolChoice",
]
