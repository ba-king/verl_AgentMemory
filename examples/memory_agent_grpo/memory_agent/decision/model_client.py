"""Call one OpenAI-compatible model acting as the Memory Agent."""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol
from urllib.parse import urlsplit

from examples.memory_agent_grpo.memory_agent.decision.model_input import (
    MemoryAgentInput,
    MemoryAgentMessage,
)
from examples.memory_agent_grpo.model_io.http_transport import (
    HttpTransport,
    HttpTransportError,
    HttpTransportErrorCode,
    UrllibHttpTransport,
)


class MemoryAgentClient(Protocol):
    """The model-call boundary required by a future inference runner."""

    def generate(self, model_input: MemoryAgentInput) -> str:
        """Return exactly the raw assistant content from one model call."""

        ...


class MemoryAgentClientErrorCode(str, Enum):
    INVALID_CONFIG = "INVALID_CONFIG"
    INVALID_MODEL_INPUT = "INVALID_MODEL_INPUT"
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


class MemoryAgentResponseFormat(str, Enum):
    """Provider-level syntax constraint for the assistant content."""

    TEXT = "text"
    JSON_OBJECT = "json_object"


class MemoryAgentThinkingMode(str, Enum):
    """Optional provider-level control for reasoning-capable chat models."""

    ENABLED = "enabled"
    DISABLED = "disabled"


class MemoryAgentClientError(RuntimeError):
    """A stable, secret-free failure at the model-call boundary."""

    def __init__(self, code: MemoryAgentClientErrorCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


@dataclass(frozen=True)
class MemoryAgentClientConfig:
    """Non-secret configuration for one OpenAI-compatible endpoint."""

    base_url: str
    model: str
    api_key_env: str
    context_window_tokens: int
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 512
    max_request_bytes: int = 262_144
    max_response_bytes: int = 1_048_576
    allow_insecure_http: bool = False
    response_format: MemoryAgentResponseFormat = MemoryAgentResponseFormat.TEXT
    thinking_mode: MemoryAgentThinkingMode | None = None


class MemoryAgentTokenCounter(Protocol):
    """Count or safely upper-bound final input tokens for one model."""

    def count_input_tokens(self, model_input: MemoryAgentInput, *, model: str) -> int:
        """Return a count that never underestimates the final model input."""

        ...


_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _is_loopback_hostname(hostname: str | None) -> bool:
    if hostname is None:
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-standard JSON constant {value!r} is not allowed.")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r} is not allowed.")
        result[key] = value
    return result


class OpenAICompatibleMemoryAgentClient:
    """Synchronous, single-attempt chat-completion client.

    API keys are read from the configured environment variable at call time.
    The client never parses, strips, repairs, retries, or executes the returned
    assistant content.
    """

    def __init__(
        self,
        config: MemoryAgentClientConfig,
        *,
        token_counter: MemoryAgentTokenCounter,
        transport: HttpTransport | None = None,
    ) -> None:
        self._validate_config(config)
        if not callable(getattr(token_counter, "count_input_tokens", None)):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "token_counter must provide count_input_tokens(...).",
            )
        self._config = config
        self._endpoint = f"{config.base_url.rstrip('/')}/chat/completions"
        self._token_counter = token_counter
        self._transport = UrllibHttpTransport() if transport is None else transport

    @property
    def config(self) -> MemoryAgentClientConfig:
        return self._config

    def generate(self, model_input: MemoryAgentInput) -> str:
        messages = self._validate_and_serialize_input(model_input)
        try:
            input_token_count = self._token_counter.count_input_tokens(
                model_input,
                model=self._config.model,
            )
        except Exception:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.TOKEN_COUNT_ERROR,
                "Memory Agent input token counting failed.",
            ) from None
        if type(input_token_count) is not int or input_token_count < 0:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.TOKEN_COUNT_ERROR,
                "Memory Agent input token counter returned an invalid count.",
            )
        if input_token_count + self._config.max_tokens > self._config.context_window_tokens:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.REQUEST_TOO_LARGE,
                "Memory Agent input plus output allowance exceeds the configured context window.",
            )
        payload = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": False,
        }
        if self._config.response_format is MemoryAgentResponseFormat.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
        if self._config.thinking_mode is not None:
            payload["thinking"] = {"type": self._config.thinking_mode.value}
        try:
            request_body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (UnicodeEncodeError, ValueError, OverflowError):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_MODEL_INPUT,
                "model_input cannot be serialized as UTF-8 JSON.",
            ) from None
        if len(request_body) > self._config.max_request_bytes:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.REQUEST_TOO_LARGE,
                f"Chat completion request exceeds {self._config.max_request_bytes} bytes.",
            )
        api_key = self._read_api_key()
        try:
            transport_result = self._transport.post(
                url=self._endpoint,
                body=request_body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout_seconds=self._config.timeout_seconds,
                max_response_bytes=self._config.max_response_bytes,
            )
        except MemoryAgentClientError:
            raise
        except HttpTransportError as exc:
            if exc.code is HttpTransportErrorCode.RESPONSE_TOO_LARGE:
                error_code = MemoryAgentClientErrorCode.RESPONSE_TOO_LARGE
            elif exc.code is HttpTransportErrorCode.REQUEST_TIMEOUT:
                error_code = MemoryAgentClientErrorCode.REQUEST_TIMEOUT
            else:
                error_code = MemoryAgentClientErrorCode.TRANSPORT_ERROR
            raise MemoryAgentClientError(error_code, "Chat completion transport failed.") from None
        except Exception:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.TRANSPORT_ERROR,
                "Chat completion transport failed.",
            ) from None
        if type(transport_result) is not tuple or len(transport_result) != 2:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.TRANSPORT_ERROR,
                "Chat completion transport returned an invalid result.",
            )
        status_code, response_body = transport_result
        if (
            type(status_code) is not int
            or not 100 <= status_code <= 599
        ):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.TRANSPORT_ERROR,
                "Chat completion transport returned an invalid HTTP status.",
            )
        if not 200 <= status_code < 300:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.HTTP_ERROR,
                f"Chat completion returned HTTP {status_code}.",
            )
        if type(response_body) is not bytes:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "Response body must be bytes.",
            )
        if len(response_body) > self._config.max_response_bytes:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.RESPONSE_TOO_LARGE,
                f"Chat completion response exceeds {self._config.max_response_bytes} bytes.",
            )
        return self._extract_raw_content(response_body)

    @staticmethod
    def _validate_config(config: object) -> None:
        if not isinstance(config, MemoryAgentClientConfig):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "config must be a MemoryAgentClientConfig.",
            )
        if not isinstance(config.base_url, str) or not config.base_url:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "base_url must be a non-empty string.",
            )
        if config.base_url != config.base_url.strip() or any(
            character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
            for character in config.base_url
        ):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "base_url must not contain whitespace or control characters.",
            )
        try:
            config.base_url.encode("ascii")
        except UnicodeEncodeError:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "base_url must use an ASCII URL representation.",
            ) from None
        if "?" in config.base_url or "#" in config.base_url:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "base_url must not contain a query or fragment delimiter.",
            )
        try:
            parsed_url = urlsplit(config.base_url)
            parsed_port = parsed_url.port
        except ValueError as exc:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "base_url is malformed.",
            ) from exc
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or parsed_url.hostname is None
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
            or (parsed_port is not None and not 1 <= parsed_port <= 65535)
        ):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "base_url must be an HTTP(S) origin or API base path without credentials, query, or fragment.",
            )
        if not isinstance(config.allow_insecure_http, bool):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "allow_insecure_http must be a boolean.",
            )
        if type(config.response_format) is not MemoryAgentResponseFormat:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "response_format must be a MemoryAgentResponseFormat.",
            )
        if config.thinking_mode is not None and (
            type(config.thinking_mode) is not MemoryAgentThinkingMode
        ):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "thinking_mode must be a MemoryAgentThinkingMode or None.",
            )
        if parsed_url.scheme == "http" and not config.allow_insecure_http:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "HTTP base_url requires explicit allow_insecure_http=True.",
            )
        if parsed_url.scheme == "http" and not _is_loopback_hostname(parsed_url.hostname):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "Insecure HTTP is restricted to an explicit loopback address.",
            )
        if config.base_url.rstrip("/").endswith("/chat/completions"):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "base_url must not include the /chat/completions endpoint.",
            )
        if not isinstance(config.model, str) or not config.model.strip():
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "model must be a non-empty string.",
            )
        if config.model != config.model.strip() or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in config.model
        ):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "model must not contain surrounding whitespace or control characters.",
            )
        try:
            config.model.encode("utf-8")
        except UnicodeEncodeError:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "model must be valid UTF-8 text.",
            ) from None
        if not isinstance(config.api_key_env, str) or _ENV_NAME_PATTERN.fullmatch(config.api_key_env) is None:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "api_key_env must be a valid environment-variable name.",
            )
        timeout_seconds = _finite_number(config.timeout_seconds)
        if timeout_seconds is None or not 0 < timeout_seconds <= 3600:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "timeout_seconds must be a finite number between 0 and 3600.",
            )
        temperature = _finite_number(config.temperature)
        if temperature is None or not 0.0 <= temperature <= 2.0:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "temperature must be a finite number between 0 and 2.",
            )
        if isinstance(config.max_tokens, bool) or not isinstance(config.max_tokens, int) or config.max_tokens < 1:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "max_tokens must be a positive integer.",
            )
        if (
            isinstance(config.context_window_tokens, bool)
            or not isinstance(config.context_window_tokens, int)
            or config.context_window_tokens < 1
            or config.max_tokens > config.context_window_tokens
        ):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_CONFIG,
                "context_window_tokens must be a positive integer at least as large as max_tokens.",
            )
        for field_name, value in (
            ("max_request_bytes", config.max_request_bytes),
            ("max_response_bytes", config.max_response_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MemoryAgentClientError(
                    MemoryAgentClientErrorCode.INVALID_CONFIG,
                    f"{field_name} must be a positive integer.",
                )

    @staticmethod
    def _validate_and_serialize_input(model_input: object) -> list[dict[str, str]]:
        if not isinstance(model_input, tuple) or len(model_input) != 2:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_MODEL_INPUT,
                "model_input must contain exactly one system message and one user message.",
            )
        expected_roles = ("system", "user")
        messages: list[dict[str, str]] = []
        for position, (message, expected_role) in enumerate(zip(model_input, expected_roles, strict=True)):
            if not isinstance(message, MemoryAgentMessage):
                raise MemoryAgentClientError(
                    MemoryAgentClientErrorCode.INVALID_MODEL_INPUT,
                    f"model_input[{position}] must be a MemoryAgentMessage.",
                )
            if message.role != expected_role:
                raise MemoryAgentClientError(
                    MemoryAgentClientErrorCode.INVALID_MODEL_INPUT,
                    f"model_input[{position}] must have role {expected_role!r}.",
                )
            if not isinstance(message.content, str) or not message.content:
                raise MemoryAgentClientError(
                    MemoryAgentClientErrorCode.INVALID_MODEL_INPUT,
                    f"model_input[{position}].content must be a non-empty string.",
                )
            try:
                message.content.encode("utf-8")
            except UnicodeEncodeError:
                raise MemoryAgentClientError(
                    MemoryAgentClientErrorCode.INVALID_MODEL_INPUT,
                    f"model_input[{position}].content must be valid UTF-8 text.",
                ) from None
            messages.append(message.to_dict())
        return messages

    def _read_api_key(self) -> str:
        api_key = os.environ.get(self._config.api_key_env)
        if api_key is None or not api_key:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.MISSING_API_KEY,
                f"Environment variable {self._config.api_key_env} is not set.",
            )
        if any(ord(character) < 0x21 or ord(character) > 0x7E for character in api_key):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_API_KEY,
                f"Environment variable {self._config.api_key_env} is not header-safe.",
            )
        return api_key

    @staticmethod
    def _extract_raw_content(response_body: bytes) -> str:
        if not isinstance(response_body, bytes):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "Response body must be bytes.",
            )
        try:
            response_text = response_body.decode("utf-8")
        except UnicodeDecodeError:
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_RESPONSE_ENCODING,
                "Chat completion response is not valid UTF-8.",
            ) from None
        try:
            payload = json.loads(
                response_text,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (json.JSONDecodeError, ValueError, RecursionError):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_RESPONSE_JSON,
                "Chat completion response is not valid JSON.",
            ) from None
        if not isinstance(payload, dict):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "Chat completion response must be a JSON object.",
            )
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "Chat completion response must contain exactly one choice object.",
            )
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "The single choice must contain a string finish_reason.",
            )
        if finish_reason != "stop":
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INCOMPLETE_COMPLETION,
                "Chat completion did not finish normally.",
            )
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "The single choice must contain one assistant message object.",
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise MemoryAgentClientError(
                MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
                "Assistant message content must be a string.",
            )
        return content


__all__ = [
    "MemoryAgentClient",
    "MemoryAgentClientConfig",
    "MemoryAgentClientError",
    "MemoryAgentClientErrorCode",
    "MemoryAgentResponseFormat",
    "MemoryAgentThinkingMode",
    "MemoryAgentTokenCounter",
    "OpenAICompatibleMemoryAgentClient",
]
