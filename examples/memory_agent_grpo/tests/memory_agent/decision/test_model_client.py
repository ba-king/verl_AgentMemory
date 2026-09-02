import io
import json
import socket
import urllib.error
from dataclasses import replace
from typing import Mapping

import pytest

from examples.memory_agent_grpo.memory_agent.decision.model_client import (
    MemoryAgentClientConfig,
    MemoryAgentClientError,
    MemoryAgentClientErrorCode,
    MemoryAgentResponseFormat,
    MemoryAgentThinkingMode,
    OpenAICompatibleMemoryAgentClient,
)
from examples.memory_agent_grpo.memory_agent.decision.model_input import (
    MemoryAgentInput,
    MemoryAgentMessage,
)
from examples.memory_agent_grpo.model_io.http_transport import (
    HttpTransportError,
    HttpTransportErrorCode,
)

_KEY_ENV = "TEST_MEMORY_AGENT_API_KEY"
_SECRET = "test-secret-that-must-not-leak"


def _config(**changes: object) -> MemoryAgentClientConfig:
    config = MemoryAgentClientConfig(
        base_url="https://api.example.test/v1/",
        model="memory-model",
        api_key_env=_KEY_ENV,
        context_window_tokens=4096,
        timeout_seconds=12.5,
        temperature=0.0,
        max_tokens=321,
    )
    return replace(config, **changes)


def _model_input(
    system_content: str = "系统规则",
    user_content: str = "task café 🚆",
) -> MemoryAgentInput:
    return (
        MemoryAgentMessage(role="system", content=system_content),
        MemoryAgentMessage(role="user", content=user_content),
    )


def _response(
    content: object,
    *,
    finish_reason: object = "stop",
    role: object = "assistant",
) -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"role": role, "content": content},
                }
            ]
        },
        ensure_ascii=False,
    ).encode("utf-8")


class _RecordingTransport:
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


class _FixedTokenCounter:
    def __init__(self, result: object = 100) -> None:
        self.result = result
        self.calls: list[tuple[MemoryAgentInput, str]] = []

    def count_input_tokens(self, model_input: MemoryAgentInput, *, model: str) -> object:
        self.calls.append((model_input, model))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _RaisingOpener:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[tuple[object, object]] = []

    def open(self, request: object, *, timeout: object) -> object:
        self.calls.append((request, timeout))
        raise self.error


class _RecordingUrlResponse:
    def __init__(self, body: bytes, status_code: int = 200) -> None:
        self.body = body
        self.status_code = status_code
        self.read_sizes: list[int] = []

    def __enter__(self) -> "_RecordingUrlResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status_code

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self.body[:size]


class _ReturningOpener:
    def __init__(self, response: _RecordingUrlResponse) -> None:
        self.response = response
        self.calls: list[tuple[object, object]] = []

    def open(self, request: object, *, timeout: object) -> _RecordingUrlResponse:
        self.calls.append((request, timeout))
        return self.response


def _client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response_body: bytes | None = None,
    status_code: int = 200,
    config: MemoryAgentClientConfig | None = None,
    token_counter: _FixedTokenCounter | None = None,
) -> tuple[OpenAICompatibleMemoryAgentClient, _RecordingTransport]:
    monkeypatch.setenv(_KEY_ENV, _SECRET)
    body = _response('{"memory_action":null}') if response_body is None else response_body
    transport = _RecordingTransport((status_code, body))
    return (
        OpenAICompatibleMemoryAgentClient(
            config or _config(),
            token_counter=token_counter or _FixedTokenCounter(),
            transport=transport,
        ),
        transport,
    )


def _raised_code(error: pytest.ExceptionInfo[MemoryAgentClientError]) -> MemoryAgentClientErrorCode:
    return error.value.code


def test_one_request_preserves_messages_and_uses_frozen_request_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_content = "  {\"memory_action\":null}\n"
    token_counter = _FixedTokenCounter(100)
    client, transport = _client(
        monkeypatch,
        response_body=_response(raw_content),
        token_counter=token_counter,
    )
    model_input = _model_input()

    result = client.generate(model_input)

    assert result == raw_content
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"] == "https://api.example.test/v1/chat/completions"
    assert call["timeout_seconds"] == 12.5
    assert call["max_response_bytes"] == 1_048_576
    assert call["headers"] == {
        "Authorization": f"Bearer {_SECRET}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    request_payload = json.loads(call["body"])
    assert request_payload == {
        "model": "memory-model",
        "messages": [message.to_dict() for message in model_input],
        "temperature": 0.0,
        "max_tokens": 321,
        "stream": False,
    }
    assert _SECRET.encode() not in call["body"]
    assert token_counter.calls == [(model_input, "memory-model")]


def test_json_object_response_format_is_sent_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, transport = _client(
        monkeypatch,
        config=_config(response_format=MemoryAgentResponseFormat.JSON_OBJECT),
    )

    assert client.generate(_model_input()) == '{"memory_action":null}'
    request_payload = json.loads(transport.calls[0]["body"])
    assert request_payload["response_format"] == {"type": "json_object"}


def test_optional_thinking_mode_is_sent_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, transport = _client(
        monkeypatch,
        config=_config(thinking_mode=MemoryAgentThinkingMode.DISABLED),
    )

    assert client.generate(_model_input()) == '{"memory_action":null}'
    request_payload = json.loads(transport.calls[0]["body"])
    assert request_payload["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize(
    "raw_content",
    [
        "",
        "   \n",
        "```json\n{\"memory_action\":null}\n```",
        "not action JSON",
        "前缀 {\"memory_action\":null}",
    ],
)
def test_assistant_content_is_returned_without_strip_parse_or_repair(
    monkeypatch: pytest.MonkeyPatch,
    raw_content: str,
) -> None:
    client, transport = _client(monkeypatch, response_body=_response(raw_content))

    assert client.generate(_model_input()) == raw_content
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "bad_input",
    [
        [],
        (),
        (MemoryAgentMessage("system", "one"),),
        (MemoryAgentMessage("user", "wrong"), MemoryAgentMessage("system", "order")),
        ("system", "user"),
        (MemoryAgentMessage("system", ""), MemoryAgentMessage("user", "content")),
    ],
)
def test_invalid_model_input_fails_before_reading_key_or_calling_transport(
    monkeypatch: pytest.MonkeyPatch,
    bad_input: object,
) -> None:
    monkeypatch.delenv(_KEY_ENV, raising=False)
    transport = _RecordingTransport((200, _response("output")))
    client = OpenAICompatibleMemoryAgentClient(
        _config(),
        token_counter=_FixedTokenCounter(),
        transport=transport,
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(bad_input)  # type: ignore[arg-type]

    assert _raised_code(error) is MemoryAgentClientErrorCode.INVALID_MODEL_INPUT
    assert transport.calls == []


@pytest.mark.parametrize(
    ("api_key", "expected_code"),
    [
        (None, MemoryAgentClientErrorCode.MISSING_API_KEY),
        ("", MemoryAgentClientErrorCode.MISSING_API_KEY),
        (" key", MemoryAgentClientErrorCode.INVALID_API_KEY),
        ("key\nvalue", MemoryAgentClientErrorCode.INVALID_API_KEY),
        ("key\x7fvalue", MemoryAgentClientErrorCode.INVALID_API_KEY),
        ("密钥", MemoryAgentClientErrorCode.INVALID_API_KEY),
    ],
)
def test_missing_or_unsafe_api_key_fails_before_transport(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str | None,
    expected_code: MemoryAgentClientErrorCode,
) -> None:
    if api_key is None:
        monkeypatch.delenv(_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(_KEY_ENV, api_key)
    transport = _RecordingTransport((200, _response("output")))
    client = OpenAICompatibleMemoryAgentClient(
        _config(),
        token_counter=_FixedTokenCounter(),
        transport=transport,
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is expected_code
    assert transport.calls == []
    assert not api_key or api_key not in str(error.value)


@pytest.mark.parametrize(
    "config",
    [
        _config(base_url=""),
        _config(base_url=" https://api.example.test"),
        _config(base_url="https://api.example.test/bad path"),
        _config(base_url="ftp://api.example.test"),
        _config(base_url="https:///v1"),
        _config(base_url="https://user:pass@api.example.test/v1"),
        _config(base_url="https://api.example.test/v1?debug=true"),
        _config(base_url="https://api.example.test/v1?"),
        _config(base_url="https://api.example.test/v1#fragment"),
        _config(base_url="https://api.example.test/v1#"),
        _config(base_url="https://api.example.test/chat/completions"),
        _config(base_url="https://api.example.test:99999/v1"),
        _config(base_url="http://api.example.test/v1"),
        _config(base_url="http://api.example.test/v1", allow_insecure_http=True),
        _config(base_url="http://127.0.0.1:8000/v1"),
        _config(base_url="https://api.例子.test/v1"),
        _config(model=""),
        _config(model=" model"),
        _config(model="model\nname"),
        _config(model="model\ud800name"),
        _config(api_key_env="NOT-AN-ENV"),
        _config(timeout_seconds=0),
        _config(timeout_seconds=3600.1),
        _config(timeout_seconds=10**400),
        _config(timeout_seconds=float("nan")),
        _config(timeout_seconds=True),
        _config(temperature=-0.1),
        _config(temperature=2.1),
        _config(temperature=10**400),
        _config(temperature=float("inf")),
        _config(max_tokens=0),
        _config(max_tokens=True),
        _config(max_tokens=4097),
        _config(context_window_tokens=0),
        _config(context_window_tokens=True),
        _config(max_request_bytes=0),
        _config(max_request_bytes=True),
        _config(max_response_bytes=0),
        _config(max_response_bytes=True),
        _config(allow_insecure_http="yes"),
        _config(response_format="json_object"),
        _config(thinking_mode="disabled"),
    ],
)
def test_invalid_configuration_fails_at_construction(config: MemoryAgentClientConfig) -> None:
    with pytest.raises(MemoryAgentClientError) as error:
        OpenAICompatibleMemoryAgentClient(config, token_counter=_FixedTokenCounter())

    assert _raised_code(error) is MemoryAgentClientErrorCode.INVALID_CONFIG


def test_non_config_object_has_stable_invalid_config_error() -> None:
    with pytest.raises(MemoryAgentClientError) as error:
        OpenAICompatibleMemoryAgentClient(
            object(),  # type: ignore[arg-type]
            token_counter=_FixedTokenCounter(),
        )

    assert _raised_code(error) is MemoryAgentClientErrorCode.INVALID_CONFIG


@pytest.mark.parametrize("base_url", ["http://localhost:8000/v1", "http://127.0.0.1/v1", "http://[::1]:8000/v1"])
def test_insecure_http_requires_explicit_loopback_opt_in(base_url: str) -> None:
    client = OpenAICompatibleMemoryAgentClient(
        _config(base_url=base_url, allow_insecure_http=True),
        token_counter=_FixedTokenCounter(),
        transport=_RecordingTransport((200, _response("output"))),
    )

    assert client.config.base_url == base_url


def test_token_budget_is_checked_before_key_or_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_KEY_ENV, raising=False)
    token_counter = _FixedTokenCounter(4000)
    transport = _RecordingTransport((200, _response("output")))
    client = OpenAICompatibleMemoryAgentClient(
        _config(max_tokens=97, context_window_tokens=4096),
        token_counter=token_counter,
        transport=transport,
    )
    model_input = _model_input()

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(model_input)

    assert _raised_code(error) is MemoryAgentClientErrorCode.REQUEST_TOO_LARGE
    assert token_counter.calls == [(model_input, "memory-model")]
    assert transport.calls == []


def test_token_budget_accepts_exact_context_window_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, transport = _client(
        monkeypatch,
        config=_config(max_tokens=96, context_window_tokens=4096),
        token_counter=_FixedTokenCounter(4000),
    )

    assert client.generate(_model_input()) == '{"memory_action":null}'
    assert len(transport.calls) == 1


@pytest.mark.parametrize("counter_result", [-1, True, 1.5, "100", RuntimeError(f"counter leaked {_SECRET}")])
def test_token_counter_failure_is_stable_and_secret_free(
    monkeypatch: pytest.MonkeyPatch,
    counter_result: object,
) -> None:
    monkeypatch.delenv(_KEY_ENV, raising=False)
    transport = _RecordingTransport((200, _response("output")))
    client = OpenAICompatibleMemoryAgentClient(
        _config(),
        token_counter=_FixedTokenCounter(counter_result),
        transport=transport,
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is MemoryAgentClientErrorCode.TOKEN_COUNT_ERROR
    assert _SECRET not in str(error.value)
    assert transport.calls == []


def test_int_subclass_from_token_counter_cannot_execute_comparison_code() -> None:
    class _HostileInt(int):
        def __lt__(self, other: object) -> bool:
            raise RuntimeError("comparison must not run")

    client = OpenAICompatibleMemoryAgentClient(
        _config(),
        token_counter=_FixedTokenCounter(_HostileInt(1)),
        transport=_RecordingTransport((200, _response("output"))),
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is MemoryAgentClientErrorCode.TOKEN_COUNT_ERROR


def test_missing_token_counter_contract_is_invalid_config() -> None:
    with pytest.raises(MemoryAgentClientError) as error:
        OpenAICompatibleMemoryAgentClient(
            _config(),
            token_counter=object(),  # type: ignore[arg-type]
        )

    assert _raised_code(error) is MemoryAgentClientErrorCode.INVALID_CONFIG


def test_invalid_utf8_message_fails_before_token_counter_or_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_KEY_ENV, raising=False)
    token_counter = _FixedTokenCounter()
    transport = _RecordingTransport((200, _response("output")))
    client = OpenAICompatibleMemoryAgentClient(
        _config(),
        token_counter=token_counter,
        transport=transport,
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input(user_content="bad\ud800text"))

    assert _raised_code(error) is MemoryAgentClientErrorCode.INVALID_MODEL_INPUT
    assert token_counter.calls == []
    assert transport.calls == []


def test_request_byte_limit_fails_before_key_or_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_KEY_ENV, raising=False)
    transport = _RecordingTransport((200, _response("output")))
    client = OpenAICompatibleMemoryAgentClient(
        _config(max_request_bytes=32),
        token_counter=_FixedTokenCounter(),
        transport=transport,
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is MemoryAgentClientErrorCode.REQUEST_TOO_LARGE
    assert transport.calls == []


@pytest.mark.parametrize("status_code", [199, 300, 400, 401, 429, 500])
def test_non_2xx_status_is_one_secret_free_failure(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    echoed_body = f'{{"error":"provider echoed {_SECRET}"}}'.encode()
    client, transport = _client(
        monkeypatch,
        response_body=echoed_body,
        status_code=status_code,
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is MemoryAgentClientErrorCode.HTTP_ERROR
    assert str(status_code) in str(error.value)
    assert _SECRET not in str(error.value)
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("response_body", "expected_code"),
    [
        (b"\xff", MemoryAgentClientErrorCode.INVALID_RESPONSE_ENCODING),
        (b"not json", MemoryAgentClientErrorCode.INVALID_RESPONSE_JSON),
        (
            b'{"extra":NaN,"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x"}}]}',
            MemoryAgentClientErrorCode.INVALID_RESPONSE_JSON,
        ),
        (
            b'{"choices":[],"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x"}}]}',
            MemoryAgentClientErrorCode.INVALID_RESPONSE_JSON,
        ),
        (b"null", MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA),
        (b"{}", MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA),
        (b'{"choices":[]}', MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA),
        (b'{"choices":[{},{}]}', MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA),
        (b'{"choices":[null]}', MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA),
        (_response("output", role="user"), MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA),
        (
            _response("output", finish_reason=None),
            MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
        ),
        (
            _response("output", finish_reason=7),
            MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
        ),
        (_response(None), MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA),
        (_response(7), MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA),
    ],
)
def test_malformed_success_response_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    response_body: bytes,
    expected_code: MemoryAgentClientErrorCode,
) -> None:
    client, transport = _client(monkeypatch, response_body=response_body)

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is expected_code
    assert len(transport.calls) == 1


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls", ""])
def test_non_stop_finish_reason_is_never_treated_as_complete(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: object,
) -> None:
    client, transport = _client(
        monkeypatch,
        response_body=_response('{"memory_action":null}', finish_reason=finish_reason),
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is MemoryAgentClientErrorCode.INCOMPLETE_COMPLETION
    assert len(transport.calls) == 1


def test_finish_reason_is_not_reflected_into_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = _client(
        monkeypatch,
        response_body=_response("output", finish_reason=_SECRET),
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is MemoryAgentClientErrorCode.INCOMPLETE_COMPLETION
    assert _SECRET not in str(error.value)


def test_injected_transport_response_size_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    client, transport = _client(
        monkeypatch,
        response_body=b"x" * 17,
        config=_config(max_response_bytes=16),
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is MemoryAgentClientErrorCode.RESPONSE_TOO_LARGE
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "transport_result",
    [
        None,
        (200,),
        (200, b"body", "extra"),
        (True, _response("output")),
        ("200", _response("output")),
        (99, _response("output")),
        (600, _response("output")),
        (200, "not bytes"),
    ],
)
def test_invalid_transport_result_has_stable_failure(
    monkeypatch: pytest.MonkeyPatch,
    transport_result: object,
) -> None:
    monkeypatch.setenv(_KEY_ENV, _SECRET)
    transport = _RecordingTransport(transport_result)
    client = OpenAICompatibleMemoryAgentClient(
        _config(),
        token_counter=_FixedTokenCounter(),
        transport=transport,
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    expected = (
        MemoryAgentClientErrorCode.INVALID_RESPONSE_SCHEMA
        if transport_result == (200, "not bytes")
        else MemoryAgentClientErrorCode.TRANSPORT_ERROR
    )
    assert _raised_code(error) is expected
    assert len(transport.calls) == 1


def test_unexpected_transport_exception_is_wrapped_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_KEY_ENV, _SECRET)
    transport = _RecordingTransport(RuntimeError(f"transport exposed {_SECRET}"))
    client = OpenAICompatibleMemoryAgentClient(
        _config(),
        token_counter=_FixedTokenCounter(),
        transport=transport,
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is MemoryAgentClientErrorCode.TRANSPORT_ERROR
    assert _SECRET not in str(error.value)
    assert len(transport.calls) == 1


def test_unknown_shared_transport_error_code_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_KEY_ENV, _SECRET)
    transport_error = HttpTransportError(
        HttpTransportErrorCode.TRANSPORT_ERROR,
        "transport failed",
    )
    transport_error.code = []  # type: ignore[assignment]
    client = OpenAICompatibleMemoryAgentClient(
        _config(),
        token_counter=_FixedTokenCounter(),
        transport=_RecordingTransport(transport_error),
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is MemoryAgentClientErrorCode.TRANSPORT_ERROR


def test_existing_client_error_from_transport_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_KEY_ENV, _SECRET)
    timeout_error = MemoryAgentClientError(
        MemoryAgentClientErrorCode.REQUEST_TIMEOUT,
        "timed out",
    )
    transport = _RecordingTransport(timeout_error)
    client = OpenAICompatibleMemoryAgentClient(
        _config(),
        token_counter=_FixedTokenCounter(),
        transport=transport,
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert error.value is timeout_error
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("urlopen_error", "expected_code"),
    [
        (socket.timeout("slow"), MemoryAgentClientErrorCode.REQUEST_TIMEOUT),
        (
            urllib.error.URLError(socket.timeout("wrapped timeout")),
            MemoryAgentClientErrorCode.REQUEST_TIMEOUT,
        ),
        (urllib.error.URLError("offline"), MemoryAgentClientErrorCode.TRANSPORT_ERROR),
    ],
)
def test_standard_transport_maps_network_failure_without_real_request(
    monkeypatch: pytest.MonkeyPatch,
    urlopen_error: Exception,
    expected_code: MemoryAgentClientErrorCode,
) -> None:
    monkeypatch.setenv(_KEY_ENV, _SECRET)

    opener = _RaisingOpener(urlopen_error)
    monkeypatch.setattr("urllib.request.build_opener", lambda *handlers: opener)
    client = OpenAICompatibleMemoryAgentClient(
        _config(),
        token_counter=_FixedTokenCounter(),
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is expected_code
    assert len(opener.calls) == 1


def test_standard_transport_converts_http_error_without_exposing_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_KEY_ENV, _SECRET)

    http_error = urllib.error.HTTPError(
        url="https://api.example.test/v1/chat/completions",
        code=429,
        msg="rate limited",
        hdrs=None,
        fp=io.BytesIO(f"provider echoed {_SECRET}".encode()),
    )
    opener = _RaisingOpener(http_error)
    monkeypatch.setattr("urllib.request.build_opener", lambda *handlers: opener)
    client = OpenAICompatibleMemoryAgentClient(
        _config(),
        token_counter=_FixedTokenCounter(),
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is MemoryAgentClientErrorCode.HTTP_ERROR
    assert _SECRET not in str(error.value)
    assert len(opener.calls) == 1


def test_standard_transport_installs_no_redirect_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_KEY_ENV, _SECRET)
    monkeypatch.setenv("HTTP_PROXY", "http://remote-proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://remote-proxy.example:8080")
    redirect_error = urllib.error.HTTPError(
        url="https://api.example.test/v1/chat/completions",
        code=302,
        msg="redirect",
        hdrs={"Location": "https://attacker.example/collect"},
        fp=io.BytesIO(b"redirect body"),
    )
    opener = _RaisingOpener(redirect_error)
    captured_handlers: list[object] = []

    def _build_opener(*handlers: object) -> _RaisingOpener:
        captured_handlers.extend(handlers)
        return opener

    monkeypatch.setattr("urllib.request.build_opener", _build_opener)
    client = OpenAICompatibleMemoryAgentClient(
        _config(
            base_url="http://127.0.0.1:8000/v1",
            allow_insecure_http=True,
        ),
        token_counter=_FixedTokenCounter(),
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is MemoryAgentClientErrorCode.HTTP_ERROR
    assert len(opener.calls) == 1
    assert len(captured_handlers) == 2
    proxy_handler = next(
        handler for handler in captured_handlers if isinstance(handler, urllib.request.ProxyHandler)
    )
    assert proxy_handler.proxies == {}
    redirect_handler = next(
        handler for handler in captured_handlers if isinstance(handler, urllib.request.HTTPRedirectHandler)
    )
    redirect_request = getattr(redirect_handler, "redirect_request")
    assert redirect_request(None, None, 302, "redirect", {}, "https://attacker.example") is None


def test_standard_transport_reads_only_response_limit_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_KEY_ENV, _SECRET)
    response = _RecordingUrlResponse(b"x" * 17)
    opener = _ReturningOpener(response)
    monkeypatch.setattr("urllib.request.build_opener", lambda *handlers: opener)
    client = OpenAICompatibleMemoryAgentClient(
        _config(max_response_bytes=16),
        token_counter=_FixedTokenCounter(),
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _raised_code(error) is MemoryAgentClientErrorCode.RESPONSE_TOO_LARGE
    assert response.read_sizes == [17]
    assert len(opener.calls) == 1


def test_config_and_error_repr_never_contain_actual_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_KEY_ENV, _SECRET)
    config = _config()
    client = OpenAICompatibleMemoryAgentClient(
        config,
        token_counter=_FixedTokenCounter(),
        transport=_RecordingTransport((500, b"error")),
    )

    with pytest.raises(MemoryAgentClientError) as error:
        client.generate(_model_input())

    assert _SECRET not in repr(config)
    assert _SECRET not in repr(client)
    assert _SECRET not in repr(error.value)
