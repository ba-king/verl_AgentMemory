import pytest

from examples.memory_agent_grpo.answer_agent.input import AnswerAgentInitialMessage
from examples.memory_agent_grpo.answer_agent.model_client import (
    AnswerAgentClientError,
    AnswerAgentClientErrorCode,
)
from examples.memory_agent_grpo.answer_agent.providers.deepseek import (
    DEEPSEEK_MAX_TOOL_COUNT,
    DeepSeekAnswerAgentClient,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.definition import ToolDefinition
from examples.memory_agent_grpo.model_io.http_transport import (
    HttpTransportError,
    HttpTransportErrorCode,
)

from ._support import (
    KEY_ENV,
    SECRET,
    FixedTokenCounter,
    RecordingTransport,
    client,
    config,
    definition,
    messages,
    response_body,
)


def _raised_code(
    error: pytest.ExceptionInfo[AnswerAgentClientError],
) -> AnswerAgentClientErrorCode:
    return error.value.code


@pytest.mark.parametrize(
    "client_config",
    [
        config(model=""),
        config(model=" model"),
        config(model="bad\nmodel"),
        config(model="bad\ud800model"),
        config(api_key_env="bad-name"),
        config(timeout_seconds=0),
        config(timeout_seconds=float("nan")),
        config(temperature=-0.1),
        config(temperature=float("inf")),
        config(max_tokens=0),
        config(max_tokens=True),
        config(context_window_tokens=320),
        config(context_window_tokens=True),
        config(max_request_bytes=0),
        config(max_response_bytes=False),
        config(tool_choice="required"),
        config(tool_choice=object()),
    ],
)
def test_invalid_configuration_fails_at_construction(client_config) -> None:
    with pytest.raises(AnswerAgentClientError) as error:
        DeepSeekAnswerAgentClient(
            client_config,
            token_counter=FixedTokenCounter(),  # type: ignore[arg-type]
            transport=RecordingTransport((200, response_body())),  # type: ignore[arg-type]
        )

    assert _raised_code(error) is AnswerAgentClientErrorCode.INVALID_CONFIG


def test_non_config_object_and_invalid_counter_fail_at_construction() -> None:
    with pytest.raises(AnswerAgentClientError) as config_error:
        DeepSeekAnswerAgentClient(  # type: ignore[arg-type]
            {},
            token_counter=FixedTokenCounter(),  # type: ignore[arg-type]
        )
    with pytest.raises(AnswerAgentClientError) as counter_error:
        DeepSeekAnswerAgentClient(
            config(),
            token_counter=object(),  # type: ignore[arg-type]
        )

    assert _raised_code(config_error) is AnswerAgentClientErrorCode.INVALID_CONFIG
    assert _raised_code(counter_error) is AnswerAgentClientErrorCode.INVALID_CONFIG


@pytest.mark.parametrize(
    "initial_messages",
    [
        [],
        (),
        (AnswerAgentInitialMessage(role="system", content="system"),),
        (
            AnswerAgentInitialMessage(role="user", content="user"),
            AnswerAgentInitialMessage(role="system", content="system"),
        ),
        ("system", "user"),
        (
            AnswerAgentInitialMessage(role="system", content=" \n"),
            AnswerAgentInitialMessage(role="user", content="user"),
        ),
        (
            AnswerAgentInitialMessage(role="system", content="bad\ud800"),
            AnswerAgentInitialMessage(role="user", content="user"),
        ),
    ],
)
def test_invalid_initial_messages_fail_before_key_or_transport(initial_messages) -> None:
    provider = DeepSeekAnswerAgentClient(
        config(),
        token_counter=FixedTokenCounter(),  # type: ignore[arg-type]
        transport=RecordingTransport((200, response_body())),  # type: ignore[arg-type]
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(initial_messages)  # type: ignore[arg-type]

    assert _raised_code(error) is AnswerAgentClientErrorCode.INVALID_INITIAL_MESSAGES


@pytest.mark.parametrize(
    "definitions",
    [
        [definition()],
        ({"name": "lookup"},),
        (definition("bad name"),),
        (definition("x" * 65),),
        (definition("same"), definition("same")),
        (definition(parameters_json_schema=('{"type":"object","properties":{},"maximum":1e9999999999999999999}')),),
    ],
)
def test_invalid_tool_definitions_fail_before_key_or_transport(definitions) -> None:
    counter = FixedTokenCounter()
    transport = RecordingTransport((200, response_body()))
    provider = DeepSeekAnswerAgentClient(
        config(),
        token_counter=counter,  # type: ignore[arg-type]
        transport=transport,  # type: ignore[arg-type]
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages(), definitions)  # type: ignore[arg-type]

    assert _raised_code(error) is AnswerAgentClientErrorCode.INVALID_TOOL_DEFINITIONS
    assert counter.calls == []
    assert transport.calls == []


def test_more_than_provider_tool_limit_fails_closed() -> None:
    definitions = tuple(definition(f"tool_{index}") for index in range(DEEPSEEK_MAX_TOOL_COUNT + 1))
    provider = DeepSeekAnswerAgentClient(
        config(),
        token_counter=FixedTokenCounter(),  # type: ignore[arg-type]
        transport=RecordingTransport((200, response_body())),  # type: ignore[arg-type]
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages(), definitions)

    assert _raised_code(error) is AnswerAgentClientErrorCode.INVALID_TOOL_DEFINITIONS


@pytest.mark.parametrize(
    "counter_result",
    [RuntimeError("counter secret"), None, True, -1, 1.5],
)
def test_token_counter_failure_is_stable_and_precedes_key_read(counter_result) -> None:
    provider = DeepSeekAnswerAgentClient(
        config(),
        token_counter=FixedTokenCounter(counter_result),  # type: ignore[arg-type]
        transport=RecordingTransport((200, response_body())),  # type: ignore[arg-type]
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages(), (definition(),))

    assert _raised_code(error) is AnswerAgentClientErrorCode.TOKEN_COUNT_ERROR
    assert "counter secret" not in str(error.value)


def test_token_budget_includes_tools_and_accepts_exact_boundary(monkeypatch) -> None:
    definitions = (definition(),)
    initial_messages = messages()
    exact_counter = FixedTokenCounter(4096 - 321)
    provider, transport, counter = client(
        monkeypatch,
        token_counter=exact_counter,
    )

    provider.generate_initial(initial_messages, definitions)

    assert len(transport.calls) == 1
    assert counter.calls == [(initial_messages, definitions, "deepseek-test-model")]


def test_token_budget_overflow_fails_before_key_and_transport() -> None:
    transport = RecordingTransport((200, response_body()))
    provider = DeepSeekAnswerAgentClient(
        config(),
        token_counter=FixedTokenCounter(4096 - 321 + 1),  # type: ignore[arg-type]
        transport=transport,  # type: ignore[arg-type]
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages(), (definition(),))

    assert _raised_code(error) is AnswerAgentClientErrorCode.REQUEST_TOO_LARGE
    assert transport.calls == []


def test_encoded_request_byte_limit_fails_before_key_and_transport() -> None:
    transport = RecordingTransport((200, response_body()))
    provider = DeepSeekAnswerAgentClient(
        config(max_request_bytes=16),
        token_counter=FixedTokenCounter(),  # type: ignore[arg-type]
        transport=transport,  # type: ignore[arg-type]
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages())

    assert _raised_code(error) is AnswerAgentClientErrorCode.REQUEST_TOO_LARGE
    assert transport.calls == []


def test_missing_and_header_unsafe_keys_fail_without_transport(monkeypatch) -> None:
    transport = RecordingTransport((200, response_body()))
    provider = DeepSeekAnswerAgentClient(
        config(),
        token_counter=FixedTokenCounter(),  # type: ignore[arg-type]
        transport=transport,  # type: ignore[arg-type]
    )

    with pytest.raises(AnswerAgentClientError) as missing_error:
        provider.generate_initial(messages())
    monkeypatch.setenv(KEY_ENV, "unsafe key with spaces")
    with pytest.raises(AnswerAgentClientError) as unsafe_error:
        provider.generate_initial(messages())

    assert _raised_code(missing_error) is AnswerAgentClientErrorCode.MISSING_API_KEY
    assert _raised_code(unsafe_error) is AnswerAgentClientErrorCode.INVALID_API_KEY
    assert transport.calls == []


@pytest.mark.parametrize(
    ("transport_error", "expected_code"),
    [
        (
            HttpTransportError(HttpTransportErrorCode.RESPONSE_TOO_LARGE, "large"),
            AnswerAgentClientErrorCode.RESPONSE_TOO_LARGE,
        ),
        (
            HttpTransportError(HttpTransportErrorCode.REQUEST_TIMEOUT, "slow"),
            AnswerAgentClientErrorCode.REQUEST_TIMEOUT,
        ),
        (
            HttpTransportError(HttpTransportErrorCode.TRANSPORT_ERROR, "offline"),
            AnswerAgentClientErrorCode.TRANSPORT_ERROR,
        ),
    ],
)
def test_shared_transport_failures_map_to_answer_client_codes(
    monkeypatch,
    transport_error: HttpTransportError,
    expected_code: AnswerAgentClientErrorCode,
) -> None:
    monkeypatch.setenv(KEY_ENV, SECRET)
    transport = RecordingTransport(transport_error)
    provider = DeepSeekAnswerAgentClient(
        config(),
        token_counter=FixedTokenCounter(),  # type: ignore[arg-type]
        transport=transport,  # type: ignore[arg-type]
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages())

    assert _raised_code(error) is expected_code
    assert len(transport.calls) == 1


def test_unexpected_transport_exception_is_wrapped_without_secret(monkeypatch) -> None:
    monkeypatch.setenv(KEY_ENV, SECRET)
    transport = RecordingTransport(RuntimeError(f"transport leaked {SECRET}"))
    provider = DeepSeekAnswerAgentClient(
        config(),
        token_counter=FixedTokenCounter(),  # type: ignore[arg-type]
        transport=transport,  # type: ignore[arg-type]
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages())

    assert _raised_code(error) is AnswerAgentClientErrorCode.TRANSPORT_ERROR
    assert SECRET not in str(error.value)
    assert len(transport.calls) == 1


def test_injected_domain_error_from_transport_is_sanitized(monkeypatch) -> None:
    monkeypatch.setenv(KEY_ENV, SECRET)
    transport = RecordingTransport(
        AnswerAgentClientError(
            AnswerAgentClientErrorCode.HTTP_ERROR,
            f"injected transport exposed {SECRET}",
        )
    )
    provider = DeepSeekAnswerAgentClient(
        config(),
        token_counter=FixedTokenCounter(),  # type: ignore[arg-type]
        transport=transport,  # type: ignore[arg-type]
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages())

    assert _raised_code(error) is AnswerAgentClientErrorCode.TRANSPORT_ERROR
    assert SECRET not in str(error.value)


def test_unknown_shared_transport_error_code_is_sanitized(monkeypatch) -> None:
    monkeypatch.setenv(KEY_ENV, SECRET)
    transport_error = HttpTransportError(
        HttpTransportErrorCode.TRANSPORT_ERROR,
        "transport failed",
    )
    transport_error.code = []  # type: ignore[assignment]
    transport = RecordingTransport(transport_error)
    provider = DeepSeekAnswerAgentClient(
        config(),
        token_counter=FixedTokenCounter(),  # type: ignore[arg-type]
        transport=transport,  # type: ignore[arg-type]
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages())

    assert _raised_code(error) is AnswerAgentClientErrorCode.TRANSPORT_ERROR


def test_int_subclass_from_token_counter_cannot_execute_comparison_code() -> None:
    class _HostileInt(int):
        def __lt__(self, other: object) -> bool:
            raise RuntimeError("comparison must not run")

    provider = DeepSeekAnswerAgentClient(
        config(),
        token_counter=FixedTokenCounter(_HostileInt(1)),  # type: ignore[arg-type]
        transport=RecordingTransport((200, response_body())),  # type: ignore[arg-type]
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages())

    assert _raised_code(error) is AnswerAgentClientErrorCode.TOKEN_COUNT_ERROR
    assert "comparison must not run" not in str(error.value)


@pytest.mark.parametrize(
    ("transport_result", "expected_code"),
    [
        (None, AnswerAgentClientErrorCode.TRANSPORT_ERROR),
        ((200,), AnswerAgentClientErrorCode.TRANSPORT_ERROR),
        ((True, response_body()), AnswerAgentClientErrorCode.TRANSPORT_ERROR),
        ((600, response_body()), AnswerAgentClientErrorCode.TRANSPORT_ERROR),
        ((200, "not bytes"), AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA),
        ((429, b"provider body"), AnswerAgentClientErrorCode.HTTP_ERROR),
    ],
)
def test_invalid_transport_result_and_http_failure_are_stable(
    monkeypatch,
    transport_result: object,
    expected_code: AnswerAgentClientErrorCode,
) -> None:
    monkeypatch.setenv(KEY_ENV, SECRET)
    transport = RecordingTransport(transport_result)
    provider = DeepSeekAnswerAgentClient(
        config(),
        token_counter=FixedTokenCounter(),  # type: ignore[arg-type]
        transport=transport,  # type: ignore[arg-type]
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages())

    assert _raised_code(error) is expected_code
    assert SECRET not in str(error.value)


def test_hostile_transport_result_subclasses_cannot_run_magic_methods(
    monkeypatch,
) -> None:
    class _HostileTuple(tuple):
        def __len__(self) -> int:
            raise RuntimeError("tuple length must not run")

    class _HostileInt(int):
        def __ge__(self, other: object) -> bool:
            raise RuntimeError("status comparison must not run")

    class _HostileBytes(bytes):
        def __len__(self) -> int:
            raise RuntimeError("body length must not run")

    monkeypatch.setenv(KEY_ENV, SECRET)
    results = (
        _HostileTuple((200, response_body())),
        (_HostileInt(200), response_body()),
        (200, _HostileBytes(response_body())),
    )
    expected_codes = (
        AnswerAgentClientErrorCode.TRANSPORT_ERROR,
        AnswerAgentClientErrorCode.TRANSPORT_ERROR,
        AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA,
    )

    for transport_result, expected_code in zip(results, expected_codes, strict=True):
        provider = DeepSeekAnswerAgentClient(
            config(),
            token_counter=FixedTokenCounter(),  # type: ignore[arg-type]
            transport=RecordingTransport(transport_result),  # type: ignore[arg-type]
        )
        with pytest.raises(AnswerAgentClientError) as error:
            provider.generate_initial(messages())
        assert _raised_code(error) is expected_code


def test_injected_response_size_is_checked_again(monkeypatch) -> None:
    provider, transport, _ = client(
        monkeypatch,
        body=b"x" * 17,
        client_config=config(max_response_bytes=16),
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages())

    assert _raised_code(error) is AnswerAgentClientErrorCode.RESPONSE_TOO_LARGE
    assert len(transport.calls) == 1


def test_config_and_errors_never_store_actual_key(monkeypatch) -> None:
    monkeypatch.setenv(KEY_ENV, SECRET)
    client_config = config()
    provider = DeepSeekAnswerAgentClient(
        client_config,
        token_counter=FixedTokenCounter(),  # type: ignore[arg-type]
        transport=RecordingTransport((500, b"error")),  # type: ignore[arg-type]
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages())

    assert SECRET not in repr(client_config)
    assert SECRET not in repr(provider)
    assert SECRET not in repr(error.value)


def test_tool_definition_objects_remain_adapter_constructed_values() -> None:
    adapter_value = ToolDefinition(
        name="adapter_tool",
        description="Adapter-owned definition.",
        parameters_json_schema='{"type":"object","properties":{}}',
    )

    assert adapter_value == definition(
        "adapter_tool",
        description="Adapter-owned definition.",
        parameters_json_schema='{"type":"object","properties":{}}',
    )
