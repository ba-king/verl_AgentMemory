import json

import pytest

from examples.memory_agent_grpo.answer_agent.model_client import (
    AnswerAgentClientError,
    AnswerAgentClientErrorCode,
)

from ._support import client, definition, messages, response_body, tool_call


def _raised_code(
    error: pytest.ExceptionInfo[AnswerAgentClientError],
) -> AnswerAgentClientErrorCode:
    return error.value.code


def _surrogate_response_body(target: str) -> bytes:
    if target == "content":
        payload = json.loads(response_body())
        payload["choices"][0]["message"]["content"] = "bad\ud800text"
    else:
        payload = json.loads(
            response_body(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[tool_call()],
            )
        )
        call = payload["choices"][0]["message"]["tool_calls"][0]
        if target == "id":
            call["id"] = "bad\ud800id"
        elif target == "name":
            call["function"]["name"] = "bad\ud800name"
        else:
            call["function"]["arguments"] = '{"nested":"\ud800"}'
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def test_stop_response_preserves_public_text_exactly(monkeypatch) -> None:
    provider, _, _ = client(
        monkeypatch,
        body=response_body(content="  final 北京 answer\n"),
    )

    response = provider.generate_initial(messages())

    assert response.text == "  final 北京 answer\n"
    assert response.tool_calls == ()


@pytest.mark.parametrize("content", [None, ""])
def test_tool_call_response_normalizes_absent_public_text_to_none(
    monkeypatch,
    content: object,
) -> None:
    provider, _, _ = client(
        monkeypatch,
        body=response_body(
            content=content,
            finish_reason="tool_calls",
            tool_calls=[tool_call(arguments='{"query":"café 北京"}')],
        ),
    )

    response = provider.generate_initial(messages(), (definition(),))

    assert response.text is None
    assert response.tool_calls[0].arguments_json == '{"query":"café 北京"}'


def test_text_plus_tool_call_is_preserved_without_final_interpretation(
    monkeypatch,
) -> None:
    provider, _, _ = client(
        monkeypatch,
        body=response_body(
            content="I will look this up.",
            finish_reason="tool_calls",
            tool_calls=[tool_call()],
        ),
    )

    response = provider.generate_initial(messages(), (definition(),))

    assert response.text == "I will look this up."
    assert response.has_tool_calls


def test_multiple_tool_calls_and_unknown_names_preserve_provider_order(
    monkeypatch,
) -> None:
    provider, _, _ = client(
        monkeypatch,
        body=response_body(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[
                tool_call("call-2", name="unknown_provider_name", arguments="{}"),
                tool_call("call-1", name="lookup", arguments='{"query":"hotel"}'),
            ],
        ),
    )

    response = provider.generate_initial(messages(), (definition(),))

    assert [call.tool_call_id for call in response.tool_calls] == ["call-2", "call-1"]
    assert [call.name for call in response.tool_calls] == [
        "unknown_provider_name",
        "lookup",
    ]


@pytest.mark.parametrize(
    ("finish_reason", "expected_code"),
    [
        ("length", AnswerAgentClientErrorCode.OUTPUT_TOKEN_LIMIT),
        ("content_filter", AnswerAgentClientErrorCode.CONTENT_FILTERED),
        (
            "insufficient_system_resource",
            AnswerAgentClientErrorCode.INSUFFICIENT_SYSTEM_RESOURCE,
        ),
    ],
)
def test_incomplete_finish_reason_has_specific_stable_failure_without_leakage(
    monkeypatch,
    finish_reason: str,
    expected_code: AnswerAgentClientErrorCode,
) -> None:
    provider, _, _ = client(
        monkeypatch,
        body=response_body(finish_reason=finish_reason),
    )

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages())

    assert _raised_code(error) is expected_code
    assert str(error.value) == (f"{expected_code.value}: DeepSeek did not complete the assistant turn normally.")


@pytest.mark.parametrize(
    "body",
    [
        response_body(
            content="text",
            finish_reason="stop",
            tool_calls=[tool_call()],
        ),
        response_body(content="text", finish_reason="tool_calls", tool_calls=[]),
        response_body(content=None, finish_reason="stop"),
        response_body(content="", finish_reason="stop"),
        response_body(content=" \n", finish_reason="stop"),
        response_body(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[tool_call(arguments="not json")],
        ),
        response_body(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[tool_call(arguments='{"x":1,"x":2}')],
        ),
        response_body(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[tool_call(arguments="[]")],
        ),
        response_body(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[tool_call("same"), tool_call("same")],
        ),
        response_body(reasoning_content="private reasoning"),
        response_body(reasoning_content={"unexpected": "object"}),
    ],
)
def test_invalid_normalized_response_fails_closed(monkeypatch, body: bytes) -> None:
    provider, _, _ = client(monkeypatch, body=body)

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages(), (definition(),))

    assert _raised_code(error) is AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA


@pytest.mark.parametrize(
    "body",
    [
        _surrogate_response_body("content"),
        _surrogate_response_body("id"),
        _surrogate_response_body("name"),
        _surrogate_response_body("arguments"),
    ],
)
def test_json_escaped_lone_surrogate_never_enters_normalized_response(
    monkeypatch,
    body: bytes,
) -> None:
    provider, _, _ = client(monkeypatch, body=body)

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages(), (definition(),))

    assert _raised_code(error) is AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA


@pytest.mark.parametrize(
    "body",
    [
        b"\xff",
        b"not json",
        b'{"choices":[],"choices":[]}',
        b'{"choices":NaN}',
    ],
)
def test_invalid_response_encoding_or_json_has_stable_failure(
    monkeypatch,
    body: bytes,
) -> None:
    provider, _, _ = client(monkeypatch, body=body)

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages())

    expected = (
        AnswerAgentClientErrorCode.INVALID_RESPONSE_ENCODING
        if body == b"\xff"
        else AnswerAgentClientErrorCode.INVALID_RESPONSE_JSON
    )
    assert _raised_code(error) is expected


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"choices": []},
        {"choices": [{}, {}]},
        {"choices": [None]},
        {"choices": [{"finish_reason": None, "message": {}}]},
        {
            "choices": [
                {
                    "finish_reason": "unknown",
                    "message": {"role": "assistant", "content": "text"},
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "user", "content": "text"},
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": 7},
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": "not a list",
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call(call_type="custom")],
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "call-1", "type": "function"}],
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call(tool_call_id=7)],
                    },
                }
            ]
        },
    ],
)
def test_invalid_provider_envelope_has_stable_schema_failure(
    monkeypatch,
    payload: object,
) -> None:
    body = json.dumps(payload).encode("utf-8")
    provider, _, _ = client(monkeypatch, body=body)

    with pytest.raises(AnswerAgentClientError) as error:
        provider.generate_initial(messages())

    assert _raised_code(error) is AnswerAgentClientErrorCode.INVALID_RESPONSE_SCHEMA
