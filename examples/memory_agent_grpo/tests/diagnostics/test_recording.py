import json
import stat
from dataclasses import dataclass
from enum import Enum

import pytest

from examples.memory_agent_grpo.diagnostics import ArtifactRecorder, RecordingHttpTransport
from examples.memory_agent_grpo.model_io.http_transport import HttpTransportError, HttpTransportErrorCode


class FakeTransport:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def post(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def call(transport, *, body=None, headers=None):
    return transport.post(
        url="https://example.test/v1/chat/completions?api_key=never-log-url",
        body=json.dumps(body or {"model": "model", "max_tokens": 4096}).encode(),
        headers=headers or {"Authorization": "Bearer private-header-key"},
        timeout_seconds=30, max_response_bytes=100_000,
    )


def test_records_complete_body_and_usage_without_changing_result(tmp_path):
    # Invalid model JSON must remain available verbatim for later diagnosis.
    content = '{"memory_action": {"type": "ADD", invalid JSON'
    response = json.dumps({
        "choices": [{"finish_reason": "length", "message": {"content": content}}],
        "usage": {"prompt_tokens": 80, "completion_tokens": 4096},
    }).encode()
    inner = FakeTransport((200, response))
    transport = RecordingHttpTransport(inner, tmp_path, "memory")
    request = {"model": "deepseek-model", "max_tokens": 4096,
               "thinking": {"type": "disabled"}, "messages": [{"content": "中文 task"}]}
    assert call(transport, body=request) == (200, response)
    event = json.loads((tmp_path / "memory_llm_calls.jsonl").read_text())
    assert event["request_body"] == request
    assert event["response_body"]["choices"][0]["message"]["content"] == content
    assert event["usage"]["completion_tokens"] == 4096
    assert event["finish_reason"] == "length"
    assert event["thinking"] == {"type": "disabled"}
    assert event["latency_seconds"] >= 0
    assert inner.calls[0]["headers"]["Authorization"] == "Bearer private-header-key"
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "memory_llm_calls.jsonl").stat().st_mode) == 0o600


@pytest.mark.parametrize("failure,expected", [
    (HttpTransportError(HttpTransportErrorCode.REQUEST_TIMEOUT, "private-header-key"), "REQUEST_TIMEOUT"),
    (RuntimeError("private-header-key never-log-exception"), "UNEXPECTED_TRANSPORT_ERROR"),
])
def test_failure_is_recorded_and_propagated_without_exception_text(tmp_path, failure, expected):
    transport = RecordingHttpTransport(FakeTransport(failure), tmp_path, "answer")
    with pytest.raises(type(failure)) as caught:
        call(transport)
    assert caught.value is failure
    saved = (tmp_path / "answer_llm_calls.jsonl").read_text()
    event = json.loads(saved)
    assert event["transport_error_code"] == expected
    assert event["response_body"] is None
    assert event["latency_seconds"] >= 0
    assert "private-header-key" not in saved
    assert "never-log-exception" not in saved
    assert "never-log-url" not in saved


def test_removes_nested_credentials_headers_and_redacts_echoed_tokens(tmp_path):
    response = json.dumps({
        "nested": {"api_key": "body-secret", "headers": {"X-Trace": "never-log-headers"}},
        "choices": [{"message": {"content": "private-header-key body-secret Bearer unknown-secret"}}],
    }).encode()
    recorder = ArtifactRecorder(tmp_path)
    transport = RecordingHttpTransport(FakeTransport((200, response)), tmp_path, "memory", recorder=recorder)
    call(transport, body={"model": "model", "nested": [{"password": "nested-password"}],
                          "messages": [{"content": "nested-password"}]})
    recorder.write_json("episode.json", {
        "public_text": "private-header-key", "API-Key": "other-secret",
        "echo-private-header-key": "a secret must also be redacted from object keys",
    })
    for artifact in tmp_path.iterdir():
        saved = artifact.read_text()
        json.loads(saved)
        for secret in ("private-header-key", "body-secret", "unknown-secret", "nested-password",
                       "other-secret", "never-log-headers", "never-log-url"):
            assert secret not in saved
        assert '"headers"' not in saved
        assert '"api_key"' not in saved
        assert '"Authorization"' not in saved


def test_non_json_http_error_and_incremental_calls_are_readable(tmp_path):
    inner = FakeTransport((429, b"rate limited"))
    transport = RecordingHttpTransport(inner, tmp_path, "answer")
    assert call(transport) == (429, b"rate limited")
    first = (tmp_path / "answer_llm_calls.jsonl").read_text()
    inner.result = (200, b'{"choices": []}')
    call(transport)
    saved = (tmp_path / "answer_llm_calls.jsonl").read_text()
    assert saved.startswith(first)
    events = [json.loads(line) for line in saved.splitlines()]
    assert [event["call_index"] for event in events] == [1, 2]
    assert events[0]["status_code"] == 429
    assert events[0]["response_body"] == "rate limited"


def test_serializes_nested_dataclasses_enums_and_paths(tmp_path):
    class Status(str, Enum):
        OK = "ok"

    @dataclass
    class Event:
        status: Status
        state: tuple

    recorder = ArtifactRecorder(tmp_path)
    event = Event(Status.OK, ({"path": tmp_path, "text": "记忆"},))
    recorder.write_json("episode.json", event)
    recorder.append_event("events.jsonl", event)
    expected = {"status": "ok", "state": [{"path": str(tmp_path), "text": "记忆"}]}
    assert json.loads((tmp_path / "episode.json").read_text()) == expected
    assert json.loads((tmp_path / "events.jsonl").read_text()) == expected


@pytest.mark.parametrize("filename", ["../outside.json", "/tmp/outside.json", "", "..", "subdir/file.json"])
def test_rejects_artifacts_outside_output_directory(tmp_path, filename):
    with pytest.raises(ValueError, match="plain filename"):
        ArtifactRecorder(tmp_path).write_json(filename, {})


def test_refuses_symlink_file_and_directory(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("untouched")
    output = tmp_path / "run"
    recorder = ArtifactRecorder(output)
    (output / "artifact.json").symlink_to(target)
    with pytest.raises(OSError):
        recorder.write_json("artifact.json", {})
    assert target.read_text() == "untouched"
    linked = tmp_path / "linked"
    linked.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        ArtifactRecorder(linked)
