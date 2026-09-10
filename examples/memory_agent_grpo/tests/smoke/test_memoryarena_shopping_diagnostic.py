from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import asdict

import pytest

from examples.memory_agent_grpo.answer_agent.providers.deepseek import DeepSeekToolChoice
from examples.memory_agent_grpo.diagnostics import ArtifactRecorder
from examples.memory_agent_grpo.memory_agent.decision.model_client import (
    MemoryAgentClientError,
    MemoryAgentClientErrorCode,
)
from examples.memory_agent_grpo.smoke import memoryarena_shopping_diagnostic as diagnostic
from examples.memory_agent_grpo.tests.adapters.memoryarena.shopping.test_memory_episode_runtime import (
    _full_task,
    _memory_runner,
)
from examples.memory_agent_grpo.tests.answer_agent.providers._support import (
    RecordingTransport,
    messages,
    response_body,
    tool_call,
)
from examples.memory_agent_grpo.tests.episode.runtime._support import NULL_ACTION, ScriptedMemoryClient, add_action
from examples.memory_agent_grpo.tests.smoke.test_memoryarena_shopping_smoke import _tool_definitions


@pytest.mark.parametrize("option", [None, "required", "auto"])
def test_diagnostic_tool_choice_reaches_recorded_http_request(tmp_path, monkeypatch, option):
    argv = ["--output-dir", str(tmp_path / "run"), "--preflight-only"]
    if option is not None:
        argv.extend(["--answer-tool-choice", option])
    args = diagnostic.parse_args(argv)
    expected = DeepSeekToolChoice.REQUIRED if option is None else DeepSeekToolChoice(option)
    assert args.answer_tool_choice is expected
    transport = RecordingTransport((200, response_body(
        content=None, finish_reason="tool_calls",
        tool_calls=[tool_call(name="click", arguments='{"item":"Buy Now"}')],
    )))
    monkeypatch.setenv(args.answer_api_key_env, "test-only-secret")
    monkeypatch.setattr(diagnostic, "UrllibHttpTransport", lambda: transport)

    recorder, answer, _ = diagnostic._build_clients(args, args.output_dir)
    answer.client.generate_initial(messages(), _tool_definitions())

    payload = json.loads(transport.calls[0]["body"])
    saved = json.loads((recorder.output_dir / "answer_llm_calls.jsonl").read_text())
    config = json.loads((recorder.output_dir / "client_config.json").read_text())
    assert len(transport.calls) == 1
    assert payload["tool_choice"] == expected.value
    assert saved["request_body"]["tool_choice"] == expected.value
    assert config["answer"]["tool_choice"] == expected.value


def test_live_confirmation_required_before_output_creation(tmp_path):
    output = tmp_path / "new"
    with pytest.raises(SystemExit):
        diagnostic.parse_args(["--output-dir", str(output)])
    assert not output.exists()


def test_existing_results_are_never_reused(tmp_path):
    existing = tmp_path / "run"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep")
    with pytest.raises(SystemExit):
        diagnostic.parse_args(["--output-dir", str(existing), "--preflight-only"])
    assert (existing / "keep.txt").read_text() == "keep"


def test_credentials_restored_even_when_run_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_ANSWER_KEY", "old")
    monkeypatch.delenv("TEST_MEMORY_KEY", raising=False)
    key_file = tmp_path / "key"
    key_file.write_text("temporary-test-key\n")
    args = Namespace(
        answer_api_key_env="TEST_ANSWER_KEY",
        memory_api_key_env="TEST_MEMORY_KEY",
        api_key_file=key_file,
        preflight_only=False,
        mode="paired",
    )
    with pytest.raises(RuntimeError), diagnostic._credentials(args):
        assert diagnostic.os.environ["TEST_ANSWER_KEY"] == "temporary-test-key"
        assert diagnostic.os.environ["TEST_MEMORY_KEY"] == "temporary-test-key"
        raise RuntimeError("failure")
    assert diagnostic.os.environ["TEST_ANSWER_KEY"] == "old"
    assert "TEST_MEMORY_KEY" not in diagnostic.os.environ


def test_invalid_output_and_recovery_survive_full_episode_serialization(tmp_path):
    recorder = ArtifactRecorder(tmp_path / "run")
    memory = diagnostic.ObservedMemoryClient(
        ScriptedMemoryClient(outcomes=["not json", *([NULL_ACTION] * 12)]),
        recorder,
    )
    runner, _, _ = _memory_runner(
        memory_client=memory,
        memory_output_correction_budget=1,
        memory_decision_observer=memory.observe_decision,
        product_observer=lambda trace: recorder.append_event("products.jsonl", trace),
    )
    result = runner.run(_full_task())
    assert result.pipeline_completed
    recorder.write_json("episode_trace.json", asdict(result))
    traces = [json.loads(line) for line in (recorder.output_dir / "memory_decisions.jsonl").read_text().splitlines()]
    assert traces[0]["trace"]["raw_output"] == "not json"
    assert traces[0]["trace"]["bank_before"] == traces[0]["trace"]["bank_after"]
    stats = diagnostic.summarize_memory_attempts(memory.attempts)
    assert stats["first_attempt_count"] == 12
    assert stats["first_attempt_valid_count"] == 11
    assert stats["correction_attempt_count"] == stats["correction_valid_count"] == 1
    assert stats["error_counts"] == {"INVALID_JSON": 1}
    episode = json.loads((recorder.output_dir / "episode_trace.json").read_text())
    assert len(episode["product_traces"]) == 6
    assert "answer_result" in episode["product_traces"][0]


def test_incomplete_completion_retains_input_and_is_not_counted_as_json_error(tmp_path):
    recorder = ArtifactRecorder(tmp_path / "run")
    memory = diagnostic.ObservedMemoryClient(
        ScriptedMemoryClient(
            outcomes=[
                MemoryAgentClientError(MemoryAgentClientErrorCode.INCOMPLETE_COMPLETION, "truncated"),
            ]
        ),
        recorder,
    )
    runner, _, _ = _memory_runner(
        memory_client=memory,
        memory_output_correction_budget=1,
        memory_decision_observer=memory.observe_decision,
    )
    result = runner.run(_full_task())
    assert not result.pipeline_completed
    assert len(memory.attempts) == 1
    stats = diagnostic.summarize_memory_attempts(memory.attempts)
    assert stats["error_counts"] == {"INCOMPLETE_COMPLETION": 1}
    assert stats["correction_attempt_count"] == 0
    events = [json.loads(line) for line in (recorder.output_dir / "memory_attempts.jsonl").read_text().splitlines()]
    assert events[0]["model_input"]
    assert events[1]["status"] == "model_call_failed"


def test_repair_is_logged_and_not_counted_as_native_valid_output(tmp_path):
    recorder = ArtifactRecorder(tmp_path / "run")
    raw = add_action("PRESERVED_MEMORY")[:-1]
    memory = diagnostic.ObservedMemoryClient(
        ScriptedMemoryClient(outcomes=[NULL_ACTION, raw, *([NULL_ACTION] * 10)]), recorder,
    )
    runner, _, _ = _memory_runner(memory_client=memory, memory_decision_observer=memory.observe_decision)
    result = runner.run(_full_task())
    assert result.pipeline_completed
    stats = diagnostic.summarize_memory_attempts(memory.attempts)
    assert stats["first_attempt_count"] == stats["first_attempt_accepted_count"] == 12
    assert stats["first_attempt_valid_count"] == 11
    assert stats["deterministic_repair_count"] == 1
    assert stats["correction_attempt_count"] == 0
    assert stats["error_counts"] == {"INVALID_JSON": 1}
    traces = [json.loads(line) for line in (recorder.output_dir / "memory_decisions.jsonl").read_text().splitlines()]
    repaired = traces[1]
    assert repaired["status"] == "repaired"
    assert repaired["raw_output_valid"] is False
    assert repaired["trace"]["raw_output"] == raw
    assert repaired["trace"]["output_repair"]["repaired_output"] == raw + "}"
