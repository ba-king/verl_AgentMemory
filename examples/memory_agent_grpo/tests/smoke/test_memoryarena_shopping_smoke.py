from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from examples.memory_agent_grpo.answer_agent.input import AnswerAgentSubtaskInput
from examples.memory_agent_grpo.answer_agent.providers.deepseek import (
    DeepSeekToolChoice,
)
from examples.memory_agent_grpo.answer_agent.runtime import (
    AnswerAgentInteractionMode,
    AnswerAgentSubtaskRunner,
    AnswerAgentSubtaskRunnerConfig,
    AnswerAgentSubtaskRunStatus,
    AnswerAgentTaskToolExecutionResult,
)
from examples.memory_agent_grpo.answer_agent.tool_call_gateway import (
    ValidatedToolCall,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.definition import (
    ToolDefinition,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.observation import (
    ToolObservation,
    ToolObservationStatus,
)
from examples.memory_agent_grpo.smoke import memoryarena_shopping_smoke as smoke
from examples.memory_agent_grpo.tests.answer_agent.providers._support import (
    RecordingTransport,
    messages,
    response_body,
    tool_call,
)


def _tool_definitions() -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="search",
            description="Search the public store.",
            parameters_json_schema=json.dumps(
                {
                    "type": "object",
                    "properties": {"keywords": {"type": "string"}},
                    "required": ["keywords"],
                    "additionalProperties": False,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
        ToolDefinition(
            name="click",
            description="Click one public page value.",
            parameters_json_schema=json.dumps(
                {
                    "type": "object",
                    "properties": {"item": {"type": "string"}},
                    "required": ["item"],
                    "additionalProperties": False,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
    )


class _TerminalThirdExecutor:
    def __init__(self) -> None:
        self.calls: list[ValidatedToolCall] = []

    def execute(
        self,
        call: ValidatedToolCall,
    ) -> AnswerAgentTaskToolExecutionResult:
        self.calls.append(call)
        return AnswerAgentTaskToolExecutionResult(
            tool_observation=ToolObservation(
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                status=ToolObservationStatus.SUCCESS,
                content=f"public page {len(self.calls)}",
            ),
            is_terminal=len(self.calls) == 3,
        )


def _scripted_runner_result():
    executor = _TerminalThirdExecutor()
    result = AnswerAgentSubtaskRunner(
        client=smoke._ScriptedShoppingAnswerClient(),
        executor=executor,
        config=AnswerAgentSubtaskRunnerConfig(
            max_inference_steps=4,
            max_tool_calls_per_inference=1,
            max_consecutive_tool_failures=2,
            interaction_mode=(AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL),
        ),
    ).run(
        AnswerAgentSubtaskInput(
            task_prompt="Use the public tools.",
            subtask_input="Public Product 1 page.",
        ),
        tool_definitions=_tool_definitions(),
    )
    return executor, result


def _passing_summary(
    *,
    mode: smoke.ShoppingSmokeMode = smoke.ShoppingSmokeMode.SCRIPTED,
    product_ordinal: int = 1,
) -> smoke.ShoppingSmokeSummary:
    return smoke.ShoppingSmokeSummary(
        verification_code=smoke.ShoppingSmokeVerificationCode.PASS,
        mode=mode,
        product_ordinal=product_ordinal,
        reduced_smoke_catalog=False,
        runner_status=AnswerAgentSubtaskRunStatus.COMPLETED,
        inference_step_count=3,
        emitted_tool_call_count=3,
        executed_tool_count=3,
        successful_tool_observation_count=3,
        error_tool_observation_count=0,
        terminal_tool_execution_count=1,
        termination_kind=smoke.ShoppingTerminationKind.PURCHASE_COMPLETED,
        purchased_target_matched=True,
        runner_error_code=None,
        provider_error_code=None,
    )


@pytest.mark.parametrize("choice", [None, DeepSeekToolChoice.REQUIRED, DeepSeekToolChoice.AUTO])
def test_shopping_client_renders_selected_tool_choice(monkeypatch, choice) -> None:
    transport = RecordingTransport((200, response_body(
        content=None, finish_reason="tool_calls",
        tool_calls=[tool_call(name="click", arguments='{"item":"Buy Now"}')],
    )))
    monkeypatch.setenv(smoke.DEFAULT_API_KEY_ENV, "test-only-secret")
    monkeypatch.setattr(
        "examples.memory_agent_grpo.answer_agent.providers.deepseek.UrllibHttpTransport", lambda: transport,
    )
    client = smoke._build_deepseek_client(
        api_key_env=smoke.DEFAULT_API_KEY_ENV,
        answer_max_tokens=2048,
        **({} if choice is None else {"answer_tool_choice": choice}),
    )

    client.generate_initial(messages(), _tool_definitions())

    expected = DeepSeekToolChoice.REQUIRED if choice is None else choice
    assert client.config.tool_choice is expected
    payload = json.loads(transport.calls[0]["body"])
    assert payload["tool_choice"] == expected.value
    assert payload["max_tokens"] == 2048
    assert payload["thinking"] == {"type": "disabled"}
    assert len(transport.calls) == 1


def test_scripted_client_exercises_three_sequential_native_calls() -> None:
    executor, result = _scripted_runner_result()

    assert result.status is AnswerAgentSubtaskRunStatus.COMPLETED
    assert [call.name for call in executor.calls] == ["search", "click", "click"]
    assert [call.arguments_json for call in executor.calls] == [
        '{"keywords":"ginseng cleansing oil gel"}',
        '{"item":"B08GP7MRPW"}',
        '{"item":"Buy Now"}',
    ]
    assert len({call.tool_call_id for call in executor.calls}) == 3


def test_summary_passes_only_after_trusted_environment_termination() -> None:
    _, result = _scripted_runner_result()

    summary = smoke.summarize_shopping_result(
        result,
        mode=smoke.ShoppingSmokeMode.SCRIPTED,
        product_ordinal=1,
        reduced_smoke_catalog=False,
        termination_evidence=smoke.ShoppingTerminationEvidence(
            smoke.ShoppingTerminationKind.PURCHASE_COMPLETED,
            purchased_target_matched=True,
        ),
    )

    assert summary.passed is True
    assert summary.to_dict() == {
        "smoke_pass": True,
        "unscored": True,
        "saved_trajectory": False,
        "mode": "scripted",
        "scripted_oracle": True,
        "reduced_smoke_catalog": False,
        "verification_code": "PASS",
        "runner_status": "COMPLETED",
        "product_ordinal": 1,
        "inference_step_count": 3,
        "emitted_tool_call_count": 3,
        "executed_tool_count": 3,
        "successful_tool_observation_count": 3,
        "error_tool_observation_count": 0,
        "terminal_tool_execution_count": 1,
        "environment_terminated": True,
        "termination_kind": "PURCHASE_COMPLETED",
        "purchase_completed": True,
        "purchased_target_matched": True,
        "runner_error_code": None,
        "provider_error_code": None,
    }


def test_summary_rejects_soft_exit_even_when_runner_received_terminal() -> None:
    _, result = _scripted_runner_result()

    summary = smoke.summarize_shopping_result(
        result,
        mode=smoke.ShoppingSmokeMode.LIVE_DEEPSEEK,
        product_ordinal=1,
        reduced_smoke_catalog=True,
        termination_evidence=smoke.ShoppingTerminationEvidence(smoke.ShoppingTerminationKind.SOFT_EXIT),
    )

    assert summary.passed is False
    assert summary.verification_code is smoke.ShoppingSmokeVerificationCode.PURCHASE_NOT_COMPLETED
    assert summary.termination_kind is smoke.ShoppingTerminationKind.SOFT_EXIT


def test_scripted_summary_requires_purchase_to_match_sealed_target() -> None:
    _, result = _scripted_runner_result()

    summary = smoke.summarize_shopping_result(
        result,
        mode=smoke.ShoppingSmokeMode.SCRIPTED,
        product_ordinal=1,
        reduced_smoke_catalog=True,
        termination_evidence=smoke.ShoppingTerminationEvidence(
            smoke.ShoppingTerminationKind.PURCHASE_COMPLETED,
            purchased_target_matched=False,
        ),
    )

    assert summary.passed is False
    assert summary.verification_code is smoke.ShoppingSmokeVerificationCode.SCRIPTED_TARGET_MISMATCH
    assert summary.to_dict()["purchased_target_matched"] is False


def test_default_cli_is_scripted_and_never_reads_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []
    key_reads: list[str] = []

    def fake_run(**kwargs: object) -> smoke.ShoppingSmokeSummary:
        calls.append(kwargs)
        print("PRIVATE-UPSTREAM-OUTPUT")
        print("PRIVATE-UPSTREAM-ERROR", file=sys.stderr)
        return _passing_summary()

    original_environment_get = smoke.os.environ.get

    def record_environment_get(
        name: str,
        default: str | None = None,
    ) -> str | None:
        if name == smoke.DEFAULT_API_KEY_ENV:
            key_reads.append(name)
        return original_environment_get(name, default)

    monkeypatch.setattr(smoke, "run_shopping_smoke", fake_run)
    monkeypatch.setattr(smoke.os.environ, "get", record_environment_get)
    monkeypatch.setattr(sys, "argv", ["shopping-smoke"])

    assert smoke.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "PRIVATE" not in captured.out
    assert key_reads == []
    assert len(calls) == 1
    assert calls[0]["mode"] is smoke.ShoppingSmokeMode.SCRIPTED
    assert calls[0]["product_ordinal"] == 1
    assert calls[0]["task_file"] is None
    assert calls[0]["upstream_items_file"] is None
    assert json.loads(captured.out)["smoke_pass"] is True


def test_live_cli_fails_closed_without_reading_or_echoing_secret_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_environment_name = "PRIVATE_DEEPSEEK_KEY_NAME"
    run_calls: list[object] = []

    monkeypatch.setattr(
        smoke,
        "run_shopping_smoke",
        lambda **kwargs: run_calls.append(kwargs),
    )
    monkeypatch.delenv(private_environment_name, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shopping-smoke",
            "--live-deepseek",
            "--api-key-env",
            private_environment_name,
        ],
    )

    assert smoke.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert private_environment_name not in captured.out
    assert run_calls == []
    assert json.loads(captured.out)["verification_code"] == "API_KEY_MISSING"


def test_live_cli_forwards_explicit_paths_and_budgets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []
    api_key_environment = "TEST_SHOPPING_LIVE_KEY"

    def fake_run(**kwargs: object) -> smoke.ShoppingSmokeSummary:
        calls.append(kwargs)
        return _passing_summary(
            mode=smoke.ShoppingSmokeMode.LIVE_DEEPSEEK,
            product_ordinal=2,
        )

    monkeypatch.setenv(api_key_environment, "PRIVATE-SECRET-VALUE")
    monkeypatch.setattr(smoke, "run_shopping_smoke", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shopping-smoke",
            "--live-deepseek",
            "--api-key-env",
            api_key_environment,
            "--memoryarena-root",
            "/tmp/memoryarena-test",
            "--task-file",
            "/tmp/task-test.json",
            "--product-ordinal",
            "2",
            "--upstream-url",
            "http://127.0.0.1:39001",
            "--upstream-python",
            "/tmp/runtime-python",
            "--upstream-items-file",
            "/tmp/reduced-items.json",
            "--max-inference-steps",
            "17",
            "--answer-max-tokens",
            "2048",
            "--answer-tool-choice",
            "auto",
        ],
    )

    assert smoke.main() == 0
    assert len(calls) == 1
    assert calls[0]["mode"] is smoke.ShoppingSmokeMode.LIVE_DEEPSEEK
    assert calls[0]["memoryarena_root"] == Path("/tmp/memoryarena-test")
    assert calls[0]["task_file"] == Path("/tmp/task-test.json")
    assert calls[0]["product_ordinal"] == 2
    assert calls[0]["upstream_url"] == "http://127.0.0.1:39001"
    assert calls[0]["upstream_python"] == Path("/tmp/runtime-python")
    assert calls[0]["upstream_items_file"] == Path("/tmp/reduced-items.json")
    assert calls[0]["max_inference_steps"] == 17
    assert calls[0]["answer_max_tokens"] == 2048
    assert calls[0]["answer_tool_choice"] is DeepSeekToolChoice.AUTO
    output = capsys.readouterr().out
    assert "PRIVATE-SECRET-VALUE" not in output
    assert api_key_environment not in output
    assert json.loads(output)["mode"] == "live_deepseek"


def test_cli_setup_failure_discards_private_output_and_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_value = "PRIVATE-TASK-OR-OBSERVATION"

    def fail(**kwargs: object) -> smoke.ShoppingSmokeSummary:
        del kwargs
        print(private_value)
        print(private_value, file=sys.stderr)
        raise smoke.ShoppingSmokeSetupError(private_value)

    monkeypatch.setattr(smoke, "run_shopping_smoke", fail)
    monkeypatch.setattr(sys, "argv", ["shopping-smoke"])

    assert smoke.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert private_value not in captured.out
    assert json.loads(captured.out)["verification_code"] == "SETUP_FAILED"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://127.0.0.1:36014",
        "http://example.com:36014",
        "http://127.0.0.1",
        "http://127.0.0.1:36014/private",
        "not-a-url",
    ],
)
def test_upstream_service_target_must_be_one_local_http_port(
    base_url: str,
) -> None:
    with pytest.raises(smoke.ShoppingSmokeSetupError):
        smoke._local_upstream_port(base_url)


def test_upstream_python_path_keeps_virtualenv_symlink(tmp_path: Path) -> None:
    real_python = tmp_path / "real-python"
    real_python.touch()
    virtualenv_python = tmp_path / "venv" / "bin" / "python"
    virtualenv_python.parent.mkdir(parents=True)
    virtualenv_python.symlink_to(real_python)

    selected = smoke._absolute_path_without_resolving_links(virtualenv_python)

    assert selected == virtualenv_python
    assert selected.resolve() == real_python


def test_upstream_service_refuses_to_reuse_occupied_port(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process_calls: list[object] = []
    monkeypatch.setattr(smoke, "_service_reachable", lambda base_url: True)
    monkeypatch.setattr(
        smoke.subprocess,
        "Popen",
        lambda *args, **kwargs: process_calls.append((args, kwargs)),
    )

    with pytest.raises(smoke.ShoppingSmokeSetupError):
        with smoke._upstream_service(
            memoryarena_root=tmp_path,
            base_url="http://127.0.0.1:36014",
            python_executable=tmp_path / "python",
            items_file=tmp_path / "items.json",
            startup_timeout_seconds=180,
            api_key_env=smoke.DEFAULT_API_KEY_ENV,
        ):
            raise AssertionError("occupied service must never be reused")

    assert process_calls == []


def test_upstream_service_preserves_child_startup_error_in_private_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "webshop.log"
    opened_descriptors: list[int] = []
    monkeypatch.setattr(smoke, "_service_reachable", lambda base_url: False)

    def failed_child(*args, **kwargs):
        del args
        descriptor = kwargs["stdout"]
        assert descriptor == kwargs["stderr"]
        opened_descriptors.append(descriptor)
        os.write(descriptor, b"ModuleNotFoundError: No module named 'webshop_dependency'\n")
        return type("FailedChild", (), {"poll": lambda self: 1})()

    monkeypatch.setattr(smoke.subprocess, "Popen", failed_child)
    with pytest.raises(smoke.ShoppingSmokeSetupError, match="did not become ready"):
        with smoke._upstream_service(
            memoryarena_root=tmp_path,
            base_url="http://127.0.0.1:36014",
            python_executable=Path(sys.executable),
            items_file=tmp_path / "items.json",
            startup_timeout_seconds=1,
            api_key_env=smoke.DEFAULT_API_KEY_ENV,
            upstream_log_path=log_path,
        ):
            pytest.fail("failed child must not yield a ready service")

    assert "ModuleNotFoundError" in log_path.read_text()
    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
    with pytest.raises(OSError):
        os.fstat(opened_descriptors[0])


@pytest.mark.parametrize("keep_log", [False, True])
def test_upstream_service_closes_parent_log_and_preserves_process_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    keep_log: bool,
) -> None:
    class RunningChild:
        terminated = False
        waited = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, *, timeout):
            assert timeout == 15.0
            self.waited = True

    child = RunningChild()
    descriptors: list[int] = []
    monkeypatch.setattr(smoke, "_service_reachable", lambda base_url: False)

    def start_child(*args, **kwargs):
        del args
        assert kwargs["stdout"] == kwargs["stderr"]
        descriptors.append(kwargs["stdout"])
        return child

    def ready(**kwargs):
        del kwargs
        if keep_log:
            with pytest.raises(OSError):
                os.fstat(descriptors[0])
        else:
            assert descriptors[0] == smoke.subprocess.DEVNULL

    monkeypatch.setattr(smoke.subprocess, "Popen", start_child)
    monkeypatch.setattr(smoke, "_wait_for_service", ready)
    with smoke._upstream_service(
        memoryarena_root=tmp_path,
        base_url="http://127.0.0.1:36014",
        python_executable=Path(sys.executable),
        items_file=tmp_path / "items.json",
        startup_timeout_seconds=1,
        api_key_env=smoke.DEFAULT_API_KEY_ENV,
        upstream_log_path=tmp_path / "webshop.log" if keep_log else None,
    ):
        assert child.terminated is False
    assert child.terminated is True
    assert child.waited is True


def test_upstream_service_does_not_overwrite_existing_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "webshop.log"
    log_path.write_text("previous failure")
    monkeypatch.setattr(smoke, "_service_reachable", lambda base_url: False)
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not start"))
    with pytest.raises(smoke.ShoppingSmokeSetupError, match="could not be started"):
        with smoke._upstream_service(
            memoryarena_root=tmp_path,
            base_url="http://127.0.0.1:36014",
            python_executable=Path(sys.executable),
            items_file=tmp_path / "items.json",
            startup_timeout_seconds=1,
            api_key_env=smoke.DEFAULT_API_KEY_ENV,
            upstream_log_path=log_path,
        ):
            pytest.fail("existing log must prevent a new launch")
    assert log_path.read_text() == "previous failure"


def test_wait_for_service_rejects_unknown_service_after_child_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poll_results = iter((None, 1))
    process = type(
        "ExitedChild",
        (),
        {"poll": lambda self: next(poll_results)},
    )()
    monkeypatch.setattr(smoke, "_service_reachable", lambda base_url: True)

    with pytest.raises(smoke.ShoppingSmokeSetupError):
        smoke._wait_for_service(
            base_url="http://127.0.0.1:36014",
            process=process,  # type: ignore[arg-type]
            timeout_seconds=1,
        )


def test_upstream_environment_does_not_inherit_model_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    custom_key_name = "CUSTOM_LIVE_MODEL_SECRET"
    monkeypatch.setenv(custom_key_name, "PRIVATE-CUSTOM-KEY")
    monkeypatch.setenv(smoke.DEFAULT_API_KEY_ENV, "PRIVATE-ANSWER-KEY")
    monkeypatch.setenv("P2R_MEMORY_API_KEY", "PRIVATE-MEMORY-KEY")
    monkeypatch.setenv("HARMLESS_SMOKE_SETTING", "kept")

    child_environment = smoke._build_upstream_child_environment(
        memoryarena_root=tmp_path,
        python_path=tmp_path / "venv" / "bin" / "python",
        items_file=tmp_path / "items.json",
        api_key_env=custom_key_name,
    )

    assert custom_key_name not in child_environment
    assert smoke.DEFAULT_API_KEY_ENV not in child_environment
    assert "P2R_MEMORY_API_KEY" not in child_environment
    assert child_environment["HARMLESS_SMOKE_SETTING"] == "kept"


def test_reduced_upstream_catalog_is_explicitly_marked(tmp_path: Path) -> None:
    shopping_data = tmp_path / "data" / "shopping"
    shopping_data.mkdir(parents=True)
    full_catalog = shopping_data / "items_shuffle.json"
    full_catalog.write_text("[]", encoding="utf-8")
    reduced_catalog = shopping_data / "product_catalog" / "beauty.json"
    reduced_catalog.parent.mkdir()
    reduced_catalog.write_text("[]", encoding="utf-8")

    selected, is_reduced = smoke._resolve_upstream_items_file(
        memoryarena_root=tmp_path,
        upstream_items_file=reduced_catalog,
    )

    assert selected == reduced_catalog.resolve()
    assert is_reduced is True


def test_relative_paths_resolve_against_memoryarena_before_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    memoryarena_root = tmp_path / "memoryarena"
    cwd = tmp_path / "cwd"
    relative_task = Path("benchmark_data/task.json")
    relative_catalog = Path("data/shopping/reduced.json")
    full_catalog = memoryarena_root / "data" / "shopping" / "items_shuffle.json"
    for base in (memoryarena_root, cwd):
        (base / relative_task).parent.mkdir(parents=True, exist_ok=True)
        (base / relative_task).write_text("{}", encoding="utf-8")
        (base / relative_catalog).parent.mkdir(parents=True, exist_ok=True)
        (base / relative_catalog).write_text("[]", encoding="utf-8")
    full_catalog.write_text("[]", encoding="utf-8")
    monkeypatch.chdir(cwd)

    selected_task = smoke._resolve_task_file(
        memoryarena_root=memoryarena_root,
        task_file=relative_task,
    )
    selected_catalog, is_reduced = smoke._resolve_upstream_items_file(
        memoryarena_root=memoryarena_root,
        upstream_items_file=relative_catalog,
    )

    assert selected_task == (memoryarena_root / relative_task).resolve()
    assert selected_catalog == (memoryarena_root / relative_catalog).resolve()
    assert is_reduced is True


def test_scripted_temp_task_contains_only_selected_public_section(
    tmp_path: Path,
) -> None:
    full_task = {
        "task_id": "beauty_item_26",
        "task_description": "Public description",
        "agent_instruction": (
            "Global public rules.\n\n"
            "Product 1:\nPublic Product 1 options.\n\n"
            "----------------\n"
            "Product 2:\nPublic Product 2 options."
        ),
        "steps": [
            {"step": 1, "target_asin": "B08GP7MRPW"},
            {"step": 2, "target_asin": "PRIVATE-TARGET-2"},
        ],
        "target_products": ["B08GP7MRPW", "PRIVATE-TARGET-2"],
        "global_constraints": {"max_steps": 2},
        "evaluation_criteria": {},
    }
    source = tmp_path / "full.json"
    destination = tmp_path / "single.json"
    source.write_text(json.dumps(full_task), encoding="utf-8")

    smoke._write_single_product_task(
        full_task_file=source,
        product_ordinal=1,
        destination=destination,
        scripted=True,
    )

    single = json.loads(destination.read_text(encoding="utf-8"))
    assert "Product 1:" in single["agent_instruction"]
    assert "Product 2:" not in single["agent_instruction"]
    assert "PRIVATE-TARGET-2" not in single["agent_instruction"]
    assert single["target_products"] == ["B08GP7MRPW"]
