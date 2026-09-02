from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping.answer_only_episode_runtime import (
    SHOPPING_ANSWER_ONLY_PRODUCT_COUNT,
    ShoppingAnswerOnlyEpisodeEvaluationPolicy,
    ShoppingAnswerOnlyEpisodeResult,
    ShoppingAnswerOnlyEpisodeRunStatus,
    ShoppingProductRunResult,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.catalog_bundle import (
    ShoppingCatalogBundleSummary,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.environment_session import (
    ShoppingTerminationEvidence,
    ShoppingTerminationKind,
)
from examples.memory_agent_grpo.answer_agent.input import AnswerAgentSubtaskInput
from examples.memory_agent_grpo.answer_agent.response import (
    AnswerAgentResponse,
    AnswerAgentToolCall,
)
from examples.memory_agent_grpo.answer_agent.runtime import (
    AnswerAgentInteractionMode,
    AnswerAgentSubtaskRunner,
    AnswerAgentSubtaskRunnerConfig,
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
from examples.memory_agent_grpo.smoke import (
    memoryarena_shopping_answer_only_episode_smoke as smoke,
)

_PRIVATE_SENTINEL = "PRIVATE-RAW-TASK-MODEL-OR-OBSERVATION"


def _full_task(
    product_count: int = SHOPPING_ANSWER_ONLY_PRODUCT_COUNT,
) -> dict[str, object]:
    targets = [f"SEALED-TARGET-{ordinal}" for ordinal in range(1, product_count + 1)]
    sections = "\n\n".join(
        f"Product {ordinal}:\nChoose the public option for Product {ordinal}."
        for ordinal in range(1, product_count + 1)
    )
    return {
        "task_id": "six_product_smoke_fixture",
        "task_description": "Choose the public products.",
        "agent_instruction": f"Use only public store evidence.\n\n{sections}",
        "steps": [
            {
                "step": ordinal,
                "target_asin": targets[ordinal - 1],
                "requirements": {"public_requirement": ordinal},
            }
            for ordinal in range(1, product_count + 1)
        ],
        "target_products": targets,
        "global_constraints": {
            "max_steps": product_count,
            "purchase_order": "sequence",
        },
        "evaluation_criteria": {
            **{
                f"step_{ordinal}_success": {"sealed": ordinal}
                for ordinal in range(1, product_count + 1)
            },
            "overall_success": {"all_steps_completed": True},
        },
    }


class _PrivateTerminalAnswerClient:
    def generate(
        self,
        conversation: object,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> AnswerAgentResponse:
        del conversation, tool_definitions
        return AnswerAgentResponse(
            text=f"{_PRIVATE_SENTINEL}-MODEL-TEXT",
            tool_calls=(
                AnswerAgentToolCall(
                    tool_call_id=f"{_PRIVATE_SENTINEL}-CALL-ID",
                    name="finish",
                    arguments_json=json.dumps(
                        {"item": f"{_PRIVATE_SENTINEL}-ARGUMENT"},
                        separators=(",", ":"),
                    ),
                ),
            ),
        )


class _PrivateTerminalExecutor:
    def execute(
        self,
        call: ValidatedToolCall,
    ) -> AnswerAgentTaskToolExecutionResult:
        return AnswerAgentTaskToolExecutionResult(
            tool_observation=ToolObservation(
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                status=ToolObservationStatus.SUCCESS,
                content=f"{_PRIVATE_SENTINEL}-TOOL-OBSERVATION",
            ),
            is_terminal=True,
        )


def _completed_episode_result() -> ShoppingAnswerOnlyEpisodeResult:
    config = AnswerAgentSubtaskRunnerConfig(
        max_inference_steps=1,
        max_tool_calls_per_inference=1,
        max_consecutive_tool_failures=1,
        interaction_mode=(
            AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL
        ),
    )
    answer_result = AnswerAgentSubtaskRunner(
        client=_PrivateTerminalAnswerClient(),
        executor=_PrivateTerminalExecutor(),
        config=config,
    ).run(
        AnswerAgentSubtaskInput(
            task_prompt=f"{_PRIVATE_SENTINEL}-TASK-PROMPT",
            subtask_input=f"{_PRIVATE_SENTINEL}-SUBTASK-INPUT",
        ),
        tool_definitions=(
            ToolDefinition(
                name="finish",
                description="Complete the public task.",
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
        ),
    )
    product_results = tuple(
        ShoppingProductRunResult(
            product_ordinal=product_ordinal,
            answer_result=answer_result,
            termination_evidence=ShoppingTerminationEvidence(
                ShoppingTerminationKind.PURCHASE_COMPLETED,
                purchased_target_matched=True,
            ),
        )
        for product_ordinal in range(1, SHOPPING_ANSWER_ONLY_PRODUCT_COUNT + 1)
    )
    return ShoppingAnswerOnlyEpisodeResult(
        status=ShoppingAnswerOnlyEpisodeRunStatus.COMPLETED,
        product_results=product_results,
        completed_product_count=SHOPPING_ANSWER_ONLY_PRODUCT_COUNT,
    )


def _passing_summary() -> smoke.ShoppingAnswerOnlyEpisodeSmokeSummary:
    return smoke.ShoppingAnswerOnlyEpisodeSmokeSummary(
        episode_result=_completed_episode_result(),
        catalog_summary=ShoppingCatalogBundleSummary(
            source_count=2,
            product_count=37_123,
            required_target_count=6,
            covered_target_count=6,
        ),
    )


@pytest.mark.parametrize("product_count", [5, 7])
def test_preflight_requires_exactly_six_products_without_sealed_leakage(
    product_count: int,
) -> None:
    task = _full_task(product_count)
    sealed_values = tuple(task["target_products"])  # type: ignore[arg-type]

    with pytest.raises(smoke.ShoppingAnswerOnlyEpisodeSmokeSetupError) as caught:
        smoke._preflight_required_targets(task)

    error_text = f"{caught.value!s} {caught.value!r}"
    assert all(sealed_value not in error_text for sealed_value in sealed_values)


def test_preflight_returns_six_trusted_targets_and_redacts_splitter_failure() -> None:
    task = _full_task()
    targets = tuple(task["target_products"])  # type: ignore[arg-type]

    assert smoke._preflight_required_targets(task) == targets

    private_target = targets[-1]
    task["agent_instruction"] = (
        f"{task['agent_instruction']}\nAccidental public leak: {private_target}"
    )
    with pytest.raises(smoke.ShoppingAnswerOnlyEpisodeSmokeSetupError) as caught:
        smoke._preflight_required_targets(task)

    assert private_target not in str(caught.value)
    assert private_target not in repr(caught.value)


def test_default_catalog_sources_resolve_against_memoryarena_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    memoryarena_root = tmp_path / "MemoryArena"
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    expected: list[Path] = []
    for relative_path in smoke.DEFAULT_CATALOG_RELATIVE_PATHS:
        source = memoryarena_root / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("[]", encoding="utf-8")
        expected.append(source.resolve())
    monkeypatch.chdir(unrelated_cwd)

    resolved = smoke._resolve_catalog_sources(
        memoryarena_root=memoryarena_root,
        catalog_sources=None,
    )

    assert resolved == tuple(expected)


def test_main_requires_explicit_live_confirmation_before_key_access(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    environment_name = "CONFIRMATION_ONLY_PRIVATE_KEY"
    environment_reads: list[str] = []
    run_calls: list[dict[str, object]] = []
    original_environment_get = smoke.os.environ.get

    def record_environment_get(
        name: str,
        default: str | None = None,
    ) -> str | None:
        environment_reads.append(name)
        return original_environment_get(name, default)

    monkeypatch.setattr(smoke.os.environ, "get", record_environment_get)
    monkeypatch.setattr(
        smoke,
        "run_shopping_answer_only_episode_smoke",
        lambda **kwargs: run_calls.append(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shopping-episode-smoke",
            "--api-key-env",
            environment_name,
        ],
    )

    assert smoke.main() == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert environment_name not in environment_reads
    assert run_calls == []
    assert json.loads(captured.out)["verification_code"] == (
        "LIVE_CONFIRMATION_REQUIRED"
    )


@pytest.mark.parametrize(
    ("environment_name", "environment_value"),
    [
        ("PRIVATE_MISSING_EPISODE_KEY", None),
        ("PRIVATE_EMPTY_EPISODE_KEY", ""),
        ("private-invalid-episode-key", "PRIVATE-SECRET-VALUE"),
    ],
)
def test_invalid_or_missing_key_fails_closed_without_running_or_echoing(
    environment_name: str,
    environment_value: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_calls: list[dict[str, object]] = []
    if environment_value is None:
        monkeypatch.delenv(environment_name, raising=False)
    else:
        monkeypatch.setenv(environment_name, environment_value)
    monkeypatch.setattr(
        smoke,
        "run_shopping_answer_only_episode_smoke",
        lambda **kwargs: run_calls.append(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shopping-episode-smoke",
            "--confirm-live-api",
            "--api-key-env",
            environment_name,
        ],
    )

    assert smoke.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert run_calls == []
    assert environment_name not in captured.out
    if environment_value:
        assert environment_value not in captured.out
    assert json.loads(captured.out)["verification_code"] == "API_KEY_MISSING"


@pytest.mark.parametrize(
    ("raised_error", "expected_code"),
    [
        (
            smoke.ShoppingAnswerOnlyEpisodeSmokeSetupError(_PRIVATE_SENTINEL),
            "SETUP_FAILED",
        ),
        (RuntimeError(_PRIVATE_SENTINEL), "UNEXPECTED_FAILURE"),
    ],
)
def test_main_suppresses_private_output_and_exception_text(
    raised_error: Exception,
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    environment_name = "TEST_EPISODE_LIVE_KEY"
    environment_value = "PRIVATE-SECRET-VALUE"

    def fail(**kwargs: object) -> smoke.ShoppingAnswerOnlyEpisodeSmokeSummary:
        del kwargs
        print(_PRIVATE_SENTINEL)
        print(_PRIVATE_SENTINEL, file=sys.stderr)
        raise raised_error

    monkeypatch.setenv(environment_name, environment_value)
    monkeypatch.setattr(smoke, "run_shopping_answer_only_episode_smoke", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shopping-episode-smoke",
            "--confirm-live-api",
            "--api-key-env",
            environment_name,
        ],
    )

    assert smoke.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert _PRIVATE_SENTINEL not in captured.out
    assert environment_name not in captured.out
    assert environment_value not in captured.out
    assert json.loads(captured.out)["verification_code"] == expected_code


def test_successful_main_forwards_arguments_and_prints_only_safe_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    environment_name = "TEST_EPISODE_SUCCESS_KEY"
    environment_value = "PRIVATE-SECRET-VALUE"
    calls: list[dict[str, object]] = []

    def succeed(**kwargs: object) -> smoke.ShoppingAnswerOnlyEpisodeSmokeSummary:
        calls.append(kwargs)
        print(_PRIVATE_SENTINEL)
        print(_PRIVATE_SENTINEL, file=sys.stderr)
        return _passing_summary()

    monkeypatch.setenv(environment_name, environment_value)
    monkeypatch.setattr(
        smoke,
        "run_shopping_answer_only_episode_smoke",
        succeed,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shopping-episode-smoke",
            "--confirm-live-api",
            "--api-key-env",
            environment_name,
            "--memoryarena-root",
            "/tmp/memoryarena-episode-test",
            "--task-file",
            "/tmp/six-product-task.json",
            "--catalog-source",
            "/tmp/catalog-one.json",
            "--catalog-source",
            "/tmp/catalog-two.json",
            "--upstream-url",
            "http://127.0.0.1:39024",
            "--upstream-python",
            "/tmp/runtime-python",
            "--upstream-startup-timeout-seconds",
            "19.5",
            "--environment-timeout-seconds",
            "41",
            "--max-observation-chars",
            "12345",
            "--max-inference-steps",
            "9",
            "--max-consecutive-tool-failures",
            "4",
            "--answer-max-tokens",
            "1536",
        ],
    )

    assert smoke.main() == 0
    assert len(calls) == 1
    assert calls[0] == {
        "memoryarena_root": Path("/tmp/memoryarena-episode-test"),
        "task_file": Path("/tmp/six-product-task.json"),
        "catalog_sources": (
            Path("/tmp/catalog-one.json"),
            Path("/tmp/catalog-two.json"),
        ),
        "upstream_url": "http://127.0.0.1:39024",
        "upstream_python": Path("/tmp/runtime-python"),
        "upstream_startup_timeout_seconds": 19.5,
        "environment_timeout_seconds": 41,
        "max_observation_chars": 12345,
        "max_inference_steps": 9,
        "max_consecutive_tool_failures": 4,
        "api_key_env": environment_name,
        "answer_max_tokens": 1536,
    }
    captured = capsys.readouterr()
    assert captured.err == ""
    assert _PRIVATE_SENTINEL not in captured.out
    assert environment_name not in captured.out
    assert environment_value not in captured.out
    output = json.loads(captured.out)
    assert output["verification_code"] == "PASS"
    assert output["smoke_pass"] is True


def test_summary_aggregates_safe_structural_evidence_without_raw_content() -> None:
    payload = _passing_summary().to_dict()

    assert payload["smoke_pass"] is True
    assert payload["status"] == "COMPLETED"
    assert payload["expected_product_count"] == 6
    assert payload["completed_product_count"] == 6
    assert payload["attempted_product_count"] == 6
    assert payload["stopped_product_ordinal"] is None
    assert payload["stop_failure_code"] is None
    assert payload["target_match_count"] == 6
    assert payload["purchase_completed_count"] == 6
    assert payload["total_inference_step_count"] == 6
    assert payload["total_emitted_tool_call_count"] == 6
    assert payload["total_executed_tool_count"] == 6
    assert payload["total_successful_tool_observation_count"] == 6
    assert payload["total_error_tool_observation_count"] == 0
    assert payload["total_terminal_tool_execution_count"] == 6
    assert payload["catalog_source_count"] == 2
    assert payload["catalog_product_count"] == 37_123
    assert payload["catalog_required_target_count"] == 6
    assert payload["catalog_covered_target_count"] == 6
    assert payload["catalog_all_targets_covered"] is True
    assert len(payload["product_results"]) == 6  # type: ignore[arg-type]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert _PRIVATE_SENTINEL not in serialized
    assert "SEALED-TARGET" not in serialized


def test_run_composes_one_service_and_one_six_product_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    full_task = _full_task()
    expected_result = _completed_episode_result()
    catalog_summary = ShoppingCatalogBundleSummary(
        source_count=2,
        product_count=11,
        required_target_count=6,
        covered_target_count=6,
    )
    memoryarena_root = tmp_path / "MemoryArena"
    memoryarena_root.mkdir()
    task_file = memoryarena_root / "task.json"
    catalog_sources = (
        memoryarena_root / "catalog-one.json",
        memoryarena_root / "catalog-two.json",
    )
    answer_client = object()
    tool_catalog = object()
    events: list[str] = []
    bundle_calls: list[dict[str, object]] = []
    service_calls: list[dict[str, object]] = []
    session_factory_calls: list[dict[str, object]] = []
    runner_init_calls: list[dict[str, object]] = []
    runner_run_calls: list[object] = []

    class FakeSessionFactory:
        def __init__(self, **kwargs: object) -> None:
            assert Path(kwargs["working_directory"]).is_dir()
            session_factory_calls.append(kwargs)

    class FakeEpisodeRunner:
        def __init__(self, **kwargs: object) -> None:
            runner_init_calls.append(kwargs)

        def run(self, task: object) -> ShoppingAnswerOnlyEpisodeResult:
            events.append("episode_run")
            runner_run_calls.append(task)
            return expected_result

    def fake_build_bundle(**kwargs: object) -> ShoppingCatalogBundleSummary:
        events.append("catalog_bundle")
        assert Path(kwargs["destination"]).parent.is_dir()
        bundle_calls.append(kwargs)
        return catalog_summary

    @contextmanager
    def fake_upstream_service(**kwargs: object) -> Iterator[None]:
        events.append("service_enter")
        service_calls.append(kwargs)
        yield
        events.append("service_exit")

    monkeypatch.setattr(smoke, "_expose_memoryarena", lambda root: root)
    monkeypatch.setattr(
        smoke,
        "_resolve_task_file",
        lambda **kwargs: task_file,
    )
    monkeypatch.setattr(smoke, "_load_full_task", lambda path: full_task)
    monkeypatch.setattr(
        smoke,
        "_resolve_catalog_sources",
        lambda **kwargs: catalog_sources,
    )
    monkeypatch.setattr(
        smoke,
        "_build_deepseek_client",
        lambda **kwargs: answer_client,
    )
    monkeypatch.setattr(smoke, "build_shopping_catalog_bundle", fake_build_bundle)
    monkeypatch.setattr(smoke, "ShoppingAnswerOnlySessionFactory", FakeSessionFactory)
    monkeypatch.setattr(smoke, "ShoppingAnswerOnlyEpisodeRunner", FakeEpisodeRunner)
    monkeypatch.setattr(smoke, "tool_definitions", lambda: tool_catalog)
    monkeypatch.setattr(smoke, "_upstream_service", fake_upstream_service)

    summary = smoke.run_shopping_answer_only_episode_smoke(
        memoryarena_root=memoryarena_root,
        task_file=task_file,
        catalog_sources=catalog_sources,
        upstream_url="http://127.0.0.1:39024",
        upstream_python=tmp_path / "python",
        upstream_startup_timeout_seconds=12.5,
        environment_timeout_seconds=42,
        max_observation_chars=15_000,
        max_inference_steps=8,
        max_consecutive_tool_failures=3,
        api_key_env="TEST_COMPOSITION_KEY",
        answer_max_tokens=1024,
    )

    assert summary.episode_result is expected_result
    assert summary.catalog_summary is catalog_summary
    assert events == [
        "catalog_bundle",
        "service_enter",
        "episode_run",
        "service_exit",
    ]
    assert len(bundle_calls) == 1
    assert bundle_calls[0]["source_files"] == catalog_sources
    assert bundle_calls[0]["required_target_asins"] == tuple(
        full_task["target_products"]  # type: ignore[arg-type]
    )
    assert len(service_calls) == 1
    assert service_calls[0]["items_file"] == bundle_calls[0]["destination"]
    assert service_calls[0]["api_key_env"] == "TEST_COMPOSITION_KEY"
    assert len(session_factory_calls) == 1
    assert session_factory_calls[0]["max_rounds"] == 8
    assert len(runner_init_calls) == 1
    assert runner_init_calls[0]["answer_client"] is answer_client
    assert runner_init_calls[0]["tool_definitions"] is tool_catalog
    runner_config = runner_init_calls[0]["config"]
    assert type(runner_config) is AnswerAgentSubtaskRunnerConfig
    assert runner_config.max_inference_steps == 8
    assert runner_config.max_tool_calls_per_inference == 1
    assert runner_config.max_consecutive_tool_failures == 3
    assert runner_config.interaction_mode is (
        AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL
    )
    assert runner_init_calls[0]["evaluation_policy"] is (
        ShoppingAnswerOnlyEpisodeEvaluationPolicy.CONTINUE_EVALUATION_FAILURES
    )
    assert runner_run_calls == [full_task]
