from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping.catalog_bundle import (
    ShoppingCatalogBundleSummary,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.memory_episode_contract import (
    ShoppingMemoryEpisodeResult,
)
from examples.memory_agent_grpo.answer_agent.interaction_mode import (
    AnswerAgentInteractionMode,
)
from examples.memory_agent_grpo.answer_agent.runtime import (
    AnswerAgentSubtaskRunnerConfig,
)
from examples.memory_agent_grpo.smoke import (
    memoryarena_shopping_memory_episode_smoke as smoke,
)
from examples.memory_agent_grpo.smoke import memoryarena_shopping_smoke as base_smoke
from examples.memory_agent_grpo.tests.adapters.memoryarena.shopping.test_memory_episode_runtime import (
    _full_task,
    _memory_runner,
)
from examples.memory_agent_grpo.tests.episode.runtime._support import (
    NULL_ACTION,
    ScriptedMemoryClient,
)

_PRIVATE_SENTINEL = "PRIVATE-MEMORY-EPISODE-SENTINEL"


def _completed_episode_result() -> ShoppingMemoryEpisodeResult:
    runner, _, _ = _memory_runner(
        memory_client=ScriptedMemoryClient(outcomes=[NULL_ACTION] * 12)
    )
    return runner.run(_full_task())


def _passing_summary() -> smoke.ShoppingMemoryEpisodeSmokeSummary:
    return smoke.ShoppingMemoryEpisodeSmokeSummary(
        episode_result=_completed_episode_result(),
        catalog_summary=ShoppingCatalogBundleSummary(
            source_count=2,
            product_count=77_433,
            required_target_count=6,
            covered_target_count=6,
        ),
        max_observation_chars=8_192,
    )


def test_summary_keeps_pipeline_and_evaluation_evidence_separate() -> None:
    payload = _passing_summary().to_dict()

    assert payload["smoke_pass"] is True
    assert payload["pipeline_completed"] is True
    assert payload["memory_enabled"] is True
    assert payload["answer_only"] is False
    assert payload["memory_agent_invocation_count"] == 12
    assert payload["committed_memory_decision_count"] == 12
    assert payload["final_bank_record_count"] == 0
    assert payload["memory_effect_observed"] is False
    assert payload["purchase_completed_count"] == 6
    assert payload["target_match_count"] == 6
    assert payload["evaluation_all_targets_matched"] is True
    assert payload["total_answer_inference_step_count"] == 12
    assert payload["total_answer_executed_tool_count"] == 12
    assert payload["total_answer_terminal_tool_execution_count"] == 6
    assert payload["catalog_product_count"] == 77_433
    assert len(payload["product_results"]) == 6  # type: ignore[arg-type]

    serialized = json.dumps(payload, ensure_ascii=False)
    for forbidden in (
        "PUBLIC_PRODUCT",
        "SEALED-TARGET",
        "raw_output",
        "interaction_events",
        "memory_task_input",
    ):
        assert forbidden not in serialized


def test_main_requires_confirmation_before_reading_either_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    answer_env = "PRIVATE_ANSWER_ENV"
    memory_env = "PRIVATE_MEMORY_ENV"
    environment_reads: list[str] = []
    original_get = smoke.os.environ.get

    def record_get(name: str, default: str | None = None) -> str | None:
        environment_reads.append(name)
        return original_get(name, default)

    monkeypatch.setattr(smoke.os.environ, "get", record_get)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shopping-memory-episode-smoke",
            "--answer-api-key-env",
            answer_env,
            "--memory-api-key-env",
            memory_env,
        ],
    )

    assert smoke.main() == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert answer_env not in environment_reads
    assert memory_env not in environment_reads
    assert json.loads(captured.out)["verification_code"] == (
        "LIVE_CONFIRMATION_REQUIRED"
    )


def test_main_requires_both_keys_without_echoing_names_or_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    answer_env = "TEST_MEMORY_EPISODE_ANSWER_KEY"
    memory_env = "TEST_MEMORY_EPISODE_MEMORY_KEY"
    secret = "PRIVATE-SECRET-VALUE"
    monkeypatch.setenv(answer_env, secret)
    monkeypatch.delenv(memory_env, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shopping-memory-episode-smoke",
            "--confirm-live-api",
            "--answer-api-key-env",
            answer_env,
            "--memory-api-key-env",
            memory_env,
        ],
    )

    assert smoke.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert answer_env not in captured.out
    assert memory_env not in captured.out
    assert secret not in captured.out
    assert json.loads(captured.out)["verification_code"] == "API_KEY_MISSING"


def test_successful_main_prints_only_allowlisted_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    answer_env = "TEST_MEMORY_EPISODE_ANSWER_KEY"
    memory_env = "TEST_MEMORY_EPISODE_MEMORY_KEY"
    secret = "PRIVATE-SECRET-VALUE"
    calls: list[dict[str, object]] = []

    def succeed(**kwargs: object) -> smoke.ShoppingMemoryEpisodeSmokeSummary:
        calls.append(kwargs)
        print(_PRIVATE_SENTINEL)
        print(_PRIVATE_SENTINEL, file=sys.stderr)
        return _passing_summary()

    monkeypatch.setenv(answer_env, secret)
    monkeypatch.setenv(memory_env, secret)
    monkeypatch.setattr(smoke, "run_shopping_memory_episode_smoke", succeed)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shopping-memory-episode-smoke",
            "--confirm-live-api",
            "--answer-api-key-env",
            answer_env,
            "--memory-api-key-env",
            memory_env,
            "--max-observation-chars",
            "7000",
            "--answer-max-tokens",
            "9000",
            "--upstream-url",
            "http://127.0.0.1:39034",
        ],
    )

    assert smoke.main() == 0
    assert len(calls) == 1
    assert calls[0]["answer_api_key_env"] == answer_env
    assert calls[0]["memory_api_key_env"] == memory_env
    assert calls[0]["max_observation_chars"] == 7000
    assert calls[0]["answer_max_tokens"] == 9000
    assert calls[0]["upstream_url"] == "http://127.0.0.1:39034"
    captured = capsys.readouterr()
    assert captured.err == ""
    assert _PRIVATE_SENTINEL not in captured.out
    assert answer_env not in captured.out
    assert memory_env not in captured.out
    assert secret not in captured.out
    assert json.loads(captured.out)["verification_code"] == "PASS"


def test_run_composes_both_agents_one_bank_runner_and_one_service(
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
    memory_client = object()
    answer_client = object()
    tool_catalog = object()
    events: list[str] = []
    service_calls: list[dict[str, object]] = []
    runner_init_calls: list[dict[str, object]] = []

    class FakeSessionFactory:
        def __init__(self, **kwargs: object) -> None:
            assert Path(kwargs["working_directory"]).is_dir()

    class FakeEpisodeRunner:
        def __init__(self, **kwargs: object) -> None:
            runner_init_calls.append(kwargs)

        def run(self, task: object) -> ShoppingMemoryEpisodeResult:
            assert task is full_task
            events.append("episode_run")
            return expected_result

    @contextmanager
    def fake_service(**kwargs: object) -> Iterator[None]:
        service_calls.append(kwargs)
        events.append("service_enter")
        yield
        events.append("service_exit")

    monkeypatch.setattr(smoke, "_expose_memoryarena", lambda root: root)
    monkeypatch.setattr(smoke, "_resolve_task_file", lambda **kwargs: task_file)
    monkeypatch.setattr(smoke, "_load_full_task", lambda path: full_task)
    monkeypatch.setattr(
        smoke,
        "_resolve_catalog_sources",
        lambda **kwargs: catalog_sources,
    )
    monkeypatch.setattr(
        smoke,
        "build_deepseek_smoke_client",
        lambda **kwargs: memory_client,
    )
    monkeypatch.setattr(
        smoke,
        "_build_deepseek_client",
        lambda **kwargs: answer_client,
    )
    monkeypatch.setattr(
        smoke,
        "build_shopping_catalog_bundle",
        lambda **kwargs: catalog_summary,
    )
    monkeypatch.setattr(smoke, "ShoppingAnswerOnlySessionFactory", FakeSessionFactory)
    monkeypatch.setattr(smoke, "ShoppingMemoryEpisodeRunner", FakeEpisodeRunner)
    monkeypatch.setattr(smoke, "tool_definitions", lambda: tool_catalog)
    monkeypatch.setattr(smoke, "_upstream_service", fake_service)

    summary = smoke.run_shopping_memory_episode_smoke(
        memoryarena_root=memoryarena_root,
        task_file=task_file,
        catalog_sources=catalog_sources,
        upstream_url="http://127.0.0.1:39034",
        upstream_python=tmp_path / "python",
        upstream_startup_timeout_seconds=12.5,
        environment_timeout_seconds=42,
        max_observation_chars=8_000,
        max_inference_steps=9,
        max_consecutive_tool_failures=3,
        answer_api_key_env="CUSTOM_ANSWER_KEY",
        memory_api_key_env="CUSTOM_MEMORY_KEY",
        answer_max_tokens=8_192,
    )

    assert summary.episode_result is expected_result
    assert summary.catalog_summary is catalog_summary
    assert events == ["service_enter", "episode_run", "service_exit"]
    assert len(runner_init_calls) == 1
    assert runner_init_calls[0]["memory_client"] is memory_client
    assert runner_init_calls[0]["answer_client"] is answer_client
    assert runner_init_calls[0]["tool_definitions"] is tool_catalog
    config = runner_init_calls[0]["config"]
    assert type(config) is AnswerAgentSubtaskRunnerConfig
    assert config.max_inference_steps == 9
    assert config.max_tool_calls_per_inference == 1
    assert config.interaction_mode is (
        AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL
    )
    assert service_calls[0]["api_key_env"] == "CUSTOM_ANSWER_KEY"
    assert service_calls[0]["additional_secret_env_names"] == (
        "CUSTOM_MEMORY_KEY",
    )


def test_custom_memory_key_is_removed_from_webshop_child_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    answer_env = "CUSTOM_ANSWER_SECRET"
    memory_env = "CUSTOM_MEMORY_SECRET"
    monkeypatch.setenv(answer_env, "answer-secret")
    monkeypatch.setenv(memory_env, "memory-secret")
    memoryarena_root = tmp_path / "MemoryArena"
    python_path = tmp_path / "runtime" / "bin" / "python"
    items_file = tmp_path / "items.json"

    child = base_smoke._build_upstream_child_environment(
        memoryarena_root=memoryarena_root,
        python_path=python_path,
        items_file=items_file,
        api_key_env=answer_env,
        additional_secret_env_names=(memory_env,),
    )

    assert answer_env not in child
    assert memory_env not in child
