"""Run one unscored Memory-aware six-Product MemoryArena Shopping episode."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from examples.memory_agent_grpo.adapters.memoryarena.shopping.answer_only_session_factory import (
    ShoppingAnswerOnlySessionFactory,
    ShoppingAnswerOnlySessionFactoryError,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.catalog_bundle import (
    ShoppingCatalogBundleError,
    ShoppingCatalogBundleSummary,
    build_shopping_catalog_bundle,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.environment_session import (
    ShoppingEnvironmentSessionError,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.memory_episode_contract import (
    ShoppingMemoryEpisodeInputError,
    ShoppingMemoryEpisodeResult,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.memory_episode_runtime import (
    ShoppingMemoryEpisodeRunner,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.task_input_adapter import (
    ShoppingTaskInputAdapter,
    ShoppingTaskInputAdapterError,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.tool_catalog import (
    MemoryArenaShoppingDependencyError,
    tool_definitions,
)
from examples.memory_agent_grpo.answer_agent.interaction_mode import (
    AnswerAgentInteractionMode,
)
from examples.memory_agent_grpo.answer_agent.runtime import (
    AnswerAgentSubtaskRunnerConfig,
)
from examples.memory_agent_grpo.smoke.deepseek_action_smoke import (
    build_deepseek_smoke_client,
)
from examples.memory_agent_grpo.smoke.memoryarena_shopping_answer_only_episode_smoke import (
    DEFAULT_CATALOG_RELATIVE_PATHS,
    ShoppingAnswerOnlyEpisodeSmokeSetupError,
    _preflight_required_targets,
    _resolve_catalog_sources,
)
from examples.memory_agent_grpo.smoke.memoryarena_shopping_smoke import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_ENVIRONMENT_TIMEOUT_SECONDS,
    DEFAULT_MAX_CONSECUTIVE_TOOL_FAILURES,
    DEFAULT_MAX_INFERENCE_STEPS,
    DEFAULT_MEMORYARENA_ROOT,
    DEFAULT_UPSTREAM_STARTUP_TIMEOUT_SECONDS,
    ShoppingSmokeSetupError,
    _build_deepseek_client,
    _expose_memoryarena,
    _load_full_task,
    _resolve_task_file,
    _upstream_service,
)

DEFAULT_MEMORY_EPISODE_UPSTREAM_URL = "http://127.0.0.1:36034"
DEFAULT_MEMORY_API_KEY_ENV = "P2R_MEMORY_API_KEY"
DEFAULT_MEMORY_EPISODE_MAX_OBSERVATION_CHARS = 8_192
DEFAULT_MEMORY_EPISODE_ANSWER_MAX_TOKENS = 8_192
DEFAULT_MEMORY_OUTPUT_CORRECTION_BUDGET = 1
_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class ShoppingMemoryEpisodeSmokeVerificationCode(str, Enum):
    """Stable top-level smoke outcomes without task or model content."""

    PASS = "PASS"
    EPISODE_FAILED = "EPISODE_FAILED"
    LIVE_CONFIRMATION_REQUIRED = "LIVE_CONFIRMATION_REQUIRED"
    API_KEY_MISSING = "API_KEY_MISSING"
    SETUP_FAILED = "SETUP_FAILED"
    UNEXPECTED_FAILURE = "UNEXPECTED_FAILURE"


def _safe_sum(
    product_payloads: list[dict[str, object | None]],
    field_name: str,
) -> int:
    total = 0
    for product_payload in product_payloads:
        value = product_payload[field_name]
        if type(value) is not int:
            raise ValueError("Episode summary counter is invalid.")
        total += value
    return total


@dataclass(frozen=True)
class ShoppingMemoryEpisodeSmokeSummary:
    """Allowlisted evidence for one real Memory-aware Shopping run."""

    episode_result: ShoppingMemoryEpisodeResult
    catalog_summary: ShoppingCatalogBundleSummary
    max_observation_chars: int

    @property
    def passed(self) -> bool:
        return self.episode_result.pipeline_completed

    def to_dict(self) -> dict[str, object | None]:
        episode_payload = self.episode_result.to_safe_dict()
        raw_products = episode_payload["product_results"]
        if type(raw_products) is not list or any(
            type(product) is not dict for product in raw_products
        ):
            raise ValueError("Episode product summaries are invalid.")
        product_payloads: list[dict[str, object | None]] = raw_products
        return {
            "smoke_pass": self.passed,
            "unscored": True,
            "answer_only": False,
            "memory_enabled": True,
            "model_mode": "live_deepseek",
            "saved_trajectory": False,
            "raw_conversation_reused_across_products": False,
            "shared_memory_bank_across_products": True,
            "evaluator_feedback_exposed_to_memory_agent": False,
            "shared_upstream_service": True,
            "temporary_catalog_bundle": True,
            "max_observation_chars": self.max_observation_chars,
            "catalog_source_count": self.catalog_summary.source_count,
            "catalog_product_count": self.catalog_summary.product_count,
            "catalog_required_target_count": (
                self.catalog_summary.required_target_count
            ),
            "catalog_covered_target_count": (
                self.catalog_summary.covered_target_count
            ),
            "catalog_all_targets_covered": (
                self.catalog_summary.all_targets_covered
            ),
            "verification_code": (
                ShoppingMemoryEpisodeSmokeVerificationCode.PASS.value
                if self.passed
                else ShoppingMemoryEpisodeSmokeVerificationCode.EPISODE_FAILED.value
            ),
            **{
                key: value
                for key, value in episode_payload.items()
                if key != "product_results"
            },
            "total_answer_inference_step_count": _safe_sum(
                product_payloads,
                "inference_step_count",
            ),
            "total_answer_emitted_tool_call_count": _safe_sum(
                product_payloads,
                "emitted_tool_call_count",
            ),
            "total_answer_executed_tool_count": _safe_sum(
                product_payloads,
                "executed_tool_count",
            ),
            "total_answer_successful_tool_observation_count": _safe_sum(
                product_payloads,
                "successful_tool_observation_count",
            ),
            "total_answer_error_tool_observation_count": _safe_sum(
                product_payloads,
                "error_tool_observation_count",
            ),
            "total_answer_terminal_tool_execution_count": _safe_sum(
                product_payloads,
                "terminal_tool_execution_count",
            ),
            "product_results": product_payloads,
        }


def run_shopping_memory_episode_smoke(
    *,
    memoryarena_root: Path,
    task_file: Path | None,
    catalog_sources: tuple[Path, ...] | None,
    upstream_url: str,
    upstream_python: Path | None,
    upstream_startup_timeout_seconds: float,
    environment_timeout_seconds: int,
    max_observation_chars: int,
    max_inference_steps: int,
    max_consecutive_tool_failures: int,
    answer_api_key_env: str,
    memory_api_key_env: str,
    answer_max_tokens: int,
    memory_output_correction_budget: int = DEFAULT_MEMORY_OUTPUT_CORRECTION_BUDGET,
) -> ShoppingMemoryEpisodeSmokeSummary:
    """Compose both agents, one persistent Bank, and six fresh sessions."""

    root = _expose_memoryarena(memoryarena_root)
    source_task = _resolve_task_file(
        memoryarena_root=root,
        task_file=task_file,
    )
    full_task_definition = _load_full_task(source_task)
    required_targets = _preflight_required_targets(full_task_definition)
    selected_catalog_sources = _resolve_catalog_sources(
        memoryarena_root=root,
        catalog_sources=catalog_sources,
    )
    runtime_python = (
        root / ".venv-runtime" / "bin" / "python"
        if upstream_python is None
        else upstream_python
    )
    memory_client = build_deepseek_smoke_client(
        api_key_env=memory_api_key_env
    )
    answer_client = _build_deepseek_client(
        api_key_env=answer_api_key_env,
        answer_max_tokens=answer_max_tokens,
    )

    with tempfile.TemporaryDirectory(
        prefix="memoryarena-shopping-memory-episode-"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        catalog_path = temporary_root / "catalog_bundle.json"
        catalog_summary = build_shopping_catalog_bundle(
            source_files=selected_catalog_sources,
            destination=catalog_path,
            required_target_asins=required_targets,
        )
        task_directory = temporary_root / "tasks"
        task_directory.mkdir()
        session_factory = ShoppingAnswerOnlySessionFactory(
            working_directory=task_directory,
            env_server_base=upstream_url,
            product_catalog_dir=root / "data" / "shopping" / "product_catalog",
            domain_data_path=root / "data" / "shopping" / "domain_data.json",
            timeout_seconds=environment_timeout_seconds,
            max_observation_chars=max_observation_chars,
            max_rounds=max_inference_steps,
        )
        episode_runner = ShoppingMemoryEpisodeRunner(
            memory_client=memory_client,
            answer_client=answer_client,
            session_factory=session_factory,
            task_input_adapter=ShoppingTaskInputAdapter(
                max_observation_chars=max_observation_chars
            ),
            tool_definitions=tool_definitions(),
            config=AnswerAgentSubtaskRunnerConfig(
                max_inference_steps=max_inference_steps,
                max_tool_calls_per_inference=1,
                max_consecutive_tool_failures=max_consecutive_tool_failures,
                interaction_mode=(
                    AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL
                ),
            ),
            memory_output_correction_budget=(
                memory_output_correction_budget
            ),
        )
        with _upstream_service(
            memoryarena_root=root,
            base_url=upstream_url,
            python_executable=runtime_python,
            items_file=catalog_path,
            startup_timeout_seconds=upstream_startup_timeout_seconds,
            api_key_env=answer_api_key_env,
            additional_secret_env_names=(memory_api_key_env,),
        ):
            episode_result = episode_runner.run(full_task_definition)

    return ShoppingMemoryEpisodeSmokeSummary(
        episode_result=episode_result,
        catalog_summary=catalog_summary,
        max_observation_chars=max_observation_chars,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memoryarena-root",
        type=Path,
        default=DEFAULT_MEMORYARENA_ROOT,
    )
    parser.add_argument("--task-file", type=Path, default=None)
    parser.add_argument(
        "--catalog-source",
        type=Path,
        action="append",
        default=None,
        help="Repeat to override the two default Beauty category catalogs.",
    )
    parser.add_argument(
        "--upstream-url",
        default=DEFAULT_MEMORY_EPISODE_UPSTREAM_URL,
    )
    parser.add_argument("--upstream-python", type=Path, default=None)
    parser.add_argument(
        "--upstream-startup-timeout-seconds",
        type=float,
        default=DEFAULT_UPSTREAM_STARTUP_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--environment-timeout-seconds",
        type=int,
        default=DEFAULT_ENVIRONMENT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--max-observation-chars",
        type=int,
        default=DEFAULT_MEMORY_EPISODE_MAX_OBSERVATION_CHARS,
    )
    parser.add_argument(
        "--max-inference-steps",
        type=int,
        default=DEFAULT_MAX_INFERENCE_STEPS,
    )
    parser.add_argument(
        "--max-consecutive-tool-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_TOOL_FAILURES,
    )
    parser.add_argument(
        "--answer-api-key-env",
        default=DEFAULT_API_KEY_ENV,
    )
    parser.add_argument(
        "--memory-api-key-env",
        default=DEFAULT_MEMORY_API_KEY_ENV,
    )
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=DEFAULT_MEMORY_EPISODE_ANSWER_MAX_TOKENS,
    )
    parser.add_argument(
        "--memory-output-correction-budget",
        type=int,
        choices=(0, 1),
        default=DEFAULT_MEMORY_OUTPUT_CORRECTION_BUDGET,
        help="Episode-wide Memory JSON correction budget; training remains 0.",
    )
    parser.add_argument(
        "--confirm-live-api",
        action="store_true",
        help="Required confirmation that this command may make paid DeepSeek calls.",
    )
    return parser.parse_args()


def _safe_failure_payload(
    code: ShoppingMemoryEpisodeSmokeVerificationCode,
) -> dict[str, object]:
    return {
        "smoke_pass": False,
        "unscored": True,
        "answer_only": False,
        "memory_enabled": True,
        "saved_trajectory": False,
        "model_mode": "live_deepseek",
        "verification_code": code.value,
    }


def _key_env_available(name: object) -> bool:
    return (
        type(name) is str
        and _ENV_NAME_PATTERN.fullmatch(name) is not None
        and bool(os.environ.get(name))
    )


def main() -> int:
    args = _parse_args()
    if not args.confirm_live_api:
        print(
            json.dumps(
                _safe_failure_payload(
                    ShoppingMemoryEpisodeSmokeVerificationCode.LIVE_CONFIRMATION_REQUIRED
                ),
                indent=2,
            )
        )
        return 2
    if not all(
        _key_env_available(name)
        for name in (args.answer_api_key_env, args.memory_api_key_env)
    ):
        print(
            json.dumps(
                _safe_failure_payload(
                    ShoppingMemoryEpisodeSmokeVerificationCode.API_KEY_MISSING
                ),
                indent=2,
            )
        )
        return 1

    try:
        with open(os.devnull, "w", encoding="utf-8") as discarded_output:
            with redirect_stdout(discarded_output), redirect_stderr(
                discarded_output
            ):
                summary = run_shopping_memory_episode_smoke(
                    memoryarena_root=args.memoryarena_root,
                    task_file=args.task_file,
                    catalog_sources=(
                        None
                        if args.catalog_source is None
                        else tuple(args.catalog_source)
                    ),
                    upstream_url=args.upstream_url,
                    upstream_python=args.upstream_python,
                    upstream_startup_timeout_seconds=(
                        args.upstream_startup_timeout_seconds
                    ),
                    environment_timeout_seconds=args.environment_timeout_seconds,
                    max_observation_chars=args.max_observation_chars,
                    max_inference_steps=args.max_inference_steps,
                    max_consecutive_tool_failures=(
                        args.max_consecutive_tool_failures
                    ),
                    answer_api_key_env=args.answer_api_key_env,
                    memory_api_key_env=args.memory_api_key_env,
                    answer_max_tokens=args.answer_max_tokens,
                    memory_output_correction_budget=(
                        args.memory_output_correction_budget
                    ),
                )
    except (
        MemoryArenaShoppingDependencyError,
        ShoppingAnswerOnlyEpisodeSmokeSetupError,
        ShoppingAnswerOnlySessionFactoryError,
        ShoppingCatalogBundleError,
        ShoppingEnvironmentSessionError,
        ShoppingMemoryEpisodeInputError,
        ShoppingSmokeSetupError,
        ShoppingTaskInputAdapterError,
        ValueError,
    ):
        print(
            json.dumps(
                _safe_failure_payload(
                    ShoppingMemoryEpisodeSmokeVerificationCode.SETUP_FAILED
                ),
                indent=2,
            )
        )
        return 1
    except Exception:
        print(
            json.dumps(
                _safe_failure_payload(
                    ShoppingMemoryEpisodeSmokeVerificationCode.UNEXPECTED_FAILURE
                ),
                indent=2,
            )
        )
        return 1

    payload = summary.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if summary.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CATALOG_RELATIVE_PATHS",
    "DEFAULT_MEMORY_API_KEY_ENV",
    "DEFAULT_MEMORY_EPISODE_ANSWER_MAX_TOKENS",
    "DEFAULT_MEMORY_EPISODE_MAX_OBSERVATION_CHARS",
    "DEFAULT_MEMORY_OUTPUT_CORRECTION_BUDGET",
    "DEFAULT_MEMORY_EPISODE_UPSTREAM_URL",
    "ShoppingMemoryEpisodeSmokeSummary",
    "ShoppingMemoryEpisodeSmokeVerificationCode",
    "run_shopping_memory_episode_smoke",
]
