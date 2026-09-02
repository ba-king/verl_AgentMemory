"""Run one unscored six-Product MemoryArena Shopping Answer-only episode."""

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

from examples.memory_agent_grpo.adapters.memoryarena.shopping.answer_only_episode_runtime import (
    SHOPPING_ANSWER_ONLY_PRODUCT_COUNT,
    ShoppingAnswerOnlyEpisodeEvaluationPolicy,
    ShoppingAnswerOnlyEpisodeInputError,
    ShoppingAnswerOnlyEpisodeResult,
    ShoppingAnswerOnlyEpisodeRunner,
)
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
from examples.memory_agent_grpo.adapters.memoryarena.shopping.task_input_adapter import (
    ShoppingTaskInputAdapter,
    ShoppingTaskInputAdapterError,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.task_splitter import (
    ShoppingTaskSplitterError,
    build_single_product_task,
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
from examples.memory_agent_grpo.smoke.memoryarena_shopping_smoke import (
    DEFAULT_ANSWER_MAX_TOKENS,
    DEFAULT_API_KEY_ENV,
    DEFAULT_ENVIRONMENT_TIMEOUT_SECONDS,
    DEFAULT_MAX_CONSECUTIVE_TOOL_FAILURES,
    DEFAULT_MAX_INFERENCE_STEPS,
    DEFAULT_MAX_OBSERVATION_CHARS,
    DEFAULT_MEMORYARENA_ROOT,
    DEFAULT_UPSTREAM_STARTUP_TIMEOUT_SECONDS,
    ShoppingSmokeSetupError,
    _build_deepseek_client,
    _expose_memoryarena,
    _load_full_task,
    _resolve_task_file,
    _upstream_service,
)

DEFAULT_EPISODE_UPSTREAM_URL = "http://127.0.0.1:36024"
DEFAULT_CATALOG_RELATIVE_PATHS = (
    Path("data/shopping/product_catalog/beauty_personal_care_skin_care.json"),
    Path("data/shopping/product_catalog/beauty_personal_care_tools_accessories.json"),
)
_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class ShoppingAnswerOnlyEpisodeSmokeVerificationCode(str, Enum):
    """Stable top-level smoke outcomes without task or model content."""

    PASS = "PASS"
    EPISODE_FAILED = "EPISODE_FAILED"
    LIVE_CONFIRMATION_REQUIRED = "LIVE_CONFIRMATION_REQUIRED"
    API_KEY_MISSING = "API_KEY_MISSING"
    SETUP_FAILED = "SETUP_FAILED"
    UNEXPECTED_FAILURE = "UNEXPECTED_FAILURE"


class ShoppingAnswerOnlyEpisodeSmokeSetupError(RuntimeError):
    """Content-free failure while composing the local six-Product run."""


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
class ShoppingAnswerOnlyEpisodeSmokeSummary:
    """Allowlisted evidence for one real Answer-only six-Product run."""

    episode_result: ShoppingAnswerOnlyEpisodeResult
    catalog_summary: ShoppingCatalogBundleSummary

    @property
    def passed(self) -> bool:
        return self.episode_result.passed

    def to_dict(self) -> dict[str, object | None]:
        episode_payload = self.episode_result.to_dict()
        raw_product_payloads = episode_payload["product_results"]
        if type(raw_product_payloads) is not list or any(
            type(product_payload) is not dict for product_payload in raw_product_payloads
        ):
            raise ValueError("Episode product summaries are invalid.")
        product_payloads: list[dict[str, object | None]] = raw_product_payloads
        return {
            "smoke_pass": self.passed,
            "unscored": True,
            "answer_only": True,
            "model_mode": "live_deepseek",
            "saved_trajectory": False,
            "memory_agent_invocation_count": 0,
            "raw_conversation_reused_across_products": False,
            "shared_upstream_service": True,
            "temporary_catalog_bundle": True,
            "catalog_source_count": self.catalog_summary.source_count,
            "catalog_product_count": self.catalog_summary.product_count,
            "catalog_required_target_count": self.catalog_summary.required_target_count,
            "catalog_covered_target_count": self.catalog_summary.covered_target_count,
            "catalog_all_targets_covered": self.catalog_summary.all_targets_covered,
            "verification_code": (
                ShoppingAnswerOnlyEpisodeSmokeVerificationCode.PASS.value
                if self.passed
                else ShoppingAnswerOnlyEpisodeSmokeVerificationCode.EPISODE_FAILED.value
            ),
            "status": episode_payload["status"],
            "expected_product_count": episode_payload["expected_product_count"],
            "completed_product_count": episode_payload["completed_product_count"],
            "attempted_product_count": episode_payload["attempted_product_count"],
            "failed_product_ordinal": episode_payload["failed_product_ordinal"],
            "failure_code": episode_payload["failure_code"],
            "stopped_product_ordinal": episode_payload["stopped_product_ordinal"],
            "stop_failure_code": episode_payload["stop_failure_code"],
            "target_match_count": sum(
                product_payload["purchased_target_matched"] is True
                for product_payload in product_payloads
            ),
            "purchase_completed_count": sum(
                product_payload["termination_kind"] == "PURCHASE_COMPLETED"
                for product_payload in product_payloads
            ),
            "total_inference_step_count": _safe_sum(
                product_payloads,
                "inference_step_count",
            ),
            "total_emitted_tool_call_count": _safe_sum(
                product_payloads,
                "emitted_tool_call_count",
            ),
            "total_executed_tool_count": _safe_sum(
                product_payloads,
                "executed_tool_count",
            ),
            "total_successful_tool_observation_count": _safe_sum(
                product_payloads,
                "successful_tool_observation_count",
            ),
            "total_error_tool_observation_count": _safe_sum(
                product_payloads,
                "error_tool_observation_count",
            ),
            "total_terminal_tool_execution_count": _safe_sum(
                product_payloads,
                "terminal_tool_execution_count",
            ),
            "product_results": product_payloads,
        }


def _preflight_required_targets(
    full_task_definition: object,
) -> tuple[str, ...]:
    """Pre-split all Products and retain targets only on the trusted side."""

    if type(full_task_definition) is not dict:
        raise ShoppingAnswerOnlyEpisodeSmokeSetupError(
            "Shopping task preflight failed."
        )
    steps = full_task_definition.get("steps")
    if type(steps) is not list or len(steps) != SHOPPING_ANSWER_ONLY_PRODUCT_COUNT:
        raise ShoppingAnswerOnlyEpisodeSmokeSetupError(
            "Shopping task must contain exactly six Products."
        )
    targets: list[str] = []
    try:
        for product_ordinal in range(1, SHOPPING_ANSWER_ONLY_PRODUCT_COUNT + 1):
            single_task = build_single_product_task(
                full_task_definition,
                product_ordinal=product_ordinal,
            )
            target_products = single_task.get("target_products")
            if (
                type(target_products) is not list
                or len(target_products) != 1
                or type(target_products[0]) is not str
                or not target_products[0].strip()
            ):
                raise ShoppingAnswerOnlyEpisodeSmokeSetupError(
                    "Shopping task target preflight failed."
                )
            targets.append(target_products[0])
    except ShoppingTaskSplitterError:
        raise ShoppingAnswerOnlyEpisodeSmokeSetupError(
            "Shopping task preflight failed."
        ) from None
    if len(set(target.upper() for target in targets)) != SHOPPING_ANSWER_ONLY_PRODUCT_COUNT:
        raise ShoppingAnswerOnlyEpisodeSmokeSetupError(
            "Shopping task targets must be unique."
        )
    return tuple(targets)


def _resolve_catalog_sources(
    *,
    memoryarena_root: Path,
    catalog_sources: tuple[Path, ...] | None,
) -> tuple[Path, ...]:
    requested = DEFAULT_CATALOG_RELATIVE_PATHS if catalog_sources is None else catalog_sources
    if type(requested) is not tuple or not requested:
        raise ShoppingAnswerOnlyEpisodeSmokeSetupError(
            "At least one Shopping catalog source is required."
        )
    resolved_sources: list[Path] = []
    for requested_path in requested:
        if not isinstance(requested_path, Path):
            raise ShoppingAnswerOnlyEpisodeSmokeSetupError(
                "Shopping catalog sources must be pathlib paths."
            )
        candidate = requested_path.expanduser()
        if not candidate.is_absolute():
            root_candidate = memoryarena_root / candidate
            cwd_candidate = Path.cwd() / candidate
            candidate = root_candidate if root_candidate.is_file() else cwd_candidate
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise ShoppingAnswerOnlyEpisodeSmokeSetupError(
                "A Shopping catalog source is unavailable."
            )
        resolved_sources.append(resolved)
    if len(set(resolved_sources)) != len(resolved_sources):
        raise ShoppingAnswerOnlyEpisodeSmokeSetupError(
            "Shopping catalog sources must be unique."
        )
    return tuple(resolved_sources)


def run_shopping_answer_only_episode_smoke(
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
    api_key_env: str,
    answer_max_tokens: int,
) -> ShoppingAnswerOnlyEpisodeSmokeSummary:
    """Run six isolated Product conversations against one local service."""

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
    answer_client = _build_deepseek_client(
        api_key_env=api_key_env,
        answer_max_tokens=answer_max_tokens,
    )

    with tempfile.TemporaryDirectory(
        prefix="memoryarena-shopping-episode-"
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
        episode_runner = ShoppingAnswerOnlyEpisodeRunner(
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
            evaluation_policy=(
                ShoppingAnswerOnlyEpisodeEvaluationPolicy.CONTINUE_EVALUATION_FAILURES
            ),
        )
        with _upstream_service(
            memoryarena_root=root,
            base_url=upstream_url,
            python_executable=runtime_python,
            items_file=catalog_path,
            startup_timeout_seconds=upstream_startup_timeout_seconds,
            api_key_env=api_key_env,
        ):
            episode_result = episode_runner.run(full_task_definition)

    return ShoppingAnswerOnlyEpisodeSmokeSummary(
        episode_result=episode_result,
        catalog_summary=catalog_summary,
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
    parser.add_argument("--upstream-url", default=DEFAULT_EPISODE_UPSTREAM_URL)
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
        default=DEFAULT_MAX_OBSERVATION_CHARS,
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
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=DEFAULT_ANSWER_MAX_TOKENS,
    )
    parser.add_argument(
        "--confirm-live-api",
        action="store_true",
        help="Required confirmation that this command may make paid DeepSeek calls.",
    )
    return parser.parse_args()


def _safe_failure_payload(
    code: ShoppingAnswerOnlyEpisodeSmokeVerificationCode,
) -> dict[str, object]:
    return {
        "smoke_pass": False,
        "unscored": True,
        "answer_only": True,
        "saved_trajectory": False,
        "memory_agent_invocation_count": 0,
        "model_mode": "live_deepseek",
        "verification_code": code.value,
    }


def main() -> int:
    args = _parse_args()
    if not args.confirm_live_api:
        print(
            json.dumps(
                _safe_failure_payload(
                    ShoppingAnswerOnlyEpisodeSmokeVerificationCode.LIVE_CONFIRMATION_REQUIRED
                ),
                indent=2,
            )
        )
        return 2
    if (
        type(args.api_key_env) is not str
        or _ENV_NAME_PATTERN.fullmatch(args.api_key_env) is None
        or not os.environ.get(args.api_key_env)
    ):
        print(
            json.dumps(
                _safe_failure_payload(
                    ShoppingAnswerOnlyEpisodeSmokeVerificationCode.API_KEY_MISSING
                ),
                indent=2,
            )
        )
        return 1

    try:
        with open(os.devnull, "w", encoding="utf-8") as discarded_output:
            with redirect_stdout(discarded_output), redirect_stderr(discarded_output):
                summary = run_shopping_answer_only_episode_smoke(
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
                    api_key_env=args.api_key_env,
                    answer_max_tokens=args.answer_max_tokens,
                )
    except (
        MemoryArenaShoppingDependencyError,
        ShoppingAnswerOnlyEpisodeInputError,
        ShoppingAnswerOnlyEpisodeSmokeSetupError,
        ShoppingAnswerOnlySessionFactoryError,
        ShoppingCatalogBundleError,
        ShoppingEnvironmentSessionError,
        ShoppingSmokeSetupError,
        ShoppingTaskInputAdapterError,
        ShoppingTaskSplitterError,
        ValueError,
    ):
        print(
            json.dumps(
                _safe_failure_payload(
                    ShoppingAnswerOnlyEpisodeSmokeVerificationCode.SETUP_FAILED
                ),
                indent=2,
            )
        )
        return 1
    except Exception:
        print(
            json.dumps(
                _safe_failure_payload(
                    ShoppingAnswerOnlyEpisodeSmokeVerificationCode.UNEXPECTED_FAILURE
                ),
                indent=2,
            )
        )
        return 1

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0 if summary.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ShoppingAnswerOnlyEpisodeSmokeSetupError",
    "ShoppingAnswerOnlyEpisodeSmokeSummary",
    "ShoppingAnswerOnlyEpisodeSmokeVerificationCode",
    "run_shopping_answer_only_episode_smoke",
]
