"""Run one unscored, environment-terminated MemoryArena Shopping smoke."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

from examples.memory_agent_grpo.adapters.memoryarena.shopping import (
    MemoryArenaShoppingDependencyError,
    ShoppingEnvironmentSession,
    ShoppingEnvironmentSessionConfig,
    ShoppingEnvironmentSessionError,
    ShoppingTaskInputAdapter,
    ShoppingTaskInputAdapterError,
    ShoppingTaskSplitterError,
    ShoppingTerminationEvidence,
    ShoppingTerminationKind,
    ShoppingToolExecutorAdapter,
    build_single_product_task,
    tool_definitions,
)
from examples.memory_agent_grpo.answer_agent.model_client import (
    AnswerAgentClientErrorCode,
)
from examples.memory_agent_grpo.answer_agent.providers.deepseek import (
    DeepSeekAnswerAgentClient,
    DeepSeekAnswerAgentClientConfig,
    DeepSeekToolChoice,
)
from examples.memory_agent_grpo.answer_agent.response import (
    AnswerAgentResponse,
    AnswerAgentToolCall,
)
from examples.memory_agent_grpo.answer_agent.runtime import (
    AnswerAgentConversation,
    AnswerAgentInteractionMode,
    AnswerAgentSubtaskErrorCode,
    AnswerAgentSubtaskRunner,
    AnswerAgentSubtaskRunnerConfig,
    AnswerAgentSubtaskRunResult,
    AnswerAgentSubtaskRunStatus,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.definition import (
    ToolDefinition,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.observation import (
    ToolObservationStatus,
)
from examples.memory_agent_grpo.smoke.deepseek_answer_agent_initial_smoke import (
    DEEPSEEK_CONTEXT_WINDOW_TOKENS,
    DEEPSEEK_MODEL,
    DeepSeekAnswerConservativeTokenCounter,
)

DEFAULT_MEMORYARENA_ROOT = Path(__file__).resolve().parents[3].parent / "MemoryArena"
DEFAULT_TASK_RELATIVE_PATH = Path("benchmark_data/fixed100_v2/shopping_tasks/056_beauty_item_26.json")
DEFAULT_PRODUCT_ORDINAL = 1
DEFAULT_UPSTREAM_URL = "http://127.0.0.1:36014"
DEFAULT_API_KEY_ENV = "P2R_ANSWER_API_KEY"
_MEMORY_API_KEY_ENV = "P2R_MEMORY_API_KEY"
DEFAULT_MAX_OBSERVATION_CHARS = 16_000
DEFAULT_MAX_INFERENCE_STEPS = 20
DEFAULT_MAX_CONSECUTIVE_TOOL_FAILURES = 2
DEFAULT_ANSWER_MAX_TOKENS = 4_096
DEFAULT_ENVIRONMENT_TIMEOUT_SECONDS = 300
DEFAULT_UPSTREAM_STARTUP_TIMEOUT_SECONDS = 180.0

_SCRIPTED_PRODUCT_1_QUERY = "ginseng cleansing oil gel"
_SCRIPTED_PRODUCT_1_TARGET = "B08GP7MRPW"
_UPSTREAM_MODULE = "env.env_systems.web_shopping_env.runtime.service.launch_lite"
_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_LOCAL_HOSTS = frozenset({"127.0.0.1", "0.0.0.0", "localhost", "::1"})


class ShoppingSmokeMode(str, Enum):
    """Whether the Answer Agent is deterministic or a paid live model."""

    SCRIPTED = "scripted"
    LIVE_DEEPSEEK = "live_deepseek"


class ShoppingSmokeVerificationCode(str, Enum):
    """Stable, content-free smoke outcomes."""

    PASS = "PASS"
    RUNNER_FAILED = "RUNNER_FAILED"
    ENVIRONMENT_DID_NOT_TERMINATE = "ENVIRONMENT_DID_NOT_TERMINATE"
    PURCHASE_NOT_COMPLETED = "PURCHASE_NOT_COMPLETED"
    SCRIPTED_TARGET_MISMATCH = "SCRIPTED_TARGET_MISMATCH"
    API_KEY_MISSING = "API_KEY_MISSING"
    SETUP_FAILED = "SETUP_FAILED"
    UNEXPECTED_FAILURE = "UNEXPECTED_FAILURE"


class ShoppingSmokeSetupError(RuntimeError):
    """Content-free failure while preparing the real local environment."""


class _ScriptedShoppingAnswerClient:
    """Oracle replay used only to verify the real environment composition."""

    _CALLS = (
        ("search", {"keywords": _SCRIPTED_PRODUCT_1_QUERY}),
        ("click", {"item": _SCRIPTED_PRODUCT_1_TARGET}),
        ("click", {"item": "Buy Now"}),
    )

    def __init__(self) -> None:
        self._next_call = 0

    def generate(
        self,
        conversation: AnswerAgentConversation,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> AnswerAgentResponse:
        del conversation, tool_definitions
        if self._next_call >= len(self._CALLS):
            return AnswerAgentResponse(text="Scripted smoke sequence exhausted.")
        name, arguments = self._CALLS[self._next_call]
        self._next_call += 1
        return AnswerAgentResponse(
            tool_calls=(
                AnswerAgentToolCall(
                    tool_call_id=f"scripted-shopping-{self._next_call}",
                    name=name,
                    arguments_json=json.dumps(
                        arguments,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                ),
            )
        )


@dataclass(frozen=True)
class ShoppingSmokeSummary:
    """Allowlisted structural counters; no task, model, or observation text."""

    verification_code: ShoppingSmokeVerificationCode
    mode: ShoppingSmokeMode
    product_ordinal: int
    reduced_smoke_catalog: bool
    runner_status: AnswerAgentSubtaskRunStatus
    inference_step_count: int
    emitted_tool_call_count: int
    executed_tool_count: int
    successful_tool_observation_count: int
    error_tool_observation_count: int
    terminal_tool_execution_count: int
    termination_kind: ShoppingTerminationKind
    purchased_target_matched: bool | None
    runner_error_code: AnswerAgentSubtaskErrorCode | None
    provider_error_code: AnswerAgentClientErrorCode | None

    @property
    def passed(self) -> bool:
        return self.verification_code is ShoppingSmokeVerificationCode.PASS

    def to_dict(self) -> dict[str, object | None]:
        return {
            "smoke_pass": self.passed,
            "unscored": True,
            "saved_trajectory": False,
            "mode": self.mode.value,
            "scripted_oracle": self.mode is ShoppingSmokeMode.SCRIPTED,
            "reduced_smoke_catalog": self.reduced_smoke_catalog,
            "verification_code": self.verification_code.value,
            "runner_status": self.runner_status.value,
            "product_ordinal": self.product_ordinal,
            "inference_step_count": self.inference_step_count,
            "emitted_tool_call_count": self.emitted_tool_call_count,
            "executed_tool_count": self.executed_tool_count,
            "successful_tool_observation_count": (self.successful_tool_observation_count),
            "error_tool_observation_count": self.error_tool_observation_count,
            "terminal_tool_execution_count": self.terminal_tool_execution_count,
            "environment_terminated": self.terminal_tool_execution_count == 1,
            "termination_kind": self.termination_kind.value,
            "purchase_completed": self.termination_kind is ShoppingTerminationKind.PURCHASE_COMPLETED,
            "purchased_target_matched": self.purchased_target_matched,
            "runner_error_code": None if self.runner_error_code is None else self.runner_error_code.value,
            "provider_error_code": (None if self.provider_error_code is None else self.provider_error_code.value),
        }


def summarize_shopping_result(
    result: AnswerAgentSubtaskRunResult,
    *,
    mode: ShoppingSmokeMode,
    product_ordinal: int,
    reduced_smoke_catalog: bool,
    termination_evidence: ShoppingTerminationEvidence,
) -> ShoppingSmokeSummary:
    """Reduce a sensitive runner result to structural evidence only."""

    outcomes = tuple(outcome for step in result.steps for outcome in step.tool_outcomes)
    terminal_count = sum(outcome.is_terminal for outcome in outcomes)
    if result.status is AnswerAgentSubtaskRunStatus.FAILED:
        verification_code = ShoppingSmokeVerificationCode.RUNNER_FAILED
    elif terminal_count != 1:
        verification_code = ShoppingSmokeVerificationCode.ENVIRONMENT_DID_NOT_TERMINATE
    elif termination_evidence.kind is not ShoppingTerminationKind.PURCHASE_COMPLETED:
        verification_code = ShoppingSmokeVerificationCode.PURCHASE_NOT_COMPLETED
    elif mode is ShoppingSmokeMode.SCRIPTED and termination_evidence.purchased_target_matched is not True:
        verification_code = ShoppingSmokeVerificationCode.SCRIPTED_TARGET_MISMATCH
    else:
        verification_code = ShoppingSmokeVerificationCode.PASS
    return ShoppingSmokeSummary(
        verification_code=verification_code,
        mode=mode,
        product_ordinal=product_ordinal,
        reduced_smoke_catalog=reduced_smoke_catalog,
        runner_status=result.status,
        inference_step_count=result.inference_step_count,
        emitted_tool_call_count=sum(len(step.response.tool_calls) for step in result.steps),
        executed_tool_count=result.executed_tool_count,
        successful_tool_observation_count=sum(
            outcome.tool_observation.status is ToolObservationStatus.SUCCESS for outcome in outcomes
        ),
        error_tool_observation_count=sum(
            outcome.tool_observation.status is ToolObservationStatus.ERROR for outcome in outcomes
        ),
        terminal_tool_execution_count=terminal_count,
        termination_kind=termination_evidence.kind,
        purchased_target_matched=termination_evidence.purchased_target_matched,
        runner_error_code=result.error_code,
        provider_error_code=result.provider_error_code,
    )


def _build_deepseek_client(
    *,
    api_key_env: str,
    answer_max_tokens: int,
) -> DeepSeekAnswerAgentClient:
    return DeepSeekAnswerAgentClient(
        DeepSeekAnswerAgentClientConfig(
            model=DEEPSEEK_MODEL,
            api_key_env=api_key_env,
            context_window_tokens=DEEPSEEK_CONTEXT_WINDOW_TOKENS,
            timeout_seconds=120.0,
            temperature=0.0,
            max_tokens=answer_max_tokens,
            tool_choice=DeepSeekToolChoice.REQUIRED,
        ),
        token_counter=DeepSeekAnswerConservativeTokenCounter(),
    )


def _expose_memoryarena(memoryarena_root: Path) -> Path:
    root = memoryarena_root.expanduser().resolve()
    expected = root / "env" / "env_systems" / "web_shopping_env" / "webshop_plus_client.py"
    if not expected.is_file():
        raise ShoppingSmokeSetupError("memoryarena_root does not contain the Shopping environment.")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def _resolve_task_file(
    *,
    memoryarena_root: Path,
    task_file: Path | None,
) -> Path:
    candidate = memoryarena_root / DEFAULT_TASK_RELATIVE_PATH if task_file is None else task_file.expanduser()
    if not candidate.is_absolute():
        root_candidate = memoryarena_root / candidate
        cwd_candidate = Path.cwd() / candidate
        candidate = root_candidate if root_candidate.is_file() else cwd_candidate
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ShoppingSmokeSetupError("Shopping task file is unavailable.")
    return resolved


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-standard JSON constant {value!r} is not allowed.")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys are not allowed.")
        result[key] = value
    return result


def _load_full_task(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, ValueError, RecursionError):
        raise ShoppingSmokeSetupError("Shopping task file is not strict UTF-8 JSON.") from None
    if type(value) is not dict:
        raise ShoppingSmokeSetupError("Shopping task must be a JSON object.")
    return value


def _write_single_product_task(
    *,
    full_task_file: Path,
    product_ordinal: int,
    destination: Path,
    scripted: bool,
) -> None:
    single_task = build_single_product_task(
        _load_full_task(full_task_file),
        product_ordinal=product_ordinal,
    )
    if scripted:
        steps = single_task.get("steps")
        if (
            product_ordinal != 1
            or type(steps) is not list
            or len(steps) != 1
            or type(steps[0]) is not dict
            or steps[0].get("target_asin") != _SCRIPTED_PRODUCT_1_TARGET
        ):
            raise ShoppingSmokeSetupError("Scripted smoke supports only its fixed Product 1 fixture.")
    try:
        destination.write_text(
            json.dumps(
                single_task,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError, UnicodeError, RecursionError):
        raise ShoppingSmokeSetupError("Single-Product temporary task creation failed.") from None


def _local_upstream_port(base_url: str) -> int:
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError:
        port = None
        parsed = None
    if (
        parsed is None
        or parsed.scheme != "http"
        or parsed.hostname not in _LOCAL_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or port is None
        or not 1 <= port <= 65_535
    ):
        raise ShoppingSmokeSetupError("upstream_url must identify one explicit local HTTP port.")
    return port


def _resolve_upstream_items_file(
    *,
    memoryarena_root: Path,
    upstream_items_file: Path | None,
) -> tuple[Path, bool]:
    """Return the service catalog and whether it differs from the full benchmark catalog."""

    full_catalog = (memoryarena_root / "data" / "shopping" / "items_shuffle.json").resolve()
    candidate = full_catalog if upstream_items_file is None else upstream_items_file.expanduser()
    if not candidate.is_absolute():
        root_candidate = memoryarena_root / candidate
        cwd_candidate = Path.cwd() / candidate
        candidate = root_candidate if root_candidate.is_file() else cwd_candidate
    selected = candidate.resolve()
    if not selected.is_file():
        raise ShoppingSmokeSetupError("Upstream WebShop items file is unavailable.")
    try:
        reduced_smoke_catalog = not os.path.samefile(selected, full_catalog)
    except OSError:
        raise ShoppingSmokeSetupError("Full WebShop items file is unavailable.") from None
    return selected, reduced_smoke_catalog


def _absolute_path_without_resolving_links(path: Path) -> Path:
    """Make a path absolute while retaining a virtualenv interpreter symlink."""

    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


def _service_reachable(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/",
            timeout=1.0,
        ) as response:
            return response.status == 200
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _wait_for_service(
    *,
    base_url: str,
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        if _service_reachable(base_url):
            if process.poll() is None:
                return
            break
        time.sleep(0.25)
    raise ShoppingSmokeSetupError("Upstream WebShop service did not become ready.")


def _build_upstream_child_environment(
    *,
    memoryarena_root: Path,
    python_path: Path,
    items_file: Path,
    api_key_env: str,
    additional_secret_env_names: tuple[str, ...] = (),
) -> dict[str, str]:
    """Build the local environment-service process env without model secrets."""

    child_environment = os.environ.copy()
    for secret_name in {
        api_key_env,
        DEFAULT_API_KEY_ENV,
        _MEMORY_API_KEY_ENV,
        *additional_secret_env_names,
    }:
        child_environment.pop(secret_name, None)
    child_environment["PYTHONUNBUFFERED"] = "1"
    child_environment["MEMORYARENA_WEBSHOP_DATA_ROOT"] = str(memoryarena_root / "data" / "shopping")
    child_environment["MEMORYARENA_WEBSHOP_ITEMS_FILE"] = str(items_file)
    child_environment["MEMORYARENA_WEBSHOP_SEARCH_ROOT"] = str(memoryarena_root / "data" / "shopping" / "search_engine")
    python_bin = str(python_path.parent)
    child_environment["PATH"] = (
        python_bin if not child_environment.get("PATH") else python_bin + os.pathsep + child_environment["PATH"]
    )
    return child_environment


@contextmanager
def _upstream_service(
    *,
    memoryarena_root: Path,
    base_url: str,
    python_executable: Path,
    items_file: Path,
    startup_timeout_seconds: float,
    api_key_env: str,
    additional_secret_env_names: tuple[str, ...] = (),
) -> Iterator[None]:
    port = _local_upstream_port(base_url)
    if _service_reachable(base_url):
        raise ShoppingSmokeSetupError("upstream_url is already in use; choose a free local port.")
    python_path = _absolute_path_without_resolving_links(python_executable)
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise ShoppingSmokeSetupError("upstream_python must be an executable Python interpreter.")
    if (
        isinstance(startup_timeout_seconds, bool)
        or not isinstance(startup_timeout_seconds, (int, float))
        or not 0 < float(startup_timeout_seconds) <= 600
    ):
        raise ShoppingSmokeSetupError("upstream startup timeout must be between 0 and 600 seconds.")

    child_environment = _build_upstream_child_environment(
        memoryarena_root=memoryarena_root,
        python_path=python_path,
        items_file=items_file,
        api_key_env=api_key_env,
        additional_secret_env_names=additional_secret_env_names,
    )
    try:
        process = subprocess.Popen(
            [
                str(python_path),
                "-m",
                _UPSTREAM_MODULE,
                "--port",
                str(port),
                "--limit_goals",
                "1",
            ],
            cwd=str(memoryarena_root),
            env=child_environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, ValueError):
        raise ShoppingSmokeSetupError("Upstream WebShop service could not be started.") from None
    try:
        _wait_for_service(
            base_url=base_url,
            process=process,
            timeout_seconds=float(startup_timeout_seconds),
        )
        yield
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15.0)


def run_shopping_smoke(
    *,
    memoryarena_root: Path,
    task_file: Path | None,
    product_ordinal: int,
    upstream_url: str,
    upstream_python: Path | None,
    upstream_items_file: Path | None,
    upstream_startup_timeout_seconds: float,
    environment_timeout_seconds: int,
    max_observation_chars: int,
    max_inference_steps: int,
    max_consecutive_tool_failures: int,
    mode: ShoppingSmokeMode,
    api_key_env: str,
    answer_max_tokens: int,
) -> ShoppingSmokeSummary:
    """Compose one real WebShop Product without persisting its trajectory."""

    if type(product_ordinal) is not int or product_ordinal <= 0:
        raise ShoppingSmokeSetupError("product_ordinal must be positive.")
    root = _expose_memoryarena(memoryarena_root)
    source_task = _resolve_task_file(
        memoryarena_root=root,
        task_file=task_file,
    )
    runtime_python = root / ".venv-runtime" / "bin" / "python" if upstream_python is None else upstream_python
    selected_items_file, reduced_smoke_catalog = _resolve_upstream_items_file(
        memoryarena_root=root,
        upstream_items_file=upstream_items_file,
    )
    if mode is ShoppingSmokeMode.SCRIPTED:
        answer_client = _ScriptedShoppingAnswerClient()
    elif mode is ShoppingSmokeMode.LIVE_DEEPSEEK:
        answer_client = _build_deepseek_client(
            api_key_env=api_key_env,
            answer_max_tokens=answer_max_tokens,
        )
    else:
        raise ShoppingSmokeSetupError("Unsupported Shopping smoke mode.")

    with tempfile.TemporaryDirectory(prefix="memoryarena-shopping-smoke-") as temporary_directory:
        single_task_file = Path(temporary_directory) / "single_product_task.json"
        _write_single_product_task(
            full_task_file=source_task,
            product_ordinal=product_ordinal,
            destination=single_task_file,
            scripted=mode is ShoppingSmokeMode.SCRIPTED,
        )
        with _upstream_service(
            memoryarena_root=root,
            base_url=upstream_url,
            python_executable=runtime_python,
            items_file=selected_items_file,
            startup_timeout_seconds=upstream_startup_timeout_seconds,
            api_key_env=api_key_env,
        ):
            session = ShoppingEnvironmentSession.from_memoryarena(
                ShoppingEnvironmentSessionConfig(
                    env_server_base=upstream_url,
                    single_product_task_file=str(single_task_file),
                    max_observation_chars=max_observation_chars,
                    timeout_seconds=environment_timeout_seconds,
                    max_rounds=max_inference_steps,
                    product_catalog_dir=str(root / "data" / "shopping" / "product_catalog"),
                    domain_data_path=str(root / "data" / "shopping" / "domain_data.json"),
                )
            )
            with session:
                initial_observation = session.reset(seed=0)
                subtask = ShoppingTaskInputAdapter(max_observation_chars=max_observation_chars).build_subtask_input(
                    initial_observation.to_dict()
                )
                result = AnswerAgentSubtaskRunner(
                    client=answer_client,
                    executor=ShoppingToolExecutorAdapter(session=session),
                    config=AnswerAgentSubtaskRunnerConfig(
                        max_inference_steps=max_inference_steps,
                        max_tool_calls_per_inference=1,
                        max_consecutive_tool_failures=(max_consecutive_tool_failures),
                        interaction_mode=(AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL),
                    ),
                ).run(subtask, tool_definitions=tool_definitions())
                termination_evidence = session.termination_evidence
    return summarize_shopping_result(
        result,
        mode=mode,
        product_ordinal=product_ordinal,
        reduced_smoke_catalog=reduced_smoke_catalog,
        termination_evidence=termination_evidence,
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
        "--product-ordinal",
        type=int,
        default=DEFAULT_PRODUCT_ORDINAL,
    )
    parser.add_argument("--upstream-url", default=DEFAULT_UPSTREAM_URL)
    parser.add_argument("--upstream-python", type=Path, default=None)
    parser.add_argument(
        "--upstream-items-file",
        type=Path,
        default=None,
        help="Optional reduced catalog for local smoke only; omit for the full benchmark catalog.",
    )
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
    parser.add_argument(
        "--live-deepseek",
        action="store_true",
        help="Opt in to paid DeepSeek calls instead of deterministic replay.",
    )
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=DEFAULT_ANSWER_MAX_TOKENS,
    )
    return parser.parse_args()


def _safe_failure_payload(
    *,
    code: ShoppingSmokeVerificationCode,
    mode: ShoppingSmokeMode,
    product_ordinal: int,
) -> dict[str, object]:
    return {
        "smoke_pass": False,
        "unscored": True,
        "saved_trajectory": False,
        "mode": mode.value,
        "scripted_oracle": mode is ShoppingSmokeMode.SCRIPTED,
        "verification_code": code.value,
        "product_ordinal": product_ordinal,
    }


def main() -> int:
    args = _parse_args()
    mode = ShoppingSmokeMode.LIVE_DEEPSEEK if args.live_deepseek else ShoppingSmokeMode.SCRIPTED
    if args.live_deepseek:
        if (
            type(args.api_key_env) is not str
            or _ENV_NAME_PATTERN.fullmatch(args.api_key_env) is None
            or not os.environ.get(args.api_key_env)
        ):
            print(
                json.dumps(
                    _safe_failure_payload(
                        code=ShoppingSmokeVerificationCode.API_KEY_MISSING,
                        mode=mode,
                        product_ordinal=args.product_ordinal,
                    ),
                    indent=2,
                )
            )
            return 1

    try:
        with open(os.devnull, "w", encoding="utf-8") as discarded_output:
            with redirect_stdout(discarded_output), redirect_stderr(discarded_output):
                summary = run_shopping_smoke(
                    memoryarena_root=args.memoryarena_root,
                    task_file=args.task_file,
                    product_ordinal=args.product_ordinal,
                    upstream_url=args.upstream_url,
                    upstream_python=args.upstream_python,
                    upstream_items_file=args.upstream_items_file,
                    upstream_startup_timeout_seconds=(args.upstream_startup_timeout_seconds),
                    environment_timeout_seconds=args.environment_timeout_seconds,
                    max_observation_chars=args.max_observation_chars,
                    max_inference_steps=args.max_inference_steps,
                    max_consecutive_tool_failures=(args.max_consecutive_tool_failures),
                    mode=mode,
                    api_key_env=args.api_key_env,
                    answer_max_tokens=args.answer_max_tokens,
                )
    except (
        MemoryArenaShoppingDependencyError,
        ShoppingEnvironmentSessionError,
        ShoppingSmokeSetupError,
        ShoppingTaskInputAdapterError,
        ShoppingTaskSplitterError,
        ValueError,
    ):
        print(
            json.dumps(
                _safe_failure_payload(
                    code=ShoppingSmokeVerificationCode.SETUP_FAILED,
                    mode=mode,
                    product_ordinal=args.product_ordinal,
                ),
                indent=2,
            )
        )
        return 1
    except Exception:
        print(
            json.dumps(
                _safe_failure_payload(
                    code=ShoppingSmokeVerificationCode.UNEXPECTED_FAILURE,
                    mode=mode,
                    product_ordinal=args.product_ordinal,
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
    "ShoppingSmokeMode",
    "ShoppingSmokeSetupError",
    "ShoppingSmokeSummary",
    "ShoppingSmokeVerificationCode",
    "run_shopping_smoke",
    "summarize_shopping_result",
]
