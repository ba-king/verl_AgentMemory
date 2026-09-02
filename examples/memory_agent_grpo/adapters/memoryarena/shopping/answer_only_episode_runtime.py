"""Six-Product Answer-only orchestration for MemoryArena Shopping."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ....answer_agent.interaction_mode import AnswerAgentInteractionMode
from ....answer_agent.model_client import AnswerAgentConversationClient
from ....answer_agent.runtime import (
    AnswerAgentSubtaskRunner,
    AnswerAgentSubtaskRunnerConfig,
    AnswerAgentSubtaskRunResult,
    AnswerAgentSubtaskRunStatus,
)
from ....answer_agent.tool_protocol.definition import ToolDefinition
from ....answer_agent.tool_protocol.observation import ToolObservationStatus
from ....answer_agent.tool_protocol.validation import validate_tool_definition
from .environment_session import (
    ShoppingEnvironmentSession,
    ShoppingTerminationEvidence,
    ShoppingTerminationKind,
)
from .executor_adapter import ShoppingToolExecutorAdapter
from .task_input_adapter import ShoppingTaskInputAdapter
from .task_splitter import ShoppingTaskSplitterError, build_single_product_task

SHOPPING_ANSWER_ONLY_PRODUCT_COUNT = 6


class ShoppingAnswerOnlyEpisodeRunStatus(str, Enum):
    """Terminal status for one ordered six-Product run."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ShoppingAnswerOnlyEpisodeEvaluationPolicy(str, Enum):
    """Control whether trusted evaluation failures stop the episode."""

    FAIL_FAST = "FAIL_FAST"
    CONTINUE_EVALUATION_FAILURES = "CONTINUE_EVALUATION_FAILURES"


class ShoppingAnswerOnlyEpisodeFailureCode(str, Enum):
    """Stable failure categories that contain no task or model content."""

    SESSION_FAILED = "SESSION_FAILED"
    ANSWER_INPUT_FAILED = "ANSWER_INPUT_FAILED"
    ANSWER_RUNNER_EXCEPTION = "ANSWER_RUNNER_EXCEPTION"
    ANSWER_RUNNER_FAILED = "ANSWER_RUNNER_FAILED"
    TERMINAL_COUNT_INVALID = "TERMINAL_COUNT_INVALID"
    PURCHASE_NOT_COMPLETED = "PURCHASE_NOT_COMPLETED"
    TARGET_MISMATCH = "TARGET_MISMATCH"


class ShoppingAnswerOnlyEpisodeInputErrorCode(str, Enum):
    """Caller errors detected before a Product session is opened."""

    PRODUCT_COUNT_INVALID = "PRODUCT_COUNT_INVALID"
    TASK_SPLIT_FAILED = "TASK_SPLIT_FAILED"


class ShoppingAnswerOnlyEpisodeInputError(ValueError):
    """Safe preflight failure that never embeds the supplied task."""

    def __init__(
        self,
        code: ShoppingAnswerOnlyEpisodeInputErrorCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


class ShoppingSingleProductSessionFactory(Protocol):
    """Open one fresh trusted session for one pre-split Product task."""

    def open_session(
        self,
        *,
        single_product_task: dict[str, object],
        product_ordinal: int,
    ) -> AbstractContextManager[ShoppingEnvironmentSession]: ...


def _terminal_tool_execution_count(result: AnswerAgentSubtaskRunResult) -> int:
    return sum(outcome.is_terminal for step in result.steps for outcome in step.tool_outcomes)


def _product_failure_code(
    result: AnswerAgentSubtaskRunResult,
    evidence: ShoppingTerminationEvidence,
) -> ShoppingAnswerOnlyEpisodeFailureCode | None:
    if result.status is not AnswerAgentSubtaskRunStatus.COMPLETED:
        return ShoppingAnswerOnlyEpisodeFailureCode.ANSWER_RUNNER_FAILED
    if _terminal_tool_execution_count(result) != 1:
        return ShoppingAnswerOnlyEpisodeFailureCode.TERMINAL_COUNT_INVALID
    if evidence.kind is not ShoppingTerminationKind.PURCHASE_COMPLETED:
        return ShoppingAnswerOnlyEpisodeFailureCode.PURCHASE_NOT_COMPLETED
    if evidence.purchased_target_matched is not True:
        return ShoppingAnswerOnlyEpisodeFailureCode.TARGET_MISMATCH
    return None


@dataclass(frozen=True)
class ShoppingProductRunResult:
    """One Product trace paired with trusted purchase evidence."""

    product_ordinal: int
    answer_result: AnswerAgentSubtaskRunResult
    termination_evidence: ShoppingTerminationEvidence

    def __post_init__(self) -> None:
        if type(self.product_ordinal) is not int or not 1 <= self.product_ordinal <= SHOPPING_ANSWER_ONLY_PRODUCT_COUNT:
            raise ValueError("product_ordinal must select one of the six Products.")
        if type(self.answer_result) is not AnswerAgentSubtaskRunResult:
            raise ValueError("answer_result must be an AnswerAgentSubtaskRunResult.")
        if type(self.termination_evidence) is not ShoppingTerminationEvidence:
            raise ValueError("termination_evidence must be ShoppingTerminationEvidence.")

    @property
    def terminal_tool_execution_count(self) -> int:
        return _terminal_tool_execution_count(self.answer_result)

    @property
    def failure_code(self) -> ShoppingAnswerOnlyEpisodeFailureCode | None:
        return _product_failure_code(
            self.answer_result,
            self.termination_evidence,
        )

    @property
    def passed(self) -> bool:
        return self.failure_code is None

    def to_dict(self) -> dict[str, object | None]:
        """Return allowlisted structural evidence without raw model content."""

        runner_error = self.answer_result.error_code
        provider_error = self.answer_result.provider_error_code
        outcomes = tuple(outcome for step in self.answer_result.steps for outcome in step.tool_outcomes)
        return {
            "product_ordinal": self.product_ordinal,
            "passed": self.passed,
            "failure_code": (None if self.failure_code is None else self.failure_code.value),
            "runner_status": self.answer_result.status.value,
            "inference_step_count": self.answer_result.inference_step_count,
            "emitted_tool_call_count": sum(len(step.response.tool_calls) for step in self.answer_result.steps),
            "executed_tool_count": self.answer_result.executed_tool_count,
            "successful_tool_observation_count": sum(
                outcome.tool_observation.status is ToolObservationStatus.SUCCESS for outcome in outcomes
            ),
            "error_tool_observation_count": sum(
                outcome.tool_observation.status is ToolObservationStatus.ERROR for outcome in outcomes
            ),
            "terminal_tool_execution_count": self.terminal_tool_execution_count,
            "termination_kind": self.termination_evidence.kind.value,
            "purchased_target_matched": (self.termination_evidence.purchased_target_matched),
            "runner_error_code": (None if runner_error is None else runner_error.value),
            "provider_error_code": (None if provider_error is None else provider_error.value),
        }


_FAILURES_WITHOUT_PRODUCT_RESULT = frozenset(
    {
        ShoppingAnswerOnlyEpisodeFailureCode.SESSION_FAILED,
        ShoppingAnswerOnlyEpisodeFailureCode.ANSWER_INPUT_FAILED,
        ShoppingAnswerOnlyEpisodeFailureCode.ANSWER_RUNNER_EXCEPTION,
    }
)

_CONTINUABLE_EVALUATION_FAILURES = frozenset(
    {
        ShoppingAnswerOnlyEpisodeFailureCode.PURCHASE_NOT_COMPLETED,
        ShoppingAnswerOnlyEpisodeFailureCode.TARGET_MISMATCH,
    }
)


@dataclass(frozen=True)
class ShoppingAnswerOnlyEpisodeResult:
    """Safe ordered result with distinct first-failure and stop evidence."""

    status: ShoppingAnswerOnlyEpisodeRunStatus
    product_results: tuple[ShoppingProductRunResult, ...]
    completed_product_count: int
    failed_product_ordinal: int | None = None
    failure_code: ShoppingAnswerOnlyEpisodeFailureCode | None = None
    stopped_product_ordinal: int | None = None
    stop_failure_code: ShoppingAnswerOnlyEpisodeFailureCode | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not ShoppingAnswerOnlyEpisodeRunStatus:
            raise ValueError("status must be a ShoppingAnswerOnlyEpisodeRunStatus.")
        if type(self.product_results) is not tuple or any(
            type(result) is not ShoppingProductRunResult for result in self.product_results
        ):
            raise ValueError("product_results must be ShoppingProductRunResult values.")
        if tuple(result.product_ordinal for result in self.product_results) != tuple(
            range(1, len(self.product_results) + 1)
        ):
            raise ValueError("product_results must be a contiguous one-based prefix.")
        if len(self.product_results) > SHOPPING_ANSWER_ONLY_PRODUCT_COUNT:
            raise ValueError("product_results cannot exceed six Products.")
        passed_count = sum(result.passed for result in self.product_results)
        if type(self.completed_product_count) is not int or self.completed_product_count != passed_count:
            raise ValueError("completed_product_count must match passed Product results.")

        if self.status is ShoppingAnswerOnlyEpisodeRunStatus.COMPLETED:
            if (
                len(self.product_results) != SHOPPING_ANSWER_ONLY_PRODUCT_COUNT
                or self.completed_product_count != SHOPPING_ANSWER_ONLY_PRODUCT_COUNT
                or any(not result.passed for result in self.product_results)
                or self.failed_product_ordinal is not None
                or self.failure_code is not None
                or self.stopped_product_ordinal is not None
                or self.stop_failure_code is not None
            ):
                raise ValueError("A completed episode requires six passed Products.")
            return

        if (
            not isinstance(
                self.failure_code,
                ShoppingAnswerOnlyEpisodeFailureCode,
            )
            or type(self.failed_product_ordinal) is not int
            or not 1 <= self.failed_product_ordinal <= SHOPPING_ANSWER_ONLY_PRODUCT_COUNT
        ):
            raise ValueError("A failed episode requires its first failed Product.")

        has_stop_ordinal = self.stopped_product_ordinal is not None
        has_stop_code = self.stop_failure_code is not None
        if has_stop_ordinal != has_stop_code:
            raise ValueError("A stop boundary requires both ordinal and failure code.")
        if has_stop_ordinal:
            if (
                type(self.stopped_product_ordinal) is not int
                or self.stopped_product_ordinal != len(self.product_results) + 1
                or not 1 <= self.stopped_product_ordinal <= SHOPPING_ANSWER_ONLY_PRODUCT_COUNT
                or self.stop_failure_code not in _FAILURES_WITHOUT_PRODUCT_RESULT
            ):
                raise ValueError("A stop boundary must follow the retained Product prefix.")

        failed_results = tuple(result for result in self.product_results if not result.passed)
        for result in self.product_results[:-1]:
            if not result.passed and result.failure_code not in _CONTINUABLE_EVALUATION_FAILURES:
                raise ValueError("Only evaluation failures may be retained before the final result.")

        if failed_results:
            first_failed_result = failed_results[0]
            if (
                first_failed_result.product_ordinal != self.failed_product_ordinal
                or first_failed_result.failure_code is not self.failure_code
            ):
                raise ValueError("First failure fields must match the earliest failed Product result.")
            if has_stop_ordinal and self.failed_product_ordinal >= self.stopped_product_ordinal:
                raise ValueError("A stop boundary must occur after the first retained failure.")
            if (
                not has_stop_ordinal
                and len(self.product_results) < SHOPPING_ANSWER_ONLY_PRODUCT_COUNT
                and self.product_results[-1].passed
            ):
                raise ValueError("An incomplete retained prefix must end in failure.")
            return

        if not has_stop_ordinal:
            raise ValueError("A failed episode requires retained or stop-boundary failure evidence.")
        if (
            self.failed_product_ordinal != self.stopped_product_ordinal
            or self.failure_code is not self.stop_failure_code
        ):
            raise ValueError("A first boundary failure must match the stop boundary.")

    @property
    def passed(self) -> bool:
        return self.status is ShoppingAnswerOnlyEpisodeRunStatus.COMPLETED

    @property
    def attempted_product_count(self) -> int:
        if self.stopped_product_ordinal is not None:
            return self.stopped_product_ordinal
        return len(self.product_results)

    def to_dict(self) -> dict[str, object | None]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "expected_product_count": SHOPPING_ANSWER_ONLY_PRODUCT_COUNT,
            "completed_product_count": self.completed_product_count,
            "attempted_product_count": self.attempted_product_count,
            "failed_product_ordinal": self.failed_product_ordinal,
            "failure_code": (None if self.failure_code is None else self.failure_code.value),
            "stopped_product_ordinal": self.stopped_product_ordinal,
            "stop_failure_code": (None if self.stop_failure_code is None else self.stop_failure_code.value),
            "product_results": [product_result.to_dict() for product_result in self.product_results],
        }


class _AnswerInputFailed(RuntimeError):
    """Internal content-free stage marker."""


class _AnswerRunnerRaised(RuntimeError):
    """Internal content-free stage marker."""


def _validated_tool_definitions(
    value: object,
) -> tuple[ToolDefinition, ...]:
    if type(value) is not tuple:
        raise ValueError("tool_definitions must be a tuple.")
    seen_names: set[str] = set()
    for position, definition in enumerate(value):
        if type(definition) is not ToolDefinition:
            raise ValueError(f"tool_definitions[{position}] must be a ToolDefinition.")
        validation = validate_tool_definition(definition)
        if not validation.is_valid:
            code = "UNKNOWN" if validation.error_code is None else validation.error_code.value
            raise ValueError(f"tool_definitions[{position}] is invalid ({code}).")
        if definition.name in seen_names:
            raise ValueError("tool definition names must be unique.")
        seen_names.add(definition.name)
    return value


def _split_six_products(
    full_task_definition: object,
) -> tuple[dict[str, object], ...]:
    if type(full_task_definition) is not dict:
        raise ShoppingAnswerOnlyEpisodeInputError(
            ShoppingAnswerOnlyEpisodeInputErrorCode.TASK_SPLIT_FAILED,
            "Shopping task preflight failed.",
        )
    steps = full_task_definition.get("steps")
    if type(steps) is not list or len(steps) != SHOPPING_ANSWER_ONLY_PRODUCT_COUNT:
        raise ShoppingAnswerOnlyEpisodeInputError(
            ShoppingAnswerOnlyEpisodeInputErrorCode.PRODUCT_COUNT_INVALID,
            "Answer-only Shopping requires exactly six Products.",
        )
    try:
        return tuple(
            build_single_product_task(
                full_task_definition,
                product_ordinal=product_ordinal,
            )
            for product_ordinal in range(
                1,
                SHOPPING_ANSWER_ONLY_PRODUCT_COUNT + 1,
            )
        )
    except ShoppingTaskSplitterError:
        raise ShoppingAnswerOnlyEpisodeInputError(
            ShoppingAnswerOnlyEpisodeInputErrorCode.TASK_SPLIT_FAILED,
            "Shopping task preflight failed.",
        ) from None


class ShoppingAnswerOnlyEpisodeRunner:
    """Run six isolated Product conversations through one injected service."""

    def __init__(
        self,
        *,
        answer_client: AnswerAgentConversationClient,
        session_factory: ShoppingSingleProductSessionFactory,
        task_input_adapter: ShoppingTaskInputAdapter,
        tool_definitions: object,
        config: AnswerAgentSubtaskRunnerConfig,
        evaluation_policy: ShoppingAnswerOnlyEpisodeEvaluationPolicy = (
            ShoppingAnswerOnlyEpisodeEvaluationPolicy.FAIL_FAST
        ),
    ) -> None:
        if not callable(getattr(answer_client, "generate", None)):
            raise ValueError("answer_client must expose a callable generate method.")
        open_session = getattr(session_factory, "open_session", None)
        if not callable(open_session):
            raise ValueError("session_factory must expose a callable open_session method.")
        if type(task_input_adapter) is not ShoppingTaskInputAdapter:
            raise ValueError("task_input_adapter must be a ShoppingTaskInputAdapter.")
        if type(config) is not AnswerAgentSubtaskRunnerConfig:
            raise ValueError("config must be an AnswerAgentSubtaskRunnerConfig.")
        if type(evaluation_policy) is not ShoppingAnswerOnlyEpisodeEvaluationPolicy:
            raise ValueError("evaluation_policy must be a ShoppingAnswerOnlyEpisodeEvaluationPolicy.")
        if (
            config.interaction_mode is not AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL
            or config.max_tool_calls_per_inference != 1
        ):
            raise ValueError("Shopping Answer-only episodes require environment-terminated single-tool mode.")

        self._answer_client = answer_client
        self._session_factory = session_factory
        self._task_input_adapter = task_input_adapter
        self._tool_definitions = _validated_tool_definitions(tool_definitions)
        self._config = config
        self._evaluation_policy = evaluation_policy

    @property
    def config(self) -> AnswerAgentSubtaskRunnerConfig:
        return self._config

    @property
    def tool_definitions(self) -> tuple[ToolDefinition, ...]:
        return self._tool_definitions

    @property
    def evaluation_policy(self) -> ShoppingAnswerOnlyEpisodeEvaluationPolicy:
        return self._evaluation_policy

    def run(
        self,
        full_task_definition: object,
    ) -> ShoppingAnswerOnlyEpisodeResult:
        """Preflight all six Products, then apply the configured failure policy."""

        single_product_tasks = _split_six_products(full_task_definition)
        product_results: list[ShoppingProductRunResult] = []
        first_failure_ordinal: int | None = None
        first_failure_code: ShoppingAnswerOnlyEpisodeFailureCode | None = None

        for product_ordinal, single_product_task in enumerate(
            single_product_tasks,
            start=1,
        ):
            product_result, boundary_failure = self._run_product(
                product_ordinal=product_ordinal,
                single_product_task=single_product_task,
            )
            if boundary_failure is not None:
                if first_failure_ordinal is None:
                    first_failure_ordinal = product_ordinal
                    first_failure_code = boundary_failure
                return ShoppingAnswerOnlyEpisodeResult(
                    status=ShoppingAnswerOnlyEpisodeRunStatus.FAILED,
                    product_results=tuple(product_results),
                    completed_product_count=sum(result.passed for result in product_results),
                    failed_product_ordinal=first_failure_ordinal,
                    failure_code=first_failure_code,
                    stopped_product_ordinal=product_ordinal,
                    stop_failure_code=boundary_failure,
                )
            assert product_result is not None
            product_results.append(product_result)
            if not product_result.passed:
                product_failure = product_result.failure_code
                assert product_failure is not None
                if first_failure_ordinal is None:
                    first_failure_ordinal = product_ordinal
                    first_failure_code = product_failure
                if (
                    self._evaluation_policy is ShoppingAnswerOnlyEpisodeEvaluationPolicy.CONTINUE_EVALUATION_FAILURES
                    and product_failure in _CONTINUABLE_EVALUATION_FAILURES
                ):
                    continue
                return ShoppingAnswerOnlyEpisodeResult(
                    status=ShoppingAnswerOnlyEpisodeRunStatus.FAILED,
                    product_results=tuple(product_results),
                    completed_product_count=sum(result.passed for result in product_results),
                    failed_product_ordinal=first_failure_ordinal,
                    failure_code=first_failure_code,
                )

        if first_failure_ordinal is not None:
            return ShoppingAnswerOnlyEpisodeResult(
                status=ShoppingAnswerOnlyEpisodeRunStatus.FAILED,
                product_results=tuple(product_results),
                completed_product_count=sum(result.passed for result in product_results),
                failed_product_ordinal=first_failure_ordinal,
                failure_code=first_failure_code,
            )
        return ShoppingAnswerOnlyEpisodeResult(
            status=ShoppingAnswerOnlyEpisodeRunStatus.COMPLETED,
            product_results=tuple(product_results),
            completed_product_count=SHOPPING_ANSWER_ONLY_PRODUCT_COUNT,
        )

    def _run_product(
        self,
        *,
        product_ordinal: int,
        single_product_task: dict[str, object],
    ) -> tuple[
        ShoppingProductRunResult | None,
        ShoppingAnswerOnlyEpisodeFailureCode | None,
    ]:
        answer_result: AnswerAgentSubtaskRunResult | None = None
        termination_evidence: ShoppingTerminationEvidence | None = None
        try:
            context_manager = self._session_factory.open_session(
                single_product_task=single_product_task,
                product_ordinal=product_ordinal,
            )
            with context_manager as session:
                if type(session) is not ShoppingEnvironmentSession:
                    raise ValueError("session_factory must yield a ShoppingEnvironmentSession.")
                initial_observation = session.reset(seed=0)
                try:
                    answer_input = self._task_input_adapter.build_subtask_input(initial_observation.to_dict())
                    if answer_input.retrieved_memories:
                        raise ValueError("Answer-only Shopping input must have empty memories.")
                except Exception:
                    raise _AnswerInputFailed from None

                try:
                    answer_result = AnswerAgentSubtaskRunner(
                        client=self._answer_client,
                        executor=ShoppingToolExecutorAdapter(session=session),
                        config=self._config,
                    ).run(
                        answer_input,
                        tool_definitions=self._tool_definitions,
                    )
                except Exception:
                    raise _AnswerRunnerRaised from None
                termination_evidence = session.termination_evidence
        except _AnswerInputFailed:
            return (
                None,
                ShoppingAnswerOnlyEpisodeFailureCode.ANSWER_INPUT_FAILED,
            )
        except _AnswerRunnerRaised:
            return (
                None,
                ShoppingAnswerOnlyEpisodeFailureCode.ANSWER_RUNNER_EXCEPTION,
            )
        except Exception:
            return None, ShoppingAnswerOnlyEpisodeFailureCode.SESSION_FAILED

        assert answer_result is not None
        assert termination_evidence is not None
        return (
            ShoppingProductRunResult(
                product_ordinal=product_ordinal,
                answer_result=answer_result,
                termination_evidence=termination_evidence,
            ),
            None,
        )


__all__ = [
    "SHOPPING_ANSWER_ONLY_PRODUCT_COUNT",
    "ShoppingAnswerOnlyEpisodeEvaluationPolicy",
    "ShoppingAnswerOnlyEpisodeFailureCode",
    "ShoppingAnswerOnlyEpisodeInputError",
    "ShoppingAnswerOnlyEpisodeInputErrorCode",
    "ShoppingAnswerOnlyEpisodeResult",
    "ShoppingAnswerOnlyEpisodeRunStatus",
    "ShoppingAnswerOnlyEpisodeRunner",
    "ShoppingProductRunResult",
    "ShoppingSingleProductSessionFactory",
]
