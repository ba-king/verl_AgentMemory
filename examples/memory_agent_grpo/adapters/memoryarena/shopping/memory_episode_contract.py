"""Contracts for one Memory-aware six-Product Shopping episode."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ....answer_agent.input import AnswerAgentSubtaskInput
from ....answer_agent.model_client import AnswerAgentClientErrorCode
from ....answer_agent.runtime import (
    AnswerAgentSubtaskErrorCode,
    AnswerAgentSubtaskRunResult,
)
from ....episode.trace import MemoryDecisionTrace
from ....memory_agent.action.protocol import MemoryDecisionPhase
from ....memory_agent.action.validation import ValidationErrorCode
from ....memory_agent.bank.bank_state import MemoryBankState
from ....memory_agent.decision.context import InteractionEvent, TaskResult
from ....memory_agent.decision.model_client import MemoryAgentClientErrorCode
from .answer_only_episode_runtime import ShoppingProductRunResult
from .environment_session import (
    ShoppingEnvironmentSession,
    ShoppingTerminationEvidence,
    ShoppingTerminationKind,
)

SHOPPING_MEMORY_EPISODE_PRODUCT_COUNT = 6


class ShoppingMemoryEpisodeRunStatus(str, Enum):
    """Whether all six Product control-flow stages completed structurally."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ShoppingMemoryEpisodeFailureStage(str, Enum):
    """Stable location of a structural episode failure."""

    SESSION = "SESSION"
    ANSWER_INPUT = "ANSWER_INPUT"
    PRE_TASK = "PRE_TASK"
    ANSWER = "ANSWER"
    POST_TASK = "POST_TASK"
    INTERNAL = "INTERNAL"


class ShoppingMemoryEpisodeFailureCode(str, Enum):
    """Content-free reasons why a structural trajectory stopped."""

    SESSION_FAILED = "SESSION_FAILED"
    ANSWER_INPUT_FAILED = "ANSWER_INPUT_FAILED"
    MEMORY_MODEL_CALL_FAILED = "MEMORY_MODEL_CALL_FAILED"
    MEMORY_OUTPUT_VALIDATION_FAILED = "MEMORY_OUTPUT_VALIDATION_FAILED"
    MEMORY_EXECUTION_REJECTED = "MEMORY_EXECUTION_REJECTED"
    MEMORY_INTERNAL_CONTRACT_ERROR = "MEMORY_INTERNAL_CONTRACT_ERROR"
    ANSWER_RUNNER_EXCEPTION = "ANSWER_RUNNER_EXCEPTION"
    ANSWER_RUNNER_FAILED = "ANSWER_RUNNER_FAILED"
    TERMINAL_COUNT_INVALID = "TERMINAL_COUNT_INVALID"
    TRACE_CONTRACT_ERROR = "TRACE_CONTRACT_ERROR"
    UNEXPECTED_EXCEPTION = "UNEXPECTED_EXCEPTION"


class ShoppingMemoryEpisodeInputErrorCode(str, Enum):
    """Preflight failures raised before any Product session is opened."""

    PRODUCT_COUNT_INVALID = "PRODUCT_COUNT_INVALID"
    TASK_SPLIT_FAILED = "TASK_SPLIT_FAILED"


class ShoppingMemoryEpisodeInputError(ValueError):
    """Safe caller error that never embeds task content."""

    def __init__(
        self,
        code: ShoppingMemoryEpisodeInputErrorCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


class ShoppingMemoryEpisodeSessionFactory(Protocol):
    """Open one fresh trusted session for one pre-split Product task."""

    def open_session(
        self,
        *,
        single_product_task: dict[str, object],
        product_ordinal: int,
    ) -> AbstractContextManager[ShoppingEnvironmentSession]: ...


@dataclass(frozen=True)
class ShoppingMemoryOutputCorrection:
    """Content-free audit record for the episode's sole correction attempt."""

    product_ordinal: int
    phase: MemoryDecisionPhase
    trigger_error_code: ValidationErrorCode
    recovered: bool

    def __post_init__(self) -> None:
        if (
            type(self.product_ordinal) is not int
            or not 1
            <= self.product_ordinal
            <= SHOPPING_MEMORY_EPISODE_PRODUCT_COUNT
            or type(self.phase) is not MemoryDecisionPhase
            or type(self.trigger_error_code) is not ValidationErrorCode
            or type(self.recovered) is not bool
        ):
            raise ValueError("Memory output correction metadata is invalid.")


@dataclass(frozen=True)
class ShoppingMemoryProductTrace:
    """One committed Product, including its complete in-memory agent trace."""

    product_ordinal: int
    memory_task_input: str
    pre_task_decision: MemoryDecisionTrace
    answer_input: AnswerAgentSubtaskInput
    answer_result: AnswerAgentSubtaskRunResult
    interaction_events: tuple[InteractionEvent, ...]
    task_result: TaskResult
    post_task_decisions: tuple[MemoryDecisionTrace, ...]
    termination_evidence: ShoppingTerminationEvidence
    bank_before: MemoryBankState
    bank_after: MemoryBankState

    def __post_init__(self) -> None:
        from .memory_episode_validation import validate_memory_product_trace

        validate_memory_product_trace(self)

    @property
    def product_result(self) -> ShoppingProductRunResult:
        return ShoppingProductRunResult(
            product_ordinal=self.product_ordinal,
            answer_result=self.answer_result,
            termination_evidence=self.termination_evidence,
        )

    @property
    def memory_decision_count(self) -> int:
        return 1 + len(self.post_task_decisions)

    @property
    def retrieved_memory_count(self) -> int:
        return len(self.answer_input.retrieved_memories)

    def to_safe_dict(self) -> dict[str, object | None]:
        """Drop task text, observations, model output, and memory contents."""

        from .memory_episode_summary import memory_product_trace_to_safe_dict

        return memory_product_trace_to_safe_dict(self)


@dataclass(frozen=True)
class ShoppingMemoryEpisodeResult:
    """A completed six-Product trace or its last committed Product prefix."""

    status: ShoppingMemoryEpisodeRunStatus
    product_traces: tuple[ShoppingMemoryProductTrace, ...]
    final_bank_state: MemoryBankState
    memory_agent_invocation_count: int
    memory_output_correction_budget: int = 0
    memory_output_correction: ShoppingMemoryOutputCorrection | None = None
    failed_product_ordinal: int | None = None
    failure_stage: ShoppingMemoryEpisodeFailureStage | None = None
    failure_code: ShoppingMemoryEpisodeFailureCode | None = None
    memory_validation_error_code: ValidationErrorCode | None = None
    memory_model_error_code: MemoryAgentClientErrorCode | None = None
    answer_error_code: AnswerAgentSubtaskErrorCode | None = None
    answer_provider_error_code: AnswerAgentClientErrorCode | None = None

    def __post_init__(self) -> None:
        from .memory_episode_validation import validate_memory_episode_result

        validate_memory_episode_result(self)

    @property
    def pipeline_completed(self) -> bool:
        return self.status is ShoppingMemoryEpisodeRunStatus.COMPLETED

    @property
    def memory_decision_count(self) -> int:
        return sum(trace.memory_decision_count for trace in self.product_traces)

    @property
    def target_match_count(self) -> int:
        return sum(
            trace.termination_evidence.purchased_target_matched is True
            for trace in self.product_traces
        )

    @property
    def purchase_completed_count(self) -> int:
        return sum(
            trace.termination_evidence.kind
            is ShoppingTerminationKind.PURCHASE_COMPLETED
            for trace in self.product_traces
        )

    @property
    def memory_effect_observed(self) -> bool:
        from .memory_episode_summary import memory_effect_observed

        return memory_effect_observed(self.product_traces)

    def to_safe_dict(self) -> dict[str, object | None]:
        from .memory_episode_summary import memory_episode_result_to_safe_dict

        return memory_episode_result_to_safe_dict(self)


__all__ = [
    "SHOPPING_MEMORY_EPISODE_PRODUCT_COUNT",
    "ShoppingMemoryEpisodeFailureCode",
    "ShoppingMemoryEpisodeFailureStage",
    "ShoppingMemoryEpisodeInputError",
    "ShoppingMemoryEpisodeInputErrorCode",
    "ShoppingMemoryEpisodeResult",
    "ShoppingMemoryEpisodeRunStatus",
    "ShoppingMemoryEpisodeSessionFactory",
    "ShoppingMemoryOutputCorrection",
    "ShoppingMemoryProductTrace",
]
