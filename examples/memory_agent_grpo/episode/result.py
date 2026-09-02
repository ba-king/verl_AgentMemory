"""Terminal result contract for one isolated episode run."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from examples.memory_agent_grpo.answer_agent.model_client import (
    AnswerAgentClientErrorCode,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState

from .input import EpisodeInput
from .trace import EpisodeTrace


class EpisodeRunStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EpisodeRunErrorCode(str, Enum):
    """Stable coarse failure locations; provider text is never stored here."""

    INITIALIZATION_FAILED = "INITIALIZATION_FAILED"
    SUBTASK_FAILED = "SUBTASK_FAILED"
    INTERNAL_CONTRACT_ERROR = "INTERNAL_CONTRACT_ERROR"


class EpisodeFailureStage(str, Enum):
    """Stable stage location with no model- or dataset-authored text."""

    INITIALIZATION = "INITIALIZATION"
    PRE_TASK = "PRE_TASK"
    ANSWER = "ANSWER"
    POST_TASK = "POST_TASK"
    INTERNAL = "INTERNAL"


class EpisodeFailureDetailCode(str, Enum):
    """Finite, secret-free reason for one uncommitted episode failure."""

    MEMORY_MODEL_CALL_FAILED = "MEMORY_MODEL_CALL_FAILED"
    MEMORY_OUTPUT_VALIDATION_FAILED = "MEMORY_OUTPUT_VALIDATION_FAILED"
    MEMORY_EXECUTION_REJECTED = "MEMORY_EXECUTION_REJECTED"

    ANSWER_PROVIDER_ERROR = "ANSWER_PROVIDER_ERROR"
    ANSWER_INVALID_MODEL_RESPONSE = "ANSWER_INVALID_MODEL_RESPONSE"
    ANSWER_TOOL_CALL_BATCH_LIMIT = "ANSWER_TOOL_CALL_BATCH_LIMIT"
    ANSWER_DUPLICATE_TOOL_CALL_ID = "ANSWER_DUPLICATE_TOOL_CALL_ID"
    ANSWER_CONSECUTIVE_TOOL_FAILURE_LIMIT = "ANSWER_CONSECUTIVE_TOOL_FAILURE_LIMIT"
    ANSWER_INFERENCE_STEP_LIMIT = "ANSWER_INFERENCE_STEP_LIMIT"
    ANSWER_RUNNER_EXCEPTION = "ANSWER_RUNNER_EXCEPTION"

    MEMORY_INTERNAL_CONTRACT_ERROR = "MEMORY_INTERNAL_CONTRACT_ERROR"
    RETRIEVAL_CONTRACT_ERROR = "RETRIEVAL_CONTRACT_ERROR"
    ANSWER_RESULT_CONTRACT_ERROR = "ANSWER_RESULT_CONTRACT_ERROR"
    EPISODE_TRACE_CONTRACT_ERROR = "EPISODE_TRACE_CONTRACT_ERROR"
    UNEXPECTED_EXCEPTION = "UNEXPECTED_EXCEPTION"


@dataclass(frozen=True)
class EpisodeRunResult:
    """A complete run or an auditable prefix ending at a typed failure."""

    status: EpisodeRunStatus
    episode_input: EpisodeInput
    trace: EpisodeTrace | None
    completed_subtask_count: int
    final_bank_state: MemoryBankState
    error_code: EpisodeRunErrorCode | None = None
    failed_subtask_ordinal: int | None = None
    failed_stage: EpisodeFailureStage | None = None
    failure_detail_code: EpisodeFailureDetailCode | None = None
    answer_provider_error_code: AnswerAgentClientErrorCode | None = None

    def to_dict(self) -> dict[str, object | None]:
        return {
            "status": self.status.value,
            "episode_input": self.episode_input.to_dict(),
            "trace": None if self.trace is None else self.trace.to_dict(),
            "completed_subtask_count": self.completed_subtask_count,
            "final_bank_state": self.final_bank_state.to_dict(),
            "error_code": (None if self.error_code is None else self.error_code.value),
            "failed_subtask_ordinal": self.failed_subtask_ordinal,
            "failed_stage": (
                None if self.failed_stage is None else self.failed_stage.value
            ),
            "failure_detail_code": (
                None
                if self.failure_detail_code is None
                else self.failure_detail_code.value
            ),
            "answer_provider_error_code": (
                None
                if self.answer_provider_error_code is None
                else self.answer_provider_error_code.value
            ),
        }


__all__ = [
    "EpisodeFailureDetailCode",
    "EpisodeFailureStage",
    "EpisodeRunErrorCode",
    "EpisodeRunResult",
    "EpisodeRunStatus",
]
