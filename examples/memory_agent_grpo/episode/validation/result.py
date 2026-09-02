"""Validation for terminal episode results and failed prefixes."""

from __future__ import annotations

from examples.memory_agent_grpo.answer_agent.model_client import (
    AnswerAgentClientErrorCode,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState

from ..result import (
    EpisodeFailureDetailCode,
    EpisodeFailureStage,
    EpisodeRunErrorCode,
    EpisodeRunResult,
    EpisodeRunStatus,
)
from .common import (
    EpisodeValidationErrorCode,
    EpisodeValidationResult,
    invalid,
    valid_bank_state,
)
from .input import validate_episode_input
from .trace import validate_episode_trace


_MEMORY_STAGE_DETAILS = {
    EpisodeFailureDetailCode.MEMORY_MODEL_CALL_FAILED,
    EpisodeFailureDetailCode.MEMORY_OUTPUT_VALIDATION_FAILED,
    EpisodeFailureDetailCode.MEMORY_EXECUTION_REJECTED,
}
_ANSWER_STAGE_DETAILS = {
    EpisodeFailureDetailCode.ANSWER_PROVIDER_ERROR,
    EpisodeFailureDetailCode.ANSWER_INVALID_MODEL_RESPONSE,
    EpisodeFailureDetailCode.ANSWER_TOOL_CALL_BATCH_LIMIT,
    EpisodeFailureDetailCode.ANSWER_DUPLICATE_TOOL_CALL_ID,
    EpisodeFailureDetailCode.ANSWER_CONSECUTIVE_TOOL_FAILURE_LIMIT,
    EpisodeFailureDetailCode.ANSWER_INFERENCE_STEP_LIMIT,
    EpisodeFailureDetailCode.ANSWER_RUNNER_EXCEPTION,
}
_INTERNAL_DETAILS = {
    EpisodeFailureDetailCode.MEMORY_INTERNAL_CONTRACT_ERROR,
    EpisodeFailureDetailCode.RETRIEVAL_CONTRACT_ERROR,
    EpisodeFailureDetailCode.ANSWER_RESULT_CONTRACT_ERROR,
    EpisodeFailureDetailCode.EPISODE_TRACE_CONTRACT_ERROR,
    EpisodeFailureDetailCode.UNEXPECTED_EXCEPTION,
}


def validate_episode_run_result(
    result: object,
) -> EpisodeValidationResult:
    if type(result) is not EpisodeRunResult:
        return invalid(
            EpisodeValidationErrorCode.WRONG_TYPE,
            "result must be an EpisodeRunResult.",
        )
    input_validation = validate_episode_input(result.episode_input)
    if not input_validation.is_valid:
        return input_validation
    if type(result.status) is not EpisodeRunStatus:
        return invalid(
            EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
            "result.status must be an EpisodeRunStatus.",
        )
    if (
        type(result.completed_subtask_count) is not int
        or result.completed_subtask_count < 0
        or result.completed_subtask_count > result.episode_input.subtask_count
        or not valid_bank_state(result.final_bank_state)
    ):
        return invalid(
            EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
            "Result counts and final Bank must be structurally valid.",
        )

    traced_count = 0
    if result.trace is None:
        if result.final_bank_state != MemoryBankState.empty():
            return invalid(
                EpisodeValidationErrorCode.BANK_CONTINUITY_MISMATCH,
                "A result without initialization trace must retain the empty Bank.",
            )
    else:
        trace_validation = validate_episode_trace(
            result.trace,
            episode_input=result.episode_input,
        )
        if not trace_validation.is_valid:
            return trace_validation
        traced_count = len(result.trace.subtasks)
        if result.final_bank_state != result.trace.final_bank_state:
            return invalid(
                EpisodeValidationErrorCode.BANK_CONTINUITY_MISMATCH,
                "Result and trace must expose the same final Bank snapshot.",
            )
    if result.completed_subtask_count != traced_count:
        return invalid(
            EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
            "completed_subtask_count must equal the completed trace prefix.",
        )

    if result.status is EpisodeRunStatus.COMPLETED:
        if (
            result.trace is None
            or traced_count != result.episode_input.subtask_count
            or result.error_code is not None
            or result.failed_subtask_ordinal is not None
            or result.failed_stage is not None
            or result.failure_detail_code is not None
            or result.answer_provider_error_code is not None
        ):
            return invalid(
                EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
                "A completed result requires the full trace and no failure fields.",
            )
        return EpisodeValidationResult()

    if type(result.error_code) is not EpisodeRunErrorCode:
        return invalid(
            EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
            "A failed result requires one typed error code.",
        )
    if (
        type(result.failed_stage) is not EpisodeFailureStage
        or type(result.failure_detail_code) is not EpisodeFailureDetailCode
    ):
        return invalid(
            EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
            "A failed result requires typed stage and detail codes.",
        )
    if result.failure_detail_code is EpisodeFailureDetailCode.ANSWER_PROVIDER_ERROR:
        if type(result.answer_provider_error_code) is not AnswerAgentClientErrorCode:
            return invalid(
                EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
                "An Answer provider failure requires one typed provider code.",
            )
    elif result.answer_provider_error_code is not None:
        return invalid(
            EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
            "A provider code is allowed only for an Answer provider failure.",
        )
    if result.completed_subtask_count >= result.episode_input.subtask_count:
        return invalid(
            EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
            "A failed result must end at a strict completed-subtask prefix.",
        )
    if result.failed_stage is EpisodeFailureStage.INITIALIZATION:
        if (
            result.error_code is not EpisodeRunErrorCode.INITIALIZATION_FAILED
            or result.failure_detail_code not in _MEMORY_STAGE_DETAILS
            or result.trace is not None
            or result.completed_subtask_count != 0
            or result.failed_subtask_ordinal is not None
        ):
            return invalid(
                EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
                "Initialization failure cannot claim initialized episode state.",
            )
    elif result.failed_stage in {
        EpisodeFailureStage.PRE_TASK,
        EpisodeFailureStage.POST_TASK,
    }:
        expected_failed = result.completed_subtask_count + 1
        if (
            result.error_code is not EpisodeRunErrorCode.SUBTASK_FAILED
            or result.failure_detail_code not in _MEMORY_STAGE_DETAILS
            or result.trace is None
            or type(result.failed_subtask_ordinal) is not int
            or result.failed_subtask_ordinal != expected_failed
        ):
            return invalid(
                EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
                "A Memory subtask failure must follow the exact prefix.",
            )
    elif result.failed_stage is EpisodeFailureStage.ANSWER:
        expected_failed = result.completed_subtask_count + 1
        if (
            result.error_code is not EpisodeRunErrorCode.SUBTASK_FAILED
            or result.failure_detail_code not in _ANSWER_STAGE_DETAILS
            or result.trace is None
            or type(result.failed_subtask_ordinal) is not int
            or result.failed_subtask_ordinal != expected_failed
        ):
            return invalid(
                EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
                "An Answer failure must follow the exact completed prefix.",
            )
    else:
        if (
            result.failed_stage is not EpisodeFailureStage.INTERNAL
            or result.error_code is not EpisodeRunErrorCode.INTERNAL_CONTRACT_ERROR
            or result.failure_detail_code not in _INTERNAL_DETAILS
        ):
            return invalid(
                EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
                "An internal failure requires an internal detail code.",
            )
        if result.trace is None:
            if result.failed_subtask_ordinal is not None:
                return invalid(
                    EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
                    "An internal initialization failure cannot name a subtask.",
                )
        elif (
            type(result.failed_subtask_ordinal) is not int
            or result.failed_subtask_ordinal != result.completed_subtask_count + 1
        ):
            return invalid(
                EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
                "Internal failure must identify the next incomplete subtask.",
            )
    return EpisodeValidationResult()


__all__ = ["validate_episode_run_result"]
