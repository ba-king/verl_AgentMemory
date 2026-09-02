from dataclasses import replace

import pytest

from examples.memory_agent_grpo.answer_agent.model_client import (
    AnswerAgentClientErrorCode,
)

from examples.memory_agent_grpo.episode.result import (
    EpisodeFailureDetailCode,
    EpisodeFailureStage,
    EpisodeRunErrorCode,
    EpisodeRunResult,
    EpisodeRunStatus,
)
from examples.memory_agent_grpo.episode.validation import (
    EpisodeValidationErrorCode,
    validate_episode_run_result,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState

from ._support import build_valid_episode


def test_completed_episode_result_requires_the_full_trace() -> None:
    _, _, result = build_valid_episode()

    assert validate_episode_run_result(result).is_valid
    assert result.failed_stage is None
    assert result.failure_detail_code is None
    assert result.answer_provider_error_code is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("failed_stage", EpisodeFailureStage.ANSWER),
        (
            "failure_detail_code",
            EpisodeFailureDetailCode.ANSWER_INFERENCE_STEP_LIMIT,
        ),
        (
            "answer_provider_error_code",
            AnswerAgentClientErrorCode.REQUEST_TIMEOUT,
        ),
    ],
)
def test_completed_result_rejects_every_failure_diagnostic(
    field_name: str,
    value: object,
) -> None:
    _, _, result = build_valid_episode()

    validation = validate_episode_run_result(replace(result, **{field_name: value}))

    assert validation.error_code is EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH


def test_completed_result_cannot_claim_wrong_final_bank() -> None:
    _, _, result = build_valid_episode()
    forged = replace(result, final_bank_state=MemoryBankState.empty())

    validation = validate_episode_run_result(forged)

    assert validation.error_code is EpisodeValidationErrorCode.BANK_CONTINUITY_MISMATCH


def test_initialization_failure_has_no_reusable_bank_state() -> None:
    episode_input, _, _ = build_valid_episode()
    result = EpisodeRunResult(
        status=EpisodeRunStatus.FAILED,
        episode_input=episode_input,
        trace=None,
        completed_subtask_count=0,
        final_bank_state=MemoryBankState.empty(),
        error_code=EpisodeRunErrorCode.INITIALIZATION_FAILED,
        failed_stage=EpisodeFailureStage.INITIALIZATION,
        failure_detail_code=(EpisodeFailureDetailCode.MEMORY_OUTPUT_VALIDATION_FAILED),
    )

    assert validate_episode_run_result(result).is_valid


def test_subtask_failure_must_follow_completed_prefix() -> None:
    episode_input, trace, _ = build_valid_episode()
    prefix = replace(
        trace,
        subtasks=(),
        final_bank_state=trace.initialization.bank_after,
    )
    result = EpisodeRunResult(
        status=EpisodeRunStatus.FAILED,
        episode_input=episode_input,
        trace=prefix,
        completed_subtask_count=0,
        final_bank_state=prefix.final_bank_state,
        error_code=EpisodeRunErrorCode.SUBTASK_FAILED,
        failed_subtask_ordinal=1,
        failed_stage=EpisodeFailureStage.PRE_TASK,
        failure_detail_code=(EpisodeFailureDetailCode.MEMORY_OUTPUT_VALIDATION_FAILED),
    )

    assert validate_episode_run_result(result).is_valid


def test_subtask_failure_ordinal_cannot_be_boolean() -> None:
    episode_input, trace, _ = build_valid_episode()
    prefix = replace(
        trace,
        subtasks=(),
        final_bank_state=trace.initialization.bank_after,
    )
    result = EpisodeRunResult(
        status=EpisodeRunStatus.FAILED,
        episode_input=episode_input,
        trace=prefix,
        completed_subtask_count=0,
        final_bank_state=prefix.final_bank_state,
        error_code=EpisodeRunErrorCode.SUBTASK_FAILED,
        failed_subtask_ordinal=True,  # type: ignore[arg-type]
        failed_stage=EpisodeFailureStage.PRE_TASK,
        failure_detail_code=(EpisodeFailureDetailCode.MEMORY_OUTPUT_VALIDATION_FAILED),
    )

    validation = validate_episode_run_result(result)

    assert validation.error_code is EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH


def test_failed_result_cannot_claim_all_subtasks_completed() -> None:
    episode_input, trace, _ = build_valid_episode()
    result = EpisodeRunResult(
        status=EpisodeRunStatus.FAILED,
        episode_input=episode_input,
        trace=trace,
        completed_subtask_count=1,
        final_bank_state=trace.final_bank_state,
        error_code=EpisodeRunErrorCode.SUBTASK_FAILED,
        failed_subtask_ordinal=2,
        failed_stage=EpisodeFailureStage.ANSWER,
        failure_detail_code=(EpisodeFailureDetailCode.ANSWER_INFERENCE_STEP_LIMIT),
    )

    validation = validate_episode_run_result(result)

    assert validation.error_code is EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH


def test_internal_failure_cannot_point_to_completed_subtask() -> None:
    episode_input, trace, _ = build_valid_episode()
    result = EpisodeRunResult(
        status=EpisodeRunStatus.FAILED,
        episode_input=episode_input,
        trace=trace,
        completed_subtask_count=1,
        final_bank_state=trace.final_bank_state,
        error_code=EpisodeRunErrorCode.INTERNAL_CONTRACT_ERROR,
        failed_subtask_ordinal=1,
        failed_stage=EpisodeFailureStage.INTERNAL,
        failure_detail_code=EpisodeFailureDetailCode.UNEXPECTED_EXCEPTION,
    )

    validation = validate_episode_run_result(result)

    assert validation.error_code is EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("failed_stage", None),
        ("failure_detail_code", None),
        ("failed_stage", "ANSWER"),
        ("failure_detail_code", "ANSWER_PROVIDER_ERROR"),
    ],
)
def test_failed_result_requires_typed_stage_and_detail(
    field_name: str,
    value: object,
) -> None:
    episode_input, trace, _ = build_valid_episode()
    prefix = replace(
        trace,
        subtasks=(),
        final_bank_state=trace.initialization.bank_after,
    )
    result = EpisodeRunResult(
        status=EpisodeRunStatus.FAILED,
        episode_input=episode_input,
        trace=prefix,
        completed_subtask_count=0,
        final_bank_state=prefix.final_bank_state,
        error_code=EpisodeRunErrorCode.SUBTASK_FAILED,
        failed_subtask_ordinal=1,
        failed_stage=EpisodeFailureStage.ANSWER,
        failure_detail_code=EpisodeFailureDetailCode.ANSWER_PROVIDER_ERROR,
        answer_provider_error_code=AnswerAgentClientErrorCode.REQUEST_TIMEOUT,
    )

    validation = validate_episode_run_result(replace(result, **{field_name: value}))

    assert validation.error_code is EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH


@pytest.mark.parametrize(
    ("failed_stage", "failure_detail_code"),
    [
        (
            EpisodeFailureStage.ANSWER,
            EpisodeFailureDetailCode.MEMORY_MODEL_CALL_FAILED,
        ),
        (
            EpisodeFailureStage.PRE_TASK,
            EpisodeFailureDetailCode.ANSWER_INFERENCE_STEP_LIMIT,
        ),
        (
            EpisodeFailureStage.POST_TASK,
            EpisodeFailureDetailCode.ANSWER_RUNNER_EXCEPTION,
        ),
        (
            EpisodeFailureStage.INTERNAL,
            EpisodeFailureDetailCode.MEMORY_OUTPUT_VALIDATION_FAILED,
        ),
    ],
)
def test_failure_detail_must_belong_to_its_stage(
    failed_stage: EpisodeFailureStage,
    failure_detail_code: EpisodeFailureDetailCode,
) -> None:
    episode_input, trace, _ = build_valid_episode()
    prefix = replace(
        trace,
        subtasks=(),
        final_bank_state=trace.initialization.bank_after,
    )
    result = EpisodeRunResult(
        status=EpisodeRunStatus.FAILED,
        episode_input=episode_input,
        trace=prefix,
        completed_subtask_count=0,
        final_bank_state=prefix.final_bank_state,
        error_code=(
            EpisodeRunErrorCode.INTERNAL_CONTRACT_ERROR
            if failed_stage is EpisodeFailureStage.INTERNAL
            else EpisodeRunErrorCode.SUBTASK_FAILED
        ),
        failed_subtask_ordinal=1,
        failed_stage=failed_stage,
        failure_detail_code=failure_detail_code,
    )

    validation = validate_episode_run_result(result)

    assert validation.error_code is EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH


def test_initialization_failure_requires_initialization_stage() -> None:
    episode_input, _, _ = build_valid_episode()
    result = EpisodeRunResult(
        status=EpisodeRunStatus.FAILED,
        episode_input=episode_input,
        trace=None,
        completed_subtask_count=0,
        final_bank_state=MemoryBankState.empty(),
        error_code=EpisodeRunErrorCode.INITIALIZATION_FAILED,
        failed_stage=EpisodeFailureStage.PRE_TASK,
        failure_detail_code=EpisodeFailureDetailCode.MEMORY_MODEL_CALL_FAILED,
    )

    validation = validate_episode_run_result(result)

    assert validation.error_code is EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH


def test_internal_failure_requires_internal_stage_and_detail() -> None:
    episode_input, _, _ = build_valid_episode()
    result = EpisodeRunResult(
        status=EpisodeRunStatus.FAILED,
        episode_input=episode_input,
        trace=None,
        completed_subtask_count=0,
        final_bank_state=MemoryBankState.empty(),
        error_code=EpisodeRunErrorCode.INTERNAL_CONTRACT_ERROR,
        failed_stage=EpisodeFailureStage.INITIALIZATION,
        failure_detail_code=(EpisodeFailureDetailCode.MEMORY_INTERNAL_CONTRACT_ERROR),
    )

    validation = validate_episode_run_result(result)

    assert validation.error_code is EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH


@pytest.mark.parametrize(
    ("failure_detail_code", "provider_error_code"),
    [
        (EpisodeFailureDetailCode.ANSWER_PROVIDER_ERROR, None),
        (
            EpisodeFailureDetailCode.ANSWER_INFERENCE_STEP_LIMIT,
            AnswerAgentClientErrorCode.REQUEST_TIMEOUT,
        ),
    ],
)
def test_answer_provider_subtype_exists_only_for_provider_failure(
    failure_detail_code: EpisodeFailureDetailCode,
    provider_error_code: AnswerAgentClientErrorCode | None,
) -> None:
    episode_input, trace, _ = build_valid_episode()
    prefix = replace(
        trace,
        subtasks=(),
        final_bank_state=trace.initialization.bank_after,
    )
    result = EpisodeRunResult(
        status=EpisodeRunStatus.FAILED,
        episode_input=episode_input,
        trace=prefix,
        completed_subtask_count=0,
        final_bank_state=prefix.final_bank_state,
        error_code=EpisodeRunErrorCode.SUBTASK_FAILED,
        failed_subtask_ordinal=1,
        failed_stage=EpisodeFailureStage.ANSWER,
        failure_detail_code=failure_detail_code,
        answer_provider_error_code=provider_error_code,
    )

    validation = validate_episode_run_result(result)

    assert validation.error_code is EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH
