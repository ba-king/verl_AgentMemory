"""Integrity checks for dataset-neutral Memory Decision Contexts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from examples.memory_agent_grpo.memory_agent.action.protocol import MemoryDecisionPhase
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryRecord
from examples.memory_agent_grpo.memory_agent.bank.validation import is_canonical_memory_id
from examples.memory_agent_grpo.memory_agent.retrieval.result import RetrievalMatch, RetrievalResult

from .context import (
    InteractionEvent,
    InteractionEventType,
    MemoryDecisionContext,
    TaskResult,
    TaskResultStatus,
)


class ContextValidationErrorCode(str, Enum):
    CONTEXT_NOT_MEMORY_DECISION_CONTEXT = "CONTEXT_NOT_MEMORY_DECISION_CONTEXT"
    PHASE_NOT_MEMORY_DECISION_PHASE = "PHASE_NOT_MEMORY_DECISION_PHASE"
    TASK_INPUT_NOT_STRING = "TASK_INPUT_NOT_STRING"
    EMPTY_TASK_INPUT = "EMPTY_TASK_INPUT"
    BANK_RECORD_COUNT_NOT_INTEGER = "BANK_RECORD_COUNT_NOT_INTEGER"
    BANK_RECORD_COUNT_OUT_OF_RANGE = "BANK_RECORD_COUNT_OUT_OF_RANGE"
    INTERACTION_EVENTS_NOT_TUPLE = "INTERACTION_EVENTS_NOT_TUPLE"
    INTERACTION_EVENT_NOT_TYPED = "INTERACTION_EVENT_NOT_TYPED"
    INTERACTION_EVENT_TYPE_NOT_ENUM = "INTERACTION_EVENT_TYPE_NOT_ENUM"
    INTERACTION_CONTENT_NOT_STRING = "INTERACTION_CONTENT_NOT_STRING"
    EMPTY_INTERACTION_CONTENT = "EMPTY_INTERACTION_CONTENT"
    TASK_RESULT_NOT_TYPED = "TASK_RESULT_NOT_TYPED"
    TASK_RESULT_STATUS_NOT_ENUM = "TASK_RESULT_STATUS_NOT_ENUM"
    FINAL_ANSWER_NOT_STRING = "FINAL_ANSWER_NOT_STRING"
    COMPLETED_TASK_HAS_EMPTY_FINAL_ANSWER = "COMPLETED_TASK_HAS_EMPTY_FINAL_ANSWER"
    REWARD_NOT_FINITE_NUMBER = "REWARD_NOT_FINITE_NUMBER"
    FEEDBACK_NOT_STRING = "FEEDBACK_NOT_STRING"
    EMPTY_FEEDBACK = "EMPTY_FEEDBACK"
    RETRIEVAL_NOT_TYPED = "RETRIEVAL_NOT_TYPED"
    RETRIEVAL_QUERY_NOT_STRING = "RETRIEVAL_QUERY_NOT_STRING"
    EMPTY_RETRIEVAL_QUERY = "EMPTY_RETRIEVAL_QUERY"
    RETRIEVAL_MATCHES_NOT_TUPLE = "RETRIEVAL_MATCHES_NOT_TUPLE"
    RETRIEVAL_MATCH_NOT_TYPED = "RETRIEVAL_MATCH_NOT_TYPED"
    RETRIEVAL_RECORD_NOT_TYPED = "RETRIEVAL_RECORD_NOT_TYPED"
    RETRIEVAL_MEMORY_ID_NOT_STRING = "RETRIEVAL_MEMORY_ID_NOT_STRING"
    EMPTY_RETRIEVAL_MEMORY_ID = "EMPTY_RETRIEVAL_MEMORY_ID"
    INVALID_RETRIEVAL_MEMORY_ID = "INVALID_RETRIEVAL_MEMORY_ID"
    RETRIEVAL_CONTENT_NOT_STRING = "RETRIEVAL_CONTENT_NOT_STRING"
    EMPTY_RETRIEVAL_CONTENT = "EMPTY_RETRIEVAL_CONTENT"
    RETRIEVAL_SCORE_NOT_FINITE_NUMBER = "RETRIEVAL_SCORE_NOT_FINITE_NUMBER"
    RETRIEVAL_SCORE_NOT_POSITIVE = "RETRIEVAL_SCORE_NOT_POSITIVE"
    DUPLICATE_MEMORY_ID_IN_RETRIEVAL = "DUPLICATE_MEMORY_ID_IN_RETRIEVAL"
    INCONSISTENT_VISIBLE_MEMORY_CONTENT = "INCONSISTENT_VISIBLE_MEMORY_CONTENT"
    PRE_TASK_HAS_TASK_EXECUTION_DATA = "PRE_TASK_HAS_TASK_EXECUTION_DATA"
    POST_TASK_MISSING_TASK_RESULT = "POST_TASK_MISSING_TASK_RESULT"
    POST_TASK_HAS_POST_TASK_RETRIEVAL = "POST_TASK_HAS_POST_TASK_RETRIEVAL"
    AFTER_RETRIEVAL_MISSING_TASK_RESULT = "AFTER_RETRIEVAL_MISSING_TASK_RESULT"
    AFTER_RETRIEVAL_MISSING_POST_TASK_RETRIEVAL = (
        "AFTER_RETRIEVAL_MISSING_POST_TASK_RETRIEVAL"
    )


@dataclass(frozen=True)
class ContextValidationResult:
    error_code: ContextValidationErrorCode | None = None
    error_message: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.error_code is None


def _invalid(code: ContextValidationErrorCode, message: str) -> ContextValidationResult:
    return ContextValidationResult(error_code=code, error_message=message)


def _validate_task_result(result: object) -> ContextValidationResult:
    if not isinstance(result, TaskResult):
        return _invalid(
            ContextValidationErrorCode.TASK_RESULT_NOT_TYPED,
            "task_result must be a TaskResult or None.",
        )
    if not isinstance(result.status, TaskResultStatus):
        return _invalid(
            ContextValidationErrorCode.TASK_RESULT_STATUS_NOT_ENUM,
            "task_result.status must be a TaskResultStatus.",
        )
    if not isinstance(result.final_answer, str):
        return _invalid(
            ContextValidationErrorCode.FINAL_ANSWER_NOT_STRING,
            "task_result.final_answer must be a string.",
        )
    if result.status is TaskResultStatus.COMPLETED and not result.final_answer.strip():
        return _invalid(
            ContextValidationErrorCode.COMPLETED_TASK_HAS_EMPTY_FINAL_ANSWER,
            "A completed task must have a non-empty final_answer.",
        )
    if result.reward is not None:
        if isinstance(result.reward, bool) or not isinstance(result.reward, (int, float)):
            return _invalid(
                ContextValidationErrorCode.REWARD_NOT_FINITE_NUMBER,
                "task_result.reward must be a finite number or None.",
            )
        if not math.isfinite(float(result.reward)):
            return _invalid(
                ContextValidationErrorCode.REWARD_NOT_FINITE_NUMBER,
                "task_result.reward must be a finite number or None.",
            )
    if result.feedback is not None:
        if not isinstance(result.feedback, str):
            return _invalid(
                ContextValidationErrorCode.FEEDBACK_NOT_STRING,
                "task_result.feedback must be a string or None.",
            )
        if not result.feedback.strip():
            return _invalid(
                ContextValidationErrorCode.EMPTY_FEEDBACK,
                "task_result.feedback must not be empty when present.",
            )
    return ContextValidationResult()


def _validate_retrieval(retrieval: object, *, field_name: str) -> ContextValidationResult:
    if not isinstance(retrieval, RetrievalResult):
        return _invalid(
            ContextValidationErrorCode.RETRIEVAL_NOT_TYPED,
            f"{field_name} must be a RetrievalResult or None.",
        )
    if not isinstance(retrieval.query, str):
        return _invalid(
            ContextValidationErrorCode.RETRIEVAL_QUERY_NOT_STRING,
            f"{field_name}.query must be a string.",
        )
    if not retrieval.query.strip():
        return _invalid(
            ContextValidationErrorCode.EMPTY_RETRIEVAL_QUERY,
            f"{field_name}.query must not be empty.",
        )
    if not isinstance(retrieval.matches, tuple):
        return _invalid(
            ContextValidationErrorCode.RETRIEVAL_MATCHES_NOT_TUPLE,
            f"{field_name}.matches must be an immutable tuple.",
        )

    seen_ids: set[str] = set()
    for position, match in enumerate(retrieval.matches):
        location = f"{field_name}.matches[{position}]"
        if not isinstance(match, RetrievalMatch):
            return _invalid(
                ContextValidationErrorCode.RETRIEVAL_MATCH_NOT_TYPED,
                f"{location} must be a RetrievalMatch.",
            )
        if not isinstance(match.record, MemoryRecord):
            return _invalid(
                ContextValidationErrorCode.RETRIEVAL_RECORD_NOT_TYPED,
                f"{location}.record must be a MemoryRecord.",
            )
        if not isinstance(match.memory_id, str):
            return _invalid(
                ContextValidationErrorCode.RETRIEVAL_MEMORY_ID_NOT_STRING,
                f"{location}.memory_id must be a string.",
            )
        if not match.memory_id.strip():
            return _invalid(
                ContextValidationErrorCode.EMPTY_RETRIEVAL_MEMORY_ID,
                f"{location}.memory_id must not be empty.",
            )
        if not is_canonical_memory_id(match.memory_id):
            return _invalid(
                ContextValidationErrorCode.INVALID_RETRIEVAL_MEMORY_ID,
                f"{location}.memory_id must use canonical M001-style format.",
            )
        if match.memory_id in seen_ids:
            return _invalid(
                ContextValidationErrorCode.DUPLICATE_MEMORY_ID_IN_RETRIEVAL,
                f"{field_name} contains duplicate memory ID {match.memory_id!r}.",
            )
        if not isinstance(match.content, str):
            return _invalid(
                ContextValidationErrorCode.RETRIEVAL_CONTENT_NOT_STRING,
                f"{location}.content must be a string.",
            )
        if not match.content.strip():
            return _invalid(
                ContextValidationErrorCode.EMPTY_RETRIEVAL_CONTENT,
                f"{location}.content must not be empty.",
            )
        if isinstance(match.score, bool) or not isinstance(match.score, (int, float)):
            return _invalid(
                ContextValidationErrorCode.RETRIEVAL_SCORE_NOT_FINITE_NUMBER,
                f"{location}.score must be a finite number.",
            )
        if not math.isfinite(float(match.score)):
            return _invalid(
                ContextValidationErrorCode.RETRIEVAL_SCORE_NOT_FINITE_NUMBER,
                f"{location}.score must be a finite number.",
            )
        if float(match.score) <= 0.0:
            return _invalid(
                ContextValidationErrorCode.RETRIEVAL_SCORE_NOT_POSITIVE,
                f"{location}.score must be positive.",
            )
        seen_ids.add(match.memory_id)
    return ContextValidationResult()


def validate_memory_decision_context(context: object) -> ContextValidationResult:
    """Validate trusted runner output without repairing or normalizing it."""

    if not isinstance(context, MemoryDecisionContext):
        return _invalid(
            ContextValidationErrorCode.CONTEXT_NOT_MEMORY_DECISION_CONTEXT,
            "context must be a MemoryDecisionContext.",
        )
    if not isinstance(context.phase, MemoryDecisionPhase):
        return _invalid(
            ContextValidationErrorCode.PHASE_NOT_MEMORY_DECISION_PHASE,
            "phase must be a MemoryDecisionPhase.",
        )
    if not isinstance(context.task_input, str):
        return _invalid(
            ContextValidationErrorCode.TASK_INPUT_NOT_STRING,
            "task_input must be a string.",
        )
    if not context.task_input.strip():
        return _invalid(
            ContextValidationErrorCode.EMPTY_TASK_INPUT,
            "task_input must not be empty.",
        )
    if isinstance(context.bank_record_count, bool) or not isinstance(context.bank_record_count, int):
        return _invalid(
            ContextValidationErrorCode.BANK_RECORD_COUNT_NOT_INTEGER,
            "bank_record_count must be an integer.",
        )
    if context.bank_record_count < 0:
        return _invalid(
            ContextValidationErrorCode.BANK_RECORD_COUNT_OUT_OF_RANGE,
            "bank_record_count must be at least zero.",
        )
    if not isinstance(context.interaction_events, tuple):
        return _invalid(
            ContextValidationErrorCode.INTERACTION_EVENTS_NOT_TUPLE,
            "interaction_events must be an immutable tuple.",
        )
    for position, event in enumerate(context.interaction_events):
        if not isinstance(event, InteractionEvent):
            return _invalid(
                ContextValidationErrorCode.INTERACTION_EVENT_NOT_TYPED,
                f"interaction_events[{position}] must be an InteractionEvent.",
            )
        if not isinstance(event.event_type, InteractionEventType):
            return _invalid(
                ContextValidationErrorCode.INTERACTION_EVENT_TYPE_NOT_ENUM,
                f"interaction_events[{position}].event_type must be an InteractionEventType.",
            )
        if not isinstance(event.content, str):
            return _invalid(
                ContextValidationErrorCode.INTERACTION_CONTENT_NOT_STRING,
                f"interaction_events[{position}].content must be a string.",
            )
        if not event.content.strip():
            return _invalid(
                ContextValidationErrorCode.EMPTY_INTERACTION_CONTENT,
                f"interaction_events[{position}].content must not be empty.",
            )

    if context.task_result is not None:
        result_validation = _validate_task_result(context.task_result)
        if not result_validation.is_valid:
            return result_validation
    for field_name, retrieval in (
        ("pre_task_retrieval", context.pre_task_retrieval),
        ("post_task_retrieval", context.post_task_retrieval),
    ):
        if retrieval is not None:
            retrieval_validation = _validate_retrieval(retrieval, field_name=field_name)
            if not retrieval_validation.is_valid:
                return retrieval_validation

    visible_contents: dict[str, str] = {}
    for retrieval in (context.pre_task_retrieval, context.post_task_retrieval):
        if retrieval is None:
            continue
        for match in retrieval.matches:
            previous_content = visible_contents.get(match.memory_id)
            if previous_content is not None and previous_content != match.content:
                return _invalid(
                    ContextValidationErrorCode.INCONSISTENT_VISIBLE_MEMORY_CONTENT,
                    f"Memory ID {match.memory_id!r} has conflicting visible content.",
                )
            visible_contents[match.memory_id] = match.content

    if context.phase is MemoryDecisionPhase.PRE_TASK:
        if (
            context.interaction_events
            or context.task_result is not None
            or context.pre_task_retrieval is not None
            or context.post_task_retrieval is not None
        ):
            return _invalid(
                ContextValidationErrorCode.PRE_TASK_HAS_TASK_EXECUTION_DATA,
                "PRE_TASK context cannot contain task execution data or retrieval results.",
            )
    elif context.phase is MemoryDecisionPhase.POST_TASK:
        if context.task_result is None:
            return _invalid(
                ContextValidationErrorCode.POST_TASK_MISSING_TASK_RESULT,
                "POST_TASK context requires task_result.",
            )
        if context.post_task_retrieval is not None:
            return _invalid(
                ContextValidationErrorCode.POST_TASK_HAS_POST_TASK_RETRIEVAL,
                "POST_TASK context cannot contain a post-task retrieval result yet.",
            )
    else:
        if context.task_result is None:
            return _invalid(
                ContextValidationErrorCode.AFTER_RETRIEVAL_MISSING_TASK_RESULT,
                "POST_TASK_AFTER_RETRIEVAL context requires task_result.",
            )
        if context.post_task_retrieval is None:
            return _invalid(
                ContextValidationErrorCode.AFTER_RETRIEVAL_MISSING_POST_TASK_RETRIEVAL,
                "POST_TASK_AFTER_RETRIEVAL context requires post_task_retrieval.",
            )

    return ContextValidationResult()


__all__ = [
    "ContextValidationErrorCode",
    "ContextValidationResult",
    "validate_memory_decision_context",
]
