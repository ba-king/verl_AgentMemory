"""Shared exact-type checks for episode validation modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from examples.memory_agent_grpo.answer_agent.input import AnswerAgentSubtaskInput
from examples.memory_agent_grpo.answer_agent.validation import (
    validate_answer_agent_subtask_input,
)
from examples.memory_agent_grpo.memory_agent.action.protocol import (
    AddArguments,
    DeleteArguments,
    MemoryAction,
    MemoryActionName,
    MemoryDecision,
    MemoryDecisionPhase,
    RetrieveArguments,
    UpdateArguments,
)
from examples.memory_agent_grpo.memory_agent.action.validation import (
    ValidationErrorCode,
    ValidationResult,
    ValidationStatus,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import (
    MemoryBankState,
    MemoryRecord,
)
from examples.memory_agent_grpo.memory_agent.bank.validation import (
    is_canonical_memory_id,
    validate_memory_bank_state,
)
from examples.memory_agent_grpo.memory_agent.decision.context import (
    InteractionEvent,
    InteractionEventType,
    MemoryDecisionContext,
    TaskResult,
    TaskResultStatus,
)
from examples.memory_agent_grpo.memory_agent.decision.context_validation import (
    validate_memory_decision_context,
)
from examples.memory_agent_grpo.memory_agent.executor.outcome import (
    ExecutionCode,
    ExecutionResult,
    ExecutionStatus,
)
from examples.memory_agent_grpo.memory_agent.retrieval.result import (
    RetrievalMatch,
    RetrievalResult,
)


class EpisodeValidationErrorCode(str, Enum):
    WRONG_TYPE = "WRONG_TYPE"
    INVALID_TEXT = "INVALID_TEXT"
    INVALID_SEED = "INVALID_SEED"
    INVALID_SUBTASK_INPUT = "INVALID_SUBTASK_INPUT"
    INVALID_SUBTASK_SEQUENCE = "INVALID_SUBTASK_SEQUENCE"
    PRELOADED_ANSWER_MEMORIES = "PRELOADED_ANSWER_MEMORIES"
    INVALID_NESTED_CONTRACT = "INVALID_NESTED_CONTRACT"
    NON_EMPTY_INITIAL_BANK = "NON_EMPTY_INITIAL_BANK"
    CONTEXT_MISMATCH = "CONTEXT_MISMATCH"
    RAW_OUTPUT_MISMATCH = "RAW_OUTPUT_MISMATCH"
    GATEWAY_MISMATCH = "GATEWAY_MISMATCH"
    BANK_TRANSITION_MISMATCH = "BANK_TRANSITION_MISMATCH"
    DECISION_SEQUENCE_MISMATCH = "DECISION_SEQUENCE_MISMATCH"
    RETRIEVAL_EXPOSURE_MISMATCH = "RETRIEVAL_EXPOSURE_MISMATCH"
    ANSWER_TRACE_MISMATCH = "ANSWER_TRACE_MISMATCH"
    TASK_RESULT_MISMATCH = "TASK_RESULT_MISMATCH"
    BANK_CONTINUITY_MISMATCH = "BANK_CONTINUITY_MISMATCH"
    EPISODE_INPUT_MISMATCH = "EPISODE_INPUT_MISMATCH"
    RESULT_SEMANTICS_MISMATCH = "RESULT_SEMANTICS_MISMATCH"


@dataclass(frozen=True)
class EpisodeValidationResult:
    error_code: EpisodeValidationErrorCode | None = None
    error_message: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.error_code is None


def invalid(
    code: EpisodeValidationErrorCode,
    message: str,
) -> EpisodeValidationResult:
    return EpisodeValidationResult(error_code=code, error_message=message)


def valid_text(value: object, *, allow_empty: bool = False) -> bool:
    if type(value) is not str or (not allow_empty and not value.strip()):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def valid_event(event: object) -> bool:
    return (
        type(event) is InteractionEvent
        and type(event.event_type) is InteractionEventType
        and valid_text(event.content)
    )


def valid_bank_state(state: object) -> bool:
    if (
        type(state) is not MemoryBankState
        or type(state.records) is not tuple
        or type(state.next_id) is not int
    ):
        return False
    for record in state.records:
        if (
            type(record) is not MemoryRecord
            or not valid_text(record.memory_id)
            or not valid_text(record.content)
        ):
            return False
    return validate_memory_bank_state(state).is_valid


def valid_retrieval(retrieval: object) -> bool:
    if (
        type(retrieval) is not RetrievalResult
        or not valid_text(retrieval.query)
        or type(retrieval.matches) is not tuple
    ):
        return False
    seen_ids: set[str] = set()
    for match in retrieval.matches:
        if (
            type(match) is not RetrievalMatch
            or type(match.record) is not MemoryRecord
            or not valid_text(match.record.memory_id)
            or not is_canonical_memory_id(match.record.memory_id)
            or match.record.memory_id in seen_ids
            or not valid_text(match.record.content)
            or type(match.score) not in (int, float)
            or not math.isfinite(float(match.score))
            or float(match.score) <= 0.0
        ):
            return False
        seen_ids.add(match.record.memory_id)
    return True


def valid_task_result(result: object, *, completed: bool) -> bool:
    if type(result) is not TaskResult:
        return False
    if (
        type(result.final_answer) is not str
        or not valid_text(result.final_answer, allow_empty=True)
        or type(result.status) is not TaskResultStatus
        or (
            result.reward is not None
            and type(result.reward) not in (int, float)
        )
        or (
            result.feedback is not None
            and not valid_text(result.feedback)
        )
    ):
        return False
    probe = MemoryDecisionContext(
        phase=MemoryDecisionPhase.POST_TASK,
        task_input="validation probe",
        bank_record_count=0,
        task_result=result,
    )
    if not validate_memory_decision_context(probe).is_valid:
        return False
    return not completed or result.status is TaskResultStatus.COMPLETED


def valid_context(context: object) -> bool:
    if (
        type(context) is not MemoryDecisionContext
        or type(context.phase) is not MemoryDecisionPhase
        or not valid_text(context.task_input)
        or type(context.bank_record_count) is not int
        or type(context.interaction_events) is not tuple
        or any(not valid_event(event) for event in context.interaction_events)
        or (
            context.task_result is not None
            and not valid_task_result(context.task_result, completed=False)
        )
        or (
            context.pre_task_retrieval is not None
            and not valid_retrieval(context.pre_task_retrieval)
        )
        or (
            context.post_task_retrieval is not None
            and not valid_retrieval(context.post_task_retrieval)
        )
    ):
        return False
    return validate_memory_decision_context(context).is_valid


def valid_answer_input(value: object) -> bool:
    if (
        type(value) is not AnswerAgentSubtaskInput
        or not valid_text(value.task_prompt)
        or not valid_text(value.subtask_input)
        or type(value.retrieved_memories) is not tuple
        or any(not valid_text(memory) for memory in value.retrieved_memories)
    ):
        return False
    return validate_answer_agent_subtask_input(value).is_valid


def valid_memory_decision(decision: object) -> bool:
    if type(decision) is not MemoryDecision:
        return False
    action = decision.memory_action
    if action is None:
        return True
    if (
        type(action) is not MemoryAction
        or type(action.action) is not MemoryActionName
    ):
        return False
    arguments = action.arguments
    if action.action is MemoryActionName.RETRIEVE:
        return type(arguments) is RetrieveArguments and valid_text(
            arguments.query
        )
    if action.action is MemoryActionName.ADD:
        return type(arguments) is AddArguments and valid_text(arguments.content)
    if action.action is MemoryActionName.UPDATE:
        return (
            type(arguments) is UpdateArguments
            and valid_text(arguments.memory_id)
            and valid_text(arguments.content)
        )
    return type(arguments) is DeleteArguments and valid_text(
        arguments.memory_id
    )


def valid_action_validation(result: object) -> bool:
    return (
        type(result) is ValidationResult
        and type(result.status) is ValidationStatus
        and (
            result.decision is None or valid_memory_decision(result.decision)
        )
        and (
            result.error_code is None
            or type(result.error_code) is ValidationErrorCode
        )
        and (
            result.error_message is None
            or valid_text(result.error_message)
        )
    )


def action_validation_signature(
    result: ValidationResult,
) -> tuple[object, ...]:
    return (
        result.status,
        None if result.decision is None else result.decision.to_dict(),
        result.error_code,
        result.error_message,
    )


def valid_execution_result(result: object) -> bool:
    return (
        type(result) is ExecutionResult
        and type(result.status) is ExecutionStatus
        and type(result.code) is ExecutionCode
        and (
            result.action is None or type(result.action) is MemoryActionName
        )
        and (
            result.record is None
            or (
                type(result.record) is MemoryRecord
                and valid_text(result.record.memory_id)
                and valid_text(result.record.content)
            )
        )
        and (
            result.retrieval is None or valid_retrieval(result.retrieval)
        )
        and (
            result.error_message is None
            or valid_text(result.error_message)
        )
    )


__all__ = [
    "EpisodeValidationErrorCode",
    "EpisodeValidationResult",
]
