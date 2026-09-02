"""Validation for initialization, subtask, and episode traces."""

from __future__ import annotations

from examples.memory_agent_grpo.answer_agent.prompts.builder import (
    build_answer_agent_initial_messages,
)
from examples.memory_agent_grpo.answer_agent.runtime.result import (
    AnswerAgentSubtaskRunResult,
    AnswerAgentSubtaskRunStatus,
)
from examples.memory_agent_grpo.memory_agent.action.protocol import (
    MemoryActionName,
    MemoryDecisionPhase,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState
from examples.memory_agent_grpo.memory_agent.decision.context import (
    InteractionEvent,
    TaskResult,
)
from examples.memory_agent_grpo.memory_agent.retrieval.result import RetrievalResult

from ..input import EpisodeInput
from ..trace import (
    EpisodeInitializationTrace,
    EpisodeSubtaskTrace,
    EpisodeTrace,
    build_subtask_interaction_events,
)
from .common import (
    EpisodeValidationErrorCode,
    EpisodeValidationResult,
    invalid,
    valid_answer_input,
    valid_bank_state,
    valid_event,
    valid_task_result,
)
from .input import (
    validate_completed_task_seed_input,
    validate_episode_input,
    validate_episode_subtask_input,
)
from .memory_decision import (
    retrieval_from_decision,
    validate_complete_memory_decision,
)


def _validate_post_task_decisions(
    decisions: object,
    *,
    task_input: str,
    task_result: TaskResult,
    interaction_events: tuple[InteractionEvent, ...],
    pre_task_retrieval: RetrievalResult | None,
    bank_before: MemoryBankState,
    bank_after: MemoryBankState,
) -> EpisodeValidationResult:
    if type(decisions) is not tuple or len(decisions) not in (1, 2):
        return invalid(
            EpisodeValidationErrorCode.DECISION_SEQUENCE_MISMATCH,
            "POST_TASK requires one decision, plus one follow-up only after retrieval.",
        )
    expected_before = bank_before
    for decision in decisions:
        validation = validate_complete_memory_decision(decision)
        if not validation.is_valid:
            return validation
        if decision.bank_before != expected_before:
            return invalid(
                EpisodeValidationErrorCode.BANK_CONTINUITY_MISMATCH,
                "POST_TASK decisions must use contiguous Bank snapshots.",
            )
        expected_before = decision.bank_after

    first = decisions[0]
    if first.context.phase is not MemoryDecisionPhase.POST_TASK:
        return invalid(
            EpisodeValidationErrorCode.DECISION_SEQUENCE_MISMATCH,
            "The first post-task decision must use POST_TASK.",
        )
    first_retrieval = retrieval_from_decision(first)
    if (first_retrieval is not None) != (len(decisions) == 2):
        return invalid(
            EpisodeValidationErrorCode.DECISION_SEQUENCE_MISMATCH,
            "Only a POST_TASK retrieval creates a second memory decision.",
        )

    for position, decision in enumerate(decisions):
        context = decision.context
        expected_phase = (
            MemoryDecisionPhase.POST_TASK
            if position == 0
            else MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL
        )
        expected_post_retrieval = None if position == 0 else first_retrieval
        if (
            context.phase is not expected_phase
            or context.task_input != task_input
            or context.task_result != task_result
            or context.interaction_events != interaction_events
            or context.pre_task_retrieval != pre_task_retrieval
            or context.post_task_retrieval != expected_post_retrieval
        ):
            return invalid(
                EpisodeValidationErrorCode.CONTEXT_MISMATCH,
                "POST_TASK context does not match its episode-stage evidence.",
            )
    if expected_before != bank_after:
        return invalid(
            EpisodeValidationErrorCode.BANK_CONTINUITY_MISMATCH,
            "Post-task Bank endpoint does not match the decision sequence.",
        )
    return EpisodeValidationResult()


def validate_initialization_trace(
    trace: object,
) -> EpisodeValidationResult:
    if type(trace) is not EpisodeInitializationTrace:
        return invalid(
            EpisodeValidationErrorCode.WRONG_TYPE,
            "trace must be an EpisodeInitializationTrace.",
        )
    seed_validation = validate_completed_task_seed_input(trace.seed)
    if not seed_validation.is_valid:
        return seed_validation
    if not valid_bank_state(trace.bank_before) or not valid_bank_state(
        trace.bank_after
    ):
        return invalid(
            EpisodeValidationErrorCode.INVALID_NESTED_CONTRACT,
            "Initialization Bank snapshots must use exact valid Bank types.",
        )
    if trace.bank_before != MemoryBankState.empty():
        return invalid(
            EpisodeValidationErrorCode.NON_EMPTY_INITIAL_BANK,
            "Every episode must initialize from a new empty Memory Bank.",
        )
    return _validate_post_task_decisions(
        trace.post_task_decisions,
        task_input=trace.seed.task_input,
        task_result=trace.seed.task_result,
        interaction_events=trace.seed.interaction_events,
        pre_task_retrieval=None,
        bank_before=trace.bank_before,
        bank_after=trace.bank_after,
    )


def validate_episode_subtask_trace(
    trace: object,
) -> EpisodeValidationResult:
    if type(trace) is not EpisodeSubtaskTrace:
        return invalid(
            EpisodeValidationErrorCode.WRONG_TYPE,
            "trace must be an EpisodeSubtaskTrace.",
        )
    subtask_validation = validate_episode_subtask_input(trace.subtask)
    if not subtask_validation.is_valid:
        return subtask_validation
    for state in (trace.bank_before, trace.bank_after):
        if not valid_bank_state(state):
            return invalid(
                EpisodeValidationErrorCode.INVALID_NESTED_CONTRACT,
                "Subtask Bank snapshots must be valid.",
            )

    pre_validation = validate_complete_memory_decision(
        trace.pre_task_decision
    )
    if not pre_validation.is_valid:
        return pre_validation
    pre = trace.pre_task_decision
    if (
        pre.context.phase is not MemoryDecisionPhase.PRE_TASK
        or pre.context.task_input != trace.subtask.memory_task_input
        or pre.bank_before != trace.bank_before
    ):
        return invalid(
            EpisodeValidationErrorCode.CONTEXT_MISMATCH,
            "PRE_TASK trace does not match the current subtask and Bank.",
        )
    pre_retrieval = retrieval_from_decision(pre)
    decision = pre.gateway_result.validation.decision
    if (
        decision is not None
        and decision.memory_action is not None
        and decision.memory_action.action is not MemoryActionName.RETRIEVE
    ):
        return invalid(
            EpisodeValidationErrorCode.DECISION_SEQUENCE_MISMATCH,
            "PRE_TASK may only skip or retrieve.",
        )

    if not valid_answer_input(trace.answer_input):
        return invalid(
            EpisodeValidationErrorCode.INVALID_NESTED_CONTRACT,
            "answer_input violates the Answer Agent contract.",
        )
    template = trace.subtask.answer_input_without_memories
    expected_memories = (
        ()
        if pre_retrieval is None
        else tuple(match.content for match in pre_retrieval.matches)
    )
    if (
        trace.answer_input.task_prompt != template.task_prompt
        or trace.answer_input.subtask_input != template.subtask_input
        or trace.answer_input.retrieved_memories != expected_memories
    ):
        return invalid(
            EpisodeValidationErrorCode.RETRIEVAL_EXPOSURE_MISMATCH,
            "Answer Agent memories must equal the exact PRE_TASK retrieval contents.",
        )
    if type(trace.answer_result) is not AnswerAgentSubtaskRunResult:
        return invalid(
            EpisodeValidationErrorCode.WRONG_TYPE,
            "answer_result must be an AnswerAgentSubtaskRunResult.",
        )
    if (
        trace.answer_result.status is not AnswerAgentSubtaskRunStatus.COMPLETED
        or trace.answer_result.conversation.initial_messages
        != build_answer_agent_initial_messages(trace.answer_input)
    ):
        return invalid(
            EpisodeValidationErrorCode.ANSWER_TRACE_MISMATCH,
            "A completed subtask trace must bind to its exact Answer Agent input.",
        )
    if (
        not valid_task_result(trace.task_result, completed=True)
        or trace.answer_result.final_answer != trace.task_result.final_answer
    ):
        return invalid(
            EpisodeValidationErrorCode.TASK_RESULT_MISMATCH,
            "TaskResult must contain the completed Answer Agent final answer.",
        )
    if type(trace.interaction_events) is not tuple or any(
        not valid_event(event) for event in trace.interaction_events
    ):
        return invalid(
            EpisodeValidationErrorCode.INVALID_NESTED_CONTRACT,
            "interaction_events must be valid immutable public events.",
        )
    if trace.interaction_events != build_subtask_interaction_events(
        trace.answer_result
    ):
        return invalid(
            EpisodeValidationErrorCode.ANSWER_TRACE_MISMATCH,
            "interaction_events must be projected from this Answer Agent trace.",
        )
    return _validate_post_task_decisions(
        trace.post_task_decisions,
        task_input=trace.subtask.memory_task_input,
        task_result=trace.task_result,
        interaction_events=trace.interaction_events,
        pre_task_retrieval=pre_retrieval,
        bank_before=pre.bank_after,
        bank_after=trace.bank_after,
    )


def validate_episode_trace(
    trace: object,
    *,
    episode_input: EpisodeInput,
) -> EpisodeValidationResult:
    input_validation = validate_episode_input(episode_input)
    if not input_validation.is_valid:
        return input_validation
    if type(trace) is not EpisodeTrace:
        return invalid(
            EpisodeValidationErrorCode.WRONG_TYPE,
            "trace must be an EpisodeTrace.",
        )
    if not valid_bank_state(trace.initial_bank_state) or not valid_bank_state(
        trace.final_bank_state
    ):
        return invalid(
            EpisodeValidationErrorCode.INVALID_NESTED_CONTRACT,
            "Episode Bank snapshots must use exact valid Bank types.",
        )
    if trace.initial_bank_state != MemoryBankState.empty():
        return invalid(
            EpisodeValidationErrorCode.NON_EMPTY_INITIAL_BANK,
            "Every episode trace must start from an empty Memory Bank.",
        )
    initialization_validation = validate_initialization_trace(
        trace.initialization
    )
    if not initialization_validation.is_valid:
        return initialization_validation
    if (
        trace.initialization.seed != episode_input.initial_completed_task
        or trace.initialization.bank_before != trace.initial_bank_state
    ):
        return invalid(
            EpisodeValidationErrorCode.EPISODE_INPUT_MISMATCH,
            "Initialization trace must use the EpisodeInput seed.",
        )
    if type(trace.subtasks) is not tuple or len(trace.subtasks) > len(
        episode_input.subtasks
    ):
        return invalid(
            EpisodeValidationErrorCode.INVALID_SUBTASK_SEQUENCE,
            "Episode trace must contain an immutable completed-subtask prefix.",
        )

    expected_bank = trace.initialization.bank_after
    for position, subtask_trace in enumerate(trace.subtasks):
        validation = validate_episode_subtask_trace(subtask_trace)
        if not validation.is_valid:
            return validation
        if subtask_trace.subtask != episode_input.subtasks[position]:
            return invalid(
                EpisodeValidationErrorCode.EPISODE_INPUT_MISMATCH,
                "Subtask trace does not match the ordered EpisodeInput.",
            )
        if subtask_trace.bank_before != expected_bank:
            return invalid(
                EpisodeValidationErrorCode.BANK_CONTINUITY_MISMATCH,
                "One Memory Bank must continue across episode subtasks.",
            )
        expected_bank = subtask_trace.bank_after
    if trace.final_bank_state != expected_bank:
        return invalid(
            EpisodeValidationErrorCode.BANK_CONTINUITY_MISMATCH,
            "Episode final Bank must equal the last completed stage Bank.",
        )
    return EpisodeValidationResult()


__all__ = [
    "validate_episode_subtask_trace",
    "validate_episode_trace",
    "validate_initialization_trace",
]
