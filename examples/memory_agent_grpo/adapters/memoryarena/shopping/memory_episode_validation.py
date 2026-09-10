"""Validation helpers for Memory-aware Shopping episode contracts."""

from __future__ import annotations

from ....answer_agent.input import AnswerAgentSubtaskInput
from ....answer_agent.prompts.builder import build_answer_agent_initial_messages
from ....answer_agent.runtime import (
    AnswerAgentSubtaskErrorCode,
    AnswerAgentSubtaskRunResult,
    AnswerAgentSubtaskRunStatus,
)
from ....answer_agent.tool_protocol.definition import ToolDefinition
from ....answer_agent.tool_protocol.validation import validate_tool_definition
from ....episode.trace import MemoryDecisionTrace, build_subtask_interaction_events
from ....memory_agent.action.protocol import MemoryActionName, MemoryDecisionPhase
from ....memory_agent.action.validation import ValidationErrorCode
from ....memory_agent.bank.bank_state import MemoryBankState
from ....memory_agent.decision.context import (
    InteractionEvent,
    TaskResult,
)
from ....memory_agent.decision.model_client import MemoryAgentClientErrorCode
from ....memory_agent.executor.outcome import ExecutionStatus
from ....memory_agent.retrieval.result import RetrievalResult
from .environment_session import (
    ShoppingTerminationEvidence,
    ShoppingTerminationKind,
)
from .memory_episode_contract import (
    SHOPPING_MEMORY_EPISODE_PRODUCT_COUNT,
    ShoppingMemoryEpisodeFailureCode,
    ShoppingMemoryEpisodeFailureStage,
    ShoppingMemoryEpisodeInputError,
    ShoppingMemoryEpisodeInputErrorCode,
    ShoppingMemoryEpisodeResult,
    ShoppingMemoryEpisodeRunStatus,
    ShoppingMemoryOutputCorrection,
    ShoppingMemoryProductTrace,
)
from .task_splitter import ShoppingTaskSplitterError, build_single_product_task


def validated_tool_definitions(value: object) -> tuple[ToolDefinition, ...]:
    if type(value) is not tuple:
        raise ValueError("tool_definitions must be a tuple.")
    definitions: tuple[ToolDefinition, ...] = value
    seen_names: set[str] = set()
    for position, definition in enumerate(definitions):
        if type(definition) is not ToolDefinition:
            raise ValueError(
                f"tool_definitions[{position}] must be a ToolDefinition."
            )
        validation = validate_tool_definition(definition)
        if not validation.is_valid:
            code = (
                "UNKNOWN"
                if validation.error_code is None
                else validation.error_code.value
            )
            raise ValueError(
                f"tool_definitions[{position}] is invalid ({code})."
            )
        if definition.name in seen_names:
            raise ValueError("tool definition names must be unique.")
        seen_names.add(definition.name)
    return definitions


def split_six_products(
    full_task_definition: object,
) -> tuple[dict[str, object], ...]:
    """Validate and split all six Products before any runtime side effect."""

    if type(full_task_definition) is not dict:
        raise ShoppingMemoryEpisodeInputError(
            ShoppingMemoryEpisodeInputErrorCode.TASK_SPLIT_FAILED,
            "Shopping task preflight failed.",
        )
    steps = full_task_definition.get("steps")
    if (
        type(steps) is not list
        or len(steps) != SHOPPING_MEMORY_EPISODE_PRODUCT_COUNT
    ):
        raise ShoppingMemoryEpisodeInputError(
            ShoppingMemoryEpisodeInputErrorCode.PRODUCT_COUNT_INVALID,
            "Memory-aware Shopping requires exactly six Products.",
        )
    try:
        return tuple(
            build_single_product_task(
                full_task_definition,
                product_ordinal=product_ordinal,
            )
            for product_ordinal in range(
                1,
                SHOPPING_MEMORY_EPISODE_PRODUCT_COUNT + 1,
            )
        )
    except ShoppingTaskSplitterError:
        raise ShoppingMemoryEpisodeInputError(
            ShoppingMemoryEpisodeInputErrorCode.TASK_SPLIT_FAILED,
            "Shopping task preflight failed.",
        ) from None


def terminal_tool_execution_count(result: AnswerAgentSubtaskRunResult) -> int:
    return sum(
        outcome.is_terminal
        for step in result.steps
        for outcome in step.tool_outcomes
    )


def retrieval_from_trace(
    trace: MemoryDecisionTrace,
) -> RetrievalResult | None:
    """Return the exact retrieval produced by a completed Memory decision."""

    decision = trace.gateway_result.validation.decision
    if decision is None or decision.memory_action is None:
        return None
    if decision.memory_action.action is not MemoryActionName.RETRIEVE:
        return None
    execution = trace.gateway_result.execution
    if (
        execution is None
        or execution.status is ExecutionStatus.REJECTED
        or type(execution.retrieval) is not RetrievalResult
    ):
        raise ValueError("A RETRIEVE decision requires a valid retrieval result.")
    return execution.retrieval


def memory_decisions(
    trace: ShoppingMemoryProductTrace,
) -> tuple[MemoryDecisionTrace, ...]:
    return (trace.pre_task_decision, *trace.post_task_decisions)


def _validate_post_decisions(
    decisions: tuple[MemoryDecisionTrace, ...],
    *,
    memory_task_input: str,
    task_result: TaskResult,
    interaction_events: tuple[InteractionEvent, ...],
    pre_task_retrieval: RetrievalResult | None,
    bank_before: MemoryBankState,
    bank_after: MemoryBankState,
) -> None:
    if len(decisions) not in (1, 2):
        raise ValueError("POST_TASK requires one decision and one optional follow-up.")
    expected_bank = bank_before
    for position, decision in enumerate(decisions):
        expected_phase = (
            MemoryDecisionPhase.POST_TASK
            if position == 0
            else MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL
        )
        expected_post_retrieval = (
            None if position == 0 else retrieval_from_trace(decisions[0])
        )
        if (
            type(decision) is not MemoryDecisionTrace
            or decision.context.phase is not expected_phase
            or decision.context.task_input != memory_task_input
            or decision.context.interaction_events != interaction_events
            or decision.context.task_result != task_result
            or decision.context.pre_task_retrieval != pre_task_retrieval
            or decision.context.post_task_retrieval != expected_post_retrieval
            or decision.bank_before != expected_bank
        ):
            raise ValueError("POST_TASK decision sequence is inconsistent.")
        expected_bank = decision.bank_after
    first_retrieval = retrieval_from_trace(decisions[0])
    if (first_retrieval is not None) != (len(decisions) == 2):
        raise ValueError("Only POST_TASK retrieval may create a follow-up decision.")
    if expected_bank != bank_after:
        raise ValueError("POST_TASK Bank endpoint is inconsistent.")


def validate_memory_product_trace(trace: object) -> None:
    """Raise when one committed Shopping Product trace is inconsistent."""

    if type(trace) is not ShoppingMemoryProductTrace:
        raise ValueError("trace must be a ShoppingMemoryProductTrace.")
    if (
        type(trace.product_ordinal) is not int
        or not 1 <= trace.product_ordinal <= SHOPPING_MEMORY_EPISODE_PRODUCT_COUNT
    ):
        raise ValueError("product_ordinal must select one of six Products.")
    if type(trace.memory_task_input) is not str or not trace.memory_task_input.strip():
        raise ValueError("memory_task_input must be non-empty text.")
    if type(trace.bank_before) is not MemoryBankState or type(
        trace.bank_after
    ) is not MemoryBankState:
        raise ValueError("Product Bank snapshots must be MemoryBankState values.")
    if type(trace.pre_task_decision) is not MemoryDecisionTrace:
        raise ValueError("pre_task_decision must be a MemoryDecisionTrace.")
    pre = trace.pre_task_decision
    if (
        pre.context.phase is not MemoryDecisionPhase.PRE_TASK
        or pre.context.task_input != trace.memory_task_input
        or pre.bank_before != trace.bank_before
    ):
        raise ValueError("PRE_TASK trace is inconsistent with the Product.")
    pre_retrieval = retrieval_from_trace(pre)
    expected_memories = (
        ()
        if pre_retrieval is None
        else tuple(match.content for match in pre_retrieval.matches)
    )
    if (
        type(trace.answer_input) is not AnswerAgentSubtaskInput
        or trace.answer_input.subtask_input != trace.memory_task_input
        or trace.answer_input.retrieved_memories != expected_memories
    ):
        raise ValueError("Answer input does not match PRE_TASK retrieval.")
    if (
        type(trace.answer_result) is not AnswerAgentSubtaskRunResult
        or trace.answer_result.status is not AnswerAgentSubtaskRunStatus.COMPLETED
        or trace.answer_result.final_answer is None
        or trace.answer_result.conversation.initial_messages
        != build_answer_agent_initial_messages(
            trace.answer_input,
            interaction_mode=trace.answer_result.config.interaction_mode,
        )
    ):
        raise ValueError("Answer result does not match the Product input.")
    if (
        type(trace.interaction_events) is not tuple
        or trace.interaction_events
        != build_subtask_interaction_events(trace.answer_result)
    ):
        raise ValueError("Interaction events do not match the Answer trace.")
    if (
        type(trace.task_result) is not TaskResult
        or trace.task_result.final_answer != trace.answer_result.final_answer
        or trace.task_result.reward is not None
        or trace.task_result.feedback is not None
    ):
        raise ValueError("TaskResult must be public and evaluator-free.")
    if type(trace.termination_evidence) is not ShoppingTerminationEvidence:
        raise ValueError("termination_evidence must be trusted Shopping evidence.")
    if (
        trace.termination_evidence.kind is ShoppingTerminationKind.NOT_TERMINATED
        or terminal_tool_execution_count(trace.answer_result) != 1
    ):
        raise ValueError("A committed Product requires one terminal tool execution.")
    if type(trace.post_task_decisions) is not tuple:
        raise ValueError("post_task_decisions must be a tuple.")
    _validate_post_decisions(
        trace.post_task_decisions,
        memory_task_input=trace.memory_task_input,
        task_result=trace.task_result,
        interaction_events=trace.interaction_events,
        pre_task_retrieval=pre_retrieval,
        bank_before=pre.bank_after,
        bank_after=trace.bank_after,
    )


def validate_memory_episode_result(result: object) -> None:
    """Raise when a completed result or committed failure prefix is invalid."""

    if type(result) is not ShoppingMemoryEpisodeResult:
        raise ValueError("result must be a ShoppingMemoryEpisodeResult.")
    if type(result.status) is not ShoppingMemoryEpisodeRunStatus:
        raise ValueError("status must be a ShoppingMemoryEpisodeRunStatus.")
    if type(result.product_traces) is not tuple or any(
        type(trace) is not ShoppingMemoryProductTrace
        for trace in result.product_traces
    ):
        raise ValueError("product_traces must contain Shopping traces.")
    if tuple(trace.product_ordinal for trace in result.product_traces) != tuple(
        range(1, len(result.product_traces) + 1)
    ):
        raise ValueError("product_traces must be a contiguous one-based prefix.")
    expected_bank = MemoryBankState.empty()
    for trace in result.product_traces:
        if trace.bank_before != expected_bank:
            raise ValueError("Product traces must share one continuous Bank.")
        expected_bank = trace.bank_after
    if type(result.final_bank_state) is not MemoryBankState or (
        result.final_bank_state != expected_bank
    ):
        raise ValueError("final_bank_state must equal the committed prefix.")
    committed_decisions = sum(
        trace.memory_decision_count for trace in result.product_traces
    )
    if (
        type(result.memory_agent_invocation_count) is not int
        or result.memory_agent_invocation_count < committed_decisions
    ):
        raise ValueError("memory_agent_invocation_count is inconsistent.")
    if (
        type(result.memory_output_correction_budget) is not int
        or result.memory_output_correction_budget not in (0, 1, 2, 3)
    ):
        raise ValueError("Memory output correction budget must be an integer from 0 to 3.")
    corrections = result.memory_output_corrections
    if type(corrections) is not tuple or len(corrections) > result.memory_output_correction_budget:
        raise ValueError("Memory output corrections must fit the episode budget.")
    correction_count = len(corrections)
    previous_key = (0, -1)
    phase_order = {phase: i for i, phase in enumerate(MemoryDecisionPhase)}
    for index, correction in enumerate(corrections):
        if (
            type(correction) is not ShoppingMemoryOutputCorrection
            or correction.product_ordinal > len(result.product_traces) + 1
            or result.memory_agent_invocation_count < committed_decisions + correction_count
        ):
            raise ValueError("Memory output correction metadata is inconsistent.")
        key = (correction.product_ordinal, phase_order[correction.phase])
        if key <= previous_key or (not correction.recovered and index != correction_count - 1):
            raise ValueError("Each decision may have one correction, in execution order.")
        previous_key = key
    correction = corrections[-1] if corrections else None

    if result.status is ShoppingMemoryEpisodeRunStatus.COMPLETED:
        if (
            len(result.product_traces) != SHOPPING_MEMORY_EPISODE_PRODUCT_COUNT
            or result.failed_product_ordinal is not None
            or result.failure_stage is not None
            or result.failure_code is not None
            or result.memory_validation_error_code is not None
            or result.memory_model_error_code is not None
            or result.answer_error_code is not None
            or result.answer_provider_error_code is not None
            or result.memory_agent_invocation_count
            != committed_decisions + correction_count
            or (correction is not None and not correction.recovered)
        ):
            raise ValueError("A completed run requires exactly six Product traces.")
        return

    if (
        type(result.failed_product_ordinal) is not int
        or result.failed_product_ordinal != len(result.product_traces) + 1
        or not 1
        <= result.failed_product_ordinal
        <= SHOPPING_MEMORY_EPISODE_PRODUCT_COUNT
        or type(result.failure_stage) is not ShoppingMemoryEpisodeFailureStage
        or type(result.failure_code) is not ShoppingMemoryEpisodeFailureCode
    ):
        raise ValueError("A failed run requires one failed next Product stage.")
    if correction is not None and not correction.recovered:
        correction_stage = (
            ShoppingMemoryEpisodeFailureStage.PRE_TASK
            if correction.phase is MemoryDecisionPhase.PRE_TASK
            else ShoppingMemoryEpisodeFailureStage.POST_TASK
        )
        if (
            correction.product_ordinal != result.failed_product_ordinal
            or result.failure_stage
            not in (correction_stage, ShoppingMemoryEpisodeFailureStage.INTERNAL)
        ):
            raise ValueError(
                "An unrecovered correction must identify the failure stage."
            )
    if (
        result.failure_code
        is ShoppingMemoryEpisodeFailureCode.MEMORY_OUTPUT_VALIDATION_FAILED
    ):
        if type(result.memory_validation_error_code) is not ValidationErrorCode:
            raise ValueError(
                "A Memory output validation failure requires its validation code."
            )
    elif result.memory_validation_error_code is not None:
        raise ValueError(
            "A Memory validation code requires an output validation failure."
        )
    if result.memory_model_error_code is not None and (
        type(result.memory_model_error_code) is not MemoryAgentClientErrorCode
        or result.failure_code
        is not ShoppingMemoryEpisodeFailureCode.MEMORY_MODEL_CALL_FAILED
    ):
        raise ValueError(
            "A Memory model error code requires a Memory model call failure."
        )
    if result.failure_stage is ShoppingMemoryEpisodeFailureStage.ANSWER:
        if result.failure_code is ShoppingMemoryEpisodeFailureCode.ANSWER_RUNNER_FAILED:
            if type(result.answer_error_code) is not AnswerAgentSubtaskErrorCode:
                raise ValueError("A returned Answer failure requires its error code.")
        elif result.answer_error_code is not None:
            raise ValueError("Only a returned Answer failure has an Answer error code.")
    elif result.answer_error_code is not None:
        raise ValueError("Answer error code is allowed only at the Answer stage.")
    if result.answer_provider_error_code is not None and (
        result.answer_error_code is not AnswerAgentSubtaskErrorCode.PROVIDER_ERROR
    ):
        raise ValueError("Provider error code requires an Answer provider failure.")


__all__ = [
    "memory_decisions",
    "retrieval_from_trace",
    "split_six_products",
    "terminal_tool_execution_count",
    "validate_memory_episode_result",
    "validate_memory_product_trace",
    "validated_tool_definitions",
]
