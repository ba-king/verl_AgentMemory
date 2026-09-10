"""Validation for one Memory Agent raw-output-to-Bank trace."""

from __future__ import annotations

from examples.memory_agent_grpo.memory_agent.action.protocol import (
    MemoryActionName,
)
from examples.memory_agent_grpo.memory_agent.action.repair import (
    MemoryOutputRepair,
    repair_missing_add_outer_brace,
)
from examples.memory_agent_grpo.memory_agent.action.validation import (
    validate_memory_decision_output,
)
from examples.memory_agent_grpo.memory_agent.bank.store import MemoryBankStore
from examples.memory_agent_grpo.memory_agent.decision.action_gateway import (
    ActionToExecutorResult,
    ActionToExecutorStatus,
)
from examples.memory_agent_grpo.memory_agent.decision.context import (
    MemoryDecisionContext,
)
from examples.memory_agent_grpo.memory_agent.executor.engine import (
    MemoryActionExecutor,
)
from examples.memory_agent_grpo.memory_agent.executor.outcome import (
    ExecutionStatus,
)
from examples.memory_agent_grpo.memory_agent.retrieval.result import RetrievalResult

from ..trace import MemoryDecisionTrace
from .common import (
    EpisodeValidationErrorCode,
    EpisodeValidationResult,
    action_validation_signature,
    invalid,
    valid_action_validation,
    valid_bank_state,
    valid_context,
    valid_execution_result,
    valid_text,
)


def validate_memory_decision_trace(
    trace: object,
) -> EpisodeValidationResult:
    if type(trace) is not MemoryDecisionTrace:
        return invalid(
            EpisodeValidationErrorCode.WRONG_TYPE,
            "trace must be a MemoryDecisionTrace.",
        )
    if type(trace.context) is not MemoryDecisionContext:
        return invalid(
            EpisodeValidationErrorCode.WRONG_TYPE,
            "trace.context must be a MemoryDecisionContext.",
        )
    if not valid_context(trace.context):
        return invalid(
            EpisodeValidationErrorCode.INVALID_NESTED_CONTRACT,
            "trace.context violates the Memory Decision Context contract.",
        )
    if not valid_text(trace.raw_output, allow_empty=True):
        return invalid(
            EpisodeValidationErrorCode.INVALID_TEXT,
            "trace.raw_output must be UTF-8 text.",
        )
    if type(trace.gateway_result) is not ActionToExecutorResult:
        return invalid(
            EpisodeValidationErrorCode.WRONG_TYPE,
            "trace.gateway_result must be an ActionToExecutorResult.",
        )
    if (
        type(trace.gateway_result.status) is not ActionToExecutorStatus
        or type(trace.gateway_result.phase) is not type(trace.context.phase)
        or not valid_action_validation(trace.gateway_result.validation)
        or (
            trace.gateway_result.execution is not None
            and not valid_execution_result(trace.gateway_result.execution)
        )
    ):
        return invalid(
            EpisodeValidationErrorCode.INVALID_NESTED_CONTRACT,
            "trace.gateway_result must use exact validated boundary types.",
        )
    for state in (trace.bank_before, trace.bank_after):
        if not valid_bank_state(state):
            return invalid(
                EpisodeValidationErrorCode.INVALID_NESTED_CONTRACT,
                "Every traced Memory Bank snapshot must be valid.",
            )
    if trace.context.bank_record_count != trace.bank_before.record_count:
        return invalid(
            EpisodeValidationErrorCode.CONTEXT_MISMATCH,
            "Context Bank count must equal the before-state record count.",
        )
    if trace.gateway_result.phase is not trace.context.phase:
        return invalid(
            EpisodeValidationErrorCode.GATEWAY_MISMATCH,
            "Gateway phase must equal the model-visible context phase.",
        )

    if trace.output_repair is not None:
        repair = trace.output_repair
        if (
            type(repair) is not MemoryOutputRepair
            or type(repair.rule) is not str
            or not valid_text(repair.repaired_output)
            or not valid_action_validation(repair.original_validation)
            or repair != repair_missing_add_outer_brace(
                trace.raw_output,
                phase=trace.context.phase,
                visible_memory_ids=trace.context.visible_memory_ids,
            )
        ):
            return invalid(
                EpisodeValidationErrorCode.RAW_OUTPUT_MISMATCH,
                "Repair must exactly match the permitted deterministic transformation.",
            )

    expected_validation = validate_memory_decision_output(
        trace.effective_output,
        phase=trace.context.phase,
        visible_memory_ids=trace.context.visible_memory_ids,
    )
    if action_validation_signature(
        trace.gateway_result.validation
    ) != action_validation_signature(expected_validation):
        return invalid(
            EpisodeValidationErrorCode.RAW_OUTPUT_MISMATCH,
            "Gateway validation must follow raw output and any verified syntax repair.",
        )

    decision = expected_validation.decision
    if not expected_validation.is_valid:
        if (
            trace.gateway_result.status
            is not ActionToExecutorStatus.VALIDATION_FAILED
            or trace.gateway_result.execution is not None
        ):
            return invalid(
                EpisodeValidationErrorCode.GATEWAY_MISMATCH,
                "Invalid output must stop at Gateway validation.",
            )
        expected_after = trace.bank_before
    elif decision is None:
        return invalid(
            EpisodeValidationErrorCode.GATEWAY_MISMATCH,
            "A valid Gateway validation must contain a MemoryDecision.",
        )
    elif decision.memory_action is None:
        if (
            trace.gateway_result.status is not ActionToExecutorStatus.SKIPPED
            or trace.gateway_result.execution is not None
        ):
            return invalid(
                EpisodeValidationErrorCode.GATEWAY_MISMATCH,
                "A null memory action must be skipped without execution.",
            )
        expected_after = trace.bank_before
    else:
        if (
            trace.gateway_result.status is not ActionToExecutorStatus.DISPATCHED
            or trace.gateway_result.execution is None
        ):
            return invalid(
                EpisodeValidationErrorCode.GATEWAY_MISMATCH,
                "A valid non-null action must be dispatched exactly once.",
            )
        executor = MemoryActionExecutor(
            store=MemoryBankStore(initial_state=trace.bank_before)
        )
        expected_execution = executor.execute(decision.memory_action)
        if (
            trace.gateway_result.execution is None
            or trace.gateway_result.execution.to_dict()
            != expected_execution.to_dict()
        ):
            return invalid(
                EpisodeValidationErrorCode.GATEWAY_MISMATCH,
                "Recorded execution must equal deterministic action execution.",
            )
        expected_after = executor.state

    if trace.bank_after != expected_after:
        return invalid(
            EpisodeValidationErrorCode.BANK_TRANSITION_MISMATCH,
            "bank_after does not match the recorded Memory Agent decision.",
        )
    return EpisodeValidationResult()


def retrieval_from_decision(
    trace: MemoryDecisionTrace,
) -> RetrievalResult | None:
    validation = trace.gateway_result.validation
    decision = validation.decision
    if (
        not validation.is_valid
        or decision is None
        or decision.memory_action is None
        or decision.memory_action.action is not MemoryActionName.RETRIEVE
    ):
        return None
    execution = trace.gateway_result.execution
    if execution is None or execution.status is ExecutionStatus.REJECTED:
        return None
    return execution.retrieval


def validate_complete_memory_decision(
    trace: MemoryDecisionTrace,
) -> EpisodeValidationResult:
    validation = validate_memory_decision_trace(trace)
    if not validation.is_valid:
        return validation
    if trace.gateway_result.status is ActionToExecutorStatus.VALIDATION_FAILED:
        return invalid(
            EpisodeValidationErrorCode.DECISION_SEQUENCE_MISMATCH,
            "A completed episode stage cannot contain invalid Memory Agent output.",
        )
    execution = trace.gateway_result.execution
    if execution is not None and execution.status is ExecutionStatus.REJECTED:
        return invalid(
            EpisodeValidationErrorCode.DECISION_SEQUENCE_MISMATCH,
            "A completed episode stage cannot contain a rejected execution.",
        )
    return EpisodeValidationResult()


__all__ = ["validate_memory_decision_trace"]
