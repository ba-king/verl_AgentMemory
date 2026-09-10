"""Execute one Memory Agent model-to-Bank decision without retries."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from examples.memory_agent_grpo.memory_agent.action.repair import repair_missing_add_outer_brace
from examples.memory_agent_grpo.memory_agent.action.validation import (
    ValidationErrorCode,
)
from examples.memory_agent_grpo.memory_agent.decision.action_gateway import (
    ActionToExecutorGateway,
    ActionToExecutorStatus,
)
from examples.memory_agent_grpo.memory_agent.decision.context import (
    MemoryDecisionContext,
)
from examples.memory_agent_grpo.memory_agent.decision.model_client import (
    MemoryAgentClient,
    MemoryAgentClientError,
    MemoryAgentClientErrorCode,
)
from examples.memory_agent_grpo.memory_agent.executor.outcome import (
    ExecutionStatus,
)
from examples.memory_agent_grpo.memory_agent.prompts.correction import (
    build_memory_agent_correction_input,
)
from examples.memory_agent_grpo.memory_agent.prompts.decision import (
    build_memory_agent_input,
)

from ..trace import MemoryDecisionTrace
from ..validation import validate_memory_decision_trace


class MemoryDecisionStepErrorCode(str, Enum):
    MODEL_CALL_FAILED = "MODEL_CALL_FAILED"
    OUTPUT_VALIDATION_FAILED = "OUTPUT_VALIDATION_FAILED"
    EXECUTION_REJECTED = "EXECUTION_REJECTED"
    INTERNAL_CONTRACT_ERROR = "INTERNAL_CONTRACT_ERROR"


class MemoryDecisionStepError(RuntimeError):
    """Stable failure for one uncommitted Memory Agent decision."""

    def __init__(
        self,
        code: MemoryDecisionStepErrorCode,
        message: str,
        *,
        validation_error_code: ValidationErrorCode | None = None,
        model_client_error_code: MemoryAgentClientErrorCode | None = None,
        rejected_output: str | None = None,
        validation_error_message: str | None = None,
    ) -> None:
        self.code = code
        self.validation_error_code = validation_error_code
        self.model_client_error_code = model_client_error_code
        # Same-decision repair evidence; deliberately excluded from exception text.
        self.rejected_output = rejected_output
        self.validation_error_message = validation_error_message
        super().__init__(f"{code.value}: {message}")


def execute_memory_decision(
    *,
    memory_client: MemoryAgentClient,
    gateway: ActionToExecutorGateway,
    context: MemoryDecisionContext,
    correction_error_code: ValidationErrorCode | None = None,
    correction_raw_output: str | None = None,
    correction_error_message: str | None = None,
    repair_add_outer_brace: bool = False,
    trace_observer: Callable[[MemoryDecisionTrace], None] | None = None,
) -> MemoryDecisionTrace:
    """Run exactly one model call and validate its deterministic transition.

    A missing ADD outer brace may be repaired deterministically before dispatch.
    The raw output and repair evidence remain separate in the trace. No model
    retry or conversion to a null action occurs here. The default remains strict;
    diagnostic callers can explicitly set repair_add_outer_brace to True.
    The caller decides whether the uncommitted episode stage should fail.
    The optional observer receives every structurally valid trace, including
    rejected model outputs, before the decision's final status is propagated.
    Observer failures stop this step without retrying an already executed action.
    """

    if type(repair_add_outer_brace) is not bool:
        raise MemoryDecisionStepError(
            MemoryDecisionStepErrorCode.INTERNAL_CONTRACT_ERROR,
            "repair_add_outer_brace must be a bool.",
        )
    if trace_observer is not None and not callable(trace_observer):
        raise MemoryDecisionStepError(
            MemoryDecisionStepErrorCode.INTERNAL_CONTRACT_ERROR,
            "Memory decision trace observer must be callable.",
        )

    try:
        model_input = (
            build_memory_agent_input(context)
            if correction_error_code is None
            else build_memory_agent_correction_input(
                context,
                validation_error_code=correction_error_code,
                rejected_output=correction_raw_output,
                validation_error_message=correction_error_message,
            )
        )
    except Exception:
        raise MemoryDecisionStepError(
            MemoryDecisionStepErrorCode.INTERNAL_CONTRACT_ERROR,
            "Memory Agent input construction failed.",
        ) from None

    bank_before = gateway.state
    try:
        raw_output = memory_client.generate(model_input)
    except MemoryAgentClientError as error:
        raise MemoryDecisionStepError(
            MemoryDecisionStepErrorCode.MODEL_CALL_FAILED,
            "Memory Agent model call failed.",
            model_client_error_code=error.code,
        ) from None
    except Exception:
        raise MemoryDecisionStepError(
            MemoryDecisionStepErrorCode.MODEL_CALL_FAILED,
            "Memory Agent model call failed.",
        ) from None

    try:
        repair = (
            repair_missing_add_outer_brace(
                raw_output,
                phase=context.phase,
                visible_memory_ids=context.visible_memory_ids,
            )
            if repair_add_outer_brace else None
        )
        effective_output = raw_output if repair is None else repair.repaired_output
        gateway_result = gateway.process_output(effective_output, context=context)
        trace = MemoryDecisionTrace(
            context=context,
            raw_output=raw_output,
            gateway_result=gateway_result,
            bank_before=bank_before,
            bank_after=gateway.state,
            output_repair=repair,
        )
        validation = validate_memory_decision_trace(trace)
    except Exception:
        raise MemoryDecisionStepError(
            MemoryDecisionStepErrorCode.INTERNAL_CONTRACT_ERROR,
            "Memory decision trace construction failed.",
        ) from None

    if not validation.is_valid:
        raise MemoryDecisionStepError(
            MemoryDecisionStepErrorCode.INTERNAL_CONTRACT_ERROR,
            "Memory decision trace violates the runtime contract.",
        )
    if trace_observer is not None:
        try:
            trace_observer(trace)
        except Exception:
            raise MemoryDecisionStepError(
                MemoryDecisionStepErrorCode.INTERNAL_CONTRACT_ERROR,
                "Memory decision trace observer failed.",
            ) from None
    if gateway_result.status is ActionToExecutorStatus.VALIDATION_FAILED:
        raise MemoryDecisionStepError(
            MemoryDecisionStepErrorCode.OUTPUT_VALIDATION_FAILED,
            "Memory Agent output failed validation.",
            validation_error_code=gateway_result.validation.error_code,
            rejected_output=raw_output,
            validation_error_message=gateway_result.validation.error_message,
        )
    execution = gateway_result.execution
    if execution is not None and execution.status is ExecutionStatus.REJECTED:
        raise MemoryDecisionStepError(
            MemoryDecisionStepErrorCode.EXECUTION_REJECTED,
            "Memory action execution was rejected.",
        )
    return trace


__all__ = [
    "MemoryDecisionStepError",
    "MemoryDecisionStepErrorCode",
    "execute_memory_decision",
]
