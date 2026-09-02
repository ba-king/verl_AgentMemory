"""Safely transfer one raw model completion to the action Executor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from examples.memory_agent_grpo.memory_agent.action.protocol import MemoryDecisionPhase
from examples.memory_agent_grpo.memory_agent.action.validation import (
    ValidationResult,
    validate_memory_decision_output,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState
from examples.memory_agent_grpo.memory_agent.executor.engine import MemoryActionExecutor
from examples.memory_agent_grpo.memory_agent.executor.outcome import ExecutionResult
from examples.memory_agent_grpo.memory_agent.decision.context import MemoryDecisionContext
from examples.memory_agent_grpo.memory_agent.decision.context_validation import (
    validate_memory_decision_context,
)


class ActionToExecutorStatus(str, Enum):
    VALIDATION_FAILED = "VALIDATION_FAILED"
    SKIPPED = "SKIPPED"
    DISPATCHED = "DISPATCHED"


@dataclass(frozen=True)
class ActionToExecutorResult:
    status: ActionToExecutorStatus
    phase: MemoryDecisionPhase
    validation: ValidationResult
    execution: ExecutionResult | None = None

    def to_dict(self) -> dict[str, object | None]:
        decision = self.validation.decision
        return {
            "status": self.status.value,
            "phase": self.phase.value,
            "validation": {
                "status": self.validation.status.value,
                "decision": None if decision is None else decision.to_dict(),
                "error_code": (
                    None if self.validation.error_code is None else self.validation.error_code.value
                ),
                "error_message": self.validation.error_message,
            },
            "execution": None if self.execution is None else self.execution.to_dict(),
        }


class ActionToExecutorGateway:
    """The sole normal gateway from raw model output to the Executor."""

    def __init__(self, *, executor: MemoryActionExecutor | None = None) -> None:
        self._executor = MemoryActionExecutor() if executor is None else executor

    @property
    def state(self) -> MemoryBankState:
        return self._executor.state

    def process_output(
        self,
        raw_output: str,
        *,
        context: MemoryDecisionContext,
    ) -> ActionToExecutorResult:
        """Validate, skip, or dispatch exactly one raw model completion."""

        context_validation = validate_memory_decision_context(context)
        if not context_validation.is_valid:
            error_code = context_validation.error_code
            code = "UNKNOWN" if error_code is None else error_code.value
            raise ValueError(
                f"Invalid MemoryDecisionContext ({code}): {context_validation.error_message}"
            )

        validation = validate_memory_decision_output(
            raw_output,
            phase=context.phase,
            visible_memory_ids=context.visible_memory_ids,
        )
        if not validation.is_valid:
            return ActionToExecutorResult(
                status=ActionToExecutorStatus.VALIDATION_FAILED,
                phase=context.phase,
                validation=validation,
            )

        decision = validation.decision
        if decision is None:
            raise RuntimeError("Valid action validation result did not contain a decision.")
        if decision.memory_action is None:
            return ActionToExecutorResult(
                status=ActionToExecutorStatus.SKIPPED,
                phase=context.phase,
                validation=validation,
            )

        execution = self._executor.execute(decision.memory_action)
        return ActionToExecutorResult(
            status=ActionToExecutorStatus.DISPATCHED,
            phase=context.phase,
            validation=validation,
            execution=execution,
        )


__all__ = [
    "ActionToExecutorGateway",
    "ActionToExecutorResult",
    "ActionToExecutorStatus",
]
