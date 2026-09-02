"""Pre-execution checks over typed actions and the current Bank snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from examples.memory_agent_grpo.memory_agent.action.protocol import (
    AddArguments,
    DeleteArguments,
    MemoryAction,
    MemoryActionName,
    RetrieveArguments,
    UpdateArguments,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState, MemoryRecord
from examples.memory_agent_grpo.memory_agent.bank.validation import validate_memory_bank_state

from .outcome import ExecutionCode


class ExecutorValidationDisposition(str, Enum):
    READY = "READY"
    NO_CHANGE = "NO_CHANGE"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ExecutorValidationResult:
    disposition: ExecutorValidationDisposition
    code: ExecutionCode | None = None
    error_message: str | None = None
    record: MemoryRecord | None = None

    @property
    def is_ready(self) -> bool:
        return self.disposition is ExecutorValidationDisposition.READY


_EXPECTED_ARGUMENT_TYPE: dict[MemoryActionName, type[object]] = {
    MemoryActionName.RETRIEVE: RetrieveArguments,
    MemoryActionName.ADD: AddArguments,
    MemoryActionName.UPDATE: UpdateArguments,
    MemoryActionName.DELETE: DeleteArguments,
}


def validate_action_for_execution(
    action: object,
    *,
    state: MemoryBankState,
) -> ExecutorValidationResult:
    """Classify a typed action without executing it or changing Bank state."""

    state_validation = validate_memory_bank_state(state)
    if not state_validation.is_valid:
        return _rejected(
            ExecutionCode.INVALID_BANK_STATE,
            f"Current Memory Bank state is invalid: {state_validation.error_code}.",
        )
    if not isinstance(action, MemoryAction):
        return _rejected(
            ExecutionCode.INVALID_ACTION,
            "Executor accepts exactly one MemoryAction.",
        )
    if not isinstance(action.action, MemoryActionName):
        return _rejected(
            ExecutionCode.INVALID_ACTION,
            "Action name must be a MemoryActionName.",
        )

    expected_argument_type = _EXPECTED_ARGUMENT_TYPE[action.action]
    if not isinstance(action.arguments, expected_argument_type):
        return _rejected(
            ExecutionCode.INVALID_ACTION,
            f"{action.action.value} received the wrong typed argument object.",
        )

    arguments = action.arguments
    if isinstance(arguments, RetrieveArguments):
        return _validate_string_argument(arguments.query, name="query")
    if isinstance(arguments, AddArguments):
        value_validation = _validate_string_argument(arguments.content, name="content")
        if not value_validation.is_ready:
            return value_validation
        for record in state.records:
            if record.content == arguments.content:
                return ExecutorValidationResult(
                    disposition=ExecutorValidationDisposition.NO_CHANGE,
                    code=ExecutionCode.EXACT_DUPLICATE_ADD,
                    record=record,
                )
        return ExecutorValidationResult(disposition=ExecutorValidationDisposition.READY)
    if isinstance(arguments, UpdateArguments):
        id_validation = _validate_string_argument(arguments.memory_id, name="memory_id")
        if not id_validation.is_ready:
            return id_validation
        content_validation = _validate_string_argument(arguments.content, name="content")
        if not content_validation.is_ready:
            return content_validation
        target = _find_record(state, arguments.memory_id)
        if target is None:
            return _rejected(
                ExecutionCode.TARGET_NOT_FOUND,
                f"Memory record {arguments.memory_id!r} does not exist.",
            )
        if target.content == arguments.content:
            return ExecutorValidationResult(
                disposition=ExecutorValidationDisposition.NO_CHANGE,
                code=ExecutionCode.UNCHANGED_UPDATE,
                record=target,
            )
        return ExecutorValidationResult(disposition=ExecutorValidationDisposition.READY)

    id_validation = _validate_string_argument(arguments.memory_id, name="memory_id")
    if not id_validation.is_ready:
        return id_validation
    if _find_record(state, arguments.memory_id) is None:
        return _rejected(
            ExecutionCode.TARGET_NOT_FOUND,
            f"Memory record {arguments.memory_id!r} does not exist.",
        )
    return ExecutorValidationResult(disposition=ExecutorValidationDisposition.READY)


def _validate_string_argument(value: object, *, name: str) -> ExecutorValidationResult:
    if not isinstance(value, str):
        return _rejected(
            ExecutionCode.INVALID_ARGUMENTS,
            f"{name} must be a string.",
        )
    if not value.strip():
        return _rejected(
            ExecutionCode.INVALID_ARGUMENTS,
            f"{name} must not be empty.",
        )
    return ExecutorValidationResult(disposition=ExecutorValidationDisposition.READY)


def _find_record(state: MemoryBankState, memory_id: str) -> MemoryRecord | None:
    for record in state.records:
        if record.memory_id == memory_id:
            return record
    return None


def _rejected(code: ExecutionCode, message: str) -> ExecutorValidationResult:
    return ExecutorValidationResult(
        disposition=ExecutorValidationDisposition.REJECTED,
        code=code,
        error_message=message,
    )


__all__ = [
    "ExecutorValidationDisposition",
    "ExecutorValidationResult",
    "validate_action_for_execution",
]
