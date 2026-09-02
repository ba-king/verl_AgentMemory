"""Typed execution outcomes returned by the deterministic action Executor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from examples.memory_agent_grpo.memory_agent.action.protocol import MemoryActionName
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryRecord
from examples.memory_agent_grpo.memory_agent.retrieval.result import RetrievalResult


class ExecutionStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    NO_CHANGE = "NO_CHANGE"
    REJECTED = "REJECTED"


class ExecutionCode(str, Enum):
    RETRIEVED = "RETRIEVED"
    ADDED = "ADDED"
    UPDATED = "UPDATED"
    DELETED = "DELETED"
    EXACT_DUPLICATE_ADD = "EXACT_DUPLICATE_ADD"
    UNCHANGED_UPDATE = "UNCHANGED_UPDATE"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    INVALID_ACTION = "INVALID_ACTION"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    INVALID_BANK_STATE = "INVALID_BANK_STATE"


@dataclass(frozen=True)
class ExecutionResult:
    status: ExecutionStatus
    code: ExecutionCode
    action: MemoryActionName | None
    record: MemoryRecord | None = None
    retrieval: RetrievalResult | None = None
    error_message: str | None = None

    @property
    def is_rejected(self) -> bool:
        return self.status is ExecutionStatus.REJECTED

    def to_dict(self) -> dict[str, object | None]:
        return {
            "status": self.status.value,
            "code": self.code.value,
            "action": None if self.action is None else self.action.value,
            "record": None if self.record is None else self.record.to_dict(),
            "retrieval": None if self.retrieval is None else self.retrieval.to_dict(),
            "error_message": self.error_message,
        }


__all__ = ["ExecutionCode", "ExecutionResult", "ExecutionStatus"]
