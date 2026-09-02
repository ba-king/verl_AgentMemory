"""Deterministic in-memory state transitions for the Memory Bank."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .bank_state import MemoryBankState, MemoryRecord
from .validation import BankValidationResult, validate_memory_bank_state


class StoreMutationStatus(str, Enum):
    """Outcome of a valid Store method call, not a Memory Agent action."""

    APPLIED = "APPLIED"
    NO_CHANGE = "NO_CHANGE"


@dataclass(frozen=True)
class StoreMutationResult:
    status: StoreMutationStatus
    state: MemoryBankState
    record: MemoryRecord

    @property
    def changed(self) -> bool:
        return self.status is StoreMutationStatus.APPLIED


class InvalidMemoryBankStateError(ValueError):
    """Raised when a Store is initialized with an invalid snapshot."""

    def __init__(self, result: BankValidationResult) -> None:
        self.validation_result = result
        code = result.error_code.value if result.error_code is not None else "UNKNOWN"
        super().__init__(f"Invalid Memory Bank state ({code}): {result.error_message}")


class MemoryRecordNotFoundError(LookupError):
    """Raised when UPDATE or DELETE targets a record absent from the Bank."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__(f"Memory record {memory_id!r} does not exist.")


class MemoryBankStore:
    """Own the current immutable snapshot and apply deterministic mutations."""

    def __init__(self, initial_state: MemoryBankState | None = None) -> None:
        state = MemoryBankState.empty() if initial_state is None else initial_state
        validation_result = validate_memory_bank_state(state)
        if not validation_result.is_valid:
            raise InvalidMemoryBankStateError(validation_result)
        self._state = state

    @property
    def state(self) -> MemoryBankState:
        return self._state

    def add(self, content: str) -> StoreMutationResult:
        """Append one unique textual memory and allocate its permanent ID."""

        content = _require_content(content)
        for record in self._state.records:
            if record.content == content:
                return StoreMutationResult(
                    status=StoreMutationStatus.NO_CHANGE,
                    state=self._state,
                    record=record,
                )

        record = MemoryRecord(
            memory_id=f"M{self._state.next_id:03d}",
            content=content,
        )
        new_state = MemoryBankState(
            records=(*self._state.records, record),
            next_id=self._state.next_id + 1,
        )
        self._commit(new_state)
        return StoreMutationResult(
            status=StoreMutationStatus.APPLIED,
            state=new_state,
            record=record,
        )

    def update(self, memory_id: str, content: str) -> StoreMutationResult:
        """Fully replace one record's content while preserving ID and position."""

        memory_id = _require_memory_id(memory_id)
        content = _require_content(content)
        position = self._find_position(memory_id)
        current_record = self._state.records[position]
        if current_record.content == content:
            return StoreMutationResult(
                status=StoreMutationStatus.NO_CHANGE,
                state=self._state,
                record=current_record,
            )

        updated_record = MemoryRecord(memory_id=memory_id, content=content)
        records = list(self._state.records)
        records[position] = updated_record
        new_state = MemoryBankState(records=tuple(records), next_id=self._state.next_id)
        self._commit(new_state)
        return StoreMutationResult(
            status=StoreMutationStatus.APPLIED,
            state=new_state,
            record=updated_record,
        )

    def delete(self, memory_id: str) -> StoreMutationResult:
        """Remove one record without making its ID available for reuse."""

        memory_id = _require_memory_id(memory_id)
        position = self._find_position(memory_id)
        deleted_record = self._state.records[position]
        new_state = MemoryBankState(
            records=self._state.records[:position] + self._state.records[position + 1 :],
            next_id=self._state.next_id,
        )
        self._commit(new_state)
        return StoreMutationResult(
            status=StoreMutationStatus.APPLIED,
            state=new_state,
            record=deleted_record,
        )

    def _find_position(self, memory_id: str) -> int:
        for position, record in enumerate(self._state.records):
            if record.memory_id == memory_id:
                return position
        raise MemoryRecordNotFoundError(memory_id)

    def _commit(self, new_state: MemoryBankState) -> None:
        validation_result = validate_memory_bank_state(new_state)
        if not validation_result.is_valid:
            raise RuntimeError(
                "Store produced an invalid Memory Bank state: "
                f"{validation_result.error_code}: {validation_result.error_message}"
            )
        self._state = new_state


def _require_content(content: object) -> str:
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    if not content.strip():
        raise ValueError("content must not be empty")
    return content


def _require_memory_id(memory_id: object) -> str:
    if not isinstance(memory_id, str):
        raise TypeError("memory_id must be a string")
    if not memory_id.strip():
        raise ValueError("memory_id must not be empty")
    return memory_id


__all__ = [
    "InvalidMemoryBankStateError",
    "MemoryBankStore",
    "MemoryRecordNotFoundError",
    "StoreMutationResult",
    "StoreMutationStatus",
]
