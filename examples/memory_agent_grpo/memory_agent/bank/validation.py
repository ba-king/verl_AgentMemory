"""Integrity checks for typed Memory Bank snapshots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .bank_state import MemoryBankState, MemoryRecord


class BankValidationErrorCode(str, Enum):
    """Stable reasons why a Memory Bank snapshot is invalid."""

    STATE_NOT_MEMORY_BANK_STATE = "STATE_NOT_MEMORY_BANK_STATE"
    NEXT_ID_NOT_INTEGER = "NEXT_ID_NOT_INTEGER"
    NEXT_ID_OUT_OF_RANGE = "NEXT_ID_OUT_OF_RANGE"
    RECORDS_NOT_TUPLE = "RECORDS_NOT_TUPLE"
    RECORD_NOT_MEMORY_RECORD = "RECORD_NOT_MEMORY_RECORD"
    MEMORY_ID_NOT_STRING = "MEMORY_ID_NOT_STRING"
    INVALID_MEMORY_ID = "INVALID_MEMORY_ID"
    CONTENT_NOT_STRING = "CONTENT_NOT_STRING"
    EMPTY_CONTENT = "EMPTY_CONTENT"
    DUPLICATE_MEMORY_ID = "DUPLICATE_MEMORY_ID"
    RECORDS_OUT_OF_ORDER = "RECORDS_OUT_OF_ORDER"
    NEXT_ID_NOT_AFTER_RECORDS = "NEXT_ID_NOT_AFTER_RECORDS"


@dataclass(frozen=True)
class BankValidationResult:
    error_code: BankValidationErrorCode | None = None
    error_message: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.error_code is None


_MEMORY_ID_PATTERN = re.compile(r"M[0-9]{3,}\Z")


def _invalid(code: BankValidationErrorCode, message: str) -> BankValidationResult:
    return BankValidationResult(error_code=code, error_message=message)


def _memory_id_ordinal(memory_id: str) -> int | None:
    if _MEMORY_ID_PATTERN.fullmatch(memory_id) is None:
        return None

    digits = memory_id[1:]
    if digits == "000" or (len(digits) > 3 and digits.startswith("0")):
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def is_canonical_memory_id(value: object) -> bool:
    """Return whether a value uses the canonical expandable M001-style format."""

    return isinstance(value, str) and _memory_id_ordinal(value) is not None


def validate_memory_bank_state(state: object) -> BankValidationResult:
    """Check one typed snapshot without mutating or repairing it.

    The first failed invariant is returned as structured data. A future store
    can use this function at construction and persistence boundaries.
    """

    if not isinstance(state, MemoryBankState):
        return _invalid(
            BankValidationErrorCode.STATE_NOT_MEMORY_BANK_STATE,
            "State must be a MemoryBankState.",
        )

    if isinstance(state.next_id, bool) or not isinstance(state.next_id, int):
        return _invalid(
            BankValidationErrorCode.NEXT_ID_NOT_INTEGER,
            "next_id must be an integer.",
        )
    if state.next_id < 1:
        return _invalid(
            BankValidationErrorCode.NEXT_ID_OUT_OF_RANGE,
            "next_id must be at least 1.",
        )
    if not isinstance(state.records, tuple):
        return _invalid(
            BankValidationErrorCode.RECORDS_NOT_TUPLE,
            "records must be an immutable tuple.",
        )

    seen_ids: set[str] = set()
    previous_ordinal = 0
    for position, record in enumerate(state.records):
        if not isinstance(record, MemoryRecord):
            return _invalid(
                BankValidationErrorCode.RECORD_NOT_MEMORY_RECORD,
                f"Record at position {position} must be a MemoryRecord.",
            )
        if not isinstance(record.memory_id, str):
            return _invalid(
                BankValidationErrorCode.MEMORY_ID_NOT_STRING,
                f"memory_id at position {position} must be a string.",
            )

        ordinal = _memory_id_ordinal(record.memory_id)
        if ordinal is None:
            return _invalid(
                BankValidationErrorCode.INVALID_MEMORY_ID,
                f"memory_id {record.memory_id!r} is not in canonical M001-style format.",
            )
        if record.memory_id in seen_ids:
            return _invalid(
                BankValidationErrorCode.DUPLICATE_MEMORY_ID,
                f"memory_id {record.memory_id!r} appears more than once.",
            )
        if ordinal <= previous_ordinal:
            return _invalid(
                BankValidationErrorCode.RECORDS_OUT_OF_ORDER,
                "records must remain in increasing creation-ID order.",
            )
        if not isinstance(record.content, str):
            return _invalid(
                BankValidationErrorCode.CONTENT_NOT_STRING,
                f"content at position {position} must be a string.",
            )
        if not record.content.strip():
            return _invalid(
                BankValidationErrorCode.EMPTY_CONTENT,
                f"content at position {position} must not be empty.",
            )

        seen_ids.add(record.memory_id)
        previous_ordinal = ordinal

    if previous_ordinal >= state.next_id:
        return _invalid(
            BankValidationErrorCode.NEXT_ID_NOT_AFTER_RECORDS,
            "next_id must be greater than every current record ID ordinal.",
        )

    return BankValidationResult()


__all__ = [
    "BankValidationErrorCode",
    "BankValidationResult",
    "is_canonical_memory_id",
    "validate_memory_bank_state",
]
