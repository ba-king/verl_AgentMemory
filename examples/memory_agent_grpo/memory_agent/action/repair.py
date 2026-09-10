"""One conservative, deterministic syntax repair; strict validation stays separate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .protocol import MemoryActionName, MemoryDecisionPhase
from .validation import (
    ValidationErrorCode,
    ValidationResult,
    validate_memory_decision_output,
)


@dataclass(frozen=True)
class MemoryOutputRepair:
    rule: str
    repaired_output: str
    original_validation: ValidationResult


def _only_outer_object_is_open(text: str) -> bool:
    """Count structural delimiters, respecting quoted and escaped characters."""

    if not text.lstrip().startswith("{"):
        return False
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or stack.pop() != {"}": "{", "]": "["}[char]:
                return False
            if not stack:
                # An already closed top-level object is never a repair candidate.
                return False
    return not in_string and stack == ["{"]


def repair_missing_add_outer_brace(
    raw_output: str,
    *,
    phase: MemoryDecisionPhase,
    visible_memory_ids: Iterable[str] = (),
) -> MemoryOutputRepair | None:
    """Append exactly one outer brace only if the result is an authorized ADD.

    Never edit, trim, reserialize, or infer content. Duplicate keys, incomplete
    strings, missing inner braces, other actions and unauthorized phases fail
    closed. This function performs no model call or Bank mutation.
    """

    visible_ids = tuple(visible_memory_ids)
    original = validate_memory_decision_output(
        raw_output, phase=phase, visible_memory_ids=visible_ids
    )
    if original.error_code is not ValidationErrorCode.INVALID_JSON:
        return None
    if not _only_outer_object_is_open(raw_output):
        return None
    repaired_output = raw_output + "}"
    repaired = validate_memory_decision_output(
        repaired_output, phase=phase, visible_memory_ids=visible_ids
    )
    if not repaired.is_valid or repaired.decision is None:
        return None
    action = repaired.decision.memory_action
    if action is None or action.action is not MemoryActionName.ADD:
        return None
    return MemoryOutputRepair(
        rule="ADD_MISSING_OUTER_BRACE_V1",
        repaired_output=repaired_output,
        original_validation=original,
    )
