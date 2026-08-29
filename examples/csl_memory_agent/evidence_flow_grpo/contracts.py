"""Strict action contract for the Evidence Flow GRPO policy.

The policy emits one JSON object and nothing else.  This module deliberately
does not extract JSON from prose, markdown fences, or model-specific wrappers:
malformed output is a training signal, not something the scorer repairs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


ACTION_KEYS = frozenset({"action", "evidence_status", "selected_memory_ids"})
RETRIEVE = "RETRIEVE"
NO_RETRIEVE = "NO_RETRIEVE"
SUFFICIENT = "SUFFICIENT"
INSUFFICIENT = "INSUFFICIENT"
NOT_NEEDED = "NOT_NEEDED"

VALID_ACTIONS = frozenset({RETRIEVE, NO_RETRIEVE})
VALID_EVIDENCE_STATUSES = frozenset({SUFFICIENT, INSUFFICIENT, NOT_NEEDED})
NON_SUFFICIENT_STATUSES = frozenset({INSUFFICIENT, NOT_NEEDED})
DEFAULT_MAX_SELECTED = 3


class ContractError(ValueError):
    """Raised when a model output violates the canonical action contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ActionDecision:
    """A validated, immutable policy action."""

    action: str
    evidence_status: str
    selected_memory_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible representation."""

        return {
            "action": self.action,
            "evidence_status": self.evidence_status,
            "selected_memory_ids": list(self.selected_memory_ids),
        }


def _reject_non_json_constant(value: str) -> None:
    raise ContractError(
        "non_json_constant", f"non-standard JSON constant is forbidden: {value}"
    )


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate_key", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_strict(text: str, *, field: str = "value") -> Any:
    """Parse one standards-compliant JSON value without duplicate object keys."""

    if not isinstance(text, str):
        raise ContractError("not_string", f"{field} must be a string")
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_non_json_constant,
        )
    except ContractError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ContractError(
            "invalid_json", f"{field} must be exactly one JSON value"
        ) from exc


def _validate_max_selected(max_selected: int) -> None:
    if (
        isinstance(max_selected, bool)
        or not isinstance(max_selected, int)
        or max_selected < 1
    ):
        raise ValueError("max_selected must be a positive integer")


def _known_id_set(known_memory_ids: Iterable[str]) -> frozenset[str]:
    if isinstance(known_memory_ids, (str, bytes)):
        raise ValueError("known_memory_ids must be an iterable of memory IDs")
    try:
        values = tuple(known_memory_ids)
    except TypeError as exc:
        raise ValueError("known_memory_ids must be an iterable of memory IDs") from exc
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError("known_memory_ids must contain only non-empty strings")
    return frozenset(values)


def validate_action_mapping(
    value: Mapping[str, Any],
    *,
    known_memory_ids: Iterable[str],
    max_selected: int = DEFAULT_MAX_SELECTED,
) -> ActionDecision:
    """Validate a decoded mapping against the canonical action contract."""

    _validate_max_selected(max_selected)
    known_ids = _known_id_set(known_memory_ids)

    if not isinstance(value, Mapping):
        raise ContractError("not_object", "action output must be a JSON object")
    if set(value) != ACTION_KEYS:
        missing = sorted(ACTION_KEYS - set(value))
        extra = sorted(set(value) - ACTION_KEYS)
        raise ContractError(
            "wrong_keys",
            f"action object has wrong keys; missing={missing}, extra={extra}",
        )

    action = value["action"]
    status = value["evidence_status"]
    selected = value["selected_memory_ids"]

    if not isinstance(action, str) or action not in VALID_ACTIONS:
        raise ContractError(
            "invalid_action",
            f"action must be one of {sorted(VALID_ACTIONS)}",
        )
    if not isinstance(status, str) or status not in VALID_EVIDENCE_STATUSES:
        raise ContractError(
            "invalid_evidence_status",
            "evidence_status must be one of " f"{sorted(VALID_EVIDENCE_STATUSES)}",
        )
    if (
        isinstance(selected, (str, bytes))
        or not isinstance(selected, Sequence)
        or not isinstance(selected, list)
    ):
        raise ContractError(
            "selected_ids_not_array",
            "selected_memory_ids must be a JSON array",
        )
    if any(not isinstance(item, str) or not item for item in selected):
        raise ContractError(
            "invalid_memory_id",
            "selected_memory_ids must contain only non-empty strings",
        )
    if len(set(selected)) != len(selected):
        raise ContractError("duplicate_memory_id", "selected_memory_ids must be unique")
    unknown_ids = sorted(set(selected) - known_ids)
    if unknown_ids:
        raise ContractError(
            "unknown_memory_id",
            f"selected_memory_ids contains unknown IDs: {unknown_ids}",
        )

    if action == RETRIEVE:
        if status != SUFFICIENT:
            raise ContractError(
                "retrieve_requires_sufficient",
                "RETRIEVE requires evidence_status=SUFFICIENT",
            )
        if not 1 <= len(selected) <= max_selected:
            raise ContractError(
                "retrieve_selection_count",
                f"RETRIEVE requires 1..{max_selected} selected IDs",
            )
    else:
        if status not in NON_SUFFICIENT_STATUSES:
            raise ContractError(
                "no_retrieve_requires_non_sufficient",
                "NO_RETRIEVE requires INSUFFICIENT or NOT_NEEDED",
            )
        if selected:
            raise ContractError(
                "no_retrieve_requires_empty_selection",
                "NO_RETRIEVE requires an empty selected_memory_ids array",
            )

    return ActionDecision(
        action=action,
        evidence_status=status,
        selected_memory_ids=tuple(selected),
    )


def parse_action_json(
    solution_str: str,
    *,
    known_memory_ids: Iterable[str],
    max_selected: int = DEFAULT_MAX_SELECTED,
) -> ActionDecision:
    """Parse and validate one model response, with no extraction or repair."""

    value = load_json_strict(solution_str, field="solution_str")
    if not isinstance(value, Mapping):
        raise ContractError("not_object", "action output must be a JSON object")
    return validate_action_mapping(
        value,
        known_memory_ids=known_memory_ids,
        max_selected=max_selected,
    )


__all__ = [
    "ACTION_KEYS",
    "DEFAULT_MAX_SELECTED",
    "INSUFFICIENT",
    "NON_SUFFICIENT_STATUSES",
    "NOT_NEEDED",
    "NO_RETRIEVE",
    "RETRIEVE",
    "SUFFICIENT",
    "VALID_ACTIONS",
    "VALID_EVIDENCE_STATUSES",
    "ActionDecision",
    "ContractError",
    "load_json_strict",
    "parse_action_json",
    "validate_action_mapping",
]
