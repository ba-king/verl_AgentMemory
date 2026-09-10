"""Strict validation boundary for untrusted Memory Agent completions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from .protocol import (
    AddArguments,
    DeleteArguments,
    MemoryAction,
    MemoryActionArguments,
    MemoryActionName,
    MemoryDecision,
    MemoryDecisionPhase,
    RetrieveArguments,
    UpdateArguments,
)


class ValidationStatus(str, Enum):
    """Top-level result consumed by a future inference runner."""

    VALID = "VALID"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    DISALLOWED_ACTION_FOR_PHASE = "DISALLOWED_ACTION_FOR_PHASE"


class ValidationErrorCode(str, Enum):
    """Stable, machine-readable reasons for non-valid results."""

    RAW_OUTPUT_NOT_STRING = "RAW_OUTPUT_NOT_STRING"
    EMPTY_OUTPUT = "EMPTY_OUTPUT"
    INVALID_JSON = "INVALID_JSON"
    DUPLICATE_JSON_KEY = "DUPLICATE_JSON_KEY"
    TOP_LEVEL_NOT_OBJECT = "TOP_LEVEL_NOT_OBJECT"
    INVALID_ENVELOPE_KEYS = "INVALID_ENVELOPE_KEYS"
    MEMORY_ACTION_NOT_OBJECT = "MEMORY_ACTION_NOT_OBJECT"
    INVALID_ACTION_KEYS = "INVALID_ACTION_KEYS"
    ACTION_NAME_NOT_STRING = "ACTION_NAME_NOT_STRING"
    UNKNOWN_ACTION = "UNKNOWN_ACTION"
    ARGUMENTS_NOT_OBJECT = "ARGUMENTS_NOT_OBJECT"
    INVALID_ARGUMENT_KEYS = "INVALID_ARGUMENT_KEYS"
    ARGUMENT_NOT_STRING = "ARGUMENT_NOT_STRING"
    EMPTY_ARGUMENT = "EMPTY_ARGUMENT"
    ACTION_NOT_ALLOWED_IN_PHASE = "ACTION_NOT_ALLOWED_IN_PHASE"
    MEMORY_ID_NOT_VISIBLE = "MEMORY_ID_NOT_VISIBLE"


@dataclass(frozen=True)
class ValidationResult:
    status: ValidationStatus
    decision: MemoryDecision | None = None
    error_code: ValidationErrorCode | None = None
    error_message: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.status is ValidationStatus.VALID


class _DuplicateKeyError(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


_EXPECTED_ARGUMENT_KEYS: dict[MemoryActionName, frozenset[str]] = {
    MemoryActionName.RETRIEVE: frozenset({"query"}),
    MemoryActionName.ADD: frozenset({"content"}),
    MemoryActionName.UPDATE: frozenset({"memory_id", "content"}),
    MemoryActionName.DELETE: frozenset({"memory_id"}),
}

_ALLOWED_ACTIONS: dict[MemoryDecisionPhase, frozenset[MemoryActionName]] = {
    MemoryDecisionPhase.PRE_TASK: frozenset({MemoryActionName.RETRIEVE}),
    MemoryDecisionPhase.POST_TASK: frozenset(MemoryActionName),
    MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL: frozenset(
        {MemoryActionName.ADD, MemoryActionName.UPDATE, MemoryActionName.DELETE}
    ),
}


def _invalid(code: ValidationErrorCode, message: str) -> ValidationResult:
    return ValidationResult(
        status=ValidationStatus.INVALID_OUTPUT,
        error_code=code,
        error_message=message,
    )


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_json(raw_output: str) -> tuple[Any | None, ValidationResult | None]:
    try:
        payload = json.loads(
            raw_output,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_non_finite_constant,
        )
    except _DuplicateKeyError as exc:
        return None, _invalid(
            ValidationErrorCode.DUPLICATE_JSON_KEY,
            f"JSON object contains the duplicate key {exc.key!r}.",
        )
    except json.JSONDecodeError as exc:
        return None, _invalid(
            ValidationErrorCode.INVALID_JSON,
            f"Output is not strict JSON: {exc.msg} (line {exc.lineno}, column {exc.colno}, character {exc.pos}).",
        )
    except (RecursionError, ValueError):
        return None, _invalid(ValidationErrorCode.INVALID_JSON, "Output is not strict JSON.")
    return payload, None


def _parse_arguments(
    action_name: MemoryActionName,
    raw_arguments: Any,
) -> tuple[MemoryActionArguments | None, ValidationResult | None]:
    if not isinstance(raw_arguments, dict):
        return None, _invalid(
            ValidationErrorCode.ARGUMENTS_NOT_OBJECT,
            "The action arguments must be a JSON object.",
        )

    expected_keys = _EXPECTED_ARGUMENT_KEYS[action_name]
    actual_keys = frozenset(raw_arguments)
    if actual_keys != expected_keys:
        return None, _invalid(
            ValidationErrorCode.INVALID_ARGUMENT_KEYS,
            f"{action_name.value} requires exactly these argument keys: {sorted(expected_keys)}.",
        )

    normalized: dict[str, str] = {}
    for key in expected_keys:
        value = raw_arguments[key]
        if not isinstance(value, str):
            return None, _invalid(
                ValidationErrorCode.ARGUMENT_NOT_STRING,
                f"Argument {key!r} must be a string.",
            )
        value = value.strip()
        if not value:
            return None, _invalid(
                ValidationErrorCode.EMPTY_ARGUMENT,
                f"Argument {key!r} must not be empty.",
            )
        normalized[key] = value

    if action_name is MemoryActionName.RETRIEVE:
        return RetrieveArguments(query=normalized["query"]), None
    if action_name is MemoryActionName.ADD:
        return AddArguments(content=normalized["content"]), None
    if action_name is MemoryActionName.UPDATE:
        return UpdateArguments(
            memory_id=normalized["memory_id"],
            content=normalized["content"],
        ), None
    return DeleteArguments(memory_id=normalized["memory_id"]), None


def validate_memory_decision_output(
    raw_output: str,
    *,
    phase: MemoryDecisionPhase,
    visible_memory_ids: Iterable[str] = (),
) -> ValidationResult:
    """Parse and authorize one raw Memory Agent completion.

    ``phase`` and ``visible_memory_ids`` are trusted caller state rather
    than model output. Invalid model output is returned as data so a future
    MemoryArena runner can log it and safely continue without side effects.
    """

    if not isinstance(phase, MemoryDecisionPhase):
        raise TypeError("phase must be a MemoryDecisionPhase")
    if not isinstance(raw_output, str):
        return _invalid(
            ValidationErrorCode.RAW_OUTPUT_NOT_STRING,
            "Model output must be a string.",
        )
    if not raw_output.strip():
        return _invalid(ValidationErrorCode.EMPTY_OUTPUT, "Model output must not be empty.")

    payload, load_error = _load_json(raw_output)
    if load_error is not None:
        return load_error
    if not isinstance(payload, dict):
        return _invalid(
            ValidationErrorCode.TOP_LEVEL_NOT_OBJECT,
            "The top-level JSON value must be an object.",
        )
    if frozenset(payload) != frozenset({"memory_action"}):
        return _invalid(
            ValidationErrorCode.INVALID_ENVELOPE_KEYS,
            "The envelope must contain exactly the 'memory_action' key.",
        )

    raw_action = payload["memory_action"]
    if raw_action is None:
        return ValidationResult(
            status=ValidationStatus.VALID,
            decision=MemoryDecision(memory_action=None),
        )
    if not isinstance(raw_action, dict):
        return _invalid(
            ValidationErrorCode.MEMORY_ACTION_NOT_OBJECT,
            "'memory_action' must be an object or null.",
        )
    if frozenset(raw_action) != frozenset({"action", "arguments"}):
        return _invalid(
            ValidationErrorCode.INVALID_ACTION_KEYS,
            "A memory action must contain exactly 'action' and 'arguments'.",
        )

    raw_action_name = raw_action["action"]
    if not isinstance(raw_action_name, str):
        return _invalid(
            ValidationErrorCode.ACTION_NAME_NOT_STRING,
            "The action name must be a string.",
        )
    try:
        action_name = MemoryActionName(raw_action_name)
    except ValueError:
        return _invalid(
            ValidationErrorCode.UNKNOWN_ACTION,
            f"Unknown memory action {raw_action_name!r}.",
        )

    arguments, arguments_error = _parse_arguments(action_name, raw_action["arguments"])
    if arguments_error is not None:
        return arguments_error
    assert arguments is not None

    if action_name not in _ALLOWED_ACTIONS[phase]:
        return ValidationResult(
            status=ValidationStatus.DISALLOWED_ACTION_FOR_PHASE,
            error_code=ValidationErrorCode.ACTION_NOT_ALLOWED_IN_PHASE,
            error_message=f"{action_name.value} is not allowed in phase {phase.value}.",
        )

    action = MemoryAction(action=action_name, arguments=arguments)
    if isinstance(arguments, (UpdateArguments, DeleteArguments)):
        visible_ids = frozenset(visible_memory_ids)
        if arguments.memory_id not in visible_ids:
            return _invalid(
                ValidationErrorCode.MEMORY_ID_NOT_VISIBLE,
                "UPDATE and DELETE may reference only a memory ID visible in the current state.",
            )

    return ValidationResult(
        status=ValidationStatus.VALID,
        decision=MemoryDecision(memory_action=action),
    )


__all__ = [
    "ValidationErrorCode",
    "ValidationResult",
    "ValidationStatus",
    "validate_memory_decision_output",
]
