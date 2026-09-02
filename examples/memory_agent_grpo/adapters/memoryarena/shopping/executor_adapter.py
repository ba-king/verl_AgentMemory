"""Translate Gateway-approved native calls into one Shopping session step."""

from __future__ import annotations

import json
import math

from ....answer_agent.runtime import AnswerAgentTaskToolExecutionResult
from ....answer_agent.tool_call_gateway.validated_call import ValidatedToolCall
from ....answer_agent.tool_protocol.observation import (
    ToolObservation,
    ToolObservationStatus,
)
from ....answer_agent.tool_protocol.validation import validate_tool_observation
from .environment_session import (
    ShoppingEnvironmentSession,
    ShoppingEnvironmentSessionError,
)

_SAFE_EXECUTION_ERROR = "Shopping action execution failed."
_TOOL_ARGUMENT_NAMES = {
    "search": "keywords",
    "click": "item",
}


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-standard JSON constant {value!r} is not allowed.")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r} is not allowed.")
        result[key] = value
    return result


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number must be finite.")
    return parsed


def _parse_canonical_arguments(value: object) -> dict[str, object]:
    if type(value) is not str or not value:
        raise ValueError("ValidatedToolCall.arguments_json must be a non-empty string.")
    try:
        parsed = json.loads(
            value,
            parse_float=_parse_finite_float,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if type(parsed) is not dict:
            raise ValueError("arguments must encode one object.")
        canonical = json.dumps(
            parsed,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        canonical.encode("utf-8")
    except (
        UnicodeEncodeError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ):
        raise ValueError("ValidatedToolCall.arguments_json must be canonical finite UTF-8 JSON.") from None
    if canonical != value:
        raise ValueError("ValidatedToolCall.arguments_json must already be canonical JSON.")
    return parsed


def _validate_call_identity(call: ValidatedToolCall) -> None:
    for field_name, value in (
        ("tool_call_id", call.tool_call_id),
        ("name", call.name),
    ):
        if type(value) is not str or not value.strip():
            raise ValueError(f"ValidatedToolCall.{field_name} must be a non-empty string.")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError(f"ValidatedToolCall.{field_name} must be valid UTF-8 text.") from None


def _safe_error_result(
    call: ValidatedToolCall,
) -> AnswerAgentTaskToolExecutionResult:
    return AnswerAgentTaskToolExecutionResult(
        tool_observation=ToolObservation(
            tool_call_id=call.tool_call_id,
            tool_name=call.name,
            status=ToolObservationStatus.ERROR,
            content=_SAFE_EXECUTION_ERROR,
        ),
        is_terminal=False,
    )


def _internal_webshop_action(
    name: str,
    arguments: dict[str, object],
) -> str | None:
    argument_name = _TOOL_ARGUMENT_NAMES.get(name)
    if argument_name is None:
        raise ValueError("ValidatedToolCall name is not a Shopping task tool.")
    if set(arguments) != {argument_name}:
        return None
    value = arguments[argument_name]
    if type(value) is not str or not value.strip():
        return None
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return None
    if any(character in value for character in "[]\r\n\x00"):
        return None
    return f"{name}[{value}]"


class ShoppingToolExecutorAdapter:
    """Execute one validated ``search`` or ``click`` in the active Product."""

    def __init__(self, *, session: object) -> None:
        if type(session) is not ShoppingEnvironmentSession:
            raise ValueError("session must be a ShoppingEnvironmentSession.")
        self._session = session

    @property
    def session(self) -> ShoppingEnvironmentSession:
        return self._session

    def execute(
        self,
        call: object,
    ) -> AnswerAgentTaskToolExecutionResult:
        """Return public page text and keep ``done`` outside model content."""

        if type(call) is not ValidatedToolCall:
            raise ValueError("call must be a ValidatedToolCall.")
        _validate_call_identity(call)
        arguments = _parse_canonical_arguments(call.arguments_json)
        action = _internal_webshop_action(call.name, arguments)
        if action is None:
            return _safe_error_result(call)

        try:
            step_result = self._session.step(action)
        except ShoppingEnvironmentSessionError:
            return _safe_error_result(call)

        observation = ToolObservation(
            tool_call_id=call.tool_call_id,
            tool_name=call.name,
            status=ToolObservationStatus.SUCCESS,
            content=step_result.public_observation.state,
        )
        validation = validate_tool_observation(observation)
        if not validation.is_valid:
            raise RuntimeError("Shopping adapter constructed an invalid ToolObservation.")
        return AnswerAgentTaskToolExecutionResult(
            tool_observation=observation,
            is_terminal=step_result.is_terminal,
        )


__all__ = ["ShoppingToolExecutorAdapter"]
