"""Integrity checks for Answer Agent subtask inputs and model responses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from .input import AnswerAgentSubtaskInput
from .response import AnswerAgentResponse, AnswerAgentToolCall


class AnswerAgentInputErrorCode(str, Enum):
    INPUT_NOT_TYPED = "INPUT_NOT_TYPED"
    TASK_PROMPT_NOT_STRING = "TASK_PROMPT_NOT_STRING"
    EMPTY_TASK_PROMPT = "EMPTY_TASK_PROMPT"
    SUBTASK_INPUT_NOT_STRING = "SUBTASK_INPUT_NOT_STRING"
    EMPTY_SUBTASK_INPUT = "EMPTY_SUBTASK_INPUT"
    RETRIEVED_MEMORIES_NOT_TUPLE = "RETRIEVED_MEMORIES_NOT_TUPLE"
    RETRIEVED_MEMORY_NOT_STRING = "RETRIEVED_MEMORY_NOT_STRING"
    EMPTY_RETRIEVED_MEMORY = "EMPTY_RETRIEVED_MEMORY"


@dataclass(frozen=True)
class AnswerAgentInputValidationResult:
    error_code: AnswerAgentInputErrorCode | None = None
    error_message: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.error_code is None


class AnswerAgentResponseErrorCode(str, Enum):
    RESPONSE_NOT_TYPED = "RESPONSE_NOT_TYPED"
    TEXT_NOT_STRING_OR_NONE = "TEXT_NOT_STRING_OR_NONE"
    EMPTY_TEXT = "EMPTY_TEXT"
    TEXT_NOT_UTF8 = "TEXT_NOT_UTF8"
    TOOL_CALLS_NOT_TUPLE = "TOOL_CALLS_NOT_TUPLE"
    TOOL_CALL_NOT_TYPED = "TOOL_CALL_NOT_TYPED"
    TOOL_CALL_ID_NOT_STRING = "TOOL_CALL_ID_NOT_STRING"
    EMPTY_TOOL_CALL_ID = "EMPTY_TOOL_CALL_ID"
    TOOL_CALL_ID_NOT_UTF8 = "TOOL_CALL_ID_NOT_UTF8"
    DUPLICATE_TOOL_CALL_ID = "DUPLICATE_TOOL_CALL_ID"
    TOOL_NAME_NOT_STRING = "TOOL_NAME_NOT_STRING"
    EMPTY_TOOL_NAME = "EMPTY_TOOL_NAME"
    TOOL_NAME_NOT_UTF8 = "TOOL_NAME_NOT_UTF8"
    ARGUMENTS_JSON_NOT_STRING = "ARGUMENTS_JSON_NOT_STRING"
    EMPTY_ARGUMENTS_JSON = "EMPTY_ARGUMENTS_JSON"
    ARGUMENTS_JSON_NOT_UTF8 = "ARGUMENTS_JSON_NOT_UTF8"
    INVALID_ARGUMENTS_JSON = "INVALID_ARGUMENTS_JSON"
    ARGUMENTS_JSON_NOT_OBJECT = "ARGUMENTS_JSON_NOT_OBJECT"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"


@dataclass(frozen=True)
class AnswerAgentResponseValidationResult:
    error_code: AnswerAgentResponseErrorCode | None = None
    error_message: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.error_code is None


def _invalid_input(
    code: AnswerAgentInputErrorCode,
    message: str,
) -> AnswerAgentInputValidationResult:
    return AnswerAgentInputValidationResult(error_code=code, error_message=message)


def _invalid_response(
    code: AnswerAgentResponseErrorCode,
    message: str,
) -> AnswerAgentResponseValidationResult:
    return AnswerAgentResponseValidationResult(error_code=code, error_message=message)


def validate_answer_agent_subtask_input(
    value: object,
) -> AnswerAgentInputValidationResult:
    """Validate the dataset-neutral input created at one subtask boundary."""

    if not isinstance(value, AnswerAgentSubtaskInput):
        return _invalid_input(
            AnswerAgentInputErrorCode.INPUT_NOT_TYPED,
            "input must be an AnswerAgentSubtaskInput.",
        )
    if not isinstance(value.task_prompt, str):
        return _invalid_input(
            AnswerAgentInputErrorCode.TASK_PROMPT_NOT_STRING,
            "task_prompt must be a string.",
        )
    if not value.task_prompt.strip():
        return _invalid_input(
            AnswerAgentInputErrorCode.EMPTY_TASK_PROMPT,
            "task_prompt must not be empty.",
        )
    if not isinstance(value.subtask_input, str):
        return _invalid_input(
            AnswerAgentInputErrorCode.SUBTASK_INPUT_NOT_STRING,
            "subtask_input must be a string.",
        )
    if not value.subtask_input.strip():
        return _invalid_input(
            AnswerAgentInputErrorCode.EMPTY_SUBTASK_INPUT,
            "subtask_input must not be empty.",
        )
    if not isinstance(value.retrieved_memories, tuple):
        return _invalid_input(
            AnswerAgentInputErrorCode.RETRIEVED_MEMORIES_NOT_TUPLE,
            "retrieved_memories must be a tuple.",
        )
    for position, memory in enumerate(value.retrieved_memories):
        if not isinstance(memory, str):
            return _invalid_input(
                AnswerAgentInputErrorCode.RETRIEVED_MEMORY_NOT_STRING,
                f"retrieved_memories[{position}] must be a string.",
            )
        if not memory.strip():
            return _invalid_input(
                AnswerAgentInputErrorCode.EMPTY_RETRIEVED_MEMORY,
                f"retrieved_memories[{position}] must not be empty.",
            )
    return AnswerAgentInputValidationResult()


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-standard JSON constant {value!r} is not allowed.")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r} is not allowed.")
        result[key] = value
    return result


def _validate_tool_call(
    tool_call: object,
    *,
    position: int,
    seen_ids: set[str],
) -> AnswerAgentResponseValidationResult:
    if not isinstance(tool_call, AnswerAgentToolCall):
        return _invalid_response(
            AnswerAgentResponseErrorCode.TOOL_CALL_NOT_TYPED,
            f"tool_calls[{position}] must be an AnswerAgentToolCall.",
        )
    if not isinstance(tool_call.tool_call_id, str):
        return _invalid_response(
            AnswerAgentResponseErrorCode.TOOL_CALL_ID_NOT_STRING,
            f"tool_calls[{position}].tool_call_id must be a string.",
        )
    if not tool_call.tool_call_id.strip():
        return _invalid_response(
            AnswerAgentResponseErrorCode.EMPTY_TOOL_CALL_ID,
            f"tool_calls[{position}].tool_call_id must not be empty.",
        )
    try:
        tool_call.tool_call_id.encode("utf-8")
    except UnicodeEncodeError:
        return _invalid_response(
            AnswerAgentResponseErrorCode.TOOL_CALL_ID_NOT_UTF8,
            f"tool_calls[{position}].tool_call_id must be valid UTF-8 text.",
        )
    if tool_call.tool_call_id in seen_ids:
        return _invalid_response(
            AnswerAgentResponseErrorCode.DUPLICATE_TOOL_CALL_ID,
            f"tool_call_id {tool_call.tool_call_id!r} appears more than once.",
        )
    seen_ids.add(tool_call.tool_call_id)
    if not isinstance(tool_call.name, str):
        return _invalid_response(
            AnswerAgentResponseErrorCode.TOOL_NAME_NOT_STRING,
            f"tool_calls[{position}].name must be a string.",
        )
    if not tool_call.name.strip():
        return _invalid_response(
            AnswerAgentResponseErrorCode.EMPTY_TOOL_NAME,
            f"tool_calls[{position}].name must not be empty.",
        )
    try:
        tool_call.name.encode("utf-8")
    except UnicodeEncodeError:
        return _invalid_response(
            AnswerAgentResponseErrorCode.TOOL_NAME_NOT_UTF8,
            f"tool_calls[{position}].name must be valid UTF-8 text.",
        )
    if not isinstance(tool_call.arguments_json, str):
        return _invalid_response(
            AnswerAgentResponseErrorCode.ARGUMENTS_JSON_NOT_STRING,
            f"tool_calls[{position}].arguments_json must be a string.",
        )
    if not tool_call.arguments_json.strip():
        return _invalid_response(
            AnswerAgentResponseErrorCode.EMPTY_ARGUMENTS_JSON,
            f"tool_calls[{position}].arguments_json must not be empty.",
        )
    try:
        tool_call.arguments_json.encode("utf-8")
    except UnicodeEncodeError:
        return _invalid_response(
            AnswerAgentResponseErrorCode.ARGUMENTS_JSON_NOT_UTF8,
            f"tool_calls[{position}].arguments_json must be valid UTF-8 text.",
        )
    try:
        arguments = json.loads(
            tool_call.arguments_json,
            parse_float=str,
            parse_int=str,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        return _invalid_response(
            AnswerAgentResponseErrorCode.INVALID_ARGUMENTS_JSON,
            f"tool_calls[{position}].arguments_json must be strict JSON.",
        )
    if not isinstance(arguments, dict):
        return _invalid_response(
            AnswerAgentResponseErrorCode.ARGUMENTS_JSON_NOT_OBJECT,
            f"tool_calls[{position}].arguments_json must encode one JSON object.",
        )
    pending: list[object] = [arguments]
    while pending:
        nested_value = pending.pop()
        if isinstance(nested_value, str):
            try:
                nested_value.encode("utf-8")
            except UnicodeEncodeError:
                return _invalid_response(
                    AnswerAgentResponseErrorCode.ARGUMENTS_JSON_NOT_UTF8,
                    f"tool_calls[{position}].arguments_json contains invalid Unicode text.",
                )
        elif isinstance(nested_value, dict):
            pending.extend(nested_value.keys())
            pending.extend(nested_value.values())
        elif isinstance(nested_value, list):
            pending.extend(nested_value)
    return AnswerAgentResponseValidationResult()


def validate_answer_agent_response(
    value: object,
) -> AnswerAgentResponseValidationResult:
    """Validate a provider-neutral response without interpreting task semantics."""

    if not isinstance(value, AnswerAgentResponse):
        return _invalid_response(
            AnswerAgentResponseErrorCode.RESPONSE_NOT_TYPED,
            "response must be an AnswerAgentResponse.",
        )
    if value.text is not None and not isinstance(value.text, str):
        return _invalid_response(
            AnswerAgentResponseErrorCode.TEXT_NOT_STRING_OR_NONE,
            "text must be a string or None.",
        )
    if isinstance(value.text, str) and not value.text.strip():
        return _invalid_response(
            AnswerAgentResponseErrorCode.EMPTY_TEXT,
            "text must be non-empty when present; use None for no text.",
        )
    if isinstance(value.text, str):
        try:
            value.text.encode("utf-8")
        except UnicodeEncodeError:
            return _invalid_response(
                AnswerAgentResponseErrorCode.TEXT_NOT_UTF8,
                "text must be valid UTF-8 text.",
            )
    if not isinstance(value.tool_calls, tuple):
        return _invalid_response(
            AnswerAgentResponseErrorCode.TOOL_CALLS_NOT_TUPLE,
            "tool_calls must be a tuple.",
        )
    seen_ids: set[str] = set()
    for position, tool_call in enumerate(value.tool_calls):
        validation = _validate_tool_call(tool_call, position=position, seen_ids=seen_ids)
        if not validation.is_valid:
            return validation
    if not value.tool_calls and value.text is None:
        return _invalid_response(
            AnswerAgentResponseErrorCode.EMPTY_RESPONSE,
            "response must contain non-empty text or at least one tool call.",
        )
    return AnswerAgentResponseValidationResult()


__all__ = [
    "AnswerAgentInputErrorCode",
    "AnswerAgentInputValidationResult",
    "AnswerAgentResponseErrorCode",
    "AnswerAgentResponseValidationResult",
    "validate_answer_agent_response",
    "validate_answer_agent_subtask_input",
]
