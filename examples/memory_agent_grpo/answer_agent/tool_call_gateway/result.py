"""Stable success and failure result returned by the task ToolCall Gateway."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .validated_call import ValidatedToolCall


class ToolCallGatewayErrorCode(str, Enum):
    TOOL_DEFINITIONS_NOT_TUPLE = "TOOL_DEFINITIONS_NOT_TUPLE"
    INVALID_TOOL_DEFINITION = "INVALID_TOOL_DEFINITION"
    DUPLICATE_TOOL_NAME = "DUPLICATE_TOOL_NAME"
    INVALID_TOOL_CALL = "INVALID_TOOL_CALL"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
    ARGUMENTS_DO_NOT_MATCH_SCHEMA = "ARGUMENTS_DO_NOT_MATCH_SCHEMA"
    CANONICALIZATION_FAILED = "CANONICALIZATION_FAILED"


@dataclass(frozen=True)
class ToolCallGatewayResult:
    """Exactly one validated call or one non-executable failure."""

    validated_call: ValidatedToolCall | None = None
    error_code: ToolCallGatewayErrorCode | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        success = (
            isinstance(self.validated_call, ValidatedToolCall)
            and self.error_code is None
            and self.error_message is None
        )
        failure = (
            self.validated_call is None
            and isinstance(self.error_code, ToolCallGatewayErrorCode)
            and isinstance(self.error_message, str)
            and bool(self.error_message.strip())
        )
        if not (success or failure):
            raise ValueError(
                "ToolCallGatewayResult must contain exactly one validated call or one non-empty typed error."
            )

    @property
    def is_valid(self) -> bool:
        return self.validated_call is not None and self.error_code is None and self.error_message is None

    def to_dict(self) -> dict[str, object | None]:
        return {
            "validated_call": (None if self.validated_call is None else self.validated_call.to_dict()),
            "error_code": None if self.error_code is None else self.error_code.value,
            "error_message": self.error_message,
        }


__all__ = ["ToolCallGatewayErrorCode", "ToolCallGatewayResult"]
