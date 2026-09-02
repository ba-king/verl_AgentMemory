"""Validation boundary between Answer Agent tool calls and task adapters."""

from .gateway import validate_tool_call_for_execution
from .result import ToolCallGatewayErrorCode, ToolCallGatewayResult
from .validated_call import ValidatedToolCall

__all__ = [
    "ToolCallGatewayErrorCode",
    "ToolCallGatewayResult",
    "ValidatedToolCall",
    "validate_tool_call_for_execution",
]
