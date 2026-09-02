"""Render one bounded correction attempt for a Memory Agent decision."""

from __future__ import annotations

from ..action.validation import ValidationErrorCode
from ..decision.context import MemoryDecisionContext
from ..decision.model_input import MemoryAgentInput, MemoryAgentMessage
from .decision import build_memory_agent_input


def build_memory_agent_correction_input(
    context: MemoryDecisionContext,
    *,
    validation_error_code: ValidationErrorCode,
) -> MemoryAgentInput:
    """Re-render the same decision with one trusted, content-free correction."""

    if type(validation_error_code) is not ValidationErrorCode:
        raise TypeError("validation_error_code must be a ValidationErrorCode.")
    system_message, user_message = build_memory_agent_input(context)
    correction_notice = (
        "\n\n<correction_notice>\n"
        "The immediately preceding completion for this same decision was "
        "rejected by the deterministic validator with error code "
        f"{validation_error_code.value}. Its text is intentionally not "
        "included. This is the only correction attempt. Re-read the allowed "
        "outputs and decision context above, then return exactly one allowed "
        "JSON object and no other text.\n"
        "</correction_notice>"
    )
    return (
        system_message,
        MemoryAgentMessage(
            role="user",
            content=user_message.content + correction_notice,
        ),
    )


__all__ = ["build_memory_agent_correction_input"]
