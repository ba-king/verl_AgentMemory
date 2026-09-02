"""Explicit interaction and completion semantics for one Answer Agent run."""

from __future__ import annotations

from enum import Enum


class AnswerAgentInteractionMode(str, Enum):
    """Who decides completion and how many calls one inference may emit."""

    TOOL_OR_FINAL = "TOOL_OR_FINAL"
    ENVIRONMENT_TERMINATED_SINGLE_TOOL = "ENVIRONMENT_TERMINATED_SINGLE_TOOL"


__all__ = ["AnswerAgentInteractionMode"]
