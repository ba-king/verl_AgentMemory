"""Typed, side-effect-free Memory Agent action protocol.

This module defines what a valid decision looks like. It does not parse or
validate untrusted model output; that boundary belongs in ``action.validation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


class MemoryActionName(str, Enum):
    """The complete set of operations accepted by the future Executor."""

    RETRIEVE = "RETRIEVE"
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class MemoryDecisionPhase(str, Enum):
    """The point at which the Memory Agent is asked to make a decision."""

    PRE_TASK = "PRE_TASK"
    POST_TASK = "POST_TASK"
    POST_TASK_AFTER_RETRIEVAL = "POST_TASK_AFTER_RETRIEVAL"


@dataclass(frozen=True)
class RetrieveArguments:
    query: str

    def to_dict(self) -> dict[str, str]:
        return {"query": self.query}


@dataclass(frozen=True)
class AddArguments:
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"content": self.content}


@dataclass(frozen=True)
class UpdateArguments:
    memory_id: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"memory_id": self.memory_id, "content": self.content}


@dataclass(frozen=True)
class DeleteArguments:
    memory_id: str

    def to_dict(self) -> dict[str, str]:
        return {"memory_id": self.memory_id}


MemoryActionArguments: TypeAlias = RetrieveArguments | AddArguments | UpdateArguments | DeleteArguments


@dataclass(frozen=True)
class MemoryAction:
    action: MemoryActionName
    arguments: MemoryActionArguments

    def to_dict(self) -> dict[str, object]:
        return {"action": self.action.value, "arguments": self.arguments.to_dict()}


@dataclass(frozen=True)
class MemoryDecision:
    """A valid decision; ``memory_action=None`` means no tool call."""

    memory_action: MemoryAction | None

    def to_dict(self) -> dict[str, object | None]:
        action = self.memory_action
        return {"memory_action": None if action is None else action.to_dict()}


__all__ = [
    "AddArguments",
    "DeleteArguments",
    "MemoryAction",
    "MemoryActionArguments",
    "MemoryActionName",
    "MemoryDecision",
    "MemoryDecisionPhase",
    "RetrieveArguments",
    "UpdateArguments",
]
