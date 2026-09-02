"""Immutable initial-input types for one Answer Agent subtask."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(frozen=True)
class AnswerAgentSubtaskInput:
    """Public evidence supplied once when one Answer Agent subtask starts."""

    task_prompt: str
    subtask_input: str
    retrieved_memories: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "task_prompt": self.task_prompt,
            "subtask_input": self.subtask_input,
            "retrieved_memories": list(self.retrieved_memories),
        }


@dataclass(frozen=True)
class AnswerAgentInitialMessage:
    """One immutable system or user message at subtask initialization."""

    role: Literal["system", "user"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


AnswerAgentInitialMessages: TypeAlias = tuple[
    AnswerAgentInitialMessage,
    AnswerAgentInitialMessage,
]


__all__ = [
    "AnswerAgentInitialMessage",
    "AnswerAgentInitialMessages",
    "AnswerAgentSubtaskInput",
]
