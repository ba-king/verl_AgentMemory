"""Immutable model-input types for one Memory Agent decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(frozen=True)
class MemoryAgentMessage:
    """One immutable chat message to be consumed by a future model client."""

    role: Literal["system", "user"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


MemoryAgentInput: TypeAlias = tuple[MemoryAgentMessage, MemoryAgentMessage]


__all__ = [
    "MemoryAgentInput",
    "MemoryAgentMessage",
]
