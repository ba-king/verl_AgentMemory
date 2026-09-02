"""Typed, immutable retrieval results returned by a Memory Bank retriever."""

from __future__ import annotations

from dataclasses import dataclass

from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryRecord


@dataclass(frozen=True)
class RetrievalMatch:
    record: MemoryRecord
    score: float

    @property
    def memory_id(self) -> str:
        return self.record.memory_id

    @property
    def content(self) -> str:
        return self.record.content

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "score": self.score,
        }


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    matches: tuple[RetrievalMatch, ...]

    @property
    def match_count(self) -> int:
        return len(self.matches)

    @property
    def memory_ids(self) -> tuple[str, ...]:
        return tuple(match.memory_id for match in self.matches)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "matches": [match.to_dict() for match in self.matches],
        }


__all__ = ["RetrievalMatch", "RetrievalResult"]
