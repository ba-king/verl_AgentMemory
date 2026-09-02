"""Side-effect-free state structures for one Memory Bank snapshot."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryRecord:
    """One persistent memory owned by the deterministic Memory Bank."""

    memory_id: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"memory_id": self.memory_id, "content": self.content}


@dataclass(frozen=True)
class MemoryBankState:
    """An immutable, ordered snapshot of all currently stored memories.

    ``next_id`` is the positive integer ordinal that a future store will use
    for the next ADD. It is retained even when records are deleted, which
    prevents an old memory ID from being reused.
    """

    records: tuple[MemoryRecord, ...]
    next_id: int

    @classmethod
    def empty(cls) -> MemoryBankState:
        return cls(records=(), next_id=1)

    @property
    def record_count(self) -> int:
        return len(self.records)

    def to_dict(self) -> dict[str, object]:
        return {
            "records": [record.to_dict() for record in self.records],
            "next_id": self.next_id,
        }


__all__ = ["MemoryBankState", "MemoryRecord"]
