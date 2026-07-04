from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import EvidencePacket, RawEpisode


@dataclass
class MemoryArenaTask:
    task_id: str
    sessions: list[list[RawEpisode]]
    final_query: str
    user_id: str
    project_id: str | None
    gold_keywords: set[str] = field(default_factory=set)


@dataclass
class MemoryArenaResult:
    task_id: str
    baseline: str
    packet: EvidencePacket
    metrics: dict[str, object]


class MemoryArenaAdapter:
    """Minimal interface shaped like a future real MemoryArena adapter."""

    def load_tasks(self) -> list[MemoryArenaTask]:
        raise NotImplementedError

    def run(self, baseline: str) -> list[MemoryArenaResult]:
        raise NotImplementedError
