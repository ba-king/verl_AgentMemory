from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .remember_pipeline import RememberPipeline
from ..schemas import RawEpisode, new_id, utc_now


@dataclass
class SessionTurn:
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    turn_id: str


@dataclass
class SessionBuffer:
    user_id: str
    project_id: str | None = None
    session_id: str = field(default_factory=lambda: new_id("session"))
    turns: list[SessionTurn] = field(default_factory=list)

    def add_turn(self, role: Literal["user", "assistant", "system", "tool"], content: str) -> None:
        self.turns.append(SessionTurn(role=role, content=content, turn_id=str(len(self.turns))))

    def transcript(self) -> str:
        return "\n".join(f"{turn.role.upper()}[{turn.turn_id}]: {turn.content}" for turn in self.turns)

    def user_episodes(self) -> list[RawEpisode]:
        created_at = utc_now()
        return [
            RawEpisode(
                user_id=self.user_id,
                project_id=self.project_id,
                session_id=self.session_id,
                turn_id=turn.turn_id,
                role="user",
                content=turn.content,
                created_at=created_at,
                metadata={"extraction_mode": "session_end", "session_transcript": self.transcript()},
            )
            for turn in self.turns
            if turn.role == "user"
        ]


class SessionMemoryPipeline:
    """Session-end memory extraction.

    Turns are buffered during the conversation. Durable memory extraction only
    happens when `consolidate_session()` is called.
    """

    def __init__(self, remember_pipeline: RememberPipeline) -> None:
        self.remember_pipeline = remember_pipeline

    def consolidate_session(self, session: SessionBuffer) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for episode in session.user_episodes():
            for result in self.remember_pipeline.ingest_episode(episode):
                result["session_id"] = session.session_id
                result["turn_id"] = episode.turn_id
                results.append(result)
        return results
