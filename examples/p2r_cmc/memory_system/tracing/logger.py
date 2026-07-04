from __future__ import annotations

from ..schemas import ActionTrace, ExposureEvent, RewardTrace
from ..store import SQLiteMemoryStore

__all__ = ["TraceLogger"]


class TraceLogger:
    """Thin trace facade used by prompt pipelines and eval adapters."""

    def __init__(self, store: SQLiteMemoryStore) -> None:
        self.store = store

    def log_action(self, trace: ActionTrace) -> None:
        self.store.append_action_trace(trace)

    def log_exposure(self, event: ExposureEvent) -> None:
        self.store.append_exposure_event(event)

    def log_reward(self, trace: RewardTrace) -> None:
        self.store.append_reward_trace(trace)
