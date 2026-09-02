"""Immutable, dataset-neutral context for one Memory Agent decision."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from examples.memory_agent_grpo.memory_agent.action.protocol import MemoryDecisionPhase
from examples.memory_agent_grpo.memory_agent.retrieval.result import RetrievalResult


class InteractionEventType(str, Enum):
    """Externally visible events produced while the Task Agent works."""

    TASK_AGENT_ACTION = "TASK_AGENT_ACTION"
    ENVIRONMENT_OBSERVATION = "ENVIRONMENT_OBSERVATION"


class TaskResultStatus(str, Enum):
    """Whether the Task Agent returned normally or failed to produce an answer."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class InteractionEvent:
    """One ordered, externally visible Task Agent or environment event."""

    event_type: InteractionEventType
    content: str

    def to_dict(self) -> dict[str, str]:
        return {
            "event_type": self.event_type.value,
            "content": self.content,
        }


@dataclass(frozen=True)
class TaskResult:
    """The Task Agent output and optional environment-provided evaluation."""

    final_answer: str
    status: TaskResultStatus = TaskResultStatus.COMPLETED
    reward: float | None = None
    feedback: str | None = None

    def to_dict(self) -> dict[str, object | None]:
        return {
            "status": self.status.value,
            "final_answer": self.final_answer,
            "reward": self.reward,
            "feedback": self.feedback,
        }


@dataclass(frozen=True)
class MemoryDecisionContext:
    """The complete visible evidence for one Memory Agent inference.

    This is a transient decision input, not a short-term memory store. The
    trusted runner assembles it from adapter-normalized public task data and
    runner-owned phase, Bank, and retrieval state. The Memory Agent does not
    author any field in it.
    """

    phase: MemoryDecisionPhase
    task_input: str
    bank_record_count: int
    interaction_events: tuple[InteractionEvent, ...] = ()
    task_result: TaskResult | None = None
    pre_task_retrieval: RetrievalResult | None = None
    post_task_retrieval: RetrievalResult | None = None

    @property
    def visible_memory_ids(self) -> tuple[str, ...]:
        """IDs actually presented in this context, in first-visible order."""

        ordered_ids: list[str] = []
        seen_ids: set[str] = set()
        for retrieval in (self.pre_task_retrieval, self.post_task_retrieval):
            if retrieval is None:
                continue
            for memory_id in retrieval.memory_ids:
                if memory_id not in seen_ids:
                    seen_ids.add(memory_id)
                    ordered_ids.append(memory_id)
        return tuple(ordered_ids)

    def to_dict(self) -> dict[str, object | None]:
        return {
            "phase": self.phase.value,
            "task_input": self.task_input,
            "bank_record_count": self.bank_record_count,
            "interaction_events": [event.to_dict() for event in self.interaction_events],
            "task_result": None if self.task_result is None else self.task_result.to_dict(),
            "pre_task_retrieval": (
                None if self.pre_task_retrieval is None else self.pre_task_retrieval.to_dict()
            ),
            "post_task_retrieval": (
                None if self.post_task_retrieval is None else self.post_task_retrieval.to_dict()
            ),
        }


__all__ = [
    "InteractionEvent",
    "InteractionEventType",
    "MemoryDecisionContext",
    "TaskResult",
    "TaskResultStatus",
]
