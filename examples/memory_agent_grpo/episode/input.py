"""Immutable, dataset-neutral inputs for one episode and its subtasks."""

from __future__ import annotations

from dataclasses import dataclass

from examples.memory_agent_grpo.answer_agent.input import AnswerAgentSubtaskInput
from examples.memory_agent_grpo.memory_agent.decision.context import (
    InteractionEvent,
    TaskResult,
)


@dataclass(frozen=True)
class CompletedTaskSeedInput:
    """One public, already-completed task used to initialize episode memory.

    A dataset-provided initial state can be adapted to this neutral form. The
    episode runner will present it to the Memory Agent as a POST_TASK decision;
    this type does not write directly to the Memory Bank.
    """

    task_input: str
    task_result: TaskResult
    interaction_events: tuple[InteractionEvent, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "task_input": self.task_input,
            "task_result": self.task_result.to_dict(),
            "interaction_events": [
                event.to_dict() for event in self.interaction_events
            ],
        }


@dataclass(frozen=True)
class EpisodeSubtaskInput:
    """The allowlisted public projection for exactly one ordered subtask.

    ``answer_input_without_memories`` must not contain retrieved memories.
    The future runner creates a new AnswerAgentSubtaskInput after PRE_TASK
    retrieval, preventing callers from preloading cross-subtask context.
    """

    ordinal: int
    memory_task_input: str
    answer_input_without_memories: AnswerAgentSubtaskInput

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "memory_task_input": self.memory_task_input,
            "answer_input_without_memories": (
                self.answer_input_without_memories.to_dict()
            ),
        }


@dataclass(frozen=True)
class EpisodeInput:
    """Trusted orchestration input; never pass this whole object to a model.

    There is intentionally no initial-bank field.  Every episode starts from
    ``MemoryBankState.empty()``; any public seed must use
    ``initial_completed_task`` and the normal Memory Agent decision path.
    """

    episode_id: str
    initial_completed_task: CompletedTaskSeedInput
    subtasks: tuple[EpisodeSubtaskInput, ...]

    @property
    def subtask_count(self) -> int:
        return len(self.subtasks)

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "initial_completed_task": self.initial_completed_task.to_dict(),
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
        }


__all__ = [
    "CompletedTaskSeedInput",
    "EpisodeInput",
    "EpisodeSubtaskInput",
]
