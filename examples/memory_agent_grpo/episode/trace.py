"""Immutable audit traces for episode-level memory and task execution."""

from __future__ import annotations

import json
from dataclasses import dataclass

from examples.memory_agent_grpo.answer_agent.input import AnswerAgentSubtaskInput
from examples.memory_agent_grpo.answer_agent.runtime.result import (
    AnswerAgentSubtaskRunResult,
)
from examples.memory_agent_grpo.memory_agent.action.repair import MemoryOutputRepair
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState
from examples.memory_agent_grpo.memory_agent.decision.action_gateway import (
    ActionToExecutorResult,
)
from examples.memory_agent_grpo.memory_agent.decision.context import (
    InteractionEvent,
    InteractionEventType,
    MemoryDecisionContext,
    TaskResult,
)

from .input import CompletedTaskSeedInput, EpisodeSubtaskInput


def build_subtask_interaction_events(
    answer_result: AnswerAgentSubtaskRunResult,
) -> tuple[InteractionEvent, ...]:
    """Project public tool calls and observations from one Answer Agent run.

    Assistant prose, provider payloads, and hidden reasoning are deliberately
    excluded. Tool arguments were already model-visible, and ToolObservation
    is the adapter-filtered public environment boundary.
    """

    if type(answer_result) is not AnswerAgentSubtaskRunResult:
        raise ValueError("answer_result must be an AnswerAgentSubtaskRunResult.")
    events: list[InteractionEvent] = []
    for step in answer_result.steps:
        if not step.tool_outcomes:
            continue
        action_content = json.dumps(
            {
                "tool_calls": [
                    outcome.tool_call.to_dict() for outcome in step.tool_outcomes
                ]
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        observation_content = json.dumps(
            {
                "tool_observations": [
                    outcome.tool_observation.to_dict() for outcome in step.tool_outcomes
                ]
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        events.extend(
            (
                InteractionEvent(
                    event_type=InteractionEventType.TASK_AGENT_ACTION,
                    content=action_content,
                ),
                InteractionEvent(
                    event_type=InteractionEventType.ENVIRONMENT_OBSERVATION,
                    content=observation_content,
                ),
            )
        )
    return tuple(events)


@dataclass(frozen=True)
class MemoryDecisionTrace:
    """One complete raw-output-to-Bank boundary for the Memory Agent."""

    context: MemoryDecisionContext
    raw_output: str
    gateway_result: ActionToExecutorResult
    bank_before: MemoryBankState
    bank_after: MemoryBankState
    output_repair: MemoryOutputRepair | None = None

    @property
    def effective_output(self) -> str:
        return self.raw_output if self.output_repair is None else self.output_repair.repaired_output

    def to_dict(self) -> dict[str, object]:
        return {
            "context": self.context.to_dict(),
            "raw_output": self.raw_output,
            "output_repair": (
                None
                if self.output_repair is None
                else {
                    "rule": self.output_repair.rule,
                    "repaired_output": self.output_repair.repaired_output,
                    "original_validation": {
                        "status": self.output_repair.original_validation.status.value,
                        "error_code": self.output_repair.original_validation.error_code.value,
                        "error_message": self.output_repair.original_validation.error_message,
                    },
                }
            ),
            "gateway_result": self.gateway_result.to_dict(),
            "bank_before": self.bank_before.to_dict(),
            "bank_after": self.bank_after.to_dict(),
        }


@dataclass(frozen=True)
class EpisodeInitializationTrace:
    """Memory decisions made from the completed seed before subtask one."""

    seed: CompletedTaskSeedInput
    post_task_decisions: tuple[MemoryDecisionTrace, ...]
    bank_before: MemoryBankState
    bank_after: MemoryBankState

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed.to_dict(),
            "post_task_decisions": [
                decision.to_dict() for decision in self.post_task_decisions
            ],
            "bank_before": self.bank_before.to_dict(),
            "bank_after": self.bank_after.to_dict(),
        }


@dataclass(frozen=True)
class EpisodeSubtaskTrace:
    """One completed subtask with only its own Answer Agent conversation."""

    subtask: EpisodeSubtaskInput
    pre_task_decision: MemoryDecisionTrace
    answer_input: AnswerAgentSubtaskInput
    answer_result: AnswerAgentSubtaskRunResult
    interaction_events: tuple[InteractionEvent, ...]
    task_result: TaskResult
    post_task_decisions: tuple[MemoryDecisionTrace, ...]
    bank_before: MemoryBankState
    bank_after: MemoryBankState

    def to_dict(self) -> dict[str, object]:
        return {
            "subtask": self.subtask.to_dict(),
            "pre_task_decision": self.pre_task_decision.to_dict(),
            "answer_input": self.answer_input.to_dict(),
            "answer_result": self.answer_result.to_dict(),
            "interaction_events": [
                event.to_dict() for event in self.interaction_events
            ],
            "task_result": self.task_result.to_dict(),
            "post_task_decisions": [
                decision.to_dict() for decision in self.post_task_decisions
            ],
            "bank_before": self.bank_before.to_dict(),
            "bank_after": self.bank_after.to_dict(),
        }


@dataclass(frozen=True)
class EpisodeTrace:
    """The initialization and ordered completed-subtask prefix of an episode."""

    initial_bank_state: MemoryBankState
    initialization: EpisodeInitializationTrace
    subtasks: tuple[EpisodeSubtaskTrace, ...]
    final_bank_state: MemoryBankState

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_bank_state": self.initial_bank_state.to_dict(),
            "initialization": self.initialization.to_dict(),
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
            "final_bank_state": self.final_bank_state.to_dict(),
        }


__all__ = [
    "EpisodeInitializationTrace",
    "EpisodeSubtaskTrace",
    "EpisodeTrace",
    "MemoryDecisionTrace",
    "build_subtask_interaction_events",
]
