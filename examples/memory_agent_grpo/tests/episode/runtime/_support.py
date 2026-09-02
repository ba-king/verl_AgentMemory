"""Scripted, network-free fixtures for EpisodeRunner tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from examples.memory_agent_grpo.answer_agent.input import (
    AnswerAgentSubtaskInput,
)
from examples.memory_agent_grpo.answer_agent.runtime.result import (
    AnswerAgentSubtaskRunResult,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.definition import (
    ToolDefinition,
)
from examples.memory_agent_grpo.episode import (
    CompletedTaskSeedInput,
    EpisodeInput,
    EpisodeSubtaskInput,
)
from examples.memory_agent_grpo.memory_agent.action.protocol import (
    MemoryDecisionPhase,
)
from examples.memory_agent_grpo.memory_agent.decision.context import TaskResult
from examples.memory_agent_grpo.memory_agent.decision.model_input import (
    MemoryAgentInput,
)

from .._support import run_answer_agent


NULL_ACTION = '{"memory_action":null}'


def add_action(content: str) -> str:
    return json.dumps(
        {
            "memory_action": {
                "action": "ADD",
                "arguments": {"content": content},
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def retrieve_action(query: str) -> str:
    return json.dumps(
        {
            "memory_action": {
                "action": "RETRIEVE",
                "arguments": {"query": query},
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def phase_from_model_input(model_input: MemoryAgentInput) -> MemoryDecisionPhase:
    user_content = model_input[1].content
    prefix = "Current decision phase: "
    first_line = user_content.splitlines()[0]
    if first_line.startswith(prefix):
        phase_value = first_line.removeprefix(prefix)
        for phase in MemoryDecisionPhase:
            if phase_value == phase.value:
                return phase
    raise AssertionError("Memory Agent input did not expose one known phase.")


@dataclass
class ScriptedMemoryClient:
    outcomes: list[object]
    event_log: list[str] | None = None
    calls: list[MemoryAgentInput] = field(default_factory=list)

    def generate(self, model_input: MemoryAgentInput) -> str:
        self.calls.append(model_input)
        phase = phase_from_model_input(model_input)
        if self.event_log is not None:
            self.event_log.append(f"memory:{phase.value}")
        position = len(self.calls) - 1
        if position >= len(self.outcomes):
            raise AssertionError("Memory Agent was called more often than scripted.")
        outcome = self.outcomes[position]
        if isinstance(outcome, BaseException):
            raise outcome
        if type(outcome) is not str:
            raise AssertionError(
                "Scripted Memory Agent outcomes must be text or errors."
            )
        return outcome

    @property
    def phases(self) -> tuple[MemoryDecisionPhase, ...]:
        return tuple(phase_from_model_input(call) for call in self.calls)

    @property
    def serialized_inputs(self) -> tuple[str, ...]:
        return tuple(
            "\n".join(message.content for message in model_input)
            for model_input in self.calls
        )


@dataclass
class ScriptedAnswerRunner:
    final_answers: list[str]
    fail_on_call: int | None = None
    event_log: list[str] | None = None
    calls: list[AnswerAgentSubtaskInput] = field(default_factory=list)
    definitions_seen: list[tuple[ToolDefinition, ...]] = field(default_factory=list)
    results: list[AnswerAgentSubtaskRunResult] = field(default_factory=list)

    def run(
        self,
        subtask: AnswerAgentSubtaskInput,
        *,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> AnswerAgentSubtaskRunResult:
        self.calls.append(subtask)
        self.definitions_seen.append(tool_definitions)
        call_number = len(self.calls)
        if self.event_log is not None:
            self.event_log.append(f"answer:{call_number}")
        if self.fail_on_call == call_number:
            raise RuntimeError("PRIVATE_ANSWER_RUNNER_FAILURE")
        if call_number > len(self.final_answers):
            raise AssertionError("Answer Agent was called more often than scripted.")
        result = run_answer_agent(
            subtask,
            final_answer=self.final_answers[call_number - 1],
        )
        self.results.append(result)
        return result


def episode_input(
    *,
    episode_id: str = "episode-runtime-test",
    subtask_count: int = 2,
    memory_task_inputs: tuple[str, ...] | None = None,
    answer_subtask_inputs: tuple[str, ...] | None = None,
) -> EpisodeInput:
    if memory_task_inputs is None:
        memory_task_inputs = tuple(
            f"MEMORY_CURRENT_{ordinal}" for ordinal in range(1, subtask_count + 1)
        )
    if answer_subtask_inputs is None:
        answer_subtask_inputs = tuple(
            f"ANSWER_CURRENT_{ordinal}" for ordinal in range(1, subtask_count + 1)
        )
    assert len(memory_task_inputs) == subtask_count
    assert len(answer_subtask_inputs) == subtask_count

    return EpisodeInput(
        episode_id=episode_id,
        initial_completed_task=CompletedTaskSeedInput(
            task_input="PUBLIC_BASE_TASK",
            task_result=TaskResult(final_answer="PUBLIC_BASE_ANSWER"),
        ),
        subtasks=tuple(
            EpisodeSubtaskInput(
                ordinal=ordinal,
                memory_task_input=memory_task_inputs[ordinal - 1],
                answer_input_without_memories=AnswerAgentSubtaskInput(
                    task_prompt="FIXED_TASK_PROMPT",
                    subtask_input=answer_subtask_inputs[ordinal - 1],
                ),
            )
            for ordinal in range(1, subtask_count + 1)
        ),
    )


__all__ = [
    "NULL_ACTION",
    "ScriptedAnswerRunner",
    "ScriptedMemoryClient",
    "add_action",
    "episode_input",
    "phase_from_model_input",
    "retrieve_action",
]
