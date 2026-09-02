from __future__ import annotations

from dataclasses import dataclass

from examples.memory_agent_grpo.answer_agent.input import AnswerAgentSubtaskInput
from examples.memory_agent_grpo.answer_agent.response import AnswerAgentResponse
from examples.memory_agent_grpo.answer_agent.runtime import (
    AnswerAgentConversation,
    AnswerAgentSubtaskRunner,
    AnswerAgentSubtaskRunnerConfig,
)
from examples.memory_agent_grpo.answer_agent.runtime.result import (
    AnswerAgentSubtaskRunResult,
)
from examples.memory_agent_grpo.answer_agent.tool_call_gateway import (
    ValidatedToolCall,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.definition import (
    ToolDefinition,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.observation import (
    ToolObservation,
)
from examples.memory_agent_grpo.episode.input import (
    CompletedTaskSeedInput,
    EpisodeInput,
    EpisodeSubtaskInput,
)
from examples.memory_agent_grpo.episode.result import (
    EpisodeRunResult,
    EpisodeRunStatus,
)
from examples.memory_agent_grpo.episode.trace import (
    EpisodeInitializationTrace,
    EpisodeSubtaskTrace,
    EpisodeTrace,
    MemoryDecisionTrace,
    build_subtask_interaction_events,
)
from examples.memory_agent_grpo.memory_agent.action.protocol import (
    MemoryDecisionPhase,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState
from examples.memory_agent_grpo.memory_agent.decision.action_gateway import (
    ActionToExecutorGateway,
)
from examples.memory_agent_grpo.memory_agent.decision.context import (
    MemoryDecisionContext,
    TaskResult,
)
from examples.memory_agent_grpo.memory_agent.executor.engine import (
    MemoryActionExecutor,
)


@dataclass
class _FinalAnswerClient:
    final_answer: str

    def generate(
        self,
        conversation: AnswerAgentConversation,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> AnswerAgentResponse:
        return AnswerAgentResponse(text=self.final_answer)


class _UnusedTaskToolExecutor:
    def execute(self, call: ValidatedToolCall) -> ToolObservation:
        raise AssertionError("No task tool should be called by this fixture.")


def run_answer_agent(
    answer_input: AnswerAgentSubtaskInput,
    *,
    final_answer: str,
) -> AnswerAgentSubtaskRunResult:
    runner = AnswerAgentSubtaskRunner(
        client=_FinalAnswerClient(final_answer),
        executor=_UnusedTaskToolExecutor(),
        config=AnswerAgentSubtaskRunnerConfig(
            max_inference_steps=2,
            max_tool_calls_per_inference=8,
            max_consecutive_tool_failures=1,
        ),
    )
    return runner.run(answer_input)


def execute_memory_decision(
    executor: MemoryActionExecutor,
    *,
    context: MemoryDecisionContext,
    raw_output: str,
) -> MemoryDecisionTrace:
    bank_before = executor.state
    gateway = ActionToExecutorGateway(executor=executor)
    gateway_result = gateway.process_output(raw_output, context=context)
    return MemoryDecisionTrace(
        context=context,
        raw_output=raw_output,
        gateway_result=gateway_result,
        bank_before=bank_before,
        bank_after=executor.state,
    )


def build_valid_episode() -> tuple[EpisodeInput, EpisodeTrace, EpisodeRunResult]:
    seed_result = TaskResult(final_answer="Alice's public completed plan")
    seed = CompletedTaskSeedInput(
        task_input="Plan a museum trip for Alice.",
        task_result=seed_result,
    )
    template = AnswerAgentSubtaskInput(
        task_prompt="Complete only the current travel task.",
        subtask_input="Plan a museum trip for Bob.",
    )
    subtask = EpisodeSubtaskInput(
        ordinal=1,
        memory_task_input="Plan a museum trip for Bob.",
        answer_input_without_memories=template,
    )
    episode_input = EpisodeInput(
        episode_id="episode-001",
        initial_completed_task=seed,
        subtasks=(subtask,),
    )

    executor = MemoryActionExecutor()
    empty_bank = executor.state
    seed_context = MemoryDecisionContext(
        phase=MemoryDecisionPhase.POST_TASK,
        task_input=seed.task_input,
        bank_record_count=0,
        task_result=seed.task_result,
    )
    seed_decision = execute_memory_decision(
        executor,
        context=seed_context,
        raw_output=(
            '{"memory_action":{"action":"ADD","arguments":'
            '{"content":"Alice museum plan: public completed plan"}}}'
        ),
    )
    initialization = EpisodeInitializationTrace(
        seed=seed,
        post_task_decisions=(seed_decision,),
        bank_before=empty_bank,
        bank_after=executor.state,
    )

    subtask_bank_before = executor.state
    pre_context = MemoryDecisionContext(
        phase=MemoryDecisionPhase.PRE_TASK,
        task_input=subtask.memory_task_input,
        bank_record_count=executor.state.record_count,
    )
    pre_decision = execute_memory_decision(
        executor,
        context=pre_context,
        raw_output=(
            '{"memory_action":{"action":"RETRIEVE","arguments":'
            '{"query":"Alice museum plan"}}}'
        ),
    )
    retrieval = pre_decision.gateway_result.execution
    assert retrieval is not None and retrieval.retrieval is not None
    retrieved_memories = tuple(match.content for match in retrieval.retrieval.matches)
    answer_input = AnswerAgentSubtaskInput(
        task_prompt=template.task_prompt,
        subtask_input=template.subtask_input,
        retrieved_memories=retrieved_memories,
    )
    answer_result = run_answer_agent(
        answer_input,
        final_answer="Bob's public completed plan",
    )
    task_result = TaskResult(
        final_answer="Bob's public completed plan",
        reward=1.0,
    )
    interaction_events = build_subtask_interaction_events(answer_result)
    post_context = MemoryDecisionContext(
        phase=MemoryDecisionPhase.POST_TASK,
        task_input=subtask.memory_task_input,
        bank_record_count=executor.state.record_count,
        interaction_events=interaction_events,
        task_result=task_result,
        pre_task_retrieval=retrieval.retrieval,
    )
    post_decision = execute_memory_decision(
        executor,
        context=post_context,
        raw_output='{"memory_action":null}',
    )
    subtask_trace = EpisodeSubtaskTrace(
        subtask=subtask,
        pre_task_decision=pre_decision,
        answer_input=answer_input,
        answer_result=answer_result,
        interaction_events=interaction_events,
        task_result=task_result,
        post_task_decisions=(post_decision,),
        bank_before=subtask_bank_before,
        bank_after=executor.state,
    )
    episode_trace = EpisodeTrace(
        initial_bank_state=MemoryBankState.empty(),
        initialization=initialization,
        subtasks=(subtask_trace,),
        final_bank_state=executor.state,
    )
    result = EpisodeRunResult(
        status=EpisodeRunStatus.COMPLETED,
        episode_input=episode_input,
        trace=episode_trace,
        completed_subtask_count=1,
        final_bank_state=executor.state,
    )
    return episode_input, episode_trace, result


__all__ = ["build_valid_episode", "execute_memory_decision", "run_answer_agent"]
