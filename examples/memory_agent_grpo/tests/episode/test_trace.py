import json
from dataclasses import replace

from examples.memory_agent_grpo.answer_agent.input import AnswerAgentSubtaskInput
from examples.memory_agent_grpo.answer_agent.response import (
    AnswerAgentResponse,
    AnswerAgentToolCall,
)
from examples.memory_agent_grpo.answer_agent.runtime import (
    AnswerAgentConversation,
    AnswerAgentSubtaskRunner,
    AnswerAgentSubtaskRunnerConfig,
)
from examples.memory_agent_grpo.answer_agent.tool_call_gateway import (
    ValidatedToolCall,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.definition import (
    ToolDefinition,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.observation import (
    ToolObservation,
    ToolObservationStatus,
)
from examples.memory_agent_grpo.episode import build_subtask_interaction_events
from examples.memory_agent_grpo.episode.validation import (
    EpisodeValidationErrorCode,
    validate_episode_subtask_trace,
    validate_episode_trace,
    validate_memory_decision_trace,
)
from examples.memory_agent_grpo.memory_agent.decision.context import (
    InteractionEvent,
    InteractionEventType,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import (
    MemoryBankState,
    MemoryRecord,
)

from ._support import build_valid_episode, run_answer_agent


class _LyingText(str):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


class _LyingBankState(MemoryBankState):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


def test_complete_episode_trace_is_valid() -> None:
    episode_input, trace, _ = build_valid_episode()

    assert validate_episode_trace(trace, episode_input=episode_input).is_valid


def test_memory_decision_trace_binds_raw_output_to_gateway_result() -> None:
    _, trace, _ = build_valid_episode()
    decision = trace.initialization.post_task_decisions[0]
    forged = replace(decision, raw_output='{"memory_action":null}')

    result = validate_memory_decision_trace(forged)

    assert result.error_code is EpisodeValidationErrorCode.RAW_OUTPUT_MISMATCH


def test_memory_decision_trace_rejects_forged_bank_after() -> None:
    _, trace, _ = build_valid_episode()
    decision = trace.initialization.post_task_decisions[0]
    forged = replace(decision, bank_after=MemoryBankState.empty())

    result = validate_memory_decision_trace(forged)

    assert result.error_code is EpisodeValidationErrorCode.BANK_TRANSITION_MISMATCH


def test_answer_input_must_equal_exact_pre_task_retrieval() -> None:
    _, trace, _ = build_valid_episode()
    subtask = trace.subtasks[0]
    forged = replace(
        subtask,
        answer_input=AnswerAgentSubtaskInput(
            task_prompt=subtask.answer_input.task_prompt,
            subtask_input=subtask.answer_input.subtask_input,
            retrieved_memories=("invented memory",),
        ),
    )

    result = validate_episode_subtask_trace(forged)

    assert result.error_code is EpisodeValidationErrorCode.RETRIEVAL_EXPOSURE_MISMATCH


def test_episode_rejects_bank_reset_between_subtasks() -> None:
    episode_input, trace, _ = build_valid_episode()
    subtask = replace(
        trace.subtasks[0],
        bank_before=MemoryBankState.empty(),
    )
    forged = replace(trace, subtasks=(subtask,))

    result = validate_episode_trace(forged, episode_input=episode_input)

    assert result.error_code in {
        EpisodeValidationErrorCode.CONTEXT_MISMATCH,
        EpisodeValidationErrorCode.BANK_CONTINUITY_MISMATCH,
    }


def test_current_answer_conversation_does_not_receive_episode_object() -> None:
    episode_input, trace, _ = build_valid_episode()
    future = replace(
        episode_input.subtasks[0],
        ordinal=2,
        memory_task_input="FUTURE_QUERY_SENTINEL",
        answer_input_without_memories=AnswerAgentSubtaskInput(
            task_prompt="Use task tools.",
            subtask_input="FUTURE_QUERY_SENTINEL",
        ),
    )
    episode_with_future = replace(
        episode_input,
        subtasks=(episode_input.subtasks[0], future),
    )

    serialized_current_conversation = str(
        trace.subtasks[0].answer_result.conversation.to_dict()
    )

    assert "FUTURE_QUERY_SENTINEL" in str(episode_with_future.to_dict())
    assert "FUTURE_QUERY_SENTINEL" not in serialized_current_conversation


def test_post_task_events_cannot_inject_unrelated_episode_content() -> None:
    _, trace, _ = build_valid_episode()
    subtask = trace.subtasks[0]
    forged_events = (
        InteractionEvent(
            event_type=InteractionEventType.ENVIRONMENT_OBSERVATION,
            content="FUTURE_SUBTASK_SECRET",
        ),
    )
    forged_post = replace(
        subtask.post_task_decisions[0],
        context=replace(
            subtask.post_task_decisions[0].context,
            interaction_events=forged_events,
        ),
    )
    forged = replace(
        subtask,
        interaction_events=forged_events,
        post_task_decisions=(forged_post,),
    )

    result = validate_episode_subtask_trace(forged)

    assert result.error_code is EpisodeValidationErrorCode.ANSWER_TRACE_MISMATCH


def test_batch_projection_emits_two_events_with_ordered_arrays() -> None:
    first_call = AnswerAgentToolCall(
        tool_call_id="call-1",
        name="Lookup",
        arguments_json='{ "query" : "first" }',
    )
    second_call = AnswerAgentToolCall(
        tool_call_id="call-2",
        name="Lookup",
        arguments_json='{"query":"second"}',
    )

    class _BatchThenFinalClient:
        def __init__(self) -> None:
            self.responses = [
                AnswerAgentResponse(tool_calls=(first_call, second_call)),
                AnswerAgentResponse(text="final answer"),
            ]

        def generate(
            self,
            conversation: AnswerAgentConversation,
            tool_definitions: tuple[ToolDefinition, ...] = (),
        ) -> AnswerAgentResponse:
            del conversation, tool_definitions
            return self.responses.pop(0)

    class _OrderedExecutor:
        def __init__(self) -> None:
            self.call_ids: list[str] = []

        def execute(self, call: ValidatedToolCall) -> ToolObservation:
            self.call_ids.append(call.tool_call_id)
            return ToolObservation(
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                status=ToolObservationStatus.SUCCESS,
                content=f"public result for {call.tool_call_id}",
            )

    executor = _OrderedExecutor()
    runner = AnswerAgentSubtaskRunner(
        client=_BatchThenFinalClient(),
        executor=executor,
        config=AnswerAgentSubtaskRunnerConfig(
            max_inference_steps=2,
            max_tool_calls_per_inference=2,
            max_consecutive_tool_failures=1,
        ),
    )
    answer_result = runner.run(
        AnswerAgentSubtaskInput(
            task_prompt="Use the public lookup tool.",
            subtask_input="Look up two public items.",
        ),
        tool_definitions=(
            ToolDefinition(
                name="Lookup",
                description="Look up one public item.",
                parameters_json_schema=(
                    '{"type":"object","properties":{"query":{"type":"string"}},'
                    '"required":["query"],"additionalProperties":false}'
                ),
            ),
        ),
    )

    events = build_subtask_interaction_events(answer_result)

    assert executor.call_ids == ["call-1", "call-2"]
    assert len(events) == 2
    assert [event.event_type for event in events] == [
        InteractionEventType.TASK_AGENT_ACTION,
        InteractionEventType.ENVIRONMENT_OBSERVATION,
    ]
    action_batch = json.loads(events[0].content)
    observation_batch = json.loads(events[1].content)
    assert action_batch == {"tool_calls": [first_call.to_dict(), second_call.to_dict()]}
    assert [
        observation["tool_call_id"]
        for observation in observation_batch["tool_observations"]
    ] == ["call-1", "call-2"]
    assert [
        observation["content"] for observation in observation_batch["tool_observations"]
    ] == ["public result for call-1", "public result for call-2"]


def test_malformed_post_task_sequence_returns_validation_error() -> None:
    _, trace, _ = build_valid_episode()
    forged = replace(
        trace.subtasks[0],
        post_task_decisions=[object()],  # type: ignore[arg-type]
    )

    result = validate_episode_subtask_trace(forged)

    assert result.error_code is EpisodeValidationErrorCode.DECISION_SEQUENCE_MISMATCH


def test_bank_subclass_cannot_fake_empty_or_unchanged_state() -> None:
    _, trace, _ = build_valid_episode()
    decision = trace.subtasks[0].post_task_decisions[0]
    lying_state = _LyingBankState(
        records=(MemoryRecord("M999", "CROSS_EPISODE_SECRET"),),
        next_id=1000,
    )
    forged = replace(decision, bank_after=lying_state)

    result = validate_memory_decision_trace(forged)

    assert result.error_code is EpisodeValidationErrorCode.INVALID_NESTED_CONTRACT


def test_string_subclass_cannot_fake_current_answer_input() -> None:
    _, trace, _ = build_valid_episode()
    subtask = trace.subtasks[0]
    forged_input = AnswerAgentSubtaskInput(
        task_prompt=subtask.answer_input.task_prompt,
        subtask_input=_LyingText("FUTURE_SUBTASK_SECRET"),
        retrieved_memories=subtask.answer_input.retrieved_memories,
    )
    forged = replace(
        subtask,
        answer_input=forged_input,
        answer_result=run_answer_agent(
            forged_input,
            final_answer=subtask.task_result.final_answer,
        ),
    )

    result = validate_episode_subtask_trace(forged)

    assert result.error_code is EpisodeValidationErrorCode.INVALID_NESTED_CONTRACT
