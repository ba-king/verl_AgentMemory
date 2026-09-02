from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

from examples.memory_agent_grpo.answer_agent.input import (
    AnswerAgentSubtaskInput,
)
from examples.memory_agent_grpo.answer_agent.model_client import (
    AnswerAgentClientError,
    AnswerAgentClientErrorCode,
)
from examples.memory_agent_grpo.answer_agent.prompts.builder import (
    build_answer_agent_initial_messages,
)
from examples.memory_agent_grpo.answer_agent.response import (
    AnswerAgentResponse,
    AnswerAgentToolCall,
)
from examples.memory_agent_grpo.answer_agent.runtime import (
    AnswerAgentConversation,
    AnswerAgentSubtaskErrorCode,
    AnswerAgentSubtaskRunner,
    AnswerAgentSubtaskRunnerConfig,
    AnswerAgentSubtaskRunResult,
    AnswerAgentSubtaskRunStatus,
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
from examples.memory_agent_grpo.episode import (
    EpisodeFailureDetailCode,
    EpisodeFailureStage,
    EpisodeRunErrorCode,
    EpisodeRunner,
    EpisodeRunnerInputError,
    EpisodeRunStatus,
    validate_episode_run_result,
)
from examples.memory_agent_grpo.episode.runtime import runner as runner_module
from examples.memory_agent_grpo.episode.validation import (
    EpisodeValidationErrorCode,
    EpisodeValidationResult,
)
from examples.memory_agent_grpo.memory_agent.action.protocol import (
    MemoryActionName,
    MemoryDecisionPhase,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import (
    MemoryBankState,
)
from examples.memory_agent_grpo.memory_agent.executor.engine import (
    MemoryActionExecutor,
)
from examples.memory_agent_grpo.memory_agent.executor.outcome import (
    ExecutionCode,
    ExecutionResult,
    ExecutionStatus,
)

from ._support import (
    NULL_ACTION,
    ScriptedAnswerRunner,
    ScriptedMemoryClient,
    add_action,
    episode_input,
    retrieve_action,
)


class _ReturningFailedAnswerRunner:
    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        subtask: AnswerAgentSubtaskInput,
        *,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> AnswerAgentSubtaskRunResult:
        del tool_definitions
        self.calls += 1
        return AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.FAILED,
            config=AnswerAgentSubtaskRunnerConfig(
                max_inference_steps=2,
                max_tool_calls_per_inference=8,
                max_consecutive_tool_failures=1,
            ),
            conversation=AnswerAgentConversation(
                build_answer_agent_initial_messages(subtask)
            ),
            steps=(),
            inference_step_count=1,
            executed_tool_count=0,
            error_code=AnswerAgentSubtaskErrorCode.PROVIDER_ERROR,
            provider_error_code=AnswerAgentClientErrorCode.TRANSPORT_ERROR,
        )


class _InvalidAnswerResultRunner:
    def run(
        self,
        subtask: AnswerAgentSubtaskInput,
        *,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> object:
        del subtask, tool_definitions
        return object()


@dataclass
class _ScriptedAnswerClient:
    responses: list[object]

    def generate(
        self,
        conversation: AnswerAgentConversation,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> AnswerAgentResponse:
        del conversation, tool_definitions
        if not self.responses:
            raise AssertionError("Answer Agent was called more often than scripted.")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response  # type: ignore[return-value]


@dataclass
class _SuccessfulTaskToolExecutor:
    calls: list[ValidatedToolCall] = field(default_factory=list)

    def execute(self, call: ValidatedToolCall) -> ToolObservation:
        self.calls.append(call)
        return ToolObservation(
            tool_call_id=call.tool_call_id,
            tool_name=call.name,
            status=ToolObservationStatus.SUCCESS,
            content="public tool result",
        )


def _lookup_definition() -> ToolDefinition:
    return ToolDefinition(
        name="lookup",
        description="Look up public task information.",
        parameters_json_schema=(
            '{"type":"object","properties":{"query":{"type":"string"}},'
            '"required":["query"],"additionalProperties":false}'
        ),
    )


def _answer_call(call_id: str, *, name: str = "lookup") -> AnswerAgentToolCall:
    return AnswerAgentToolCall(
        tool_call_id=call_id,
        name=name,
        arguments_json='{"query":"public query"}',
    )


def _actual_failed_answer_runner(
    error_code: AnswerAgentSubtaskErrorCode,
) -> AnswerAgentSubtaskRunner:
    if error_code is AnswerAgentSubtaskErrorCode.PROVIDER_ERROR:
        responses: list[object] = [
            AnswerAgentClientError(
                AnswerAgentClientErrorCode.REQUEST_TIMEOUT,
                "private provider detail",
            )
        ]
        max_steps, max_batch, max_failures = 3, 2, 2
    elif error_code is AnswerAgentSubtaskErrorCode.INVALID_MODEL_RESPONSE:
        responses = [object()]
        max_steps, max_batch, max_failures = 3, 2, 2
    elif error_code is AnswerAgentSubtaskErrorCode.TOOL_CALL_BATCH_LIMIT:
        responses = [
            AnswerAgentResponse(
                tool_calls=(_answer_call("call-1"), _answer_call("call-2"))
            )
        ]
        max_steps, max_batch, max_failures = 3, 1, 2
    elif error_code is AnswerAgentSubtaskErrorCode.DUPLICATE_TOOL_CALL_ID:
        responses = [
            AnswerAgentResponse(tool_calls=(_answer_call("reused-id"),)),
            AnswerAgentResponse(tool_calls=(_answer_call("reused-id"),)),
        ]
        max_steps, max_batch, max_failures = 3, 2, 2
    elif error_code is AnswerAgentSubtaskErrorCode.CONSECUTIVE_TOOL_FAILURE_LIMIT:
        responses = [
            AnswerAgentResponse(
                tool_calls=(_answer_call("unknown-call", name="unknown"),)
            )
        ]
        max_steps, max_batch, max_failures = 3, 2, 1
    elif error_code is AnswerAgentSubtaskErrorCode.INFERENCE_STEP_LIMIT:
        responses = [
            AnswerAgentResponse(
                tool_calls=(_answer_call("unknown-call", name="unknown"),)
            )
        ]
        max_steps, max_batch, max_failures = 1, 2, 2
    else:
        raise AssertionError("Unexpected Answer Agent error code fixture.")
    return AnswerAgentSubtaskRunner(
        client=_ScriptedAnswerClient(responses),
        executor=_SuccessfulTaskToolExecutor(),
        config=AnswerAgentSubtaskRunnerConfig(
            max_inference_steps=max_steps,
            max_tool_calls_per_inference=max_batch,
            max_consecutive_tool_failures=max_failures,
        ),
    )


def test_two_round_null_path_has_exact_phase_order_and_current_only_inputs() -> None:
    event_log: list[str] = []
    memory_client = ScriptedMemoryClient(
        outcomes=[NULL_ACTION] * 5,
        event_log=event_log,
    )
    answer_runner = ScriptedAnswerRunner(
        final_answers=["ROUND_ONE_RAW_ANSWER", "ROUND_TWO_RAW_ANSWER"],
        event_log=event_log,
    )
    value = episode_input(
        episode_id="EPISODE_ID_MUST_NOT_REACH_MODELS",
        memory_task_inputs=("MEMORY_ONE", "FUTURE_MEMORY_SENTINEL"),
        answer_subtask_inputs=("ANSWER_ONE", "FUTURE_ANSWER_SENTINEL"),
    )

    result = EpisodeRunner(
        memory_client=memory_client,
        answer_runner=answer_runner,
    ).run(value)

    assert result.status is EpisodeRunStatus.COMPLETED
    assert result.completed_subtask_count == 2
    assert result.trace is not None
    assert validate_episode_run_result(result).is_valid
    assert event_log == [
        "memory:POST_TASK",
        "memory:PRE_TASK",
        "answer:1",
        "memory:POST_TASK",
        "memory:PRE_TASK",
        "answer:2",
        "memory:POST_TASK",
    ]
    assert memory_client.phases == (
        MemoryDecisionPhase.POST_TASK,
        MemoryDecisionPhase.PRE_TASK,
        MemoryDecisionPhase.POST_TASK,
        MemoryDecisionPhase.PRE_TASK,
        MemoryDecisionPhase.POST_TASK,
    )

    before_second_round = memory_client.serialized_inputs[:3]
    assert all(
        "FUTURE_MEMORY_SENTINEL" not in model_input
        and "FUTURE_ANSWER_SENTINEL" not in model_input
        for model_input in before_second_round
    )
    assert all(
        "EPISODE_ID_MUST_NOT_REACH_MODELS" not in model_input
        for model_input in memory_client.serialized_inputs
    )
    assert answer_runner.calls[0].subtask_input == "ANSWER_ONE"
    assert answer_runner.calls[1].subtask_input == "FUTURE_ANSWER_SENTINEL"
    assert all(call.task_prompt == "FIXED_TASK_PROMPT" for call in answer_runner.calls)
    assert all(not call.retrieved_memories for call in answer_runner.calls)
    assert all(
        not result.conversation.tool_exchanges for result in answer_runner.results
    )
    assert "ROUND_ONE_RAW_ANSWER" not in answer_runner.calls[1].to_dict().values()


def test_task_tool_definitions_are_forwarded_unchanged() -> None:
    definition = ToolDefinition(
        name="lookup",
        description="Look up public task information.",
        parameters_json_schema=(
            '{"type":"object","properties":{"query":{"type":"string"}},'
            '"required":["query"],"additionalProperties":false}'
        ),
    )
    memory_client = ScriptedMemoryClient(outcomes=[NULL_ACTION] * 3)
    answer_runner = ScriptedAnswerRunner(final_answers=["ROUND_ONE_ANSWER"])

    result = EpisodeRunner(
        memory_client=memory_client,
        answer_runner=answer_runner,
        tool_definitions=(definition,),
    ).run(episode_input(subtask_count=1))

    assert result.status is EpisodeRunStatus.COMPLETED
    assert answer_runner.definitions_seen == [(definition,)]
    assert validate_episode_run_result(result).is_valid


@pytest.mark.parametrize(
    ("post_query", "expected_match_count"),
    [("SEEDTOKEN", 1), ("ABSENTTOKEN", 0)],
)
def test_post_task_retrieval_always_creates_exactly_one_follow_up(
    post_query: str,
    expected_match_count: int,
) -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[
            add_action("SEEDTOKEN persistent memory"),
            NULL_ACTION,
            retrieve_action(post_query),
            NULL_ACTION,
        ]
    )
    answer_runner = ScriptedAnswerRunner(final_answers=["ROUND_ONE_ANSWER"])

    result = EpisodeRunner(
        memory_client=memory_client,
        answer_runner=answer_runner,
    ).run(episode_input(subtask_count=1))

    assert result.status is EpisodeRunStatus.COMPLETED
    assert result.trace is not None
    subtask_trace = result.trace.subtasks[0]
    assert len(subtask_trace.post_task_decisions) == 2
    first, follow_up = subtask_trace.post_task_decisions
    retrieval = first.gateway_result.execution
    assert retrieval is not None
    assert retrieval.retrieval is not None
    assert retrieval.retrieval.match_count == expected_match_count
    assert follow_up.context.phase is (MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL)
    assert follow_up.context.post_task_retrieval == retrieval.retrieval
    assert memory_client.phases == (
        MemoryDecisionPhase.POST_TASK,
        MemoryDecisionPhase.PRE_TASK,
        MemoryDecisionPhase.POST_TASK,
        MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
    )
    assert validate_episode_run_result(result).is_valid


def test_seed_retrieval_uses_the_same_post_task_follow_up_path() -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[
            retrieve_action("ABSENT_SEED_QUERY"),
            add_action("SEED_FOLLOW_UP_MEMORY"),
            retrieve_action("SEED_FOLLOW_UP_MEMORY"),
            NULL_ACTION,
        ]
    )
    answer_runner = ScriptedAnswerRunner(final_answers=["ROUND_ONE_ANSWER"])

    result = EpisodeRunner(
        memory_client=memory_client,
        answer_runner=answer_runner,
    ).run(episode_input(subtask_count=1))

    assert result.status is EpisodeRunStatus.COMPLETED
    assert result.trace is not None
    initialization = result.trace.initialization
    assert len(initialization.post_task_decisions) == 2
    assert initialization.post_task_decisions[1].context.phase is (
        MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL
    )
    assert tuple(
        (record.memory_id, record.content)
        for record in initialization.bank_after.records
    ) == (("M001", "SEED_FOLLOW_UP_MEMORY"),)
    assert answer_runner.calls[0].retrieved_memories == ("SEED_FOLLOW_UP_MEMORY",)
    assert memory_client.phases == (
        MemoryDecisionPhase.POST_TASK,
        MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
        MemoryDecisionPhase.PRE_TASK,
        MemoryDecisionPhase.POST_TASK,
    )
    assert validate_episode_run_result(result).is_valid


def test_real_bank_continues_and_retrieval_is_injected_exactly() -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[
            add_action("SEEDTOKEN persistent memory"),
            retrieve_action("SEEDTOKEN"),
            add_action("ROUNDONETOKEN persistent memory"),
            retrieve_action("ROUNDONETOKEN"),
            NULL_ACTION,
        ]
    )
    answer_runner = ScriptedAnswerRunner(
        final_answers=["ROUND_ONE_ANSWER", "ROUND_TWO_ANSWER"]
    )

    result = EpisodeRunner(
        memory_client=memory_client,
        answer_runner=answer_runner,
    ).run(episode_input())

    assert result.status is EpisodeRunStatus.COMPLETED
    assert result.trace is not None
    initialization = result.trace.initialization
    first, second = result.trace.subtasks
    assert initialization.bank_before == MemoryBankState.empty()
    assert tuple(
        (record.memory_id, record.content)
        for record in initialization.bank_after.records
    ) == (("M001", "SEEDTOKEN persistent memory"),)
    assert first.bank_before == initialization.bank_after
    assert first.pre_task_decision.bank_before == first.bank_before
    assert first.pre_task_decision.bank_after == first.bank_before
    assert second.bank_before == first.bank_after
    assert result.final_bank_state == second.bank_after
    assert tuple(
        (record.memory_id, record.content) for record in result.final_bank_state.records
    ) == (
        ("M001", "SEEDTOKEN persistent memory"),
        ("M002", "ROUNDONETOKEN persistent memory"),
    )
    assert result.final_bank_state.next_id == 3
    assert answer_runner.calls[0].retrieved_memories == ("SEEDTOKEN persistent memory",)
    assert answer_runner.calls[1].retrieved_memories == (
        "ROUNDONETOKEN persistent memory",
    )
    assert "M001" not in answer_runner.calls[0].retrieved_memories[0]
    assert "M002" not in answer_runner.calls[1].retrieved_memories[0]
    assert validate_episode_run_result(result).is_valid


def test_reusing_runner_still_starts_each_episode_from_a_fresh_bank() -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[
            add_action("CROSSRUN_A_SECRET"),
            NULL_ACTION,
            NULL_ACTION,
            add_action("RUN_B_MEMORY"),
            retrieve_action("CROSSRUN_A_SECRET"),
            NULL_ACTION,
        ]
    )
    answer_runner = ScriptedAnswerRunner(final_answers=["RUN_A_ANSWER", "RUN_B_ANSWER"])
    runner = EpisodeRunner(
        memory_client=memory_client,
        answer_runner=answer_runner,
    )

    first = runner.run(episode_input(episode_id="run-a", subtask_count=1))
    second = runner.run(episode_input(episode_id="run-b", subtask_count=1))

    assert first.status is EpisodeRunStatus.COMPLETED
    assert second.status is EpisodeRunStatus.COMPLETED
    assert second.trace is not None
    assert second.trace.initial_bank_state == MemoryBankState.empty()
    assert second.trace.initialization.bank_before == MemoryBankState.empty()
    assert tuple(
        (record.memory_id, record.content)
        for record in second.trace.initialization.bank_after.records
    ) == (("M001", "RUN_B_MEMORY"),)
    pre_execution = second.trace.subtasks[0].pre_task_decision.gateway_result.execution
    assert pre_execution is not None
    assert pre_execution.retrieval is not None
    assert pre_execution.retrieval.matches == ()
    assert answer_runner.calls[1].retrieved_memories == ()
    assert all(
        record.content != "CROSSRUN_A_SECRET"
        for record in second.final_bank_state.records
    )
    assert validate_episode_run_result(first).is_valid
    assert validate_episode_run_result(second).is_valid


@pytest.mark.parametrize(
    "initial_outcome",
    [RuntimeError("PRIVATE_PROVIDER_DETAIL"), "not valid JSON"],
)
def test_initialization_failure_returns_empty_uncommitted_result(
    initial_outcome: object,
) -> None:
    memory_client = ScriptedMemoryClient(outcomes=[initial_outcome])
    answer_runner = ScriptedAnswerRunner(final_answers=["MUST_NOT_RUN"])

    result = EpisodeRunner(
        memory_client=memory_client,
        answer_runner=answer_runner,
    ).run(episode_input(subtask_count=1))

    assert result.status is EpisodeRunStatus.FAILED
    assert result.error_code is EpisodeRunErrorCode.INITIALIZATION_FAILED
    assert result.failed_stage is EpisodeFailureStage.INITIALIZATION
    expected_detail = (
        EpisodeFailureDetailCode.MEMORY_MODEL_CALL_FAILED
        if isinstance(initial_outcome, RuntimeError)
        else EpisodeFailureDetailCode.MEMORY_OUTPUT_VALIDATION_FAILED
    )
    assert result.failure_detail_code is expected_detail
    assert result.answer_provider_error_code is None
    assert result.trace is None
    assert result.completed_subtask_count == 0
    assert result.failed_subtask_ordinal is None
    assert result.final_bank_state == MemoryBankState.empty()
    assert len(memory_client.calls) == 1
    assert answer_runner.calls == []
    assert "PRIVATE_PROVIDER_DETAIL" not in str(result.to_dict())
    assert validate_episode_run_result(result).is_valid


def test_initialization_execution_rejection_has_stable_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_action(
        executor: MemoryActionExecutor,
        action: object,
    ) -> ExecutionResult:
        del executor, action
        return ExecutionResult(
            status=ExecutionStatus.REJECTED,
            code=ExecutionCode.INVALID_BANK_STATE,
            action=MemoryActionName.ADD,
            error_message="stable rejection",
        )

    monkeypatch.setattr(MemoryActionExecutor, "execute", reject_action)
    memory_client = ScriptedMemoryClient(
        outcomes=[
            (
                '{"memory_action":{"action":"ADD","arguments":'
                '{"content":"public memory"}}}'
            )
        ]
    )

    result = EpisodeRunner(
        memory_client=memory_client,
        answer_runner=ScriptedAnswerRunner(final_answers=["MUST_NOT_RUN"]),
    ).run(episode_input(subtask_count=1))

    assert result.status is EpisodeRunStatus.FAILED
    assert result.error_code is EpisodeRunErrorCode.INITIALIZATION_FAILED
    assert result.failed_stage is EpisodeFailureStage.INITIALIZATION
    assert (
        result.failure_detail_code is EpisodeFailureDetailCode.MEMORY_EXECUTION_REJECTED
    )
    assert result.trace is None
    assert result.final_bank_state == MemoryBankState.empty()
    assert validate_episode_run_result(result).is_valid


def test_returned_answer_failure_stops_before_post_task() -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[NULL_ACTION, NULL_ACTION, "MUST_NOT_BE_USED"]
    )
    answer_runner = _ReturningFailedAnswerRunner()

    result = EpisodeRunner(
        memory_client=memory_client,
        answer_runner=answer_runner,
    ).run(episode_input(subtask_count=1))

    assert result.status is EpisodeRunStatus.FAILED
    assert result.error_code is EpisodeRunErrorCode.SUBTASK_FAILED
    assert result.failed_stage is EpisodeFailureStage.ANSWER
    assert result.failure_detail_code is EpisodeFailureDetailCode.ANSWER_PROVIDER_ERROR
    assert (
        result.answer_provider_error_code is AnswerAgentClientErrorCode.TRANSPORT_ERROR
    )
    assert result.completed_subtask_count == 0
    assert result.failed_subtask_ordinal == 1
    assert result.trace is not None
    assert result.trace.subtasks == ()
    assert result.final_bank_state == result.trace.initialization.bank_after
    assert answer_runner.calls == 1
    assert memory_client.phases == (
        MemoryDecisionPhase.POST_TASK,
        MemoryDecisionPhase.PRE_TASK,
    )
    assert validate_episode_run_result(result).is_valid


def test_invalid_answer_result_is_an_internal_contract_failure() -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[NULL_ACTION, NULL_ACTION, "MUST_NOT_BE_USED"]
    )

    result = EpisodeRunner(
        memory_client=memory_client,
        answer_runner=_InvalidAnswerResultRunner(),  # type: ignore[arg-type]
    ).run(episode_input(subtask_count=1))

    assert result.status is EpisodeRunStatus.FAILED
    assert result.error_code is EpisodeRunErrorCode.INTERNAL_CONTRACT_ERROR
    assert result.failed_stage is EpisodeFailureStage.INTERNAL
    assert (
        result.failure_detail_code
        is EpisodeFailureDetailCode.ANSWER_RESULT_CONTRACT_ERROR
    )
    assert result.answer_provider_error_code is None
    assert result.completed_subtask_count == 0
    assert result.failed_subtask_ordinal == 1
    assert result.trace is not None
    assert result.trace.subtasks == ()
    assert validate_episode_run_result(result).is_valid


@pytest.mark.parametrize(
    ("answer_error_code", "expected_detail"),
    [
        (
            AnswerAgentSubtaskErrorCode.PROVIDER_ERROR,
            EpisodeFailureDetailCode.ANSWER_PROVIDER_ERROR,
        ),
        (
            AnswerAgentSubtaskErrorCode.INVALID_MODEL_RESPONSE,
            EpisodeFailureDetailCode.ANSWER_INVALID_MODEL_RESPONSE,
        ),
        (
            AnswerAgentSubtaskErrorCode.TOOL_CALL_BATCH_LIMIT,
            EpisodeFailureDetailCode.ANSWER_TOOL_CALL_BATCH_LIMIT,
        ),
        (
            AnswerAgentSubtaskErrorCode.DUPLICATE_TOOL_CALL_ID,
            EpisodeFailureDetailCode.ANSWER_DUPLICATE_TOOL_CALL_ID,
        ),
        (
            AnswerAgentSubtaskErrorCode.CONSECUTIVE_TOOL_FAILURE_LIMIT,
            EpisodeFailureDetailCode.ANSWER_CONSECUTIVE_TOOL_FAILURE_LIMIT,
        ),
        (
            AnswerAgentSubtaskErrorCode.INFERENCE_STEP_LIMIT,
            EpisodeFailureDetailCode.ANSWER_INFERENCE_STEP_LIMIT,
        ),
    ],
)
def test_returned_answer_failure_exposes_only_stable_nested_codes(
    answer_error_code: AnswerAgentSubtaskErrorCode,
    expected_detail: EpisodeFailureDetailCode,
) -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[NULL_ACTION, NULL_ACTION, "MUST_NOT_BE_USED"]
    )

    result = EpisodeRunner(
        memory_client=memory_client,
        answer_runner=_actual_failed_answer_runner(answer_error_code),
        tool_definitions=(_lookup_definition(),),
    ).run(episode_input(subtask_count=1))

    assert result.status is EpisodeRunStatus.FAILED
    assert result.error_code is EpisodeRunErrorCode.SUBTASK_FAILED
    assert result.failed_stage is EpisodeFailureStage.ANSWER
    assert result.failure_detail_code is expected_detail
    expected_provider_code = (
        AnswerAgentClientErrorCode.REQUEST_TIMEOUT
        if answer_error_code is AnswerAgentSubtaskErrorCode.PROVIDER_ERROR
        else None
    )
    assert result.answer_provider_error_code is expected_provider_code
    assert result.completed_subtask_count == 0
    assert result.failed_subtask_ordinal == 1
    assert result.trace is not None
    assert result.trace.subtasks == ()
    assert validate_episode_run_result(result).is_valid


@pytest.mark.parametrize("failure_stage", ["PRE", "ANSWER", "POST"])
def test_second_subtask_failure_exposes_only_first_completed_prefix(
    failure_stage: str,
) -> None:
    q2_pre = "invalid pre output" if failure_stage == "PRE" else NULL_ACTION
    outcomes: list[object] = [
        NULL_ACTION,
        NULL_ACTION,
        add_action("ROUND_ONE_COMMITTED_MEMORY"),
        q2_pre,
    ]
    if failure_stage == "POST":
        outcomes.append("invalid post output")
    memory_client = ScriptedMemoryClient(outcomes=outcomes)
    answer_runner = ScriptedAnswerRunner(
        final_answers=["ROUND_ONE_ANSWER", "ROUND_TWO_ANSWER"],
        fail_on_call=2 if failure_stage == "ANSWER" else None,
    )

    result = EpisodeRunner(
        memory_client=memory_client,
        answer_runner=answer_runner,
    ).run(episode_input())

    assert result.status is EpisodeRunStatus.FAILED
    assert result.error_code is EpisodeRunErrorCode.SUBTASK_FAILED
    expected_stage = {
        "PRE": EpisodeFailureStage.PRE_TASK,
        "ANSWER": EpisodeFailureStage.ANSWER,
        "POST": EpisodeFailureStage.POST_TASK,
    }[failure_stage]
    expected_detail = {
        "PRE": EpisodeFailureDetailCode.MEMORY_OUTPUT_VALIDATION_FAILED,
        "ANSWER": EpisodeFailureDetailCode.ANSWER_RUNNER_EXCEPTION,
        "POST": EpisodeFailureDetailCode.MEMORY_OUTPUT_VALIDATION_FAILED,
    }[failure_stage]
    assert result.failed_stage is expected_stage
    assert result.failure_detail_code is expected_detail
    assert result.answer_provider_error_code is None
    assert result.completed_subtask_count == 1
    assert result.failed_subtask_ordinal == 2
    assert result.trace is not None
    assert len(result.trace.subtasks) == 1
    committed = result.trace.subtasks[0].bank_after
    assert result.trace.final_bank_state == committed
    assert result.final_bank_state == committed
    assert tuple(record.content for record in committed.records) == (
        "ROUND_ONE_COMMITTED_MEMORY",
    )
    expected_answer_calls = 1 if failure_stage == "PRE" else 2
    assert len(answer_runner.calls) == expected_answer_calls
    expected_memory_calls = 5 if failure_stage == "POST" else 4
    assert len(memory_client.calls) == expected_memory_calls
    assert validate_episode_run_result(result).is_valid


def test_failed_subtask_trace_validation_discards_uncommitted_bank_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uncommitted_content = "UNCOMMITTED_POST_TASK_MEMORY"
    memory_client = ScriptedMemoryClient(
        outcomes=[
            NULL_ACTION,
            NULL_ACTION,
            add_action(uncommitted_content),
        ]
    )

    def reject_subtask_trace(value: object) -> EpisodeValidationResult:
        del value
        return EpisodeValidationResult(
            error_code=EpisodeValidationErrorCode.RESULT_SEMANTICS_MISMATCH,
            error_message="stable test rejection",
        )

    monkeypatch.setattr(
        runner_module,
        "validate_episode_subtask_trace",
        reject_subtask_trace,
    )

    result = EpisodeRunner(
        memory_client=memory_client,
        answer_runner=ScriptedAnswerRunner(final_answers=["ROUND_ONE_ANSWER"]),
    ).run(episode_input(subtask_count=1))

    assert result.status is EpisodeRunStatus.FAILED
    assert result.error_code is EpisodeRunErrorCode.INTERNAL_CONTRACT_ERROR
    assert result.failed_stage is EpisodeFailureStage.INTERNAL
    assert (
        result.failure_detail_code
        is EpisodeFailureDetailCode.EPISODE_TRACE_CONTRACT_ERROR
    )
    assert result.completed_subtask_count == 0
    assert result.failed_subtask_ordinal == 1
    assert result.trace is not None
    assert result.trace.subtasks == ()
    assert result.final_bank_state == result.trace.initialization.bank_after
    assert result.final_bank_state == MemoryBankState.empty()
    assert uncommitted_content not in str(result.to_dict())
    assert validate_episode_run_result(result).is_valid


def test_invalid_episode_input_is_rejected_before_any_agent_call() -> None:
    valid = episode_input()
    invalid = replace(
        valid,
        subtasks=(replace(valid.subtasks[0], ordinal=2), valid.subtasks[1]),
    )
    memory_client = ScriptedMemoryClient(outcomes=[NULL_ACTION])
    answer_runner = ScriptedAnswerRunner(final_answers=["MUST_NOT_RUN"])
    runner = EpisodeRunner(
        memory_client=memory_client,
        answer_runner=answer_runner,
    )

    with pytest.raises(EpisodeRunnerInputError):
        runner.run(invalid)

    assert memory_client.calls == []
    assert answer_runner.calls == []
