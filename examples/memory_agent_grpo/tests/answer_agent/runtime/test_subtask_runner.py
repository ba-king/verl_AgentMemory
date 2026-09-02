from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from examples.memory_agent_grpo.answer_agent.input import AnswerAgentSubtaskInput
from examples.memory_agent_grpo.answer_agent.model_client import (
    AnswerAgentClientError,
    AnswerAgentClientErrorCode,
)
from examples.memory_agent_grpo.answer_agent.response import (
    AnswerAgentResponse,
    AnswerAgentToolCall,
)
from examples.memory_agent_grpo.answer_agent.runtime import (
    AnswerAgentConversation,
    AnswerAgentInteractionMode,
    AnswerAgentSubtaskErrorCode,
    AnswerAgentSubtaskRunner,
    AnswerAgentSubtaskRunnerConfig,
    AnswerAgentSubtaskRunStatus,
    AnswerAgentTaskToolExecutionResult,
    AnswerAgentToolCallDisposition,
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


@dataclass
class _ScriptedClient:
    responses: list[object]
    calls: list[tuple[AnswerAgentConversation, tuple[ToolDefinition, ...]]] = field(default_factory=list)

    def generate(
        self,
        conversation: AnswerAgentConversation,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> AnswerAgentResponse:
        self.calls.append((conversation, tool_definitions))
        if not self.responses:
            raise AssertionError("unexpected extra model call")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value  # type: ignore[return-value]


@dataclass
class _ScriptedExecutor:
    results: list[object] = field(default_factory=list)
    calls: list[ValidatedToolCall] = field(default_factory=list)

    def execute(self, call: ValidatedToolCall) -> ToolObservation:
        self.calls.append(call)
        if self.results:
            value = self.results.pop(0)
            if isinstance(value, Exception):
                raise value
            return value  # type: ignore[return-value]
        return _observation(call.tool_call_id, name=call.name)


def _subtask() -> AnswerAgentSubtaskInput:
    return AnswerAgentSubtaskInput(
        task_prompt="Use the supplied task tools.",
        subtask_input="Current public task.",
        retrieved_memories=("Selected memory.",),
    )


def _definition() -> ToolDefinition:
    return ToolDefinition(
        name="CitySearch",
        description="Search for cities in one state.",
        parameters_json_schema=(
            '{"type":"object","properties":{"state":{"type":"string"}},'
            '"required":["state"],"additionalProperties":false}'
        ),
    )


def _call(
    call_id: str,
    *,
    arguments_json: str = '{"state":"New York"}',
    name: str = "CitySearch",
) -> AnswerAgentToolCall:
    return AnswerAgentToolCall(
        tool_call_id=call_id,
        name=name,
        arguments_json=arguments_json,
    )


def _batch(
    *calls: AnswerAgentToolCall,
    text: str | None = None,
) -> AnswerAgentResponse:
    return AnswerAgentResponse(text=text, tool_calls=tuple(calls))


def _observation(
    call_id: str,
    *,
    name: str = "CitySearch",
    status: ToolObservationStatus = ToolObservationStatus.SUCCESS,
    content: str | None = None,
) -> ToolObservation:
    return ToolObservation(
        tool_call_id=call_id,
        tool_name=name,
        status=status,
        content=(f"public result for {call_id}" if content is None else content),
    )


def _runner(
    client: object,
    executor: object,
    *,
    max_steps: int = 5,
    max_batch: int = 3,
    max_failures: int = 2,
    interaction_mode: AnswerAgentInteractionMode = (AnswerAgentInteractionMode.TOOL_OR_FINAL),
) -> AnswerAgentSubtaskRunner:
    return AnswerAgentSubtaskRunner(
        client=client,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        config=AnswerAgentSubtaskRunnerConfig(
            max_inference_steps=max_steps,
            max_tool_calls_per_inference=max_batch,
            max_consecutive_tool_failures=max_failures,
            interaction_mode=interaction_mode,
        ),
    )


def _execution_result(
    call_id: str,
    *,
    terminal: bool,
    content: str | None = None,
) -> AnswerAgentTaskToolExecutionResult:
    return AnswerAgentTaskToolExecutionResult(
        tool_observation=_observation(call_id, content=content),
        is_terminal=terminal,
    )


def test_direct_text_response_completes_without_tool_execution() -> None:
    client = _ScriptedClient([AnswerAgentResponse(text="final plan")])
    executor = _ScriptedExecutor()

    result = _runner(client, executor).run(
        _subtask(),
        tool_definitions=(_definition(),),
    )

    assert result.status is AnswerAgentSubtaskRunStatus.COMPLETED
    assert result.final_answer == "final plan"
    assert result.inference_step_count == 1
    assert result.executed_tool_count == 0
    assert executor.calls == []
    assert result.conversation.tool_exchanges == ()


def test_environment_mode_uses_single_tool_prompt_and_stops_on_executor_terminal() -> None:
    client = _ScriptedClient(
        [
            _batch(_call("call-1")),
            _batch(_call("call-2")),
            AnswerAgentResponse(text="must not be requested"),
        ]
    )
    executor = _ScriptedExecutor(
        results=[
            _execution_result("call-1", terminal=False),
            _execution_result(
                "call-2",
                terminal=True,
                content="Public environment terminal state.",
            ),
        ]
    )

    result = _runner(
        client,
        executor,
        max_batch=1,
        interaction_mode=(AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL),
    ).run(_subtask(), tool_definitions=(_definition(),))

    assert result.status is AnswerAgentSubtaskRunStatus.COMPLETED
    assert result.final_answer == "Public environment terminal state."
    assert result.inference_step_count == 2
    assert result.executed_tool_count == 2
    assert len(client.calls) == 2
    assert len(result.conversation.tool_exchanges) == 2
    assert result.steps[-1].tool_outcomes[0].is_terminal is True
    system_prompt = client.calls[0][0].initial_messages[0].content
    assert "TOOL or FINAL" not in system_prompt
    assert "multiple task tools" not in system_prompt


def test_environment_terminal_remains_authoritative_on_error_observation() -> None:
    client = _ScriptedClient([_batch(_call("call-1"))])
    executor = _ScriptedExecutor(
        results=[
            AnswerAgentTaskToolExecutionResult(
                tool_observation=_observation(
                    "call-1",
                    status=ToolObservationStatus.ERROR,
                    content="Public terminal error state.",
                ),
                is_terminal=True,
            )
        ]
    )

    result = _runner(
        client,
        executor,
        max_batch=1,
        max_failures=1,
        interaction_mode=(AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL),
    ).run(_subtask(), tool_definitions=(_definition(),))

    assert result.status is AnswerAgentSubtaskRunStatus.COMPLETED
    assert result.final_answer == "Public terminal error state."
    assert len(client.calls) == 1


def test_environment_mode_rejects_text_without_a_tool_call() -> None:
    client = _ScriptedClient([AnswerAgentResponse(text="I am done.")])
    executor = _ScriptedExecutor()

    result = _runner(
        client,
        executor,
        max_batch=1,
        interaction_mode=(AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL),
    ).run(_subtask(), tool_definitions=(_definition(),))

    assert result.status is AnswerAgentSubtaskRunStatus.FAILED
    assert result.error_code is (AnswerAgentSubtaskErrorCode.INTERACTION_MODE_VIOLATION)
    assert result.executed_tool_count == 0
    assert executor.calls == []


def test_environment_mode_executes_single_tool_with_auxiliary_text() -> None:
    client = _ScriptedClient([_batch(_call("call-1"), text="I will call a tool.")])
    executor = _ScriptedExecutor(
        results=[_execution_result("call-1", terminal=True)],
    )

    result = _runner(
        client,
        executor,
        max_batch=1,
        interaction_mode=(AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL),
    ).run(_subtask(), tool_definitions=(_definition(),))

    assert result.status is AnswerAgentSubtaskRunStatus.COMPLETED
    assert result.executed_tool_count == 1
    assert result.steps[0].response.text == "I will call a tool."
    assert [call.tool_call_id for call in executor.calls] == ["call-1"]


def test_environment_mode_rejects_multi_call_response_before_execution() -> None:
    client = _ScriptedClient([_batch(_call("call-1"), _call("call-2"))])
    executor = _ScriptedExecutor()

    result = _runner(
        client,
        executor,
        max_batch=1,
        interaction_mode=(AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL),
    ).run(_subtask(), tool_definitions=(_definition(),))

    assert result.error_code is AnswerAgentSubtaskErrorCode.TOOL_CALL_BATCH_LIMIT
    assert result.executed_tool_count == 0
    assert executor.calls == []


def test_environment_mode_requires_typed_executor_terminal_contract() -> None:
    client = _ScriptedClient([_batch(_call("call-1"))])
    executor = _ScriptedExecutor(results=[_observation("call-1")])

    with pytest.raises(ValueError, match="AnswerAgentTaskToolExecutionResult"):
        _runner(
            client,
            executor,
            max_batch=1,
            interaction_mode=(AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL),
        ).run(_subtask(), tool_definitions=(_definition(),))


def test_default_mode_accepts_nonterminal_typed_executor_result() -> None:
    client = _ScriptedClient([_batch(_call("call-1")), AnswerAgentResponse(text="final plan")])
    executor = _ScriptedExecutor(results=[_execution_result("call-1", terminal=False)])

    result = _runner(client, executor).run(
        _subtask(),
        tool_definitions=(_definition(),),
    )

    assert result.status is AnswerAgentSubtaskRunStatus.COMPLETED
    assert result.final_answer == "final plan"


def test_two_valid_calls_execute_in_order_and_share_one_exchange() -> None:
    first = _call(
        "call-1",
        arguments_json='{ "state" : "New York" }',
    )
    second = _call(
        "call-2",
        arguments_json='{ "state" : "California" }',
    )
    batch = _batch(first, second, text="I will inspect both states.")
    client = _ScriptedClient([batch, AnswerAgentResponse(text="final plan")])
    executor = _ScriptedExecutor()

    result = _runner(client, executor, max_batch=2).run(
        _subtask(),
        tool_definitions=(_definition(),),
    )

    assert result.status is AnswerAgentSubtaskRunStatus.COMPLETED
    assert result.inference_step_count == 2
    assert result.executed_tool_count == 2
    assert [call.tool_call_id for call in executor.calls] == [
        "call-1",
        "call-2",
    ]
    assert [call.arguments_json for call in executor.calls] == [
        '{"state":"New York"}',
        '{"state":"California"}',
    ]
    first_step = result.steps[0]
    assert [outcome.disposition for outcome in first_step.tool_outcomes] == [
        AnswerAgentToolCallDisposition.EXECUTED,
        AnswerAgentToolCallDisposition.EXECUTED,
    ]
    assert len(result.conversation.tool_exchanges) == 1
    exchange = result.conversation.tool_exchanges[0]
    assert exchange.assistant_response == batch
    assert [observation.tool_call_id for observation in exchange.tool_observations] == [
        "call-1",
        "call-2",
    ]
    continued_exchange = client.calls[1][0].tool_exchanges[0]
    assert continued_exchange == exchange
    assert continued_exchange.assistant_response.tool_calls[0].arguments_json == '{ "state" : "New York" }'


@pytest.mark.parametrize("invalid_position", [0, 1])
def test_any_gateway_rejection_prevalidates_the_batch_before_zero_execution(
    invalid_position: int,
) -> None:
    calls = [
        _call("call-1"),
        _call("call-2"),
    ]
    calls[invalid_position] = _call(
        calls[invalid_position].tool_call_id,
        arguments_json="{}",
    )
    client = _ScriptedClient(
        [
            _batch(*calls),
            AnswerAgentResponse(text="corrected final plan"),
        ]
    )
    executor = _ScriptedExecutor()

    result = _runner(client, executor, max_batch=2).run(
        _subtask(),
        tool_definitions=(_definition(),),
    )

    assert result.status is AnswerAgentSubtaskRunStatus.COMPLETED
    assert result.executed_tool_count == 0
    assert executor.calls == []
    dispositions = [outcome.disposition for outcome in result.steps[0].tool_outcomes]
    assert dispositions[invalid_position] is (AnswerAgentToolCallDisposition.GATEWAY_REJECTED)
    assert dispositions[1 - invalid_position] is (AnswerAgentToolCallDisposition.BATCH_SKIPPED)
    assert all(
        outcome.tool_observation.status is ToolObservationStatus.ERROR for outcome in result.steps[0].tool_outcomes
    )
    assert len(client.calls[1][0].tool_exchanges) == 1


def test_all_valid_calls_execute_even_after_one_error_observation() -> None:
    client = _ScriptedClient(
        [
            _batch(_call("call-1"), _call("call-2")),
            AnswerAgentResponse(text="final plan after mixed evidence"),
        ]
    )
    executor = _ScriptedExecutor(
        results=[
            _observation(
                "call-1",
                status=ToolObservationStatus.ERROR,
                content="Tool execution failed.",
            ),
            _observation("call-2"),
        ]
    )

    result = _runner(client, executor, max_batch=2).run(
        _subtask(),
        tool_definitions=(_definition(),),
    )

    assert result.status is AnswerAgentSubtaskRunStatus.COMPLETED
    assert [call.tool_call_id for call in executor.calls] == [
        "call-1",
        "call-2",
    ]
    assert result.executed_tool_count == 2
    assert all(
        outcome.disposition is AnswerAgentToolCallDisposition.EXECUTED for outcome in result.steps[0].tool_outcomes
    )
    assert [outcome.tool_observation.status for outcome in result.steps[0].tool_outcomes] == [
        ToolObservationStatus.ERROR,
        ToolObservationStatus.SUCCESS,
    ]


def test_failure_budget_counts_one_failed_batch_not_each_error_observation() -> None:
    client = _ScriptedClient(
        [
            _batch(_call("batch-1-a"), _call("batch-1-b")),
            _batch(_call("batch-2-a"), _call("batch-2-b")),
            AnswerAgentResponse(text="must not be reached"),
        ]
    )
    executor = _ScriptedExecutor(
        results=[
            _observation("batch-1-a", status=ToolObservationStatus.ERROR),
            _observation("batch-1-b", status=ToolObservationStatus.ERROR),
            _observation("batch-2-a"),
            _observation("batch-2-b", status=ToolObservationStatus.ERROR),
        ]
    )

    result = _runner(
        client,
        executor,
        max_batch=2,
        max_failures=2,
    ).run(_subtask(), tool_definitions=(_definition(),))

    assert result.error_code is (AnswerAgentSubtaskErrorCode.CONSECUTIVE_TOOL_FAILURE_LIMIT)
    assert result.inference_step_count == 2
    assert result.executed_tool_count == 4
    assert len(client.calls) == 2
    assert len(result.conversation.tool_exchanges) == 2


def test_all_success_batch_resets_consecutive_failure_count() -> None:
    client = _ScriptedClient(
        [
            _batch(_call("failed-1-a"), _call("failed-1-b")),
            _batch(_call("success-a"), _call("success-b")),
            _batch(_call("failed-2")),
            AnswerAgentResponse(text="final plan"),
        ]
    )
    executor = _ScriptedExecutor(
        results=[
            _observation("failed-1-a", status=ToolObservationStatus.ERROR),
            _observation("failed-1-b", status=ToolObservationStatus.ERROR),
            _observation("success-a"),
            _observation("success-b"),
            _observation("failed-2", status=ToolObservationStatus.ERROR),
        ]
    )

    result = _runner(
        client,
        executor,
        max_batch=2,
        max_failures=2,
    ).run(_subtask(), tool_definitions=(_definition(),))

    assert result.status is AnswerAgentSubtaskRunStatus.COMPLETED
    assert result.inference_step_count == 4
    assert result.executed_tool_count == 5


def test_duplicate_historical_id_rejects_entire_new_batch_before_execution() -> None:
    client = _ScriptedClient(
        [
            _batch(_call("already-used")),
            _batch(_call("new-sibling"), _call("already-used")),
        ]
    )
    executor = _ScriptedExecutor()

    result = _runner(client, executor, max_batch=2).run(
        _subtask(),
        tool_definitions=(_definition(),),
    )

    assert result.error_code is AnswerAgentSubtaskErrorCode.DUPLICATE_TOOL_CALL_ID
    assert result.inference_step_count == 2
    assert result.executed_tool_count == 1
    assert [call.tool_call_id for call in executor.calls] == ["already-used"]
    assert len(result.steps) == 2
    assert result.steps[-1].tool_outcomes == ()
    assert len(result.conversation.tool_exchanges) == 1


def test_duplicate_ids_within_one_model_response_are_invalid_before_execution() -> None:
    client = _ScriptedClient([_batch(_call("same"), _call("same"))])
    executor = _ScriptedExecutor()

    result = _runner(client, executor, max_batch=2).run(
        _subtask(),
        tool_definitions=(_definition(),),
    )

    assert result.error_code is AnswerAgentSubtaskErrorCode.INVALID_MODEL_RESPONSE
    assert result.steps == ()
    assert executor.calls == []


def test_tool_call_batch_limit_is_terminal_and_executes_none() -> None:
    client = _ScriptedClient(
        [
            _batch(
                _call("call-1"),
                _call("call-2"),
                _call("call-3"),
            )
        ]
    )
    executor = _ScriptedExecutor()

    result = _runner(client, executor, max_batch=2).run(
        _subtask(),
        tool_definitions=(_definition(),),
    )

    assert result.error_code is AnswerAgentSubtaskErrorCode.TOOL_CALL_BATCH_LIMIT
    assert result.inference_step_count == 1
    assert result.executed_tool_count == 0
    assert executor.calls == []
    assert result.steps[0].tool_outcomes == ()
    assert result.conversation.tool_exchanges == ()


def test_one_multi_call_batch_consumes_one_inference_step_at_limit() -> None:
    client = _ScriptedClient([_batch(_call("call-1"), _call("call-2"))])
    executor = _ScriptedExecutor()

    result = _runner(
        client,
        executor,
        max_steps=1,
        max_batch=2,
    ).run(_subtask(), tool_definitions=(_definition(),))

    assert result.error_code is AnswerAgentSubtaskErrorCode.INFERENCE_STEP_LIMIT
    assert result.inference_step_count == 1
    assert result.executed_tool_count == 2
    assert len(result.steps) == 1
    assert len(result.steps[0].tool_outcomes) == 2
    assert len(result.conversation.tool_exchanges) == 1


def test_provider_error_is_terminal_and_retains_only_the_safe_code() -> None:
    client = _ScriptedClient(
        [
            AnswerAgentClientError(
                AnswerAgentClientErrorCode.REQUEST_TIMEOUT,
                "private provider detail",
            )
        ]
    )
    executor = _ScriptedExecutor()

    result = _runner(client, executor).run(
        _subtask(),
        tool_definitions=(_definition(),),
    )

    assert result.error_code is AnswerAgentSubtaskErrorCode.PROVIDER_ERROR
    assert result.provider_error_code is AnswerAgentClientErrorCode.REQUEST_TIMEOUT
    assert result.to_dict()["provider_error_code"] == "REQUEST_TIMEOUT"
    assert "private provider detail" not in str(result.to_dict())
    assert result.inference_step_count == 1
    assert result.steps == ()


def test_invalid_client_response_is_terminal_and_never_executed() -> None:
    client = _ScriptedClient([object()])
    executor = _ScriptedExecutor()

    result = _runner(client, executor).run(
        _subtask(),
        tool_definitions=(_definition(),),
    )

    assert result.error_code is AnswerAgentSubtaskErrorCode.INVALID_MODEL_RESPONSE
    assert result.inference_step_count == 1
    assert result.steps == ()
    assert executor.calls == []


@pytest.mark.parametrize(
    "definitions",
    [
        [_definition()],
        (object(),),
        (_definition(), _definition()),
    ],
)
def test_invalid_tool_catalog_fails_before_model_call(
    definitions: object,
) -> None:
    client = _ScriptedClient([AnswerAgentResponse(text="unused")])
    executor = _ScriptedExecutor()

    with pytest.raises(ValueError):
        _runner(client, executor).run(
            _subtask(),
            tool_definitions=definitions,  # type: ignore[arg-type]
        )

    assert client.calls == []
    assert executor.calls == []


@pytest.mark.parametrize(
    ("max_steps", "max_batch", "max_failures"),
    [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (True, 1, 1),
        (1, 1.5, 1),
        (1, 1, "2"),
    ],
)
def test_invalid_runner_budget_is_rejected(
    max_steps: object,
    max_batch: object,
    max_failures: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        AnswerAgentSubtaskRunnerConfig(
            max_inference_steps=max_steps,  # type: ignore[arg-type]
            max_tool_calls_per_inference=max_batch,  # type: ignore[arg-type]
            max_consecutive_tool_failures=max_failures,  # type: ignore[arg-type]
        )


def test_environment_mode_requires_configured_single_call_limit() -> None:
    with pytest.raises(ValueError, match="max_tool_calls_per_inference=1"):
        AnswerAgentSubtaskRunnerConfig(
            max_inference_steps=2,
            max_tool_calls_per_inference=2,
            max_consecutive_tool_failures=1,
            interaction_mode=(AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL),
        )


def test_runner_config_rejects_untyped_interaction_mode() -> None:
    with pytest.raises(ValueError, match="interaction_mode"):
        AnswerAgentSubtaskRunnerConfig(
            max_inference_steps=2,
            max_tool_calls_per_inference=1,
            max_consecutive_tool_failures=1,
            interaction_mode="TOOL_OR_FINAL",  # type: ignore[arg-type]
        )


def test_mismatched_first_observation_stops_before_later_execution() -> None:
    client = _ScriptedClient([_batch(_call("call-1"), _call("call-2"))])
    executor = _ScriptedExecutor(
        results=[
            ToolObservation(
                "wrong-id",
                "CitySearch",
                ToolObservationStatus.SUCCESS,
                "result",
            )
        ]
    )

    with pytest.raises(ValueError, match="must match"):
        _runner(client, executor, max_batch=2).run(
            _subtask(),
            tool_definitions=(_definition(),),
        )

    assert [call.tool_call_id for call in executor.calls] == ["call-1"]
