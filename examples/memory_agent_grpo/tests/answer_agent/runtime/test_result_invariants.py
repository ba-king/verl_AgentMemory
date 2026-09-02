import pytest

from examples.memory_agent_grpo.answer_agent.input import AnswerAgentInitialMessage
from examples.memory_agent_grpo.answer_agent.model_client import (
    AnswerAgentClientErrorCode,
)
from examples.memory_agent_grpo.answer_agent.response import (
    AnswerAgentResponse,
    AnswerAgentToolCall,
)
from examples.memory_agent_grpo.answer_agent.runtime import (
    AnswerAgentConversation,
    AnswerAgentInferenceStep,
    AnswerAgentInteractionMode,
    AnswerAgentSubtaskErrorCode,
    AnswerAgentSubtaskRunnerConfig,
    AnswerAgentSubtaskRunResult,
    AnswerAgentSubtaskRunStatus,
    AnswerAgentTaskToolExecutionResult,
    AnswerAgentToolCallDisposition,
    AnswerAgentToolCallOutcome,
    AnswerAgentToolExchange,
)
from examples.memory_agent_grpo.answer_agent.tool_call_gateway import (
    ToolCallGatewayErrorCode,
    ToolCallGatewayResult,
    ValidatedToolCall,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.observation import (
    ToolObservation,
    ToolObservationStatus,
)


def _initial_messages() -> tuple[
    AnswerAgentInitialMessage,
    AnswerAgentInitialMessage,
]:
    return (
        AnswerAgentInitialMessage("system", "system"),
        AnswerAgentInitialMessage("user", "user"),
    )


def _call(
    call_id: str,
    *,
    arguments_json: str = '{"state":"New York"}',
) -> AnswerAgentToolCall:
    return AnswerAgentToolCall(
        call_id,
        "CitySearch",
        arguments_json,
    )


def _response(*call_ids: str) -> AnswerAgentResponse:
    return AnswerAgentResponse(tool_calls=tuple(_call(call_id) for call_id in call_ids))


def _valid_gateway(
    tool_call: AnswerAgentToolCall,
    *,
    call_id: str | None = None,
    arguments_json: str | None = None,
) -> ToolCallGatewayResult:
    return ToolCallGatewayResult(
        validated_call=ValidatedToolCall(
            tool_call.tool_call_id if call_id is None else call_id,
            tool_call.name,
            (tool_call.arguments_json if arguments_json is None else arguments_json),
        )
    )


def _rejected_gateway() -> ToolCallGatewayResult:
    return ToolCallGatewayResult(
        error_code=ToolCallGatewayErrorCode.ARGUMENTS_DO_NOT_MATCH_SCHEMA,
        error_message="stable gateway rejection",
    )


def _observation(
    tool_call: AnswerAgentToolCall,
    *,
    call_id: str | None = None,
    status: ToolObservationStatus = ToolObservationStatus.SUCCESS,
) -> ToolObservation:
    return ToolObservation(
        tool_call.tool_call_id if call_id is None else call_id,
        tool_call.name,
        status,
        "public result",
    )


def _outcome(
    tool_call: AnswerAgentToolCall,
    *,
    disposition: AnswerAgentToolCallDisposition = (AnswerAgentToolCallDisposition.EXECUTED),
    status: ToolObservationStatus = ToolObservationStatus.SUCCESS,
    is_terminal: bool = False,
) -> AnswerAgentToolCallOutcome:
    gateway = (
        _rejected_gateway()
        if disposition is AnswerAgentToolCallDisposition.GATEWAY_REJECTED
        else _valid_gateway(tool_call)
    )
    return AnswerAgentToolCallOutcome(
        tool_call=tool_call,
        gateway_result=gateway,
        tool_observation=_observation(tool_call, status=status),
        disposition=disposition,
        is_terminal=is_terminal,
    )


def _tool_step(
    *call_ids: str,
    inference_step: int = 1,
    statuses: tuple[ToolObservationStatus, ...] | None = None,
    dispositions: tuple[AnswerAgentToolCallDisposition, ...] | None = None,
    terminal_positions: tuple[int, ...] = (),
) -> AnswerAgentInferenceStep:
    response = _response(*call_ids)
    if statuses is None:
        statuses = (ToolObservationStatus.SUCCESS,) * len(call_ids)
    if dispositions is None:
        dispositions = (AnswerAgentToolCallDisposition.EXECUTED,) * len(call_ids)
    assert len(statuses) == len(call_ids)
    assert len(dispositions) == len(call_ids)
    return AnswerAgentInferenceStep(
        inference_step=inference_step,
        response=response,
        tool_outcomes=tuple(
            _outcome(
                tool_call,
                disposition=disposition,
                status=status,
                is_terminal=position in terminal_positions,
            )
            for position, (tool_call, disposition, status) in enumerate(
                zip(
                    response.tool_calls,
                    dispositions,
                    statuses,
                    strict=True,
                )
            )
        ),
    )


def _exchange(step: AnswerAgentInferenceStep) -> AnswerAgentToolExchange:
    return AnswerAgentToolExchange(
        assistant_response=step.response,
        tool_observations=tuple(outcome.tool_observation for outcome in step.tool_outcomes),
    )


def _conversation(
    *steps: AnswerAgentInferenceStep,
) -> AnswerAgentConversation:
    return AnswerAgentConversation(
        _initial_messages(),
        tuple(_exchange(step) for step in steps if step.tool_outcomes),
    )


def _config(
    *,
    max_steps: int = 5,
    max_batch: int = 3,
    max_failures: int = 2,
    interaction_mode: AnswerAgentInteractionMode = (AnswerAgentInteractionMode.TOOL_OR_FINAL),
) -> AnswerAgentSubtaskRunnerConfig:
    return AnswerAgentSubtaskRunnerConfig(
        max_inference_steps=max_steps,
        max_tool_calls_per_inference=max_batch,
        max_consecutive_tool_failures=max_failures,
        interaction_mode=interaction_mode,
    )


def test_execution_result_validates_public_observation_and_terminal_flag() -> None:
    tool_call = _call("call-1")
    result = AnswerAgentTaskToolExecutionResult(
        tool_observation=_observation(tool_call),
        is_terminal=True,
    )

    assert result.is_terminal is True
    serialized_observation = result.to_dict()["tool_observation"]
    assert isinstance(serialized_observation, dict)
    assert serialized_observation["content"] == "public result"
    with pytest.raises(ValueError, match="is_terminal"):
        AnswerAgentTaskToolExecutionResult(
            tool_observation=_observation(tool_call),
            is_terminal=1,  # type: ignore[arg-type]
        )


def test_environment_terminated_result_requires_terminal_last_call() -> None:
    first = _tool_step("call-1")
    terminal = _tool_step(
        "call-2",
        inference_step=2,
        terminal_positions=(0,),
    )
    config = _config(
        max_batch=1,
        interaction_mode=(AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL),
    )

    result = AnswerAgentSubtaskRunResult(
        status=AnswerAgentSubtaskRunStatus.COMPLETED,
        config=config,
        conversation=_conversation(first, terminal),
        steps=(first, terminal),
        inference_step_count=2,
        executed_tool_count=2,
        final_answer="public result",
    )

    assert result.steps[-1].tool_outcomes[0].is_terminal is True
    serialized_config = result.to_dict()["config"]
    assert isinstance(serialized_config, dict)
    assert serialized_config["interaction_mode"] == ("ENVIRONMENT_TERMINATED_SINGLE_TOOL")


def test_default_result_rejects_forged_terminal_tool_outcome() -> None:
    terminal = _tool_step("call-1", terminal_positions=(0,))
    final = AnswerAgentInferenceStep(
        inference_step=2,
        response=AnswerAgentResponse(text="final answer"),
    )

    with pytest.raises(ValueError, match="environment-terminated mode"):
        AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.COMPLETED,
            config=_config(max_batch=1),
            conversation=_conversation(terminal),
            steps=(terminal, final),
            inference_step_count=2,
            executed_tool_count=1,
            final_answer="final answer",
        )


def test_outcome_rejects_gateway_identity_unrelated_to_model_call() -> None:
    tool_call = _call("model-id")

    with pytest.raises(ValueError, match="Validated call identity"):
        AnswerAgentToolCallOutcome(
            tool_call=tool_call,
            gateway_result=_valid_gateway(tool_call, call_id="other-id"),
            tool_observation=_observation(tool_call),
            disposition=AnswerAgentToolCallDisposition.EXECUTED,
        )


def test_outcome_rejects_gateway_arguments_unrelated_to_model_call() -> None:
    tool_call = _call("call-1")

    with pytest.raises(ValueError, match="canonical response arguments"):
        AnswerAgentToolCallOutcome(
            tool_call=tool_call,
            gateway_result=_valid_gateway(
                tool_call,
                arguments_json='{"state":"California"}',
            ),
            tool_observation=_observation(tool_call),
            disposition=AnswerAgentToolCallDisposition.EXECUTED,
        )


def test_outcome_rejects_observation_identity_unrelated_to_call() -> None:
    tool_call = _call("call-1")

    with pytest.raises(ValueError, match="observation identity"):
        AnswerAgentToolCallOutcome(
            tool_call=tool_call,
            gateway_result=_valid_gateway(tool_call),
            tool_observation=_observation(tool_call, call_id="other-id"),
            disposition=AnswerAgentToolCallDisposition.EXECUTED,
        )


@pytest.mark.parametrize(
    ("disposition", "gateway_valid", "status", "message"),
    [
        (
            AnswerAgentToolCallDisposition.EXECUTED,
            False,
            ToolObservationStatus.ERROR,
            "valid Gateway call",
        ),
        (
            AnswerAgentToolCallDisposition.GATEWAY_REJECTED,
            True,
            ToolObservationStatus.ERROR,
            "GATEWAY_REJECTED",
        ),
        (
            AnswerAgentToolCallDisposition.BATCH_SKIPPED,
            False,
            ToolObservationStatus.ERROR,
            "BATCH_SKIPPED",
        ),
        (
            AnswerAgentToolCallDisposition.BATCH_SKIPPED,
            True,
            ToolObservationStatus.SUCCESS,
            "non-executed",
        ),
    ],
)
def test_outcome_disposition_cannot_forge_gateway_or_observation_state(
    disposition: AnswerAgentToolCallDisposition,
    gateway_valid: bool,
    status: ToolObservationStatus,
    message: str,
) -> None:
    tool_call = _call("call-1")
    gateway = _valid_gateway(tool_call) if gateway_valid else _rejected_gateway()

    with pytest.raises(ValueError, match=message):
        AnswerAgentToolCallOutcome(
            tool_call=tool_call,
            gateway_result=gateway,
            tool_observation=_observation(tool_call, status=status),
            disposition=disposition,
        )


def test_step_rejects_partial_outcome_coverage() -> None:
    response = _response("call-1", "call-2")

    with pytest.raises(ValueError, match="cover every"):
        AnswerAgentInferenceStep(
            inference_step=1,
            response=response,
            tool_outcomes=(_outcome(response.tool_calls[0]),),
        )


def test_step_rejects_outcomes_in_a_different_order() -> None:
    response = _response("call-1", "call-2")

    with pytest.raises(ValueError, match="match response order"):
        AnswerAgentInferenceStep(
            inference_step=1,
            response=response,
            tool_outcomes=(
                _outcome(response.tool_calls[1]),
                _outcome(response.tool_calls[0]),
            ),
        )


def test_step_rejects_partial_execution_when_one_gateway_call_is_rejected() -> None:
    response = _response("rejected", "must-not-execute")

    with pytest.raises(ValueError, match="batch"):
        AnswerAgentInferenceStep(
            inference_step=1,
            response=response,
            tool_outcomes=(
                _outcome(
                    response.tool_calls[0],
                    disposition=(AnswerAgentToolCallDisposition.GATEWAY_REJECTED),
                    status=ToolObservationStatus.ERROR,
                ),
                _outcome(response.tool_calls[1]),
            ),
        )


def test_step_rejects_batch_skipped_without_any_gateway_rejection() -> None:
    response = _response("call-1", "call-2")

    with pytest.raises(ValueError, match="BATCH_SKIPPED"):
        AnswerAgentInferenceStep(
            inference_step=1,
            response=response,
            tool_outcomes=(
                _outcome(
                    response.tool_calls[0],
                    disposition=AnswerAgentToolCallDisposition.BATCH_SKIPPED,
                    status=ToolObservationStatus.ERROR,
                ),
                _outcome(response.tool_calls[1]),
            ),
        )


def test_result_rejects_conversation_unrelated_to_batched_step_trace() -> None:
    step = _tool_step("step-a", "step-b")
    unrelated = _tool_step("other-a", "other-b")

    with pytest.raises(ValueError, match="ordered step exchanges"):
        AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.FAILED,
            config=_config(max_steps=1),
            conversation=_conversation(unrelated),
            steps=(step,),
            inference_step_count=1,
            executed_tool_count=2,
            error_code=AnswerAgentSubtaskErrorCode.INFERENCE_STEP_LIMIT,
        )


def test_result_rejects_forged_executed_tool_count() -> None:
    step = _tool_step(
        "rejected",
        "skipped",
        dispositions=(
            AnswerAgentToolCallDisposition.GATEWAY_REJECTED,
            AnswerAgentToolCallDisposition.BATCH_SKIPPED,
        ),
        statuses=(ToolObservationStatus.ERROR, ToolObservationStatus.ERROR),
    )

    with pytest.raises(ValueError, match="executed_tool_count"):
        AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.FAILED,
            config=_config(max_steps=1, max_failures=1),
            conversation=_conversation(step),
            steps=(step,),
            inference_step_count=1,
            executed_tool_count=2,
            error_code=(AnswerAgentSubtaskErrorCode.CONSECUTIVE_TOOL_FAILURE_LIMIT),
        )


def test_completed_result_accepts_multiple_executions_in_one_inference() -> None:
    tool_step = _tool_step("call-1", "call-2")
    final_step = AnswerAgentInferenceStep(
        inference_step=2,
        response=AnswerAgentResponse(text="final answer"),
    )

    result = AnswerAgentSubtaskRunResult(
        status=AnswerAgentSubtaskRunStatus.COMPLETED,
        config=_config(max_batch=2),
        conversation=_conversation(tool_step),
        steps=(tool_step, final_step),
        inference_step_count=2,
        executed_tool_count=2,
        final_answer="final answer",
    )

    assert result.executed_tool_count == 2
    assert result.inference_step_count == 2
    assert [
        outcome["disposition"]
        for outcome in result.to_dict()["steps"][0]["tool_outcomes"]  # type: ignore[index,union-attr]
    ] == ["EXECUTED", "EXECUTED"]


def test_result_rejects_processed_batch_over_configured_limit() -> None:
    oversized = _tool_step("call-1", "call-2", "call-3")
    final_step = AnswerAgentInferenceStep(
        inference_step=2,
        response=AnswerAgentResponse(text="final answer"),
    )

    with pytest.raises(ValueError, match="max_tool_calls_per_inference"):
        AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.COMPLETED,
            config=_config(max_batch=2),
            conversation=_conversation(oversized),
            steps=(oversized, final_step),
            inference_step_count=2,
            executed_tool_count=3,
            final_answer="final answer",
        )


def test_result_rejects_final_answer_not_returned_by_terminal_response() -> None:
    final_step = AnswerAgentInferenceStep(
        inference_step=1,
        response=AnswerAgentResponse(text="actual model answer"),
    )

    with pytest.raises(ValueError, match="terminal text-only response"):
        AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.COMPLETED,
            config=_config(),
            conversation=AnswerAgentConversation(_initial_messages()),
            steps=(final_step,),
            inference_step_count=1,
            executed_tool_count=0,
            final_answer="forged answer",
        )


def test_result_rejects_completion_beyond_configured_step_budget() -> None:
    first = _tool_step("call-1", "call-2", inference_step=1)
    second = _tool_step("call-3", inference_step=2)
    final = AnswerAgentInferenceStep(
        inference_step=3,
        response=AnswerAgentResponse(text="actual model answer"),
    )

    with pytest.raises(ValueError, match="model-call attempt"):
        AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.COMPLETED,
            config=_config(max_steps=2),
            conversation=_conversation(first, second),
            steps=(first, second, final),
            inference_step_count=3,
            executed_tool_count=3,
            final_answer="actual model answer",
        )


def test_two_errors_in_one_batch_count_as_only_one_failure() -> None:
    step = _tool_step(
        "call-1",
        "call-2",
        statuses=(ToolObservationStatus.ERROR, ToolObservationStatus.ERROR),
    )

    with pytest.raises(ValueError, match="reached first at the terminal step"):
        AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.FAILED,
            config=_config(max_steps=1, max_failures=2),
            conversation=_conversation(step),
            steps=(step,),
            inference_step_count=1,
            executed_tool_count=2,
            error_code=(AnswerAgentSubtaskErrorCode.CONSECUTIVE_TOOL_FAILURE_LIMIT),
        )


def test_two_failed_batches_can_reach_failure_limit() -> None:
    first = _tool_step(
        "batch-1-a",
        "batch-1-b",
        statuses=(ToolObservationStatus.ERROR, ToolObservationStatus.ERROR),
    )
    second = _tool_step(
        "batch-2",
        inference_step=2,
        statuses=(ToolObservationStatus.ERROR,),
    )

    result = AnswerAgentSubtaskRunResult(
        status=AnswerAgentSubtaskRunStatus.FAILED,
        config=_config(max_steps=2, max_failures=2),
        conversation=_conversation(first, second),
        steps=(first, second),
        inference_step_count=2,
        executed_tool_count=3,
        error_code=AnswerAgentSubtaskErrorCode.CONSECUTIVE_TOOL_FAILURE_LIMIT,
    )

    assert result.error_code is (AnswerAgentSubtaskErrorCode.CONSECUTIVE_TOOL_FAILURE_LIMIT)


def test_successful_batch_resets_failure_sequence_in_result_contract() -> None:
    first_failed = _tool_step(
        "failed-1",
        statuses=(ToolObservationStatus.ERROR,),
    )
    success = _tool_step("success", inference_step=2)
    second_failed = _tool_step(
        "failed-2",
        inference_step=3,
        statuses=(ToolObservationStatus.ERROR,),
    )

    with pytest.raises(ValueError, match="reached first at the terminal step"):
        AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.FAILED,
            config=_config(max_steps=3, max_failures=2),
            conversation=_conversation(first_failed, success, second_failed),
            steps=(first_failed, success, second_failed),
            inference_step_count=3,
            executed_tool_count=3,
            error_code=(AnswerAgentSubtaskErrorCode.CONSECUTIVE_TOOL_FAILURE_LIMIT),
        )


def test_result_rejects_continuing_after_batch_failure_limit() -> None:
    first = _tool_step(
        "failed-1",
        statuses=(ToolObservationStatus.ERROR,),
    )
    second = _tool_step(
        "failed-2",
        inference_step=2,
        statuses=(ToolObservationStatus.ERROR,),
    )
    third = _tool_step("must-not-run", inference_step=3)

    with pytest.raises(ValueError, match="cannot continue past"):
        AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.FAILED,
            config=_config(max_steps=3, max_failures=2),
            conversation=_conversation(first, second, third),
            steps=(first, second, third),
            inference_step_count=3,
            executed_tool_count=3,
            error_code=AnswerAgentSubtaskErrorCode.INFERENCE_STEP_LIMIT,
        )


def test_batch_limit_terminal_result_requires_an_oversized_unprocessed_batch() -> None:
    terminal = AnswerAgentInferenceStep(
        inference_step=1,
        response=_response("call-1", "call-2", "call-3"),
    )

    result = AnswerAgentSubtaskRunResult(
        status=AnswerAgentSubtaskRunStatus.FAILED,
        config=_config(max_steps=1, max_batch=2),
        conversation=AnswerAgentConversation(_initial_messages()),
        steps=(terminal,),
        inference_step_count=1,
        executed_tool_count=0,
        error_code=AnswerAgentSubtaskErrorCode.TOOL_CALL_BATCH_LIMIT,
    )

    assert result.error_code is AnswerAgentSubtaskErrorCode.TOOL_CALL_BATCH_LIMIT


def test_batch_limit_claim_is_rejected_at_or_below_the_limit() -> None:
    terminal = AnswerAgentInferenceStep(
        inference_step=1,
        response=_response("call-1", "call-2"),
    )

    with pytest.raises(ValueError, match="oversized terminal batch"):
        AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.FAILED,
            config=_config(max_steps=1, max_batch=2),
            conversation=AnswerAgentConversation(_initial_messages()),
            steps=(terminal,),
            inference_step_count=1,
            executed_tool_count=0,
            error_code=AnswerAgentSubtaskErrorCode.TOOL_CALL_BATCH_LIMIT,
        )


def test_duplicate_terminal_batch_may_match_any_prior_batch_position() -> None:
    first = _tool_step("prior-a", "shared")
    terminal = AnswerAgentInferenceStep(
        inference_step=2,
        response=_response("new-sibling", "shared"),
    )

    result = AnswerAgentSubtaskRunResult(
        status=AnswerAgentSubtaskRunStatus.FAILED,
        config=_config(max_steps=2, max_batch=2),
        conversation=_conversation(first),
        steps=(first, terminal),
        inference_step_count=2,
        executed_tool_count=2,
        error_code=AnswerAgentSubtaskErrorCode.DUPLICATE_TOOL_CALL_ID,
    )

    assert result.error_code is AnswerAgentSubtaskErrorCode.DUPLICATE_TOOL_CALL_ID


def test_duplicate_error_without_prior_matching_id_is_rejected() -> None:
    terminal = AnswerAgentInferenceStep(
        inference_step=1,
        response=_response("new-a", "new-b"),
    )

    with pytest.raises(ValueError, match="previously used"):
        AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.FAILED,
            config=_config(max_steps=1, max_batch=2),
            conversation=AnswerAgentConversation(_initial_messages()),
            steps=(terminal,),
            inference_step_count=1,
            executed_tool_count=0,
            error_code=AnswerAgentSubtaskErrorCode.DUPLICATE_TOOL_CALL_ID,
        )


def test_inference_limit_accepts_a_complete_multi_call_final_batch() -> None:
    step = _tool_step("call-1", "call-2")

    result = AnswerAgentSubtaskRunResult(
        status=AnswerAgentSubtaskRunStatus.FAILED,
        config=_config(max_steps=1, max_batch=2),
        conversation=_conversation(step),
        steps=(step,),
        inference_step_count=1,
        executed_tool_count=2,
        error_code=AnswerAgentSubtaskErrorCode.INFERENCE_STEP_LIMIT,
    )

    assert result.inference_step_count == 1
    assert result.executed_tool_count == 2


def test_inference_limit_claim_before_budget_is_rejected() -> None:
    step = _tool_step("call-1", "call-2")

    with pytest.raises(ValueError, match="exhausting the step budget"):
        AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.FAILED,
            config=_config(max_steps=2, max_batch=2),
            conversation=_conversation(step),
            steps=(step,),
            inference_step_count=1,
            executed_tool_count=2,
            error_code=AnswerAgentSubtaskErrorCode.INFERENCE_STEP_LIMIT,
        )


@pytest.mark.parametrize(
    "provider_error_code",
    ["private provider detail", " ", "bad\ud800", "REQUEST_TIMEOUT"],
)
def test_provider_failure_rejects_untyped_error_text(
    provider_error_code: object,
) -> None:
    with pytest.raises(ValueError, match="AnswerAgentClientErrorCode"):
        AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.FAILED,
            config=_config(),
            conversation=AnswerAgentConversation(_initial_messages()),
            steps=(),
            inference_step_count=1,
            executed_tool_count=0,
            error_code=AnswerAgentSubtaskErrorCode.PROVIDER_ERROR,
            provider_error_code=provider_error_code,  # type: ignore[arg-type]
        )


def test_provider_failure_serializes_only_stable_enum_value() -> None:
    result = AnswerAgentSubtaskRunResult(
        status=AnswerAgentSubtaskRunStatus.FAILED,
        config=_config(),
        conversation=AnswerAgentConversation(_initial_messages()),
        steps=(),
        inference_step_count=1,
        executed_tool_count=0,
        error_code=AnswerAgentSubtaskErrorCode.PROVIDER_ERROR,
        provider_error_code=AnswerAgentClientErrorCode.REQUEST_TIMEOUT,
    )

    assert result.to_dict()["provider_error_code"] == "REQUEST_TIMEOUT"
