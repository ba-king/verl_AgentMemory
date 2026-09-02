"""Dataset-neutral loop for one fixed Answer Agent subtask."""

from __future__ import annotations

from typing import Protocol

from ..input import AnswerAgentSubtaskInput
from ..interaction_mode import AnswerAgentInteractionMode
from ..model_client import (
    AnswerAgentClientError,
    AnswerAgentConversationClient,
)
from ..prompts.builder import build_answer_agent_initial_messages
from ..response import AnswerAgentResponse, AnswerAgentToolCall
from ..tool_call_gateway import (
    ToolCallGatewayResult,
    ValidatedToolCall,
    validate_tool_call_for_execution,
)
from ..tool_protocol.definition import ToolDefinition
from ..tool_protocol.observation import ToolObservation, ToolObservationStatus
from ..tool_protocol.validation import validate_tool_definition
from ..validation import validate_answer_agent_response
from .config import AnswerAgentSubtaskRunnerConfig
from .conversation import AnswerAgentConversation, AnswerAgentToolExchange
from .result import (
    AnswerAgentInferenceStep,
    AnswerAgentSubtaskErrorCode,
    AnswerAgentSubtaskRunResult,
    AnswerAgentSubtaskRunStatus,
    AnswerAgentTaskToolExecutionResult,
    AnswerAgentToolCallDisposition,
    AnswerAgentToolCallOutcome,
)


class AnswerAgentTaskToolExecutor(Protocol):
    """Execute one Gateway-approved task call and return public evidence."""

    def execute(
        self,
        call: ValidatedToolCall,
    ) -> ToolObservation | AnswerAgentTaskToolExecutionResult: ...


def _validate_tool_definitions(
    tool_definitions: object,
) -> tuple[ToolDefinition, ...]:
    if type(tool_definitions) is not tuple:
        raise ValueError("tool_definitions must be a tuple.")
    seen_names: set[str] = set()
    for position, definition in enumerate(tool_definitions):
        if type(definition) is not ToolDefinition:
            raise ValueError(f"tool_definitions[{position}] must be a ToolDefinition.")
        validation = validate_tool_definition(definition)
        if not validation.is_valid:
            code = "UNKNOWN" if validation.error_code is None else validation.error_code.value
            raise ValueError(f"tool_definitions[{position}] is invalid ({code}).")
        if definition.name in seen_names:
            raise ValueError("tool definition names must be unique.")
        seen_names.add(definition.name)
    return tool_definitions


def _gateway_error_observation(
    tool_call: AnswerAgentToolCall,
    gateway_result: ToolCallGatewayResult,
) -> ToolObservation:
    if gateway_result.is_valid or gateway_result.error_code is None:
        raise RuntimeError("Gateway error observation requires a rejected call.")
    return ToolObservation(
        tool_call_id=tool_call.tool_call_id,
        tool_name=tool_call.name,
        status=ToolObservationStatus.ERROR,
        content=(
            f"Tool call rejected by validation ({gateway_result.error_code.value}). Correct the call and try again."
        ),
    )


def _batch_skipped_observation(
    tool_call: AnswerAgentToolCall,
) -> ToolObservation:
    return ToolObservation(
        tool_call_id=tool_call.tool_call_id,
        tool_name=tool_call.name,
        status=ToolObservationStatus.ERROR,
        content=(
            "Tool call skipped because another call in the same batch was rejected. Correct the batch and try again."
        ),
    )


class AnswerAgentSubtaskRunner:
    """Run only the active subtask; retain no cross-subtask raw history."""

    def __init__(
        self,
        *,
        client: AnswerAgentConversationClient,
        executor: AnswerAgentTaskToolExecutor,
        config: AnswerAgentSubtaskRunnerConfig,
    ) -> None:
        if not callable(getattr(client, "generate", None)):
            raise ValueError("client must expose a callable generate method.")
        if not callable(getattr(executor, "execute", None)):
            raise ValueError("executor must expose a callable execute method.")
        if type(config) is not AnswerAgentSubtaskRunnerConfig:
            raise ValueError("config must be an AnswerAgentSubtaskRunnerConfig.")
        self._client = client
        self._executor = executor
        self._config = config

    @property
    def config(self) -> AnswerAgentSubtaskRunnerConfig:
        return self._config

    def run(
        self,
        subtask: AnswerAgentSubtaskInput,
        *,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> AnswerAgentSubtaskRunResult:
        definitions = _validate_tool_definitions(tool_definitions)
        conversation = AnswerAgentConversation(
            build_answer_agent_initial_messages(
                subtask,
                interaction_mode=self._config.interaction_mode,
            )
        )
        steps: list[AnswerAgentInferenceStep] = []
        executed_tool_count = 0
        consecutive_tool_failures = 0
        seen_call_ids: set[str] = set()

        for inference_step in range(1, self._config.max_inference_steps + 1):
            try:
                response = self._client.generate(conversation, definitions)
            except AnswerAgentClientError as error:
                return AnswerAgentSubtaskRunResult(
                    status=AnswerAgentSubtaskRunStatus.FAILED,
                    config=self._config,
                    conversation=conversation,
                    steps=tuple(steps),
                    inference_step_count=inference_step,
                    executed_tool_count=executed_tool_count,
                    error_code=AnswerAgentSubtaskErrorCode.PROVIDER_ERROR,
                    provider_error_code=error.code,
                )

            response_validation = validate_answer_agent_response(response)
            if type(response) is not AnswerAgentResponse or not response_validation.is_valid:
                return AnswerAgentSubtaskRunResult(
                    status=AnswerAgentSubtaskRunStatus.FAILED,
                    config=self._config,
                    conversation=conversation,
                    steps=tuple(steps),
                    inference_step_count=inference_step,
                    executed_tool_count=executed_tool_count,
                    error_code=(AnswerAgentSubtaskErrorCode.INVALID_MODEL_RESPONSE),
                )

            environment_terminated_mode = (
                self._config.interaction_mode is AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL
            )
            if not response.tool_calls and environment_terminated_mode:
                steps.append(
                    AnswerAgentInferenceStep(
                        inference_step=inference_step,
                        response=response,
                    )
                )
                return AnswerAgentSubtaskRunResult(
                    status=AnswerAgentSubtaskRunStatus.FAILED,
                    config=self._config,
                    conversation=conversation,
                    steps=tuple(steps),
                    inference_step_count=inference_step,
                    executed_tool_count=executed_tool_count,
                    error_code=(AnswerAgentSubtaskErrorCode.INTERACTION_MODE_VIOLATION),
                )

            if not response.tool_calls:
                step = AnswerAgentInferenceStep(
                    inference_step=inference_step,
                    response=response,
                )
                steps.append(step)
                assert response.text is not None
                return AnswerAgentSubtaskRunResult(
                    status=AnswerAgentSubtaskRunStatus.COMPLETED,
                    config=self._config,
                    conversation=conversation,
                    steps=tuple(steps),
                    inference_step_count=inference_step,
                    executed_tool_count=executed_tool_count,
                    final_answer=response.text,
                )

            if len(response.tool_calls) > self._config.max_tool_calls_per_inference:
                steps.append(
                    AnswerAgentInferenceStep(
                        inference_step=inference_step,
                        response=response,
                    )
                )
                return AnswerAgentSubtaskRunResult(
                    status=AnswerAgentSubtaskRunStatus.FAILED,
                    config=self._config,
                    conversation=conversation,
                    steps=tuple(steps),
                    inference_step_count=inference_step,
                    executed_tool_count=executed_tool_count,
                    error_code=(AnswerAgentSubtaskErrorCode.TOOL_CALL_BATCH_LIMIT),
                )

            if any(tool_call.tool_call_id in seen_call_ids for tool_call in response.tool_calls):
                steps.append(
                    AnswerAgentInferenceStep(
                        inference_step=inference_step,
                        response=response,
                    )
                )
                return AnswerAgentSubtaskRunResult(
                    status=AnswerAgentSubtaskRunStatus.FAILED,
                    config=self._config,
                    conversation=conversation,
                    steps=tuple(steps),
                    inference_step_count=inference_step,
                    executed_tool_count=executed_tool_count,
                    error_code=(AnswerAgentSubtaskErrorCode.DUPLICATE_TOOL_CALL_ID),
                )

            gateway_results = tuple(
                validate_tool_call_for_execution(
                    tool_call,
                    tool_definitions=definitions,
                )
                for tool_call in response.tool_calls
            )
            batch_is_valid = all(gateway_result.is_valid for gateway_result in gateway_results)
            outcomes: list[AnswerAgentToolCallOutcome] = []
            for tool_call, gateway_result in zip(
                response.tool_calls,
                gateway_results,
                strict=True,
            ):
                if not batch_is_valid:
                    if gateway_result.is_valid:
                        observation = _batch_skipped_observation(tool_call)
                        disposition = AnswerAgentToolCallDisposition.BATCH_SKIPPED
                    else:
                        observation = _gateway_error_observation(
                            tool_call,
                            gateway_result,
                        )
                        disposition = AnswerAgentToolCallDisposition.GATEWAY_REJECTED
                else:
                    validated_call = gateway_result.validated_call
                    assert validated_call is not None
                    execution_result = self._executor.execute(validated_call)
                    if environment_terminated_mode:
                        if type(execution_result) is not (AnswerAgentTaskToolExecutionResult):
                            raise ValueError(
                                "ENVIRONMENT_TERMINATED_SINGLE_TOOL executor must "
                                "return an AnswerAgentTaskToolExecutionResult."
                            )
                        observation = execution_result.tool_observation
                        is_terminal = execution_result.is_terminal
                    else:
                        if type(execution_result) is AnswerAgentTaskToolExecutionResult:
                            if execution_result.is_terminal:
                                raise ValueError("TOOL_OR_FINAL executor must not report terminal.")
                            observation = execution_result.tool_observation
                        else:
                            observation = execution_result
                        is_terminal = False
                    executed_tool_count += 1
                    disposition = AnswerAgentToolCallDisposition.EXECUTED
                outcomes.append(
                    AnswerAgentToolCallOutcome(
                        tool_call=tool_call,
                        gateway_result=gateway_result,
                        tool_observation=observation,
                        disposition=disposition,
                        is_terminal=(is_terminal if disposition is AnswerAgentToolCallDisposition.EXECUTED else False),
                    )
                )

            outcome_tuple = tuple(outcomes)

            exchange = AnswerAgentToolExchange(
                assistant_response=response,
                tool_observations=tuple(outcome.tool_observation for outcome in outcome_tuple),
            )
            conversation = conversation.append_tool_exchange(exchange)
            seen_call_ids.update(tool_call.tool_call_id for tool_call in response.tool_calls)
            steps.append(
                AnswerAgentInferenceStep(
                    inference_step=inference_step,
                    response=response,
                    tool_outcomes=outcome_tuple,
                )
            )

            terminal_outcomes = tuple(outcome for outcome in outcome_tuple if outcome.is_terminal)
            if terminal_outcomes:
                assert len(terminal_outcomes) == 1
                return AnswerAgentSubtaskRunResult(
                    status=AnswerAgentSubtaskRunStatus.COMPLETED,
                    config=self._config,
                    conversation=conversation,
                    steps=tuple(steps),
                    inference_step_count=inference_step,
                    executed_tool_count=executed_tool_count,
                    final_answer=(terminal_outcomes[0].tool_observation.content),
                )

            batch_failed = any(
                outcome.tool_observation.status is ToolObservationStatus.ERROR for outcome in outcome_tuple
            )
            consecutive_tool_failures = consecutive_tool_failures + 1 if batch_failed else 0

            if consecutive_tool_failures >= self._config.max_consecutive_tool_failures:
                return AnswerAgentSubtaskRunResult(
                    status=AnswerAgentSubtaskRunStatus.FAILED,
                    config=self._config,
                    conversation=conversation,
                    steps=tuple(steps),
                    inference_step_count=inference_step,
                    executed_tool_count=executed_tool_count,
                    error_code=(AnswerAgentSubtaskErrorCode.CONSECUTIVE_TOOL_FAILURE_LIMIT),
                )

        return AnswerAgentSubtaskRunResult(
            status=AnswerAgentSubtaskRunStatus.FAILED,
            config=self._config,
            conversation=conversation,
            steps=tuple(steps),
            inference_step_count=self._config.max_inference_steps,
            executed_tool_count=executed_tool_count,
            error_code=AnswerAgentSubtaskErrorCode.INFERENCE_STEP_LIMIT,
        )


__all__ = ["AnswerAgentSubtaskRunner", "AnswerAgentTaskToolExecutor"]
