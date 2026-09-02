"""Dataset-neutral orchestration for one isolated memory episode."""

from __future__ import annotations

from typing import Protocol

from examples.memory_agent_grpo.answer_agent.input import (
    AnswerAgentSubtaskInput,
)
from examples.memory_agent_grpo.answer_agent.model_client import (
    AnswerAgentClientErrorCode,
)
from examples.memory_agent_grpo.answer_agent.runtime.result import (
    AnswerAgentSubtaskErrorCode,
    AnswerAgentSubtaskRunResult,
    AnswerAgentSubtaskRunStatus,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.definition import (
    ToolDefinition,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.validation import (
    validate_tool_definition,
)
from examples.memory_agent_grpo.memory_agent.action.protocol import (
    MemoryActionName,
    MemoryDecisionPhase,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import (
    MemoryBankState,
)
from examples.memory_agent_grpo.memory_agent.decision.action_gateway import (
    ActionToExecutorGateway,
)
from examples.memory_agent_grpo.memory_agent.decision.context import (
    InteractionEvent,
    MemoryDecisionContext,
    TaskResult,
)
from examples.memory_agent_grpo.memory_agent.decision.model_client import (
    MemoryAgentClient,
)
from examples.memory_agent_grpo.memory_agent.executor.engine import (
    MemoryActionExecutor,
)
from examples.memory_agent_grpo.memory_agent.retrieval.result import (
    RetrievalResult,
)

from ..input import EpisodeInput, EpisodeSubtaskInput
from ..result import (
    EpisodeFailureDetailCode,
    EpisodeFailureStage,
    EpisodeRunErrorCode,
    EpisodeRunResult,
    EpisodeRunStatus,
)
from ..trace import (
    EpisodeInitializationTrace,
    EpisodeSubtaskTrace,
    EpisodeTrace,
    MemoryDecisionTrace,
    build_subtask_interaction_events,
)
from ..validation import (
    validate_episode_input,
    validate_episode_run_result,
    validate_episode_subtask_trace,
    validate_initialization_trace,
)
from .memory_decision import (
    MemoryDecisionStepError,
    MemoryDecisionStepErrorCode,
    execute_memory_decision,
)


class EpisodeAnswerRunner(Protocol):
    """The current-subtask Answer Agent boundary required by EpisodeRunner."""

    def run(
        self,
        subtask: AnswerAgentSubtaskInput,
        *,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> AnswerAgentSubtaskRunResult: ...


class EpisodeRunnerInputError(ValueError):
    """Stable caller error raised before any episode side effect."""


class _EpisodeStageFailed(RuntimeError):
    """Typed, secret-free evidence for one uncommitted stage failure."""

    def __init__(
        self,
        *,
        stage: EpisodeFailureStage,
        detail_code: EpisodeFailureDetailCode,
        answer_provider_error_code: AnswerAgentClientErrorCode | None = None,
    ) -> None:
        self.stage = stage
        self.detail_code = detail_code
        self.answer_provider_error_code = answer_provider_error_code
        super().__init__(f"{stage.value}:{detail_code.value}")


_MEMORY_FAILURE_DETAILS = {
    MemoryDecisionStepErrorCode.MODEL_CALL_FAILED: (
        EpisodeFailureDetailCode.MEMORY_MODEL_CALL_FAILED
    ),
    MemoryDecisionStepErrorCode.OUTPUT_VALIDATION_FAILED: (
        EpisodeFailureDetailCode.MEMORY_OUTPUT_VALIDATION_FAILED
    ),
    MemoryDecisionStepErrorCode.EXECUTION_REJECTED: (
        EpisodeFailureDetailCode.MEMORY_EXECUTION_REJECTED
    ),
    MemoryDecisionStepErrorCode.INTERNAL_CONTRACT_ERROR: (
        EpisodeFailureDetailCode.MEMORY_INTERNAL_CONTRACT_ERROR
    ),
}

_ANSWER_FAILURE_DETAILS = {
    AnswerAgentSubtaskErrorCode.PROVIDER_ERROR: (
        EpisodeFailureDetailCode.ANSWER_PROVIDER_ERROR
    ),
    AnswerAgentSubtaskErrorCode.INVALID_MODEL_RESPONSE: (
        EpisodeFailureDetailCode.ANSWER_INVALID_MODEL_RESPONSE
    ),
    AnswerAgentSubtaskErrorCode.TOOL_CALL_BATCH_LIMIT: (
        EpisodeFailureDetailCode.ANSWER_TOOL_CALL_BATCH_LIMIT
    ),
    AnswerAgentSubtaskErrorCode.DUPLICATE_TOOL_CALL_ID: (
        EpisodeFailureDetailCode.ANSWER_DUPLICATE_TOOL_CALL_ID
    ),
    AnswerAgentSubtaskErrorCode.CONSECUTIVE_TOOL_FAILURE_LIMIT: (
        EpisodeFailureDetailCode.ANSWER_CONSECUTIVE_TOOL_FAILURE_LIMIT
    ),
    AnswerAgentSubtaskErrorCode.INFERENCE_STEP_LIMIT: (
        EpisodeFailureDetailCode.ANSWER_INFERENCE_STEP_LIMIT
    ),
}

_INTERNAL_FAILURE_DETAILS = {
    EpisodeFailureDetailCode.MEMORY_INTERNAL_CONTRACT_ERROR,
    EpisodeFailureDetailCode.RETRIEVAL_CONTRACT_ERROR,
    EpisodeFailureDetailCode.ANSWER_RESULT_CONTRACT_ERROR,
    EpisodeFailureDetailCode.EPISODE_TRACE_CONTRACT_ERROR,
    EpisodeFailureDetailCode.UNEXPECTED_EXCEPTION,
}


def _validated_tool_definitions(
    value: object,
) -> tuple[ToolDefinition, ...]:
    if type(value) is not tuple:
        raise EpisodeRunnerInputError("tool_definitions must be a tuple.")
    definitions: tuple[ToolDefinition, ...] = value
    seen_names: set[str] = set()
    for position, definition in enumerate(definitions):
        if type(definition) is not ToolDefinition:
            raise EpisodeRunnerInputError(
                f"tool_definitions[{position}] must be a ToolDefinition."
            )
        validation = validate_tool_definition(definition)
        if not validation.is_valid:
            code = (
                "UNKNOWN"
                if validation.error_code is None
                else validation.error_code.value
            )
            raise EpisodeRunnerInputError(
                f"tool_definitions[{position}] is invalid ({code})."
            )
        if definition.name in seen_names:
            raise EpisodeRunnerInputError("tool_definitions must use unique names.")
        seen_names.add(definition.name)
    return definitions


def _retrieval_from_trace(
    trace: MemoryDecisionTrace,
) -> RetrievalResult | None:
    decision = trace.gateway_result.validation.decision
    if decision is None or decision.memory_action is None:
        return None
    if decision.memory_action.action is not MemoryActionName.RETRIEVE:
        return None
    execution = trace.gateway_result.execution
    if execution is None or type(execution.retrieval) is not RetrievalResult:
        raise _EpisodeStageFailed(
            stage=EpisodeFailureStage.INTERNAL,
            detail_code=EpisodeFailureDetailCode.RETRIEVAL_CONTRACT_ERROR,
        )
    return execution.retrieval


def _is_internal_memory_failure(error: MemoryDecisionStepError) -> bool:
    return error.code is MemoryDecisionStepErrorCode.INTERNAL_CONTRACT_ERROR


def _execute_memory_stage(
    *,
    memory_client: MemoryAgentClient,
    gateway: ActionToExecutorGateway,
    context: MemoryDecisionContext,
    stage: EpisodeFailureStage,
) -> MemoryDecisionTrace:
    try:
        return execute_memory_decision(
            memory_client=memory_client,
            gateway=gateway,
            context=context,
        )
    except MemoryDecisionStepError as error:
        detail_code = _MEMORY_FAILURE_DETAILS[error.code]
        failure_stage = (
            EpisodeFailureStage.INTERNAL
            if _is_internal_memory_failure(error)
            else stage
        )
        raise _EpisodeStageFailed(
            stage=failure_stage,
            detail_code=detail_code,
        ) from None


def _answer_failure(
    result: AnswerAgentSubtaskRunResult,
) -> _EpisodeStageFailed:
    error_code = result.error_code
    if error_code is None or error_code not in _ANSWER_FAILURE_DETAILS:
        return _EpisodeStageFailed(
            stage=EpisodeFailureStage.INTERNAL,
            detail_code=(EpisodeFailureDetailCode.ANSWER_RESULT_CONTRACT_ERROR),
        )
    return _EpisodeStageFailed(
        stage=EpisodeFailureStage.ANSWER,
        detail_code=_ANSWER_FAILURE_DETAILS[error_code],
        answer_provider_error_code=result.provider_error_code,
    )


def _coarse_error_code(
    failure: _EpisodeStageFailed,
) -> EpisodeRunErrorCode:
    if failure.detail_code in _INTERNAL_FAILURE_DETAILS:
        return EpisodeRunErrorCode.INTERNAL_CONTRACT_ERROR
    if failure.stage is EpisodeFailureStage.INITIALIZATION:
        return EpisodeRunErrorCode.INITIALIZATION_FAILED
    return EpisodeRunErrorCode.SUBTASK_FAILED


class EpisodeRunner:
    """Run one EpisodeInput with a fresh Bank and current-only conversations."""

    def __init__(
        self,
        *,
        memory_client: MemoryAgentClient,
        answer_runner: EpisodeAnswerRunner,
        tool_definitions: object = (),
    ) -> None:
        if not callable(getattr(memory_client, "generate", None)):
            raise EpisodeRunnerInputError(
                "memory_client must expose a callable generate method."
            )
        if not callable(getattr(answer_runner, "run", None)):
            raise EpisodeRunnerInputError(
                "answer_runner must expose a callable run method."
            )
        self._memory_client = memory_client
        self._answer_runner = answer_runner
        self._tool_definitions = _validated_tool_definitions(tool_definitions)

    @property
    def tool_definitions(self) -> tuple[ToolDefinition, ...]:
        return self._tool_definitions

    def run(self, episode_input: object) -> EpisodeRunResult:
        """Run one complete episode or return its last committed prefix."""

        input_validation = validate_episode_input(episode_input)
        if not input_validation.is_valid or type(episode_input) is not EpisodeInput:
            code = (
                "UNKNOWN"
                if input_validation.error_code is None
                else input_validation.error_code.value
            )
            raise EpisodeRunnerInputError(
                f"episode_input violates the EpisodeInput contract ({code})."
            )

        executor = MemoryActionExecutor()
        gateway = ActionToExecutorGateway(executor=executor)
        initial_bank = gateway.state

        try:
            initialization = self._run_initialization(
                episode_input,
                gateway=gateway,
                initial_bank=initial_bank,
            )
        except _EpisodeStageFailed as failure:
            return self._failed_result(
                episode_input=episode_input,
                initialization=None,
                completed_subtasks=(),
                initial_bank=initial_bank,
                error_code=_coarse_error_code(failure),
                failed_subtask_ordinal=None,
                failure=failure,
            )
        except Exception:
            failure = _EpisodeStageFailed(
                stage=EpisodeFailureStage.INTERNAL,
                detail_code=EpisodeFailureDetailCode.UNEXPECTED_EXCEPTION,
            )
            return self._failed_result(
                episode_input=episode_input,
                initialization=None,
                completed_subtasks=(),
                initial_bank=initial_bank,
                error_code=EpisodeRunErrorCode.INTERNAL_CONTRACT_ERROR,
                failed_subtask_ordinal=None,
                failure=failure,
            )

        completed_subtasks: list[EpisodeSubtaskTrace] = []
        for subtask in episode_input.subtasks:
            try:
                subtask_trace = self._run_subtask(
                    subtask,
                    gateway=gateway,
                )
            except _EpisodeStageFailed as failure:
                return self._failed_result(
                    episode_input=episode_input,
                    initialization=initialization,
                    completed_subtasks=tuple(completed_subtasks),
                    initial_bank=initial_bank,
                    error_code=_coarse_error_code(failure),
                    failed_subtask_ordinal=subtask.ordinal,
                    failure=failure,
                )
            except Exception:
                failure = _EpisodeStageFailed(
                    stage=EpisodeFailureStage.INTERNAL,
                    detail_code=(EpisodeFailureDetailCode.UNEXPECTED_EXCEPTION),
                )
                return self._failed_result(
                    episode_input=episode_input,
                    initialization=initialization,
                    completed_subtasks=tuple(completed_subtasks),
                    initial_bank=initial_bank,
                    error_code=EpisodeRunErrorCode.INTERNAL_CONTRACT_ERROR,
                    failed_subtask_ordinal=subtask.ordinal,
                    failure=failure,
                )
            completed_subtasks.append(subtask_trace)

        trace = EpisodeTrace(
            initial_bank_state=initial_bank,
            initialization=initialization,
            subtasks=tuple(completed_subtasks),
            final_bank_state=gateway.state,
        )
        result = EpisodeRunResult(
            status=EpisodeRunStatus.COMPLETED,
            episode_input=episode_input,
            trace=trace,
            completed_subtask_count=len(completed_subtasks),
            final_bank_state=gateway.state,
        )
        self._require_valid_result(result)
        return result

    def _run_initialization(
        self,
        episode_input: EpisodeInput,
        *,
        gateway: ActionToExecutorGateway,
        initial_bank: MemoryBankState,
    ) -> EpisodeInitializationTrace:
        seed = episode_input.initial_completed_task
        decisions = self._run_post_task_decisions(
            task_input=seed.task_input,
            task_result=seed.task_result,
            interaction_events=seed.interaction_events,
            pre_task_retrieval=None,
            gateway=gateway,
            failure_stage=EpisodeFailureStage.INITIALIZATION,
        )
        trace = EpisodeInitializationTrace(
            seed=seed,
            post_task_decisions=decisions,
            bank_before=initial_bank,
            bank_after=gateway.state,
        )
        validation = validate_initialization_trace(trace)
        if not validation.is_valid:
            raise _EpisodeStageFailed(
                stage=EpisodeFailureStage.INTERNAL,
                detail_code=(EpisodeFailureDetailCode.EPISODE_TRACE_CONTRACT_ERROR),
            )
        return trace

    def _run_subtask(
        self,
        subtask: EpisodeSubtaskInput,
        *,
        gateway: ActionToExecutorGateway,
    ) -> EpisodeSubtaskTrace:
        bank_before = gateway.state
        pre_context = MemoryDecisionContext(
            phase=MemoryDecisionPhase.PRE_TASK,
            task_input=subtask.memory_task_input,
            bank_record_count=bank_before.record_count,
        )
        pre_task_decision = _execute_memory_stage(
            memory_client=self._memory_client,
            gateway=gateway,
            context=pre_context,
            stage=EpisodeFailureStage.PRE_TASK,
        )
        pre_task_retrieval = _retrieval_from_trace(pre_task_decision)
        retrieved_memories = (
            ()
            if pre_task_retrieval is None
            else tuple(match.content for match in pre_task_retrieval.matches)
        )
        template = subtask.answer_input_without_memories
        answer_input = AnswerAgentSubtaskInput(
            task_prompt=template.task_prompt,
            subtask_input=template.subtask_input,
            retrieved_memories=retrieved_memories,
        )
        try:
            answer_result = self._answer_runner.run(
                answer_input,
                tool_definitions=self._tool_definitions,
            )
        except Exception:
            raise _EpisodeStageFailed(
                stage=EpisodeFailureStage.ANSWER,
                detail_code=(EpisodeFailureDetailCode.ANSWER_RUNNER_EXCEPTION),
            ) from None
        if type(answer_result) is not AnswerAgentSubtaskRunResult:
            raise _EpisodeStageFailed(
                stage=EpisodeFailureStage.INTERNAL,
                detail_code=(EpisodeFailureDetailCode.ANSWER_RESULT_CONTRACT_ERROR),
            )
        if answer_result.status is not AnswerAgentSubtaskRunStatus.COMPLETED:
            raise _answer_failure(answer_result)
        if answer_result.final_answer is None:
            raise _EpisodeStageFailed(
                stage=EpisodeFailureStage.INTERNAL,
                detail_code=(EpisodeFailureDetailCode.ANSWER_RESULT_CONTRACT_ERROR),
            )

        interaction_events = build_subtask_interaction_events(answer_result)
        task_result = TaskResult(final_answer=answer_result.final_answer)
        post_task_decisions = self._run_post_task_decisions(
            task_input=subtask.memory_task_input,
            task_result=task_result,
            interaction_events=interaction_events,
            pre_task_retrieval=pre_task_retrieval,
            gateway=gateway,
            failure_stage=EpisodeFailureStage.POST_TASK,
        )
        trace = EpisodeSubtaskTrace(
            subtask=subtask,
            pre_task_decision=pre_task_decision,
            answer_input=answer_input,
            answer_result=answer_result,
            interaction_events=interaction_events,
            task_result=task_result,
            post_task_decisions=post_task_decisions,
            bank_before=bank_before,
            bank_after=gateway.state,
        )
        validation = validate_episode_subtask_trace(trace)
        if not validation.is_valid:
            raise _EpisodeStageFailed(
                stage=EpisodeFailureStage.INTERNAL,
                detail_code=(EpisodeFailureDetailCode.EPISODE_TRACE_CONTRACT_ERROR),
            )
        return trace

    def _run_post_task_decisions(
        self,
        *,
        task_input: str,
        task_result: TaskResult,
        interaction_events: tuple[InteractionEvent, ...],
        pre_task_retrieval: RetrievalResult | None,
        gateway: ActionToExecutorGateway,
        failure_stage: EpisodeFailureStage,
    ) -> tuple[MemoryDecisionTrace, ...]:
        first_context = MemoryDecisionContext(
            phase=MemoryDecisionPhase.POST_TASK,
            task_input=task_input,
            bank_record_count=gateway.state.record_count,
            interaction_events=interaction_events,
            task_result=task_result,
            pre_task_retrieval=pre_task_retrieval,
        )
        first = _execute_memory_stage(
            memory_client=self._memory_client,
            gateway=gateway,
            context=first_context,
            stage=failure_stage,
        )
        post_task_retrieval = _retrieval_from_trace(first)
        if post_task_retrieval is None:
            return (first,)

        second_context = MemoryDecisionContext(
            phase=MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
            task_input=task_input,
            bank_record_count=gateway.state.record_count,
            interaction_events=interaction_events,
            task_result=task_result,
            pre_task_retrieval=pre_task_retrieval,
            post_task_retrieval=post_task_retrieval,
        )
        second = _execute_memory_stage(
            memory_client=self._memory_client,
            gateway=gateway,
            context=second_context,
            stage=failure_stage,
        )
        return (first, second)

    def _failed_result(
        self,
        *,
        episode_input: EpisodeInput,
        initialization: EpisodeInitializationTrace | None,
        completed_subtasks: tuple[EpisodeSubtaskTrace, ...],
        initial_bank: MemoryBankState,
        error_code: EpisodeRunErrorCode,
        failed_subtask_ordinal: int | None,
        failure: _EpisodeStageFailed,
    ) -> EpisodeRunResult:
        if initialization is None:
            trace = None
            final_bank = initial_bank
        else:
            final_bank = (
                initialization.bank_after
                if not completed_subtasks
                else completed_subtasks[-1].bank_after
            )
            trace = EpisodeTrace(
                initial_bank_state=initial_bank,
                initialization=initialization,
                subtasks=completed_subtasks,
                final_bank_state=final_bank,
            )
        result = EpisodeRunResult(
            status=EpisodeRunStatus.FAILED,
            episode_input=episode_input,
            trace=trace,
            completed_subtask_count=len(completed_subtasks),
            final_bank_state=final_bank,
            error_code=error_code,
            failed_subtask_ordinal=failed_subtask_ordinal,
            failed_stage=failure.stage,
            failure_detail_code=failure.detail_code,
            answer_provider_error_code=(failure.answer_provider_error_code),
        )
        self._require_valid_result(result)
        return result

    @staticmethod
    def _require_valid_result(result: EpisodeRunResult) -> None:
        validation = validate_episode_run_result(result)
        if not validation.is_valid:
            code = (
                "UNKNOWN"
                if validation.error_code is None
                else validation.error_code.value
            )
            raise RuntimeError(f"EpisodeRunner constructed an invalid result ({code}).")


__all__ = [
    "EpisodeAnswerRunner",
    "EpisodeRunner",
    "EpisodeRunnerInputError",
]
