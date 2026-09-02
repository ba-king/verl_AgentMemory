"""Execute one Memory-aware six-Product MemoryArena Shopping episode."""

from __future__ import annotations

from dataclasses import dataclass

from ....answer_agent.input import AnswerAgentSubtaskInput
from ....answer_agent.interaction_mode import AnswerAgentInteractionMode
from ....answer_agent.model_client import (
    AnswerAgentClientErrorCode,
    AnswerAgentConversationClient,
)
from ....answer_agent.runtime import (
    AnswerAgentSubtaskErrorCode,
    AnswerAgentSubtaskRunner,
    AnswerAgentSubtaskRunnerConfig,
    AnswerAgentSubtaskRunStatus,
)
from ....answer_agent.tool_protocol.definition import ToolDefinition
from ....episode.runtime.memory_decision import (
    MemoryDecisionStepError,
    MemoryDecisionStepErrorCode,
    execute_memory_decision,
)
from ....episode.trace import MemoryDecisionTrace, build_subtask_interaction_events
from ....memory_agent.action.protocol import MemoryDecisionPhase
from ....memory_agent.action.validation import ValidationErrorCode
from ....memory_agent.bank.bank_state import MemoryBankState
from ....memory_agent.decision.action_gateway import ActionToExecutorGateway
from ....memory_agent.decision.context import (
    InteractionEvent,
    MemoryDecisionContext,
    TaskResult,
)
from ....memory_agent.decision.model_client import (
    MemoryAgentClient,
    MemoryAgentClientErrorCode,
)
from ....memory_agent.executor.engine import MemoryActionExecutor
from ....memory_agent.retrieval.result import RetrievalResult
from .environment_session import ShoppingEnvironmentSession, ShoppingTerminationKind
from .executor_adapter import ShoppingToolExecutorAdapter
from .memory_episode_contract import (
    ShoppingMemoryEpisodeFailureCode,
    ShoppingMemoryEpisodeFailureStage,
    ShoppingMemoryEpisodeResult,
    ShoppingMemoryEpisodeRunStatus,
    ShoppingMemoryEpisodeSessionFactory,
    ShoppingMemoryOutputCorrection,
    ShoppingMemoryProductTrace,
)
from .memory_episode_validation import (
    retrieval_from_trace,
    split_six_products,
    terminal_tool_execution_count,
    validated_tool_definitions,
)
from .task_input_adapter import ShoppingTaskInputAdapter


class _ProductStageFailed(RuntimeError):
    """Private marker carrying only finite, safe failure metadata."""

    def __init__(
        self,
        *,
        stage: ShoppingMemoryEpisodeFailureStage,
        code: ShoppingMemoryEpisodeFailureCode,
        memory_validation_error_code: ValidationErrorCode | None = None,
        memory_model_error_code: MemoryAgentClientErrorCode | None = None,
        answer_error_code: AnswerAgentSubtaskErrorCode | None = None,
        answer_provider_error_code: AnswerAgentClientErrorCode | None = None,
    ) -> None:
        self.stage = stage
        self.code = code
        self.memory_validation_error_code = memory_validation_error_code
        self.memory_model_error_code = memory_model_error_code
        self.answer_error_code = answer_error_code
        self.answer_provider_error_code = answer_provider_error_code
        super().__init__(f"{stage.value}:{code.value}")


@dataclass
class _RunCounters:
    memory_agent_invocation_count: int = 0
    memory_output_correction: ShoppingMemoryOutputCorrection | None = None


_MEMORY_FAILURE_CODES = {
    MemoryDecisionStepErrorCode.MODEL_CALL_FAILED: (
        ShoppingMemoryEpisodeFailureCode.MEMORY_MODEL_CALL_FAILED
    ),
    MemoryDecisionStepErrorCode.OUTPUT_VALIDATION_FAILED: (
        ShoppingMemoryEpisodeFailureCode.MEMORY_OUTPUT_VALIDATION_FAILED
    ),
    MemoryDecisionStepErrorCode.EXECUTION_REJECTED: (
        ShoppingMemoryEpisodeFailureCode.MEMORY_EXECUTION_REJECTED
    ),
    MemoryDecisionStepErrorCode.INTERNAL_CONTRACT_ERROR: (
        ShoppingMemoryEpisodeFailureCode.MEMORY_INTERNAL_CONTRACT_ERROR
    ),
}


def _execute_memory_stage(
    *,
    memory_client: MemoryAgentClient,
    gateway: ActionToExecutorGateway,
    context: MemoryDecisionContext,
    stage: ShoppingMemoryEpisodeFailureStage,
    product_ordinal: int,
    memory_output_correction_budget: int,
    counters: _RunCounters,
) -> MemoryDecisionTrace:
    counters.memory_agent_invocation_count += 1
    try:
        return execute_memory_decision(
            memory_client=memory_client,
            gateway=gateway,
            context=context,
        )
    except MemoryDecisionStepError as error:
        if (
            error.code is MemoryDecisionStepErrorCode.OUTPUT_VALIDATION_FAILED
            and memory_output_correction_budget == 1
            and counters.memory_output_correction is None
            and type(error.validation_error_code) is ValidationErrorCode
        ):
            trigger_error_code = error.validation_error_code
            counters.memory_agent_invocation_count += 1
            try:
                corrected_trace = execute_memory_decision(
                    memory_client=memory_client,
                    gateway=gateway,
                    context=context,
                    correction_error_code=trigger_error_code,
                )
            except MemoryDecisionStepError as corrected_error:
                counters.memory_output_correction = (
                    ShoppingMemoryOutputCorrection(
                        product_ordinal=product_ordinal,
                        phase=context.phase,
                        trigger_error_code=trigger_error_code,
                        recovered=False,
                    )
                )
                error = corrected_error
            else:
                counters.memory_output_correction = (
                    ShoppingMemoryOutputCorrection(
                        product_ordinal=product_ordinal,
                        phase=context.phase,
                        trigger_error_code=trigger_error_code,
                        recovered=True,
                    )
                )
                return corrected_trace
        code = _MEMORY_FAILURE_CODES[error.code]
        failure_stage = (
            ShoppingMemoryEpisodeFailureStage.INTERNAL
            if error.code is MemoryDecisionStepErrorCode.INTERNAL_CONTRACT_ERROR
            else stage
        )
        raise _ProductStageFailed(
            stage=failure_stage,
            code=code,
            memory_validation_error_code=error.validation_error_code,
            memory_model_error_code=error.model_client_error_code,
        ) from None


def _run_post_task_decisions(
    *,
    memory_client: MemoryAgentClient,
    gateway: ActionToExecutorGateway,
    memory_task_input: str,
    task_result: TaskResult,
    interaction_events: tuple[InteractionEvent, ...],
    pre_task_retrieval: RetrievalResult | None,
    product_ordinal: int,
    memory_output_correction_budget: int,
    counters: _RunCounters,
) -> tuple[MemoryDecisionTrace, ...]:
    first = _execute_memory_stage(
        memory_client=memory_client,
        gateway=gateway,
        context=MemoryDecisionContext(
            phase=MemoryDecisionPhase.POST_TASK,
            task_input=memory_task_input,
            bank_record_count=gateway.state.record_count,
            interaction_events=interaction_events,
            task_result=task_result,
            pre_task_retrieval=pre_task_retrieval,
        ),
        stage=ShoppingMemoryEpisodeFailureStage.POST_TASK,
        product_ordinal=product_ordinal,
        memory_output_correction_budget=memory_output_correction_budget,
        counters=counters,
    )
    post_task_retrieval = retrieval_from_trace(first)
    if post_task_retrieval is None:
        return (first,)
    second = _execute_memory_stage(
        memory_client=memory_client,
        gateway=gateway,
        context=MemoryDecisionContext(
            phase=MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
            task_input=memory_task_input,
            bank_record_count=gateway.state.record_count,
            interaction_events=interaction_events,
            task_result=task_result,
            pre_task_retrieval=pre_task_retrieval,
            post_task_retrieval=post_task_retrieval,
        ),
        stage=ShoppingMemoryEpisodeFailureStage.POST_TASK,
        product_ordinal=product_ordinal,
        memory_output_correction_budget=memory_output_correction_budget,
        counters=counters,
    )
    return (first, second)


class ShoppingMemoryEpisodeRunner:
    """Run six fresh Shopping sessions through one persistent in-memory Bank."""

    def __init__(
        self,
        *,
        memory_client: MemoryAgentClient,
        answer_client: AnswerAgentConversationClient,
        session_factory: ShoppingMemoryEpisodeSessionFactory,
        task_input_adapter: ShoppingTaskInputAdapter,
        tool_definitions: object,
        config: AnswerAgentSubtaskRunnerConfig,
        memory_output_correction_budget: int = 0,
    ) -> None:
        if not callable(getattr(memory_client, "generate", None)):
            raise ValueError("memory_client must expose a callable generate method.")
        if not callable(getattr(answer_client, "generate", None)):
            raise ValueError("answer_client must expose a callable generate method.")
        if not callable(getattr(session_factory, "open_session", None)):
            raise ValueError("session_factory must expose a callable open_session method.")
        if type(task_input_adapter) is not ShoppingTaskInputAdapter:
            raise ValueError("task_input_adapter must be a ShoppingTaskInputAdapter.")
        if type(config) is not AnswerAgentSubtaskRunnerConfig:
            raise ValueError("config must be an AnswerAgentSubtaskRunnerConfig.")
        if (
            type(memory_output_correction_budget) is not int
            or memory_output_correction_budget not in (0, 1)
        ):
            raise ValueError(
                "memory_output_correction_budget must be exactly 0 or 1."
            )
        if (
            config.interaction_mode
            is not AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL
            or config.max_tool_calls_per_inference != 1
        ):
            raise ValueError(
                "Shopping Memory episodes require environment-terminated "
                "single-tool mode."
            )
        self._memory_client = memory_client
        self._answer_client = answer_client
        self._session_factory = session_factory
        self._task_input_adapter = task_input_adapter
        self._tool_definitions = validated_tool_definitions(tool_definitions)
        self._config = config
        self._memory_output_correction_budget = (
            memory_output_correction_budget
        )

    @property
    def config(self) -> AnswerAgentSubtaskRunnerConfig:
        return self._config

    @property
    def tool_definitions(self) -> tuple[ToolDefinition, ...]:
        return self._tool_definitions

    def run(self, full_task_definition: object) -> ShoppingMemoryEpisodeResult:
        """Run all Products unless a structural agent/session failure occurs."""

        single_product_tasks = split_six_products(full_task_definition)
        gateway = ActionToExecutorGateway(executor=MemoryActionExecutor())
        committed_bank = gateway.state
        committed_products: list[ShoppingMemoryProductTrace] = []
        counters = _RunCounters()

        for product_ordinal, single_product_task in enumerate(
            single_product_tasks,
            start=1,
        ):
            try:
                trace = self._run_product(
                    product_ordinal=product_ordinal,
                    single_product_task=single_product_task,
                    gateway=gateway,
                    counters=counters,
                )
            except _ProductStageFailed as failure:
                return self._failed_result(
                    committed_products=committed_products,
                    committed_bank=committed_bank,
                    counters=counters,
                    product_ordinal=product_ordinal,
                    failure=failure,
                )
            except Exception:
                return self._failed_result(
                    committed_products=committed_products,
                    committed_bank=committed_bank,
                    counters=counters,
                    product_ordinal=product_ordinal,
                    failure=_ProductStageFailed(
                        stage=ShoppingMemoryEpisodeFailureStage.INTERNAL,
                        code=(
                            ShoppingMemoryEpisodeFailureCode.UNEXPECTED_EXCEPTION
                        ),
                    ),
                )
            committed_products.append(trace)
            committed_bank = trace.bank_after

        return ShoppingMemoryEpisodeResult(
            status=ShoppingMemoryEpisodeRunStatus.COMPLETED,
            product_traces=tuple(committed_products),
            final_bank_state=committed_bank,
            memory_agent_invocation_count=counters.memory_agent_invocation_count,
            memory_output_correction_budget=(
                self._memory_output_correction_budget
            ),
            memory_output_correction=counters.memory_output_correction,
        )

    def _failed_result(
        self,
        *,
        committed_products: list[ShoppingMemoryProductTrace],
        committed_bank: MemoryBankState,
        counters: _RunCounters,
        product_ordinal: int,
        failure: _ProductStageFailed,
    ) -> ShoppingMemoryEpisodeResult:
        return ShoppingMemoryEpisodeResult(
            status=ShoppingMemoryEpisodeRunStatus.FAILED,
            product_traces=tuple(committed_products),
            final_bank_state=committed_bank,
            memory_agent_invocation_count=counters.memory_agent_invocation_count,
            memory_output_correction_budget=(
                self._memory_output_correction_budget
            ),
            memory_output_correction=counters.memory_output_correction,
            failed_product_ordinal=product_ordinal,
            failure_stage=failure.stage,
            failure_code=failure.code,
            memory_validation_error_code=failure.memory_validation_error_code,
            memory_model_error_code=failure.memory_model_error_code,
            answer_error_code=failure.answer_error_code,
            answer_provider_error_code=failure.answer_provider_error_code,
        )

    def _run_product(
        self,
        *,
        product_ordinal: int,
        single_product_task: dict[str, object],
        gateway: ActionToExecutorGateway,
        counters: _RunCounters,
    ) -> ShoppingMemoryProductTrace:
        bank_before = gateway.state
        try:
            context_manager = self._session_factory.open_session(
                single_product_task=single_product_task,
                product_ordinal=product_ordinal,
            )
            with context_manager as session:
                if type(session) is not ShoppingEnvironmentSession:
                    raise ValueError(
                        "session_factory must yield a ShoppingEnvironmentSession."
                    )
                initial_observation = session.reset(seed=0)
                try:
                    template = self._task_input_adapter.build_subtask_input(
                        initial_observation.to_dict()
                    )
                    if template.retrieved_memories:
                        raise ValueError(
                            "Shopping adapter must not preload retrieved memories."
                        )
                except Exception:
                    raise _ProductStageFailed(
                        stage=ShoppingMemoryEpisodeFailureStage.ANSWER_INPUT,
                        code=(
                            ShoppingMemoryEpisodeFailureCode.ANSWER_INPUT_FAILED
                        ),
                    ) from None

                memory_task_input = template.subtask_input
                pre_task_decision = _execute_memory_stage(
                    memory_client=self._memory_client,
                    gateway=gateway,
                    context=MemoryDecisionContext(
                        phase=MemoryDecisionPhase.PRE_TASK,
                        task_input=memory_task_input,
                        bank_record_count=gateway.state.record_count,
                    ),
                    stage=ShoppingMemoryEpisodeFailureStage.PRE_TASK,
                    product_ordinal=product_ordinal,
                    memory_output_correction_budget=(
                        self._memory_output_correction_budget
                    ),
                    counters=counters,
                )
                pre_task_retrieval = retrieval_from_trace(pre_task_decision)
                answer_input = AnswerAgentSubtaskInput(
                    task_prompt=template.task_prompt,
                    subtask_input=template.subtask_input,
                    retrieved_memories=(
                        ()
                        if pre_task_retrieval is None
                        else tuple(
                            match.content
                            for match in pre_task_retrieval.matches
                        )
                    ),
                )
                try:
                    answer_result = AnswerAgentSubtaskRunner(
                        client=self._answer_client,
                        executor=ShoppingToolExecutorAdapter(session=session),
                        config=self._config,
                    ).run(
                        answer_input,
                        tool_definitions=self._tool_definitions,
                    )
                except Exception:
                    raise _ProductStageFailed(
                        stage=ShoppingMemoryEpisodeFailureStage.ANSWER,
                        code=(
                            ShoppingMemoryEpisodeFailureCode.ANSWER_RUNNER_EXCEPTION
                        ),
                    ) from None
                termination_evidence = session.termination_evidence
        except _ProductStageFailed:
            raise
        except Exception:
            raise _ProductStageFailed(
                stage=ShoppingMemoryEpisodeFailureStage.SESSION,
                code=ShoppingMemoryEpisodeFailureCode.SESSION_FAILED,
            ) from None

        if answer_result.status is not AnswerAgentSubtaskRunStatus.COMPLETED:
            raise _ProductStageFailed(
                stage=ShoppingMemoryEpisodeFailureStage.ANSWER,
                code=ShoppingMemoryEpisodeFailureCode.ANSWER_RUNNER_FAILED,
                answer_error_code=answer_result.error_code,
                answer_provider_error_code=answer_result.provider_error_code,
            )
        if (
            answer_result.final_answer is None
            or terminal_tool_execution_count(answer_result) != 1
            or termination_evidence.kind is ShoppingTerminationKind.NOT_TERMINATED
        ):
            raise _ProductStageFailed(
                stage=ShoppingMemoryEpisodeFailureStage.INTERNAL,
                code=ShoppingMemoryEpisodeFailureCode.TERMINAL_COUNT_INVALID,
            )

        interaction_events = build_subtask_interaction_events(answer_result)
        task_result = TaskResult(final_answer=answer_result.final_answer)
        post_task_decisions = _run_post_task_decisions(
            memory_client=self._memory_client,
            gateway=gateway,
            memory_task_input=memory_task_input,
            task_result=task_result,
            interaction_events=interaction_events,
            pre_task_retrieval=pre_task_retrieval,
            product_ordinal=product_ordinal,
            memory_output_correction_budget=(
                self._memory_output_correction_budget
            ),
            counters=counters,
        )
        try:
            return ShoppingMemoryProductTrace(
                product_ordinal=product_ordinal,
                memory_task_input=memory_task_input,
                pre_task_decision=pre_task_decision,
                answer_input=answer_input,
                answer_result=answer_result,
                interaction_events=interaction_events,
                task_result=task_result,
                post_task_decisions=post_task_decisions,
                termination_evidence=termination_evidence,
                bank_before=bank_before,
                bank_after=gateway.state,
            )
        except Exception:
            raise _ProductStageFailed(
                stage=ShoppingMemoryEpisodeFailureStage.INTERNAL,
                code=ShoppingMemoryEpisodeFailureCode.TRACE_CONTRACT_ERROR,
            ) from None


__all__ = ["ShoppingMemoryEpisodeRunner"]
