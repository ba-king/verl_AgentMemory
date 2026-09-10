import json

import pytest

from examples.memory_agent_grpo.memory_agent.action.protocol import MemoryActionName, MemoryDecisionPhase
from examples.memory_agent_grpo.memory_agent.action.validation import (
    ValidationErrorCode,
    ValidationStatus,
    validate_memory_decision_output,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState, MemoryRecord
from examples.memory_agent_grpo.memory_agent.bank.store import MemoryBankStore
from examples.memory_agent_grpo.memory_agent.decision.action_gateway import (
    ActionToExecutorGateway,
    ActionToExecutorResult,
    ActionToExecutorStatus,
)
from examples.memory_agent_grpo.memory_agent.decision.context import (
    MemoryDecisionContext,
    TaskResult,
)
from examples.memory_agent_grpo.memory_agent.executor.engine import MemoryActionExecutor
from examples.memory_agent_grpo.memory_agent.executor.outcome import ExecutionCode, ExecutionResult, ExecutionStatus
from examples.memory_agent_grpo.memory_agent.retrieval.result import RetrievalMatch, RetrievalResult


def _completion(action: str, arguments: dict[str, object]) -> str:
    return json.dumps({"memory_action": {"action": action, "arguments": arguments}})


def _gateway(*records: MemoryRecord, next_id: int) -> ActionToExecutorGateway:
    state = MemoryBankState(records=records, next_id=next_id)
    executor = MemoryActionExecutor(store=MemoryBankStore(state))
    return ActionToExecutorGateway(executor=executor)


def _context(
    phase: MemoryDecisionPhase,
    *visible_memory_ids: str,
) -> MemoryDecisionContext:
    retrieval = RetrievalResult(
        query="visible memories",
        matches=tuple(
            RetrievalMatch(MemoryRecord(memory_id, f"content for {memory_id}"), 1.0)
            for memory_id in visible_memory_ids
        ),
    )
    if phase is MemoryDecisionPhase.PRE_TASK:
        if visible_memory_ids:
            raise ValueError("PRE_TASK helper cannot expose memories before retrieval.")
        return MemoryDecisionContext(
            phase=phase,
            task_input="task",
            bank_record_count=0,
        )
    if phase is MemoryDecisionPhase.POST_TASK:
        return MemoryDecisionContext(
            phase=phase,
            task_input="task",
            bank_record_count=len(visible_memory_ids),
            task_result=TaskResult("answer"),
            pre_task_retrieval=retrieval if visible_memory_ids else None,
        )
    return MemoryDecisionContext(
        phase=phase,
        task_input="task",
        bank_record_count=len(visible_memory_ids),
        task_result=TaskResult("answer"),
        post_task_retrieval=retrieval,
    )


class _CountingExecutor(MemoryActionExecutor):
    def __init__(self, *, store: MemoryBankStore) -> None:
        super().__init__(store=store)
        self.execute_call_count = 0

    def execute(self, action: object) -> ExecutionResult:
        self.execute_call_count += 1
        return super().execute(action)


def test_invalid_json_fails_validation_without_execution() -> None:
    gateway = ActionToExecutorGateway()
    state_before = gateway.state

    result = gateway.process_output("not json", context=_context(MemoryDecisionPhase.POST_TASK))

    assert result.status is ActionToExecutorStatus.VALIDATION_FAILED
    assert result.validation.error_code is ValidationErrorCode.INVALID_JSON
    assert result.execution is None
    assert gateway.state is state_before


def test_non_string_model_output_fails_validation_without_execution() -> None:
    gateway = ActionToExecutorGateway()
    state_before = gateway.state

    result = gateway.process_output(
        None,  # type: ignore[arg-type]
        context=_context(MemoryDecisionPhase.POST_TASK),
    )

    assert result.status is ActionToExecutorStatus.VALIDATION_FAILED
    assert result.validation.error_code is ValidationErrorCode.RAW_OUTPUT_NOT_STRING
    assert result.execution is None
    assert gateway.state is state_before


@pytest.mark.parametrize(
    ("raw_output", "phase", "visible_memory_ids", "expected_status"),
    [
        ("not json", MemoryDecisionPhase.POST_TASK, (), ActionToExecutorStatus.VALIDATION_FAILED),
        (
            _completion("ADD", {"content": "memory"}),
            MemoryDecisionPhase.PRE_TASK,
            (),
            ActionToExecutorStatus.VALIDATION_FAILED,
        ),
        (
            _completion("DELETE", {"memory_id": "M001"}),
            MemoryDecisionPhase.POST_TASK,
            (),
            ActionToExecutorStatus.VALIDATION_FAILED,
        ),
        ('{"memory_action": null}', MemoryDecisionPhase.POST_TASK, (), ActionToExecutorStatus.SKIPPED),
    ],
)
def test_non_dispatching_branches_never_call_executor(
    raw_output: str,
    phase: MemoryDecisionPhase,
    visible_memory_ids: tuple[str, ...],
    expected_status: ActionToExecutorStatus,
) -> None:
    state = MemoryBankState(records=(MemoryRecord("M001", "memory"),), next_id=2)
    executor = _CountingExecutor(store=MemoryBankStore(state))
    gateway = ActionToExecutorGateway(executor=executor)

    result = gateway.process_output(
        raw_output,
        context=_context(phase, *visible_memory_ids),
    )

    assert result.status is expected_status
    assert result.execution is None
    assert executor.execute_call_count == 0
    assert gateway.state is state


@pytest.mark.parametrize("phase", list(MemoryDecisionPhase))
def test_valid_null_is_skipped_in_every_phase(phase: MemoryDecisionPhase) -> None:
    gateway = ActionToExecutorGateway()
    state_before = gateway.state

    result = gateway.process_output('{"memory_action": null}', context=_context(phase))

    assert result.status is ActionToExecutorStatus.SKIPPED
    assert result.validation.status is ValidationStatus.VALID
    assert result.execution is None
    assert gateway.state is state_before


def test_pre_task_mutation_is_blocked_before_executor() -> None:
    gateway = ActionToExecutorGateway()
    state_before = gateway.state

    result = gateway.process_output(
        _completion("ADD", {"content": "memory"}),
        context=_context(MemoryDecisionPhase.PRE_TASK),
    )

    assert result.status is ActionToExecutorStatus.VALIDATION_FAILED
    assert result.validation.status is ValidationStatus.DISALLOWED_ACTION_FOR_PHASE
    assert result.execution is None
    assert gateway.state is state_before


def test_second_post_task_retrieval_is_blocked_before_executor() -> None:
    gateway = ActionToExecutorGateway()

    result = gateway.process_output(
        _completion("RETRIEVE", {"query": "hotel"}),
        context=_context(MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL),
    )

    assert result.status is ActionToExecutorStatus.VALIDATION_FAILED
    assert result.validation.error_code is ValidationErrorCode.ACTION_NOT_ALLOWED_IN_PHASE
    assert result.execution is None


def test_invisible_update_target_is_blocked_even_when_record_exists() -> None:
    existing = MemoryRecord("M001", "old memory")
    gateway = _gateway(existing, next_id=2)
    state_before = gateway.state

    result = gateway.process_output(
        _completion("UPDATE", {"memory_id": "M001", "content": "new memory"}),
        context=_context(MemoryDecisionPhase.POST_TASK),
    )

    assert result.status is ActionToExecutorStatus.VALIDATION_FAILED
    assert result.validation.error_code is ValidationErrorCode.MEMORY_ID_NOT_VISIBLE
    assert result.execution is None
    assert gateway.state is state_before


def test_visible_update_is_dispatched_and_applied() -> None:
    gateway = _gateway(MemoryRecord("M001", "old memory"), next_id=2)

    result = gateway.process_output(
        _completion("UPDATE", {"memory_id": "M001", "content": "  new memory  "}),
        context=_context(MemoryDecisionPhase.POST_TASK, "M001"),
    )

    assert result.status is ActionToExecutorStatus.DISPATCHED
    assert result.execution is not None
    assert result.execution.status is ExecutionStatus.SUCCEEDED
    assert result.execution.code is ExecutionCode.UPDATED
    assert gateway.state.records == (MemoryRecord("M001", "new memory"),)


def test_post_task_add_is_dispatched_and_applied() -> None:
    gateway = ActionToExecutorGateway()

    result = gateway.process_output(
        _completion("ADD", {"content": "new memory"}),
        context=_context(MemoryDecisionPhase.POST_TASK),
    )

    assert result.status is ActionToExecutorStatus.DISPATCHED
    assert result.execution is not None
    assert result.execution.code is ExecutionCode.ADDED
    assert gateway.state.records == (MemoryRecord("M001", "new memory"),)


def test_duplicate_add_is_dispatched_and_executor_reports_no_change() -> None:
    existing = MemoryRecord("M001", "same memory")
    gateway = _gateway(existing, next_id=2)
    state_before = gateway.state

    result = gateway.process_output(
        _completion("ADD", {"content": "same memory"}),
        context=_context(MemoryDecisionPhase.POST_TASK),
    )

    assert result.status is ActionToExecutorStatus.DISPATCHED
    assert result.execution is not None
    assert result.execution.status is ExecutionStatus.NO_CHANGE
    assert result.execution.code is ExecutionCode.EXACT_DUPLICATE_ADD
    assert gateway.state is state_before


def test_visible_delete_after_post_retrieval_is_dispatched() -> None:
    gateway = _gateway(MemoryRecord("M001", "memory"), next_id=2)

    result = gateway.process_output(
        _completion("DELETE", {"memory_id": "M001"}),
        context=_context(MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL, "M001"),
    )

    assert result.status is ActionToExecutorStatus.DISPATCHED
    assert result.execution is not None
    assert result.execution.code is ExecutionCode.DELETED
    assert gateway.state.records == ()
    assert gateway.state.next_id == 2


def test_pre_task_retrieval_is_dispatched_without_mutating_bank() -> None:
    gateway = _gateway(MemoryRecord("M001", "Paris hotel"), next_id=2)
    state_before = gateway.state

    result = gateway.process_output(
        _completion("RETRIEVE", {"query": "hotel"}),
        context=_context(MemoryDecisionPhase.PRE_TASK),
    )

    assert result.status is ActionToExecutorStatus.DISPATCHED
    assert result.execution is not None
    assert result.execution.code is ExecutionCode.RETRIEVED
    assert result.execution.retrieval is not None
    assert result.execution.retrieval.memory_ids == ("M001",)
    assert gateway.state is state_before


def test_authorized_but_stale_target_is_rejected_by_executor() -> None:
    gateway = ActionToExecutorGateway()
    state_before = gateway.state

    result = gateway.process_output(
        _completion("DELETE", {"memory_id": "M999"}),
        context=_context(MemoryDecisionPhase.POST_TASK, "M999"),
    )

    assert result.status is ActionToExecutorStatus.DISPATCHED
    assert result.validation.status is ValidationStatus.VALID
    assert result.execution is not None
    assert result.execution.status is ExecutionStatus.REJECTED
    assert result.execution.code is ExecutionCode.TARGET_NOT_FOUND
    assert gateway.state is state_before


def test_invalid_context_remains_a_programmer_error_without_execution() -> None:
    gateway = ActionToExecutorGateway()
    invalid_context = MemoryDecisionContext(
        phase="POST_TASK",  # type: ignore[arg-type]
        task_input="task",
        bank_record_count=0,
        task_result=TaskResult("answer"),
    )

    with pytest.raises(ValueError, match="PHASE_NOT_MEMORY_DECISION_PHASE"):
        gateway.process_output(
            '{"memory_action": null}',
            context=invalid_context,
        )


def test_skipped_result_serializes_nested_validation_without_execution() -> None:
    validation = validate_memory_decision_output(
        '{"memory_action": null}',
        phase=MemoryDecisionPhase.PRE_TASK,
    )
    result = ActionToExecutorResult(
        status=ActionToExecutorStatus.SKIPPED,
        phase=MemoryDecisionPhase.PRE_TASK,
        validation=validation,
    )

    assert result.to_dict() == {
        "status": "SKIPPED",
        "phase": "PRE_TASK",
        "validation": {
            "status": "VALID",
            "decision": {"memory_action": None},
            "error_code": None,
            "error_message": None,
        },
        "execution": None,
    }


def test_validation_failure_serializes_stable_error_code() -> None:
    validation = validate_memory_decision_output(
        "not json",
        phase=MemoryDecisionPhase.POST_TASK,
    )
    result = ActionToExecutorResult(
        status=ActionToExecutorStatus.VALIDATION_FAILED,
        phase=MemoryDecisionPhase.POST_TASK,
        validation=validation,
    )

    payload = result.to_dict()

    assert payload["validation"] == {
        "status": "INVALID_OUTPUT",
        "decision": None,
        "error_code": "INVALID_JSON",
        "error_message": "Output is not strict JSON: Expecting value (line 1, column 1, character 0).",
    }


def test_dispatched_result_serializes_nested_executor_rejection() -> None:
    validation = validate_memory_decision_output(
        '{"memory_action":{"action":"DELETE","arguments":{"memory_id":"M999"}}}',
        phase=MemoryDecisionPhase.POST_TASK,
        visible_memory_ids={"M999"},
    )
    execution = ExecutionResult(
        status=ExecutionStatus.REJECTED,
        code=ExecutionCode.TARGET_NOT_FOUND,
        action=MemoryActionName.DELETE,
        error_message="target disappeared",
    )
    result = ActionToExecutorResult(
        status=ActionToExecutorStatus.DISPATCHED,
        phase=MemoryDecisionPhase.POST_TASK,
        validation=validation,
        execution=execution,
    )

    assert result.to_dict()["execution"] == execution.to_dict()


def test_dispatched_result_serializes_nested_no_change() -> None:
    validation = validate_memory_decision_output(
        '{"memory_action":{"action":"ADD","arguments":{"content":"memory"}}}',
        phase=MemoryDecisionPhase.POST_TASK,
    )
    execution = ExecutionResult(
        status=ExecutionStatus.NO_CHANGE,
        code=ExecutionCode.EXACT_DUPLICATE_ADD,
        action=MemoryActionName.ADD,
        record=MemoryRecord("M001", "memory"),
    )
    result = ActionToExecutorResult(
        status=ActionToExecutorStatus.DISPATCHED,
        phase=MemoryDecisionPhase.POST_TASK,
        validation=validation,
        execution=execution,
    )

    assert result.to_dict()["execution"] == execution.to_dict()
