from examples.memory_agent_grpo.memory_agent.action.protocol import (
    AddArguments,
    DeleteArguments,
    MemoryAction,
    MemoryActionName,
    MemoryDecision,
    RetrieveArguments,
    UpdateArguments,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState, MemoryRecord
from examples.memory_agent_grpo.memory_agent.bank.store import MemoryBankStore
from examples.memory_agent_grpo.memory_agent.executor.engine import MemoryActionExecutor
from examples.memory_agent_grpo.memory_agent.executor.outcome import ExecutionCode, ExecutionStatus


def _action(name: MemoryActionName, arguments: object) -> MemoryAction:
    return MemoryAction(action=name, arguments=arguments)  # type: ignore[arg-type]


def _executor(*records: MemoryRecord, next_id: int) -> MemoryActionExecutor:
    state = MemoryBankState(records=records, next_id=next_id)
    return MemoryActionExecutor(store=MemoryBankStore(state))


def test_retrieve_routes_to_bm25_without_changing_bank() -> None:
    executor = _executor(
        MemoryRecord("M001", "Paris hotel"),
        MemoryRecord("M002", "Tokyo flight"),
        next_id=3,
    )
    state_before = executor.state

    result = executor.execute(_action(MemoryActionName.RETRIEVE, RetrieveArguments("hotel")))

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.code is ExecutionCode.RETRIEVED
    assert result.retrieval is not None
    assert result.retrieval.memory_ids == ("M001",)
    assert result.record is None
    assert executor.state is state_before


def test_empty_retrieval_is_still_successful() -> None:
    executor = _executor(MemoryRecord("M001", "Paris hotel"), next_id=2)

    result = executor.execute(_action(MemoryActionName.RETRIEVE, RetrieveArguments("museum")))

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.code is ExecutionCode.RETRIEVED
    assert result.retrieval is not None
    assert result.retrieval.matches == ()


def test_add_routes_to_store_and_returns_created_record() -> None:
    executor = MemoryActionExecutor()

    result = executor.execute(_action(MemoryActionName.ADD, AddArguments("new memory")))

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.code is ExecutionCode.ADDED
    assert result.record == MemoryRecord("M001", "new memory")
    assert executor.state.records == (result.record,)
    assert executor.state.next_id == 2


def test_duplicate_add_returns_no_change_without_consuming_id() -> None:
    existing = MemoryRecord("M001", "same memory")
    executor = _executor(existing, next_id=2)
    state_before = executor.state

    result = executor.execute(_action(MemoryActionName.ADD, AddArguments("same memory")))

    assert result.status is ExecutionStatus.NO_CHANGE
    assert result.code is ExecutionCode.EXACT_DUPLICATE_ADD
    assert result.record is existing
    assert executor.state is state_before
    assert executor.state.next_id == 2


def test_update_routes_to_store_and_preserves_id() -> None:
    executor = _executor(MemoryRecord("M001", "old memory"), next_id=2)

    result = executor.execute(
        _action(MemoryActionName.UPDATE, UpdateArguments("M001", "new memory"))
    )

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.code is ExecutionCode.UPDATED
    assert result.record == MemoryRecord("M001", "new memory")
    assert executor.state.records == (result.record,)
    assert executor.state.next_id == 2


def test_unchanged_update_returns_no_change_with_same_snapshot() -> None:
    existing = MemoryRecord("M001", "same memory")
    executor = _executor(existing, next_id=2)
    state_before = executor.state

    result = executor.execute(
        _action(MemoryActionName.UPDATE, UpdateArguments("M001", "same memory"))
    )

    assert result.status is ExecutionStatus.NO_CHANGE
    assert result.code is ExecutionCode.UNCHANGED_UPDATE
    assert result.record is existing
    assert executor.state is state_before


def test_missing_update_is_rejected_without_changing_bank() -> None:
    executor = _executor(MemoryRecord("M001", "memory"), next_id=2)
    state_before = executor.state

    result = executor.execute(
        _action(MemoryActionName.UPDATE, UpdateArguments("M999", "updated"))
    )

    assert result.status is ExecutionStatus.REJECTED
    assert result.code is ExecutionCode.TARGET_NOT_FOUND
    assert result.error_message is not None
    assert executor.state is state_before


def test_delete_routes_to_store_without_reusing_id() -> None:
    executor = _executor(
        MemoryRecord("M001", "first"),
        MemoryRecord("M002", "second"),
        next_id=3,
    )

    result = executor.execute(_action(MemoryActionName.DELETE, DeleteArguments("M001")))

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.code is ExecutionCode.DELETED
    assert result.record == MemoryRecord("M001", "first")
    assert executor.state.records == (MemoryRecord("M002", "second"),)
    assert executor.state.next_id == 3


def test_missing_delete_is_rejected_without_changing_bank() -> None:
    executor = _executor(MemoryRecord("M001", "memory"), next_id=2)
    state_before = executor.state

    result = executor.execute(_action(MemoryActionName.DELETE, DeleteArguments("M999")))

    assert result.status is ExecutionStatus.REJECTED
    assert result.code is ExecutionCode.TARGET_NOT_FOUND
    assert executor.state is state_before


def test_malformed_typed_action_is_rejected_without_calling_components() -> None:
    executor = _executor(MemoryRecord("M001", "memory"), next_id=2)
    state_before = executor.state
    malformed = _action(MemoryActionName.ADD, RetrieveArguments("wrong type"))

    result = executor.execute(malformed)

    assert result.status is ExecutionStatus.REJECTED
    assert result.code is ExecutionCode.INVALID_ACTION
    assert executor.state is state_before


def test_non_string_argument_field_is_rejected_without_changing_bank() -> None:
    executor = _executor(MemoryRecord("M001", "memory"), next_id=2)
    state_before = executor.state
    malformed = _action(
        MemoryActionName.ADD,
        AddArguments(content=1),  # type: ignore[arg-type]
    )

    result = executor.execute(malformed)

    assert result.status is ExecutionStatus.REJECTED
    assert result.code is ExecutionCode.INVALID_ARGUMENTS
    assert executor.state is state_before


def test_absent_decision_is_not_an_executable_action() -> None:
    executor = MemoryActionExecutor()
    state_before = executor.state

    result = executor.execute(MemoryDecision(memory_action=None))

    assert result.status is ExecutionStatus.REJECTED
    assert result.code is ExecutionCode.INVALID_ACTION
    assert result.action is None
    assert executor.state is state_before
