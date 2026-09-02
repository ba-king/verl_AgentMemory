import pytest

from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState, MemoryRecord
from examples.memory_agent_grpo.memory_agent.bank.store import (
    InvalidMemoryBankStateError,
    MemoryBankStore,
    MemoryRecordNotFoundError,
    StoreMutationStatus,
)
from examples.memory_agent_grpo.memory_agent.bank.validation import BankValidationErrorCode


def _state(*records: MemoryRecord, next_id: int) -> MemoryBankState:
    return MemoryBankState(records=records, next_id=next_id)


def test_store_starts_empty_by_default() -> None:
    store = MemoryBankStore()

    assert store.state == MemoryBankState.empty()


def test_store_accepts_an_existing_valid_snapshot() -> None:
    initial_state = _state(MemoryRecord("M001", "memory"), next_id=2)

    store = MemoryBankStore(initial_state)

    assert store.state is initial_state


def test_store_rejects_an_invalid_initial_snapshot() -> None:
    invalid_state = _state(MemoryRecord("M000", "memory"), next_id=1)

    with pytest.raises(InvalidMemoryBankStateError) as exc_info:
        MemoryBankStore(invalid_state)

    assert exc_info.value.validation_result.error_code is BankValidationErrorCode.INVALID_MEMORY_ID


def test_add_appends_a_record_and_advances_next_id() -> None:
    store = MemoryBankStore()
    previous_state = store.state

    result = store.add("first memory")

    assert result.status is StoreMutationStatus.APPLIED
    assert result.changed
    assert result.record == MemoryRecord("M001", "first memory")
    assert result.state is store.state
    assert store.state == _state(MemoryRecord("M001", "first memory"), next_id=2)
    assert previous_state == MemoryBankState.empty()


def test_add_uses_four_digits_after_m999() -> None:
    initial_state = _state(MemoryRecord("M999", "existing memory"), next_id=1000)
    store = MemoryBankStore(initial_state)

    result = store.add("new memory")

    assert result.record.memory_id == "M1000"
    assert store.state.next_id == 1001
    assert [record.memory_id for record in store.state.records] == ["M999", "M1000"]


def test_add_appends_after_an_id_gap_instead_of_reusing_it() -> None:
    store = MemoryBankStore(
        _state(
            MemoryRecord("M001", "first"),
            MemoryRecord("M003", "third"),
            next_id=4,
        )
    )

    result = store.add("fourth")

    assert result.record.memory_id == "M004"
    assert [record.memory_id for record in store.state.records] == ["M001", "M003", "M004"]
    assert store.state.next_id == 5


def test_exact_duplicate_add_is_no_change_and_does_not_consume_an_id() -> None:
    initial_state = _state(MemoryRecord("M001", "same memory"), next_id=2)
    store = MemoryBankStore(initial_state)

    result = store.add("same memory")

    assert result.status is StoreMutationStatus.NO_CHANGE
    assert not result.changed
    assert result.record is initial_state.records[0]
    assert result.state is initial_state
    assert store.state is initial_state
    assert store.state.next_id == 2


def test_update_replaces_content_but_preserves_id_position_and_next_id() -> None:
    first = MemoryRecord("M001", "first")
    second = MemoryRecord("M002", "old second")
    third = MemoryRecord("M003", "third")
    initial_state = _state(first, second, third, next_id=4)
    store = MemoryBankStore(initial_state)

    result = store.update("M002", "new second")

    assert result.status is StoreMutationStatus.APPLIED
    assert result.record == MemoryRecord("M002", "new second")
    assert store.state.records == (first, result.record, third)
    assert store.state.next_id == 4
    assert initial_state.records == (first, second, third)


def test_unchanged_update_retains_the_same_snapshot() -> None:
    initial_state = _state(MemoryRecord("M001", "same memory"), next_id=2)
    store = MemoryBankStore(initial_state)

    result = store.update("M001", "same memory")

    assert result.status is StoreMutationStatus.NO_CHANGE
    assert not result.changed
    assert result.state is initial_state
    assert result.record is initial_state.records[0]


def test_missing_update_target_raises_without_changing_state() -> None:
    initial_state = _state(MemoryRecord("M001", "memory"), next_id=2)
    store = MemoryBankStore(initial_state)

    with pytest.raises(MemoryRecordNotFoundError) as exc_info:
        store.update("M999", "replacement")

    assert exc_info.value.memory_id == "M999"
    assert store.state is initial_state


def test_delete_removes_the_record_but_preserves_next_id() -> None:
    first = MemoryRecord("M001", "first")
    second = MemoryRecord("M002", "second")
    third = MemoryRecord("M003", "third")
    initial_state = _state(first, second, third, next_id=4)
    store = MemoryBankStore(initial_state)

    result = store.delete("M002")

    assert result.status is StoreMutationStatus.APPLIED
    assert result.record is second
    assert store.state.records == (first, third)
    assert store.state.next_id == 4
    assert initial_state.records == (first, second, third)


def test_add_after_delete_uses_a_new_id_instead_of_filling_the_gap() -> None:
    store = MemoryBankStore(
        _state(
            MemoryRecord("M001", "first"),
            MemoryRecord("M002", "second"),
            MemoryRecord("M003", "third"),
            next_id=4,
        )
    )

    store.delete("M002")
    result = store.add("fourth")

    assert result.record.memory_id == "M004"
    assert [record.memory_id for record in store.state.records] == ["M001", "M003", "M004"]
    assert store.state.next_id == 5


def test_missing_delete_target_raises_without_changing_state() -> None:
    initial_state = _state(MemoryRecord("M001", "memory"), next_id=2)
    store = MemoryBankStore(initial_state)

    with pytest.raises(MemoryRecordNotFoundError) as exc_info:
        store.delete("M999")

    assert exc_info.value.memory_id == "M999"
    assert store.state is initial_state


@pytest.mark.parametrize("content", [None, 1, True])
def test_mutations_reject_non_string_content(content: object) -> None:
    initial_state = _state(MemoryRecord("M001", "memory"), next_id=2)
    store = MemoryBankStore(initial_state)

    with pytest.raises(TypeError, match="content must be a string"):
        store.add(content)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="content must be a string"):
        store.update("M001", content)  # type: ignore[arg-type]
    assert store.state is initial_state


@pytest.mark.parametrize("content", ["", "  \n\t"])
def test_mutations_reject_empty_content(content: str) -> None:
    initial_state = _state(MemoryRecord("M001", "memory"), next_id=2)
    store = MemoryBankStore(initial_state)

    with pytest.raises(ValueError, match="content must not be empty"):
        store.add(content)
    with pytest.raises(ValueError, match="content must not be empty"):
        store.update("M001", content)
    assert store.state is initial_state


@pytest.mark.parametrize("memory_id", [None, 1, True])
def test_update_and_delete_reject_non_string_ids(memory_id: object) -> None:
    initial_state = _state(MemoryRecord("M001", "memory"), next_id=2)
    store = MemoryBankStore(initial_state)

    with pytest.raises(TypeError, match="memory_id must be a string"):
        store.update(memory_id, "replacement")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="memory_id must be a string"):
        store.delete(memory_id)  # type: ignore[arg-type]
    assert store.state is initial_state


@pytest.mark.parametrize("memory_id", ["", "  \n\t"])
def test_update_and_delete_reject_empty_ids(memory_id: str) -> None:
    initial_state = _state(MemoryRecord("M001", "memory"), next_id=2)
    store = MemoryBankStore(initial_state)

    with pytest.raises(ValueError, match="memory_id must not be empty"):
        store.update(memory_id, "replacement")
    with pytest.raises(ValueError, match="memory_id must not be empty"):
        store.delete(memory_id)
    assert store.state is initial_state
