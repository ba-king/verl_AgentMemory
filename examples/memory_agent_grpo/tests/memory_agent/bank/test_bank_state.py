from dataclasses import FrozenInstanceError

import pytest

from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState, MemoryRecord


def test_empty_state_has_no_records_and_starts_at_one() -> None:
    state = MemoryBankState.empty()

    assert state.records == ()
    assert state.next_id == 1
    assert state.record_count == 0
    assert state.to_dict() == {"records": [], "next_id": 1}


def test_state_serialization_preserves_record_order() -> None:
    state = MemoryBankState(
        records=(
            MemoryRecord(memory_id="M001", content="first memory"),
            MemoryRecord(memory_id="M003", content="third allocated memory"),
        ),
        next_id=4,
    )

    assert state.record_count == 2
    assert state.to_dict() == {
        "records": [
            {"memory_id": "M001", "content": "first memory"},
            {"memory_id": "M003", "content": "third allocated memory"},
        ],
        "next_id": 4,
    }


def test_state_and_records_are_immutable() -> None:
    record = MemoryRecord(memory_id="M001", content="memory")
    state = MemoryBankState(records=(record,), next_id=2)

    with pytest.raises(FrozenInstanceError):
        record.content = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        state.next_id = 3  # type: ignore[misc]
