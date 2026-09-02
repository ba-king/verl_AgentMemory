from examples.memory_agent_grpo.memory_agent.action.protocol import MemoryActionName
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryRecord
from examples.memory_agent_grpo.memory_agent.executor.outcome import (
    ExecutionCode,
    ExecutionResult,
    ExecutionStatus,
)


def test_successful_mutation_result_serializes_without_a_full_bank_snapshot() -> None:
    result = ExecutionResult(
        status=ExecutionStatus.SUCCEEDED,
        code=ExecutionCode.ADDED,
        action=MemoryActionName.ADD,
        record=MemoryRecord("M001", "memory"),
    )

    assert not result.is_rejected
    assert result.to_dict() == {
        "status": "SUCCEEDED",
        "code": "ADDED",
        "action": "ADD",
        "record": {"memory_id": "M001", "content": "memory"},
        "retrieval": None,
        "error_message": None,
    }


def test_rejected_result_may_have_no_action_name() -> None:
    result = ExecutionResult(
        status=ExecutionStatus.REJECTED,
        code=ExecutionCode.INVALID_ACTION,
        action=None,
        error_message="invalid",
    )

    assert result.is_rejected
    assert result.to_dict()["action"] is None
