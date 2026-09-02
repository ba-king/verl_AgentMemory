import pytest

from examples.memory_agent_grpo.memory_agent.action.protocol import (
    AddArguments,
    DeleteArguments,
    MemoryAction,
    MemoryActionName,
    RetrieveArguments,
    UpdateArguments,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState, MemoryRecord
from examples.memory_agent_grpo.memory_agent.executor.outcome import ExecutionCode
from examples.memory_agent_grpo.memory_agent.executor.validation import (
    ExecutorValidationDisposition,
    validate_action_for_execution,
)


def _state(*records: MemoryRecord, next_id: int) -> MemoryBankState:
    return MemoryBankState(records=records, next_id=next_id)


@pytest.mark.parametrize(
    "action",
    [
        MemoryAction(MemoryActionName.RETRIEVE, RetrieveArguments("hotel")),
        MemoryAction(MemoryActionName.ADD, AddArguments("new memory")),
        MemoryAction(MemoryActionName.UPDATE, UpdateArguments("M001", "updated")),
        MemoryAction(MemoryActionName.DELETE, DeleteArguments("M001")),
    ],
)
def test_well_formed_executable_actions_are_ready(action: MemoryAction) -> None:
    state = _state(MemoryRecord("M001", "old memory"), next_id=2)

    result = validate_action_for_execution(action, state=state)

    assert result.disposition is ExecutorValidationDisposition.READY
    assert result.code is None


def test_executor_accepts_only_one_memory_action() -> None:
    result = validate_action_for_execution(None, state=MemoryBankState.empty())

    assert result.disposition is ExecutorValidationDisposition.REJECTED
    assert result.code is ExecutionCode.INVALID_ACTION


def test_action_name_must_use_the_enum() -> None:
    action = MemoryAction(action="ADD", arguments=AddArguments("memory"))  # type: ignore[arg-type]

    result = validate_action_for_execution(action, state=MemoryBankState.empty())

    assert result.code is ExecutionCode.INVALID_ACTION


@pytest.mark.parametrize(
    "action",
    [
        MemoryAction(MemoryActionName.RETRIEVE, AddArguments("memory")),
        MemoryAction(MemoryActionName.ADD, RetrieveArguments("memory")),
        MemoryAction(MemoryActionName.UPDATE, DeleteArguments("M001")),
        MemoryAction(MemoryActionName.DELETE, UpdateArguments("M001", "memory")),
    ],
)
def test_action_and_argument_types_must_match(action: MemoryAction) -> None:
    result = validate_action_for_execution(action, state=MemoryBankState.empty())

    assert result.disposition is ExecutorValidationDisposition.REJECTED
    assert result.code is ExecutionCode.INVALID_ACTION


@pytest.mark.parametrize(
    "action",
    [
        MemoryAction(MemoryActionName.RETRIEVE, RetrieveArguments("")),
        MemoryAction(MemoryActionName.ADD, AddArguments("  \n")),
        MemoryAction(MemoryActionName.UPDATE, UpdateArguments("M001", "")),
        MemoryAction(MemoryActionName.UPDATE, UpdateArguments("", "memory")),
        MemoryAction(MemoryActionName.DELETE, DeleteArguments("  ")),
    ],
)
def test_empty_argument_values_are_rejected(action: MemoryAction) -> None:
    result = validate_action_for_execution(action, state=MemoryBankState.empty())

    assert result.disposition is ExecutorValidationDisposition.REJECTED
    assert result.code is ExecutionCode.INVALID_ARGUMENTS


@pytest.mark.parametrize(
    "action",
    [
        MemoryAction(MemoryActionName.RETRIEVE, RetrieveArguments(query=1)),  # type: ignore[arg-type]
        MemoryAction(MemoryActionName.ADD, AddArguments(content=True)),  # type: ignore[arg-type]
        MemoryAction(
            MemoryActionName.UPDATE,
            UpdateArguments(memory_id=1, content="memory"),  # type: ignore[arg-type]
        ),
        MemoryAction(
            MemoryActionName.UPDATE,
            UpdateArguments(memory_id="M001", content=None),  # type: ignore[arg-type]
        ),
        MemoryAction(MemoryActionName.DELETE, DeleteArguments(memory_id=1)),  # type: ignore[arg-type]
    ],
)
def test_non_string_argument_values_are_rejected(action: MemoryAction) -> None:
    state = _state(MemoryRecord("M001", "memory"), next_id=2)

    result = validate_action_for_execution(action, state=state)

    assert result.disposition is ExecutorValidationDisposition.REJECTED
    assert result.code is ExecutionCode.INVALID_ARGUMENTS


def test_invalid_bank_state_is_rejected() -> None:
    invalid_state = _state(MemoryRecord("M000", "memory"), next_id=1)
    action = MemoryAction(MemoryActionName.ADD, AddArguments("new memory"))

    result = validate_action_for_execution(action, state=invalid_state)

    assert result.disposition is ExecutorValidationDisposition.REJECTED
    assert result.code is ExecutionCode.INVALID_BANK_STATE


def test_duplicate_add_is_classified_as_no_change() -> None:
    existing = MemoryRecord("M001", "same memory")
    action = MemoryAction(MemoryActionName.ADD, AddArguments("same memory"))

    result = validate_action_for_execution(action, state=_state(existing, next_id=2))

    assert result.disposition is ExecutorValidationDisposition.NO_CHANGE
    assert result.code is ExecutionCode.EXACT_DUPLICATE_ADD
    assert result.record is existing


def test_unchanged_update_is_classified_as_no_change() -> None:
    existing = MemoryRecord("M001", "same memory")
    action = MemoryAction(MemoryActionName.UPDATE, UpdateArguments("M001", "same memory"))

    result = validate_action_for_execution(action, state=_state(existing, next_id=2))

    assert result.disposition is ExecutorValidationDisposition.NO_CHANGE
    assert result.code is ExecutionCode.UNCHANGED_UPDATE
    assert result.record is existing


@pytest.mark.parametrize(
    "action",
    [
        MemoryAction(MemoryActionName.UPDATE, UpdateArguments("M999", "updated")),
        MemoryAction(MemoryActionName.DELETE, DeleteArguments("M999")),
    ],
)
def test_missing_mutation_target_is_rejected(action: MemoryAction) -> None:
    state = _state(MemoryRecord("M001", "memory"), next_id=2)

    result = validate_action_for_execution(action, state=state)

    assert result.disposition is ExecutorValidationDisposition.REJECTED
    assert result.code is ExecutionCode.TARGET_NOT_FOUND


def test_update_may_match_another_records_content_under_current_contract() -> None:
    state = _state(
        MemoryRecord("M001", "shared content"),
        MemoryRecord("M002", "old content"),
        next_id=3,
    )
    action = MemoryAction(MemoryActionName.UPDATE, UpdateArguments("M002", "shared content"))

    result = validate_action_for_execution(action, state=state)

    assert result.disposition is ExecutorValidationDisposition.READY
