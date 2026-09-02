import pytest

from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState, MemoryRecord
from examples.memory_agent_grpo.memory_agent.bank.validation import (
    BankValidationErrorCode,
    is_canonical_memory_id,
    validate_memory_bank_state,
)


def _state(*records: MemoryRecord, next_id: int = 1) -> MemoryBankState:
    return MemoryBankState(records=records, next_id=next_id)


@pytest.mark.parametrize(
    "state",
    [
        MemoryBankState.empty(),
        _state(next_id=1001),
        _state(MemoryRecord("M001", "first"), MemoryRecord("M003", "third"), next_id=4),
        _state(
            MemoryRecord("M999", "last three-digit ID"),
            MemoryRecord("M1000", "first four-digit ID"),
            next_id=1001,
        ),
    ],
)
def test_valid_states_pass_integrity_checks(state: MemoryBankState) -> None:
    result = validate_memory_bank_state(state)

    assert result.is_valid
    assert result.error_code is None
    assert result.error_message is None


@pytest.mark.parametrize("state", [None, {}, []])
def test_state_must_use_the_typed_model(state: object) -> None:
    result = validate_memory_bank_state(state)

    assert not result.is_valid
    assert result.error_code is BankValidationErrorCode.STATE_NOT_MEMORY_BANK_STATE


@pytest.mark.parametrize("next_id", [True, 2.0, "2"])
def test_next_id_must_be_an_integer(next_id: object) -> None:
    state = MemoryBankState(records=(), next_id=next_id)  # type: ignore[arg-type]

    result = validate_memory_bank_state(state)

    assert result.error_code is BankValidationErrorCode.NEXT_ID_NOT_INTEGER


@pytest.mark.parametrize("next_id", [0, -1])
def test_next_id_must_be_positive(next_id: int) -> None:
    result = validate_memory_bank_state(_state(next_id=next_id))

    assert result.error_code is BankValidationErrorCode.NEXT_ID_OUT_OF_RANGE


def test_records_must_be_an_immutable_tuple() -> None:
    state = MemoryBankState(records=[], next_id=1)  # type: ignore[arg-type]

    result = validate_memory_bank_state(state)

    assert result.error_code is BankValidationErrorCode.RECORDS_NOT_TUPLE


def test_every_record_must_use_the_typed_model() -> None:
    state = MemoryBankState(records=({"memory_id": "M001", "content": "memory"},), next_id=2)  # type: ignore[arg-type]

    result = validate_memory_bank_state(state)

    assert result.error_code is BankValidationErrorCode.RECORD_NOT_MEMORY_RECORD


def test_memory_id_must_be_a_string() -> None:
    state = _state(MemoryRecord(memory_id=1, content="memory"), next_id=2)  # type: ignore[arg-type]

    result = validate_memory_bank_state(state)

    assert result.error_code is BankValidationErrorCode.MEMORY_ID_NOT_STRING


@pytest.mark.parametrize(
    "memory_id",
    ["M000", "M01", "M0001", "M1", "M-001", "m001", "M00A", " M001 ", "M００１"],
)
def test_memory_id_must_use_canonical_expandable_format(memory_id: str) -> None:
    result = validate_memory_bank_state(_state(MemoryRecord(memory_id, "memory"), next_id=2))

    assert result.error_code is BankValidationErrorCode.INVALID_MEMORY_ID


@pytest.mark.parametrize("memory_id", ["M001", "M999", "M1000", "M123456"])
def test_public_memory_id_check_accepts_canonical_ids(memory_id: str) -> None:
    assert is_canonical_memory_id(memory_id)


@pytest.mark.parametrize("memory_id", [None, 1, "M000", "M01", "m001", "<M001>"])
def test_public_memory_id_check_rejects_noncanonical_values(memory_id: object) -> None:
    assert not is_canonical_memory_id(memory_id)


def test_content_must_be_a_string() -> None:
    record = MemoryRecord(memory_id="M001", content=1)  # type: ignore[arg-type]

    result = validate_memory_bank_state(_state(record, next_id=2))

    assert result.error_code is BankValidationErrorCode.CONTENT_NOT_STRING


@pytest.mark.parametrize("content", ["", "  \n\t"])
def test_content_must_contain_non_whitespace_text(content: str) -> None:
    result = validate_memory_bank_state(_state(MemoryRecord("M001", content), next_id=2))

    assert result.error_code is BankValidationErrorCode.EMPTY_CONTENT


def test_duplicate_ids_are_rejected() -> None:
    state = _state(
        MemoryRecord("M001", "first"),
        MemoryRecord("M001", "duplicate"),
        next_id=2,
    )

    result = validate_memory_bank_state(state)

    assert result.error_code is BankValidationErrorCode.DUPLICATE_MEMORY_ID


def test_records_must_remain_in_creation_order() -> None:
    state = _state(
        MemoryRecord("M002", "second"),
        MemoryRecord("M001", "first"),
        next_id=3,
    )

    result = validate_memory_bank_state(state)

    assert result.error_code is BankValidationErrorCode.RECORDS_OUT_OF_ORDER


@pytest.mark.parametrize(
    "state",
    [
        _state(MemoryRecord("M001", "first"), next_id=1),
        _state(MemoryRecord("M002", "second"), next_id=2),
    ],
)
def test_next_id_must_be_after_all_current_records(state: MemoryBankState) -> None:
    result = validate_memory_bank_state(state)

    assert result.error_code is BankValidationErrorCode.NEXT_ID_NOT_AFTER_RECORDS
