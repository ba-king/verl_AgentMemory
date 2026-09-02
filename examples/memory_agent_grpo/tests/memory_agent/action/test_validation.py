import json
import sys

import pytest

from examples.memory_agent_grpo.memory_agent.action import validation as validation_module
from examples.memory_agent_grpo.memory_agent.action.protocol import (
    AddArguments,
    DeleteArguments,
    MemoryActionName,
    MemoryDecisionPhase,
    RetrieveArguments,
    UpdateArguments,
)
from examples.memory_agent_grpo.memory_agent.action.validation import (
    ValidationErrorCode,
    ValidationStatus,
    validate_memory_decision_output,
)


def _completion(action: str, arguments: object) -> str:
    return json.dumps(
        {"memory_action": {"action": action, "arguments": arguments}},
        ensure_ascii=False,
    )


@pytest.mark.parametrize("phase", list(MemoryDecisionPhase))
def test_null_is_a_valid_absent_action_in_every_phase(phase: MemoryDecisionPhase) -> None:
    result = validate_memory_decision_output('{"memory_action": null}', phase=phase)

    assert result.status is ValidationStatus.VALID
    assert result.error_code is None
    assert result.decision is not None
    assert result.decision.memory_action is None
    assert result.decision.to_dict() == {"memory_action": None}


def test_retrieve_is_valid_in_pre_task_and_trims_query() -> None:
    result = validate_memory_decision_output(
        _completion("RETRIEVE", {"query": "  Kevin hotel price  "}),
        phase=MemoryDecisionPhase.PRE_TASK,
    )

    assert result.is_valid
    assert result.decision is not None
    action = result.decision.memory_action
    assert action is not None
    assert action.action is MemoryActionName.RETRIEVE
    assert action.arguments == RetrieveArguments(query="Kevin hotel price")


def test_add_is_valid_in_post_task_and_trims_content() -> None:
    result = validate_memory_decision_output(
        _completion("ADD", {"content": "  Kevin selected Hotel A.  "}),
        phase=MemoryDecisionPhase.POST_TASK,
    )

    assert result.is_valid
    assert result.decision is not None
    action = result.decision.memory_action
    assert action is not None
    assert action.arguments == AddArguments(content="Kevin selected Hotel A.")


def test_update_is_valid_only_for_a_visible_id() -> None:
    result = validate_memory_decision_output(
        _completion("UPDATE", {"memory_id": " M003 ", "content": "New complete content"}),
        phase=MemoryDecisionPhase.POST_TASK,
        visible_memory_ids={"M003", "M008"},
    )

    assert result.is_valid
    assert result.decision is not None
    action = result.decision.memory_action
    assert action is not None
    assert action.arguments == UpdateArguments(memory_id="M003", content="New complete content")


def test_delete_is_valid_after_post_task_retrieval_for_a_visible_id() -> None:
    result = validate_memory_decision_output(
        _completion("DELETE", {"memory_id": "M003"}),
        phase=MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
        visible_memory_ids={"M003"},
    )

    assert result.is_valid
    assert result.decision is not None
    action = result.decision.memory_action
    assert action is not None
    assert action.arguments == DeleteArguments(memory_id="M003")


@pytest.mark.parametrize(
    ("phase", "action", "arguments"),
    [
        (MemoryDecisionPhase.PRE_TASK, "RETRIEVE", {"query": "Kevin hotel"}),
        (MemoryDecisionPhase.POST_TASK, "RETRIEVE", {"query": "Kevin hotel"}),
        (MemoryDecisionPhase.POST_TASK, "ADD", {"content": "new memory"}),
        (
            MemoryDecisionPhase.POST_TASK,
            "UPDATE",
            {"memory_id": "M001", "content": "updated memory"},
        ),
        (MemoryDecisionPhase.POST_TASK, "DELETE", {"memory_id": "M001"}),
        (
            MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
            "ADD",
            {"content": "new memory"},
        ),
        (
            MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
            "UPDATE",
            {"memory_id": "M001", "content": "updated memory"},
        ),
        (
            MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
            "DELETE",
            {"memory_id": "M001"},
        ),
    ],
)
def test_complete_allowed_phase_action_matrix(
    phase: MemoryDecisionPhase,
    action: str,
    arguments: dict[str, str],
) -> None:
    result = validate_memory_decision_output(
        _completion(action, arguments),
        phase=phase,
        visible_memory_ids={"M001"},
    )

    assert result.status is ValidationStatus.VALID
    assert result.decision is not None
    assert result.decision.memory_action is not None
    assert result.decision.memory_action.action is MemoryActionName(action)


@pytest.mark.parametrize("action", ["ADD", "UPDATE", "DELETE"])
def test_mutations_are_disallowed_in_pre_task(action: str) -> None:
    arguments = {
        "ADD": {"content": "content"},
        "UPDATE": {"memory_id": "M001", "content": "content"},
        "DELETE": {"memory_id": "M001"},
    }[action]
    result = validate_memory_decision_output(
        _completion(action, arguments),
        phase=MemoryDecisionPhase.PRE_TASK,
        visible_memory_ids={"M001"},
    )

    assert result.status is ValidationStatus.DISALLOWED_ACTION_FOR_PHASE
    assert result.error_code is ValidationErrorCode.ACTION_NOT_ALLOWED_IN_PHASE
    assert result.decision is None


def test_second_post_task_retrieval_is_disallowed() -> None:
    result = validate_memory_decision_output(
        _completion("RETRIEVE", {"query": "Kevin hotel"}),
        phase=MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
    )

    assert result.status is ValidationStatus.DISALLOWED_ACTION_FOR_PHASE
    assert result.error_code is ValidationErrorCode.ACTION_NOT_ALLOWED_IN_PHASE


@pytest.mark.parametrize(
    ("action", "arguments"),
    [
        ("UPDATE", {"memory_id": "M999", "content": "new content"}),
        ("DELETE", {"memory_id": "M999"}),
    ],
)
def test_update_and_delete_reject_invisible_ids(action: str, arguments: dict[str, str]) -> None:
    result = validate_memory_decision_output(
        _completion(action, arguments),
        phase=MemoryDecisionPhase.POST_TASK,
        visible_memory_ids={"M001"},
    )

    assert result.status is ValidationStatus.INVALID_OUTPUT
    assert result.error_code is ValidationErrorCode.MEMORY_ID_NOT_VISIBLE
    assert result.decision is None


@pytest.mark.parametrize(
    ("raw_output", "error_code"),
    [
        ("", ValidationErrorCode.EMPTY_OUTPUT),
        ("   \n", ValidationErrorCode.EMPTY_OUTPUT),
        ("not json", ValidationErrorCode.INVALID_JSON),
        ('{"memory_action": NaN}', ValidationErrorCode.INVALID_JSON),
        ('{"memory_action": Infinity}', ValidationErrorCode.INVALID_JSON),
        ('{"memory_action": -Infinity}', ValidationErrorCode.INVALID_JSON),
        (
            'Here is the action: {"memory_action": null}',
            ValidationErrorCode.INVALID_JSON,
        ),
        (
            '```json\n{"memory_action": null}\n```',
            ValidationErrorCode.INVALID_JSON,
        ),
        ("null", ValidationErrorCode.TOP_LEVEL_NOT_OBJECT),
        ("[]", ValidationErrorCode.TOP_LEVEL_NOT_OBJECT),
        ('{"action": null}', ValidationErrorCode.INVALID_ENVELOPE_KEYS),
        (
            '{"memory_action": null, "reasoning": "none"}',
            ValidationErrorCode.INVALID_ENVELOPE_KEYS,
        ),
        ('{"memory_action": "ADD"}', ValidationErrorCode.MEMORY_ACTION_NOT_OBJECT),
        (
            '{"memory_action":{"action":"ADD"}}',
            ValidationErrorCode.INVALID_ACTION_KEYS,
        ),
        (
            '{"memory_action":{"action":"ADD","arguments":{"content":"x"},"reasoning":"y"}}',
            ValidationErrorCode.INVALID_ACTION_KEYS,
        ),
        (
            '{"memory_action":{"action":1,"arguments":{"content":"x"}}}',
            ValidationErrorCode.ACTION_NAME_NOT_STRING,
        ),
        (
            '{"memory_action":{"action":"NO_OP","arguments":{}}}',
            ValidationErrorCode.UNKNOWN_ACTION,
        ),
        (
            '{"memory_action":{"action":"add","arguments":{"content":"x"}}}',
            ValidationErrorCode.UNKNOWN_ACTION,
        ),
        (
            '{"memory_action":{"action":"ADD","arguments":"x"}}',
            ValidationErrorCode.ARGUMENTS_NOT_OBJECT,
        ),
        (
            '{"memory_action":{"action":"ADD","arguments":{}}}',
            ValidationErrorCode.INVALID_ARGUMENT_KEYS,
        ),
        (
            '{"memory_action":{"action":"ADD","arguments":{"content":"x","query":"y"}}}',
            ValidationErrorCode.INVALID_ARGUMENT_KEYS,
        ),
        (
            '{"memory_action":{"action":"ADD","arguments":{"content":3}}}',
            ValidationErrorCode.ARGUMENT_NOT_STRING,
        ),
        (
            '{"memory_action":{"action":"ADD","arguments":{"content":"  "}}}',
            ValidationErrorCode.EMPTY_ARGUMENT,
        ),
    ],
)
def test_invalid_outputs_fail_closed(raw_output: str, error_code: ValidationErrorCode) -> None:
    result = validate_memory_decision_output(raw_output, phase=MemoryDecisionPhase.POST_TASK)

    assert result.status is ValidationStatus.INVALID_OUTPUT
    assert result.error_code is error_code
    assert result.decision is None


@pytest.mark.parametrize(
    "raw_output",
    [
        '{"memory_action":null,"memory_action":null}',
        '{"memory_action":{"action":"ADD","action":"DELETE","arguments":{"content":"x"}}}',
        '{"memory_action":{"action":"ADD","arguments":{"content":"x","content":"y"}}}',
    ],
)
def test_duplicate_json_keys_are_rejected_at_every_level(raw_output: str) -> None:
    result = validate_memory_decision_output(raw_output, phase=MemoryDecisionPhase.POST_TASK)

    assert result.status is ValidationStatus.INVALID_OUTPUT
    assert result.error_code is ValidationErrorCode.DUPLICATE_JSON_KEY


def test_deeply_nested_json_fails_closed_instead_of_raising_recursion_error() -> None:
    depth = sys.getrecursionlimit() + 100
    raw_output = '{"memory_action":' + "[" * depth + "null" + "]" * depth + "}"

    result = validate_memory_decision_output(raw_output, phase=MemoryDecisionPhase.POST_TASK)

    assert result.status is ValidationStatus.INVALID_OUTPUT
    assert result.error_code in {
        ValidationErrorCode.INVALID_JSON,
        ValidationErrorCode.MEMORY_ACTION_NOT_OBJECT,
    }
    assert result.decision is None


def test_json_decoder_recursion_error_is_mapped_to_invalid_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_recursion_error(*args: object, **kwargs: object) -> None:
        raise RecursionError

    monkeypatch.setattr(validation_module.json, "loads", raise_recursion_error)

    result = validate_memory_decision_output('{"memory_action": null}', phase=MemoryDecisionPhase.POST_TASK)

    assert result.status is ValidationStatus.INVALID_OUTPUT
    assert result.error_code is ValidationErrorCode.INVALID_JSON
    assert result.decision is None


def test_non_string_raw_output_is_rejected_without_throwing() -> None:
    result = validate_memory_decision_output(None, phase=MemoryDecisionPhase.POST_TASK)  # type: ignore[arg-type]

    assert result.status is ValidationStatus.INVALID_OUTPUT
    assert result.error_code is ValidationErrorCode.RAW_OUTPUT_NOT_STRING


def test_caller_must_supply_a_typed_phase() -> None:
    with pytest.raises(TypeError, match="phase must be a MemoryDecisionPhase"):
        validate_memory_decision_output(
            '{"memory_action": null}',
            phase="POST_TASK",  # type: ignore[arg-type]
        )
