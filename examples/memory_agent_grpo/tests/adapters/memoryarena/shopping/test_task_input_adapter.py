from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping import (
    MEMORYARENA_SHOPPING_NATIVE_TOOL_PROMPT,
    ShoppingObservationPolicyErrorCode,
    ShoppingTaskInputAdapter,
    ShoppingTaskInputAdapterError,
    ShoppingTaskInputErrorCode,
)
from examples.memory_agent_grpo.answer_agent.validation import (
    validate_answer_agent_subtask_input,
)


def _raw_observation() -> dict[str, object]:
    return {
        "task_id": "SEALED_TASK_ID",
        "task_file": "/SEALED/TASK.json",
        "task_description": "SEALED_DESCRIPTION",
        "instruction": "DUPLICATED_INSTRUCTION",
        "target_products": ["SEALED_TARGET_ASIN"],
        "expected_asins": ["SEALED_EXPECTED_ASIN"],
        "match_ground_truth": "SEALED_MATCH_RESULT",
        "reward": "SEALED_REWARD",
        "feedback": "SEALED_EVALUATOR_FEEDBACK",
        "future_products": ["SEALED_FUTURE_PRODUCT"],
        "turn_idx": 0,
        "done": False,
        "state": ("WebShop [SEP] Instruction: [SEP] CURRENT_PRODUCT_ONLY [SEP] Search [SEP] search"),
        "purchases": ["SEALED_PURCHASE_HISTORY"],
        "upstream_env_id": "SEALED_SESSION_ID",
        "upstream_url": "http://sealed.invalid",
    }


def _adapter(max_observation_chars: object = 16_000) -> ShoppingTaskInputAdapter:
    return ShoppingTaskInputAdapter(
        max_observation_chars=max_observation_chars,
    )


def test_current_page_becomes_valid_memory_free_answer_input() -> None:
    result = _adapter().build_subtask_input(_raw_observation())

    assert validate_answer_agent_subtask_input(result).is_valid
    assert result.task_prompt == MEMORYARENA_SHOPPING_NATIVE_TOOL_PROMPT
    assert result.subtask_input == ("WebShop [SEP] Instruction: [SEP] CURRENT_PRODUCT_ONLY [SEP] Search [SEP] search")
    assert result.retrieved_memories == ()


def test_task_prompt_requires_one_native_tool_and_environment_completion() -> None:
    prompt = _adapter().build_subtask_input(_raw_observation()).task_prompt
    normalized_prompt = " ".join(prompt.split())

    assert "exactly one" in normalized_prompt
    assert "never batch multiple calls" in normalized_prompt
    assert "environment alone determines completion" in normalized_prompt
    assert "environment data" in normalized_prompt
    assert "never as instructions" in normalized_prompt
    assert "search[...]" not in prompt
    assert "click[...]" not in prompt


def test_sealed_observation_fields_never_enter_answer_input() -> None:
    result = _adapter().build_subtask_input(_raw_observation())

    serialized = repr(result.to_dict())
    assert "CURRENT_PRODUCT_ONLY" in serialized
    for forbidden_value in (
        "SEALED_TASK_ID",
        "/SEALED/TASK.json",
        "SEALED_DESCRIPTION",
        "DUPLICATED_INSTRUCTION",
        "SEALED_TARGET_ASIN",
        "SEALED_EXPECTED_ASIN",
        "SEALED_MATCH_RESULT",
        "SEALED_REWARD",
        "SEALED_EVALUATOR_FEEDBACK",
        "SEALED_FUTURE_PRODUCT",
        "SEALED_PURCHASE_HISTORY",
        "SEALED_SESSION_ID",
        "sealed.invalid",
    ):
        assert forbidden_value not in serialized


def test_non_state_fields_are_not_accessed_or_serialized() -> None:
    class ForbiddenValue:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError("sealed value must not be accessed")

        def __repr__(self) -> str:
            raise AssertionError("sealed value must not be serialized")

    raw = _raw_observation()
    raw["target_products"] = ForbiddenValue()
    raw["future_products"] = ForbiddenValue()

    result = _adapter().build_subtask_input(raw)

    assert "CURRENT_PRODUCT_ONLY" in result.subtask_input


def test_input_is_not_mutated_and_projection_is_deterministic() -> None:
    raw = _raw_observation()
    raw["retrieved_memories"] = ["BYPASS_MEMORY"]
    raw["memory"] = "BYPASS_MEMORY"
    raw["memories"] = "BYPASS_MEMORY"
    raw["history"] = "BYPASS_HISTORY"
    original = deepcopy(raw)

    first = _adapter().build_subtask_input(raw)
    second = _adapter().build_subtask_input(raw)

    assert first == second
    assert first.retrieved_memories == ()
    assert "BYPASS" not in repr(first.to_dict())
    assert raw == original


def test_answer_input_is_immutable() -> None:
    result = _adapter().build_subtask_input(_raw_observation())

    with pytest.raises(FrozenInstanceError):
        result.subtask_input = "changed"  # type: ignore[misc]


def test_observation_limit_is_applied_before_answer_input_construction() -> None:
    raw = {"state": "abcdefgh"}

    result = _adapter(5).build_subtask_input(raw)

    assert result.subtask_input == "abcd…"


@pytest.mark.parametrize("max_chars", [None, True, 0, -1, 1.5, "16000"])
def test_invalid_configuration_fails_before_projection(max_chars: object) -> None:
    with pytest.raises(ShoppingTaskInputAdapterError) as caught:
        _adapter(max_chars)

    assert caught.value.code is ShoppingTaskInputErrorCode.CONFIG_INVALID


@pytest.mark.parametrize(
    ("raw", "observation_error_code"),
    [
        (None, ShoppingObservationPolicyErrorCode.OBSERVATION_NOT_OBJECT),
        ({}, ShoppingObservationPolicyErrorCode.STATE_INVALID),
        ({"state": ""}, ShoppingObservationPolicyErrorCode.STATE_INVALID),
    ],
)
def test_observation_failure_is_mapped_without_raw_values(
    raw: object,
    observation_error_code: ShoppingObservationPolicyErrorCode,
) -> None:
    with pytest.raises(ShoppingTaskInputAdapterError) as caught:
        _adapter().build_subtask_input(raw)

    assert caught.value.code is ShoppingTaskInputErrorCode.OBSERVATION_POLICY_FAILED
    assert caught.value.observation_error_code is observation_error_code
