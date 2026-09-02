from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping import (
    ShoppingObservationPolicyError,
    ShoppingObservationPolicyErrorCode,
    ShoppingPublicObservation,
    project_public_observation,
)


def _raw_observation() -> dict[str, object]:
    return {
        "task_id": "sealed-task-id",
        "task_file": "/sealed/task.json",
        "task_description": "sealed-description",
        "instruction": "duplicated-instruction",
        "target_products": ["SECRET-ASIN"],
        "turn_idx": 3,
        "done": False,
        "state": "WebShop [SEP] Search [SEP] search",
        "purchases": [{"asin": "PRIVATE-PURCHASE"}],
        "upstream_env_id": "private-session",
        "upstream_url": "http://private.invalid",
    }


def _project(
    raw: object,
    *,
    max_observation_chars: object = 16_000,
) -> ShoppingPublicObservation:
    return project_public_observation(
        raw,
        max_observation_chars=max_observation_chars,
    )


def test_projection_allowlists_only_exact_state_without_mutation() -> None:
    raw = _raw_observation()
    original = deepcopy(raw)

    public = _project(raw)

    assert public == ShoppingPublicObservation(state=raw["state"])
    assert public.to_dict() == {"state": raw["state"]}
    assert raw == original


def test_projection_excludes_targets_metadata_and_control_fields() -> None:
    public_dict = _project(_raw_observation()).to_dict()

    serialized = repr(public_dict)
    for forbidden_value in (
        "SECRET-ASIN",
        "PRIVATE-PURCHASE",
        "sealed-task-id",
        "private-session",
        "private.invalid",
    ):
        assert forbidden_value not in serialized
    assert tuple(public_dict) == ("state",)


@pytest.mark.parametrize("raw", [None, [], (), "state"])
def test_non_object_observation_fails_closed(raw: object) -> None:
    with pytest.raises(ShoppingObservationPolicyError) as caught:
        _project(raw)

    assert caught.value.code is ShoppingObservationPolicyErrorCode.OBSERVATION_NOT_OBJECT


@pytest.mark.parametrize(
    "state",
    [None, 1, "", "   ", "\ud800"],
)
def test_missing_or_invalid_state_fails_closed(state: object) -> None:
    raw = {"state": state, "target_products": ["DO-NOT-REPORT"]}

    with pytest.raises(ShoppingObservationPolicyError) as caught:
        _project(raw)

    assert caught.value.code is ShoppingObservationPolicyErrorCode.STATE_INVALID
    assert "DO-NOT-REPORT" not in str(caught.value)


def test_public_observation_is_immutable() -> None:
    public = ShoppingPublicObservation(state="visible")

    with pytest.raises(FrozenInstanceError):
        public.state = "changed"  # type: ignore[misc]


def test_projection_strips_outer_whitespace_but_preserves_page_format() -> None:
    raw = {"state": " \nWebShop [SEP]\nSearch [SEP] 搜索\n "}

    public = _project(raw)

    assert public.state == "WebShop [SEP]\nSearch [SEP] 搜索"


@pytest.mark.parametrize(
    ("state", "max_chars", "expected"),
    [
        ("abcd", 4, "abcd"),
        ("abcde", 4, "abc…"),
        ("ab", 1, "…"),
    ],
)
def test_projection_applies_deterministic_character_limit(
    state: str,
    max_chars: int,
    expected: str,
) -> None:
    public = _project(
        {"state": state},
        max_observation_chars=max_chars,
    )

    assert public.state == expected
    assert len(public.state) <= max_chars


@pytest.mark.parametrize("max_chars", [None, True, 0, -1, 1.5, "16"])
def test_invalid_character_limit_fails_closed(max_chars: object) -> None:
    with pytest.raises(ShoppingObservationPolicyError) as caught:
        _project(
            {"state": "visible"},
            max_observation_chars=max_chars,
        )

    assert caught.value.code is ShoppingObservationPolicyErrorCode.MAX_OBSERVATION_CHARS_INVALID


@pytest.mark.parametrize(
    "state",
    [
        "Invalid Action.\n\nWebShop [SEP] Search",
        "WEBSHOPPLUS TASK COMPLETED\nPurchased: B000000000",
        "WEBSHOPPLUS SOFT EXIT\nReason: repeated action",
    ],
)
def test_native_error_and_terminal_states_remain_public(state: str) -> None:
    assert _project({"state": state}).state == state


@pytest.mark.parametrize("state", ["", "   ", "\ud800"])
def test_direct_public_observation_construction_rejects_invalid_state(
    state: str,
) -> None:
    with pytest.raises(ValueError):
        ShoppingPublicObservation(state=state)
