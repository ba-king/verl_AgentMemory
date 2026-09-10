from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping.environment_session import (
    ShoppingEnvironmentSession,
    ShoppingEnvironmentSessionConfig,
    ShoppingEnvironmentSessionError,
    ShoppingEnvironmentSessionErrorCode,
    ShoppingTerminationKind,
)


def _single_product_task_definition() -> dict[str, object]:
    return {
        "task_id": "sealed-task-id",
        "agent_instruction": (
            "You are shopping. Compare all public candidates.\n\nProduct 1:\nBuy the lowest-priced public option."
        ),
        "steps": [
            {
                "step": 1,
                "target_asin": "SEALED-TARGET-ASIN",
                "requirements": {"attributes": ["SEALED-EVALUATOR-DATA"]},
            }
        ],
        "target_products": ["SEALED-TARGET-ASIN"],
        "evaluation_criteria": {"SEALED-EVALUATOR": True},
    }


class _ForbiddenReward:
    def __get__(self, instance: object, owner: object) -> object:
        raise AssertionError("reward must not be read")


@dataclass
class _FakeStepOutput:
    state: object
    done: object
    reward = _ForbiddenReward()


@dataclass
class _FakeClient:
    enable_feedback: object = False
    task_def: object = field(default_factory=_single_product_task_definition)
    observations: list[object] = field(
        default_factory=lambda: ["WebShop [SEP] Instruction: [SEP] Product 1 [SEP] Search [SEP] search"]
    )
    step_outputs: list[object] = field(default_factory=list)
    reset_calls: list[object] = field(default_factory=list)
    step_calls: list[object] = field(default_factory=list)
    close_calls: int = 0
    purchased_asins: object = field(default_factory=list)
    episode_done: object = False

    def reset(self, seed: object) -> object:
        self.reset_calls.append(seed)
        return {
            "target_products": ["DO-NOT-READ"],
            "reward": "DO-NOT-READ",
        }

    def observe(self) -> object:
        return self.observations[-1]

    def step(self, action: object) -> object:
        self.step_calls.append(action)
        if not self.step_outputs:
            raise RuntimeError("private upstream failure")
        return self.step_outputs.pop(0)

    def close(self) -> None:
        self.close_calls += 1


@dataclass
class _AvailableActionsClient(_FakeClient):
    available_actions: list[object] = field(default_factory=list)
    get_calls: list[str] = field(default_factory=list)

    def _get(self, path: str) -> object:
        self.get_calls.append(path)
        response = self.available_actions.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _session(
    client: object | None = None,
    *,
    max_observation_chars: object = 16_000,
) -> ShoppingEnvironmentSession:
    return ShoppingEnvironmentSession(
        upstream_client=_FakeClient() if client is None else client,
        max_observation_chars=max_observation_chars,
    )


def test_reset_returns_only_public_state_and_ignores_reset_payload() -> None:
    client = _FakeClient()
    session = _session(client)

    observation = session.reset(seed=7)

    assert client.reset_calls == [7]
    assert observation.to_dict() == {"state": ("WebShop [SEP] Instruction: [SEP] Product 1 [SEP] Search [SEP] search")}
    serialized = repr(observation.to_dict())
    assert "SEALED" not in serialized
    assert "DO-NOT-READ" not in serialized


def test_step_returns_public_state_and_separate_trusted_done_without_reward() -> None:
    client = _FakeClient(
        purchased_asins=["SEALED-TARGET-ASIN"],
        episode_done=True,
        step_outputs=[
            _FakeStepOutput(
                state="WebShop [SEP] Results [SEP] ITEM-1 [SEP] clickables",
                done=False,
            ),
            _FakeStepOutput(
                state="WEBSHOPPLUS TASK COMPLETED\nPurchased Items: ITEM-1",
                done=True,
            ),
        ],
    )
    session = _session(client)
    session.reset()

    first = session.step("search[public keywords]")
    second = session.step("click[Buy Now]")

    assert client.step_calls == ["search[public keywords]", "click[Buy Now]"]
    assert first.public_observation.state.startswith("WebShop [SEP] Results")
    assert first.is_terminal is False
    assert first.termination_evidence.kind is ShoppingTerminationKind.NOT_TERMINATED
    assert second.public_observation.state.startswith("WEBSHOPPLUS TASK COMPLETED")
    assert second.is_terminal is True
    assert second.termination_evidence.kind is ShoppingTerminationKind.PURCHASE_COMPLETED
    assert second.termination_evidence.purchased_target_matched is True
    assert session.is_terminal is True
    assert session.termination_evidence == second.termination_evidence


def test_clickable_items_are_fetched_fresh_and_keep_exact_labels() -> None:
    first_page = ["buy now", " Mixed Case ", "红色", ""]
    client = _AvailableActionsClient(
        available_actions=[
            {"clickables": first_page},
            {"clickables": ["< prev", "back to search"]},
            {"clickables": []},
        ],
    )
    session = _session(client)
    session.reset(seed=7)

    first = session.get_clickable_items()
    second = session.get_clickable_items()
    empty = session.get_clickable_items()
    first_page.append("later mutation")

    assert first == ("buy now", " Mixed Case ", "红色", "")
    assert second == ("< prev", "back to search")
    assert empty == ()
    assert client.get_calls == ["available_actions"] * 3
    assert client.reset_calls == [7]
    assert client.step_calls == []
    assert session.is_terminal is False
    assert session.termination_evidence.kind is ShoppingTerminationKind.NOT_TERMINATED


def test_clickable_items_ignore_private_response_fields_and_task_definition() -> None:
    class ForbiddenValue:
        def __repr__(self) -> str:
            raise AssertionError("private value must not be formatted")

    client = _AvailableActionsClient(
        available_actions=[
            {
                "clickables": ["public item"],
                "reward": ForbiddenValue(),
                "target_products": ForbiddenValue(),
                "has_search_bar": ForbiddenValue(),
            }
        ],
    )
    session = _session(client)
    session.reset()
    # This public read does not re-read the evaluator-bearing task definition.
    client.task_def = None

    assert session.get_clickable_items() == ("public item",)
    assert client.step_calls == []


def test_available_actions_capability_is_checked_only_when_called() -> None:
    client = _FakeClient(step_outputs=[_FakeStepOutput(state="public page", done=False)])
    session = _session(client)
    session.reset()
    session.step("search[keywords]")

    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        session.get_clickable_items()

    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.CLIENT_INVALID
    assert client.step_calls == ["search[keywords]"]


def test_available_actions_noncallable_capability_is_rejected() -> None:
    class InvalidClient(_FakeClient):
        _get = "PRIVATE-NONCALLABLE"

    session = _session(InvalidClient())
    session.reset()

    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        session.get_clickable_items()

    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.CLIENT_INVALID
    assert "PRIVATE-NONCALLABLE" not in str(caught.value)


def test_available_actions_attribute_error_is_sanitized() -> None:
    class InvalidClient(_FakeClient):
        @property
        def _get(self) -> object:
            raise RuntimeError("PRIVATE-ATTRIBUTE-FAILURE")

    session = _session(InvalidClient())
    session.reset()

    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        session.get_clickable_items()

    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.CLIENT_INVALID
    rendered = "".join(traceback.format_exception(caught.value))
    assert "RuntimeError: PRIVATE-ATTRIBUTE-FAILURE" not in rendered
    assert caught.value.__suppress_context__ is True


def test_available_actions_request_error_is_sanitized() -> None:
    client = _AvailableActionsClient(available_actions=[RuntimeError("PRIVATE-REQUEST-FAILURE")])
    session = _session(client)
    session.reset()

    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        session.get_clickable_items()

    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.AVAILABLE_ACTIONS_FAILED
    assert "PRIVATE-REQUEST-FAILURE" not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__suppress_context__ is True
    assert client.get_calls == ["available_actions"]
    assert client.step_calls == []
    assert session.is_terminal is False


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        "PRIVATE-INVALID-RESPONSE",
        {},
        {"clickables": None},
        {"clickables": "PRIVATE-INVALID-RESPONSE"},
        {"clickables": ("public item",)},
        {"clickables": [123]},
        {"clickables": [True]},
        {"clickables": [b"public item"]},
        {"clickables": ["PRIVATE-INVALID-RESPONSE\ud800"]},
    ],
)
def test_invalid_available_actions_schema_is_rejected(response: object) -> None:
    client = _AvailableActionsClient(available_actions=[response])
    session = _session(client)
    session.reset()

    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        session.get_clickable_items()

    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.AVAILABLE_ACTIONS_OUTPUT_INVALID
    assert "PRIVATE-INVALID-RESPONSE" not in str(caught.value)
    assert client.step_calls == []


def test_clickable_items_require_an_active_reset_session() -> None:
    client = _AvailableActionsClient(
        available_actions=[{"clickables": ["public item"]}],
        step_outputs=[_FakeStepOutput(state="Task complete", done=True)],
    )
    session = _session(client)

    with pytest.raises(ShoppingEnvironmentSessionError) as before_reset:
        session.get_clickable_items()
    assert before_reset.value.code is ShoppingEnvironmentSessionErrorCode.SESSION_NOT_RESET

    session.reset()
    session.step("click[Buy Now]")
    with pytest.raises(ShoppingEnvironmentSessionError) as after_terminal:
        session.get_clickable_items()
    assert after_terminal.value.code is ShoppingEnvironmentSessionErrorCode.SESSION_TERMINATED

    session.reset(seed=1)
    assert session.get_clickable_items() == ("public item",)
    session.close()
    with pytest.raises(ShoppingEnvironmentSessionError) as after_close:
        session.get_clickable_items()
    assert after_close.value.code is ShoppingEnvironmentSessionErrorCode.SESSION_CLOSED
    assert client.get_calls == ["available_actions"]


def test_soft_exit_is_terminal_but_not_purchase_completion() -> None:
    client = _FakeClient(
        episode_done=True,
        step_outputs=[
            _FakeStepOutput(
                state="Public page\nWEBSHOPPLUS SOFT EXIT\nReason: repeated action",
                done=True,
            )
        ],
    )
    session = _session(client)
    session.reset()

    result = session.step("search[repeated]")

    assert result.is_terminal is True
    assert result.termination_evidence.kind is ShoppingTerminationKind.SOFT_EXIT
    assert result.termination_evidence.purchased_target_matched is None


def test_completed_purchase_records_only_boolean_target_match() -> None:
    client = _FakeClient(
        purchased_asins=["DIFFERENT-ASIN"],
        episode_done=True,
        step_outputs=[
            _FakeStepOutput(
                state="WEBSHOPPLUS TASK COMPLETED\nPurchased Items: DIFFERENT-ASIN",
                done=True,
            )
        ],
    )
    session = _session(client)
    session.reset()

    result = session.step("click[Buy Now]")

    assert result.termination_evidence.kind is ShoppingTerminationKind.PURCHASE_COMPLETED
    assert result.termination_evidence.purchased_target_matched is False


@pytest.mark.parametrize(
    ("state", "purchased_asins", "episode_done"),
    [
        ("Task complete", ["SEALED-TARGET-ASIN"], True),
        ("WEBSHOPPLUS TASK COMPLETED", [], True),
        ("WEBSHOPPLUS TASK COMPLETED", ["SEALED-TARGET-ASIN"], False),
        ("WEBSHOPPLUS TASK COMPLETED", ["SEALED-TARGET-ASIN", "EXTRA"], True),
    ],
)
def test_done_without_complete_trusted_purchase_evidence_is_other_terminal(
    state: str,
    purchased_asins: list[str],
    episode_done: bool,
) -> None:
    client = _FakeClient(
        purchased_asins=purchased_asins,
        episode_done=episode_done,
        step_outputs=[_FakeStepOutput(state=state, done=True)],
    )
    session = _session(client)
    session.reset()

    result = session.step("click[Buy Now]")

    assert result.termination_evidence.kind is ShoppingTerminationKind.OTHER_TERMINAL
    assert result.termination_evidence.purchased_target_matched is None


def test_environment_state_is_capped_before_leaving_session() -> None:
    client = _FakeClient(
        observations=["abcdefgh"],
        step_outputs=[_FakeStepOutput(state="12345678", done=False)],
    )
    session = _session(client, max_observation_chars=5)

    initial = session.reset()
    result = session.step("search[keywords]")

    assert initial.state == "abcd…"
    assert result.public_observation.state == "1234…"


def test_step_requires_reset_and_terminal_session_rejects_more_actions() -> None:
    client = _FakeClient(step_outputs=[_FakeStepOutput(state="Task complete", done=True)])
    session = _session(client)

    with pytest.raises(ShoppingEnvironmentSessionError) as before_reset:
        session.step("search[keywords]")
    assert before_reset.value.code is ShoppingEnvironmentSessionErrorCode.SESSION_NOT_RESET

    session.reset()
    session.step("click[Buy Now]")
    with pytest.raises(ShoppingEnvironmentSessionError) as after_terminal:
        session.step("click[Buy Now]")
    assert after_terminal.value.code is ShoppingEnvironmentSessionErrorCode.SESSION_TERMINATED
    assert client.step_calls == ["click[Buy Now]"]


def test_reset_reopens_a_terminal_single_product_session() -> None:
    client = _FakeClient(step_outputs=[_FakeStepOutput(state="Task complete", done=True)])
    session = _session(client)
    session.reset()
    session.step("click[Buy Now]")

    session.reset(seed=1)

    assert session.is_terminal is False
    assert session.termination_evidence.kind is ShoppingTerminationKind.NOT_TERMINATED
    assert client.reset_calls == [0, 1]


def test_close_is_idempotent_and_prevents_reuse() -> None:
    client = _FakeClient()
    session = _session(client)

    session.close()
    session.close()

    assert client.close_calls == 1
    assert session.is_closed is True
    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        session.reset()
    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.SESSION_CLOSED


@pytest.mark.parametrize("enable_feedback", [True, 1, None, "false"])
def test_feedback_must_be_exactly_disabled(enable_feedback: object) -> None:
    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        _session(_FakeClient(enable_feedback=enable_feedback))

    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.FEEDBACK_NOT_DISABLED


def test_feedback_is_rechecked_before_every_environment_call() -> None:
    client = _FakeClient()
    session = _session(client)
    client.enable_feedback = True

    with pytest.raises(ShoppingEnvironmentSessionError) as reset_error:
        session.reset()

    assert reset_error.value.code is ShoppingEnvironmentSessionErrorCode.FEEDBACK_NOT_DISABLED
    assert client.reset_calls == []


def test_public_instruction_is_rechecked_before_step() -> None:
    client = _FakeClient(step_outputs=[_FakeStepOutput(state="public", done=False)])
    session = _session(client)
    session.reset()
    assert type(client.task_def) is dict
    client.task_def["agent_instruction"] = "Product 1:\nBuy SEALED-TARGET-ASIN"

    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        session.step("search[keywords]")

    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.PUBLIC_INSTRUCTION_INVALID
    assert client.step_calls == []


def test_full_task_is_rejected_before_environment_reset() -> None:
    task_definition = _single_product_task_definition()
    task_definition["steps"] = [
        {"target_asin": "TARGET-1"},
        {"target_asin": "TARGET-2"},
    ]
    client = _FakeClient(task_def=task_definition)

    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        _session(client)

    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.SINGLE_PRODUCT_TASK_INVALID
    assert client.reset_calls == []


def test_task_with_extra_evaluator_targets_is_rejected() -> None:
    task_definition = _single_product_task_definition()
    task_definition["target_products"] = [
        "SEALED-TARGET-ASIN",
        "SEALED-FUTURE-TARGET",
    ]

    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        _session(_FakeClient(task_def=task_definition))

    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.SINGLE_PRODUCT_TASK_INVALID


@pytest.mark.parametrize(
    "instruction",
    [
        "Product 1:\nFirst\nProduct 2:\nFuture",
        (
            "*** What You Need to Buy at Current Step ***\n"
            "use the ground-truth items in Purchase History as references\n"
            "Product 1:\nCurrent"
        ),
        "*** Purchase History ***\nOld target\nProduct 1:\nCurrent",
        "*** purchase history ***\nOld target\nProduct 1:\nCurrent",
        "Product 1:\nBuy SEALED-TARGET-ASIN",
    ],
)
def test_future_evaluator_or_target_instruction_is_rejected(
    instruction: str,
) -> None:
    task_definition = _single_product_task_definition()
    task_definition["agent_instruction"] = instruction

    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        _session(_FakeClient(task_def=task_definition))

    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.PUBLIC_INSTRUCTION_INVALID
    assert "SEALED-TARGET-ASIN" not in str(caught.value)


def test_invalid_upstream_done_is_rejected_without_reading_reward() -> None:
    client = _FakeClient(step_outputs=[_FakeStepOutput(state="public", done=1)])
    session = _session(client)
    session.reset()

    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        session.step("search[keywords]")

    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.STEP_OUTPUT_INVALID


def test_private_upstream_exception_is_not_exposed() -> None:
    client = _FakeClient(step_outputs=[])
    session = _session(client)
    session.reset()

    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        session.step("search[keywords]")

    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.STEP_FAILED
    assert "private upstream failure" not in str(caught.value)


def test_from_memoryarena_is_lazy_and_forces_safe_client_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_kwargs: list[dict[str, object]] = []
    client = _FakeClient()

    def fake_client_type(**kwargs: object) -> _FakeClient:
        created_kwargs.append(kwargs)
        return client

    def fake_import(name: str) -> object:
        assert name == ("env.env_systems.web_shopping_env.webshop_plus_client")
        return SimpleNamespace(WebshopPlusEnvClient=fake_client_type)

    monkeypatch.setattr(
        "examples.memory_agent_grpo.adapters.memoryarena.shopping.environment_session.importlib.import_module",
        fake_import,
    )
    config = ShoppingEnvironmentSessionConfig(
        env_server_base="http://127.0.0.1:36004",
        single_product_task_file="/tmp/product-1.json",
        max_observation_chars=12_000,
        timeout_seconds=17,
        max_rounds=25,
        product_catalog_dir="/catalog",
        domain_data_path="/domain.json",
    )

    session = ShoppingEnvironmentSession.from_memoryarena(config)

    assert session.max_observation_chars == 12_000
    assert created_kwargs == [
        {
            "env_server_base": "http://127.0.0.1:36004",
            "task_file": "/tmp/product-1.json",
            "data_len": 1,
            "timeout": 17,
            "max_rounds": 25,
            "product_catalog_dir": "/catalog",
            "domain_data_path": "/domain.json",
            "enable_feedback": False,
            "action_format": "react",
        }
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"env_server_base": ""},
        {"single_product_task_file": ""},
        {"max_observation_chars": 0},
        {"timeout_seconds": True},
        {"max_rounds": 0},
        {"product_catalog_dir": ""},
    ],
)
def test_invalid_session_configuration_is_rejected(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "env_server_base": "http://127.0.0.1:36004",
        "single_product_task_file": "/tmp/product-1.json",
        "max_observation_chars": 16_000,
    }
    values.update(kwargs)

    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        ShoppingEnvironmentSessionConfig(**values)  # type: ignore[arg-type]

    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.CONFIG_INVALID
