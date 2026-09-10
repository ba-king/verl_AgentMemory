from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping import tool_definitions
from examples.memory_agent_grpo.adapters.memoryarena.shopping.environment_session import (
    ShoppingEnvironmentSession,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.executor_adapter import (
    ShoppingToolExecutorAdapter,
)
from examples.memory_agent_grpo.answer_agent.input import AnswerAgentSubtaskInput
from examples.memory_agent_grpo.answer_agent.interaction_mode import AnswerAgentInteractionMode
from examples.memory_agent_grpo.answer_agent.response import AnswerAgentResponse, AnswerAgentToolCall
from examples.memory_agent_grpo.answer_agent.runtime import AnswerAgentSubtaskRunnerConfig
from examples.memory_agent_grpo.answer_agent.runtime.subtask_runner import AnswerAgentSubtaskRunner
from examples.memory_agent_grpo.answer_agent.tool_call_gateway import (
    ValidatedToolCall,
    validate_tool_call_for_execution,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.observation import (
    ToolObservationStatus,
)


@dataclass
class _StepOutput:
    state: object
    done: object
    reward: object = "SEALED-REWARD"


@dataclass
class _Client:
    enable_feedback: object = False
    task_def: object = field(
        default_factory=lambda: {
            "agent_instruction": "Public rules\n\nProduct 1:\nBuy one item.",
            "steps": [{"target_asin": "SEALED-TARGET"}],
            "target_products": ["SEALED-TARGET"],
        }
    )
    step_outputs: list[object] = field(default_factory=list)
    step_calls: list[str] = field(default_factory=list)
    available_actions: object = field(
        default_factory=lambda: {
            "has_search_bar": False,
            "clickables": ["buy now", "item-123"],
            "target_asin": "SEALED-TARGET",
        }
    )
    get_calls: list[str] = field(default_factory=list)
    get_error: Exception | None = None

    def reset(self, seed: object) -> object:
        return None

    def observe(self) -> str:
        return "WebShop [SEP] Search [SEP] search"

    def _get(self, path: str) -> object:
        self.get_calls.append(path)
        if self.get_error is not None:
            raise self.get_error
        return self.available_actions

    def step(self, action: str) -> object:
        self.step_calls.append(action)
        if not self.step_outputs:
            raise RuntimeError("private environment exception")
        return self.step_outputs.pop(0)

    def close(self) -> None:
        return None


def _definitions():
    return tool_definitions(
        [
            {
                "name": "search",
                "description": "Search products.",
                "parameters": {
                    "type": "object",
                    "properties": {"keywords": {"type": "string"}},
                    "required": ["keywords"],
                },
            },
            {
                "name": "click",
                "description": "Click one visible item.",
                "parameters": {
                    "type": "object",
                    "properties": {"item": {"type": "string"}},
                    "required": ["item"],
                },
            },
        ]
    )


def _validated_call(
    *,
    tool_call_id: str = "call-1",
    name: str = "search",
    arguments_json: str = '{"keywords":"face cleanser"}',
) -> ValidatedToolCall:
    return ValidatedToolCall(
        tool_call_id=tool_call_id,
        name=name,
        arguments_json=arguments_json,
    )


def _adapter(client: _Client) -> ShoppingToolExecutorAdapter:
    session = ShoppingEnvironmentSession(
        upstream_client=client,
        max_observation_chars=16_000,
    )
    session.reset()
    return ShoppingToolExecutorAdapter(session=session)


def test_search_is_translated_to_one_internal_react_action() -> None:
    client = _Client(
        step_outputs=[_StepOutput(state="public search results", done=False)],
        get_error=RuntimeError("available-actions endpoint should not be used by search"),
    )

    result = _adapter(client).execute(_validated_call())

    assert client.step_calls == ["search[face cleanser]"]
    assert result.is_terminal is False
    assert result.tool_observation.status is ToolObservationStatus.SUCCESS
    assert result.tool_observation.content == "public search results"
    assert "SEALED" not in repr(result.to_dict())
    assert client.get_calls == []


def test_gateway_argument_whitespace_is_not_rewritten() -> None:
    client = _Client(step_outputs=[_StepOutput(state="public", done=False)])

    result = _adapter(client).execute(_validated_call(arguments_json='{"keywords":"  face cleanser  "}'))

    assert result.tool_observation.status is ToolObservationStatus.SUCCESS
    assert client.step_calls == ["search[  face cleanser  ]"]


def test_click_terminal_state_is_preserved_for_runner_completion() -> None:
    client = _Client(
        step_outputs=[
            _StepOutput(
                state=("WEBSHOPPLUS TASK COMPLETED\nPurchased Items: PUBLIC-ITEM"),
                done=True,
            )
        ]
    )
    call = _validated_call(
        tool_call_id="call-buy",
        name="click",
        arguments_json='{"item":"Buy Now"}',
    )

    result = _adapter(client).execute(call)

    assert client.step_calls == ["click[Buy Now]"]
    assert result.is_terminal is True
    assert result.tool_observation.tool_call_id == "call-buy"
    assert result.tool_observation.tool_name == "click"
    assert result.tool_observation.content.startswith("WEBSHOPPLUS TASK COMPLETED")


@pytest.mark.parametrize(
    ("model_call", "expected_action"),
    [
        (
            AnswerAgentToolCall(
                tool_call_id="search-1",
                name="search",
                arguments_json='{"keywords": "cleanser"}',
            ),
            "search[cleanser]",
        ),
        (
            AnswerAgentToolCall(
                tool_call_id="click-1",
                name="click",
                arguments_json='{"item": "ITEM-123"}',
            ),
            "click[ITEM-123]",
        ),
    ],
)
def test_real_gateway_output_executes_without_argument_rewriting(
    model_call: AnswerAgentToolCall,
    expected_action: str,
) -> None:
    gateway_result = validate_tool_call_for_execution(
        model_call,
        tool_definitions=_definitions(),
    )
    assert gateway_result.validated_call is not None
    client = _Client(step_outputs=[_StepOutput(state="public", done=False)])

    result = _adapter(client).execute(gateway_result.validated_call)

    assert result.tool_observation.status is ToolObservationStatus.SUCCESS
    assert client.step_calls == [expected_action]


@pytest.mark.parametrize(
    ("name", "arguments_json"),
    [
        ("search", '{"keywords":""}'),
        ("search", '{"keywords":"unsafe]click[Buy Now"}'),
        ("search", '{"keywords":"line\\nbreak"}'),
        ("click", '{"item":""}'),
        ("click", '{"item":"unsafe["}'),
    ],
)
def test_unsafe_action_text_is_rejected_without_environment_step(
    name: str,
    arguments_json: str,
) -> None:
    client = _Client()

    result = _adapter(client).execute(_validated_call(name=name, arguments_json=arguments_json))

    assert client.step_calls == []
    assert result.is_terminal is False
    assert result.tool_observation.status is ToolObservationStatus.ERROR
    assert result.tool_observation.content == "Shopping action execution failed."


def test_private_session_failure_becomes_one_public_generic_error() -> None:
    client = _Client(step_outputs=[])

    result = _adapter(client).execute(_validated_call())

    assert client.step_calls == ["search[face cleanser]"]
    assert result.tool_observation.status is ToolObservationStatus.ERROR
    assert result.tool_observation.content == "Shopping action execution failed."
    assert "private" not in result.tool_observation.content


def test_gateway_permitted_extra_argument_is_safely_rejected() -> None:
    model_call = AnswerAgentToolCall(
        tool_call_id="search-extra",
        name="search",
        arguments_json='{"keywords":"cleanser","unused":"value"}',
    )
    gateway_result = validate_tool_call_for_execution(
        model_call,
        tool_definitions=_definitions(),
    )
    assert gateway_result.validated_call is not None
    client = _Client()

    result = _adapter(client).execute(gateway_result.validated_call)

    assert result.tool_observation.status is ToolObservationStatus.ERROR
    assert client.step_calls == []


@pytest.mark.parametrize(
    "call",
    [
        object(),
        _validated_call(tool_call_id=""),
        _validated_call(name="unknown"),
        _validated_call(arguments_json="not-json"),
        _validated_call(arguments_json='{"keywords": "not canonical"}'),
        _validated_call(arguments_json='{"keywords":"first","keywords":"second"}'),
    ],
)
def test_forged_non_gateway_calls_never_reach_environment(call: object) -> None:
    client = _Client()

    with pytest.raises(ValueError):
        _adapter(client).execute(call)

    assert client.step_calls == []


@pytest.mark.parametrize("session", [None, object(), _Client()])
def test_executor_requires_validated_shopping_session(session: object) -> None:
    with pytest.raises(ValueError, match="ShoppingEnvironmentSession"):
        ShoppingToolExecutorAdapter(session=session)


def test_buy_now_on_description_page_returns_error_and_current_buttons():
    client = _Client(available_actions={"clickables": ["back to search", "< prev"], "target_asin": "SEALED"})
    result = _adapter(client).execute(_validated_call(name="click", arguments_json='{"item":"Buy Now"}'))

    assert client.get_calls == ["available_actions"]
    assert client.step_calls == []
    assert result.is_terminal is False
    assert result.tool_observation.status is ToolObservationStatus.ERROR
    assert "not clickable" in result.tool_observation.content
    assert json.loads(result.tool_observation.content.split("Current clickable items: ")[1]) == [
        "back to search",
        "< prev",
    ]
    assert "SEALED" not in result.tool_observation.content


@pytest.mark.parametrize(
    ("item", "keys", "valid"),
    [
        ("BuY NoW", ["buy now"], True),
        ("ITEM-123", ["item-123"], True),
        ("< Prev", ["< prev"], True),
        (" Buy Now ", ["buy now"], False),
        ("RED", ["RED"], False),  # Uppercase radio keys are not lowercased by upstream.
        ("Search", ["search"], False),  # Upstream explicitly excludes click[search].
        ("Buy Now", [], False),
    ],
)
def test_click_membership_matches_upstream_case_and_search_rules(item, keys, valid):
    client = _Client(available_actions={"clickables": keys}, step_outputs=[_StepOutput("same page", False)])
    call = _validated_call(name="click", arguments_json=json.dumps({"item": item}, separators=(",", ":")))
    result = _adapter(client).execute(call)

    assert (result.tool_observation.status is ToolObservationStatus.SUCCESS) is valid
    assert client.step_calls == ([f"click[{item}]"] if valid else [])
    if item == "Search":
        assert result.tool_observation.content.endswith("[]")


@pytest.mark.parametrize("payload", [None, {}, {"clickables": "buy now"}, {"clickables": [42]}])
def test_unknown_clickability_does_not_execute_or_claim_invalid_button(payload):
    client = _Client(available_actions=payload)
    result = _adapter(client).execute(_validated_call(name="click", arguments_json='{"item":"Buy Now"}'))
    assert result.tool_observation.status is ToolObservationStatus.ERROR
    assert "Unable to verify" in result.tool_observation.content
    assert "not clickable" not in result.tool_observation.content
    assert client.step_calls == []


def test_clickability_network_error_is_private_and_does_not_execute():
    client = _Client(get_error=RuntimeError("private endpoint failure: SECRET"))
    result = _adapter(client).execute(_validated_call(name="click", arguments_json='{"item":"Buy Now"}'))
    assert result.tool_observation.status is ToolObservationStatus.ERROR
    assert "Unable to verify" in result.tool_observation.content
    assert "SECRET" not in repr(result)
    assert client.step_calls == []


def test_valid_click_that_keeps_page_unchanged_is_not_rejected():
    client = _Client(step_outputs=[_StepOutput("WebShop [SEP] Search [SEP] search", False)])
    result = _adapter(client).execute(_validated_call(name="click", arguments_json='{"item":"ITEM-123"}'))
    assert result.tool_observation.status is ToolObservationStatus.SUCCESS
    assert client.step_calls == ["click[ITEM-123]"]


class _ScriptedAnswer:
    def __init__(self, items):
        self.items = iter(items)
        self.conversations = []

    def generate(self, conversation, tool_definitions=()):
        self.conversations.append(conversation)
        return AnswerAgentResponse(
            tool_calls=(
                AnswerAgentToolCall(
                    tool_call_id=f"call-{len(self.conversations)}",
                    name="click",
                    arguments_json=json.dumps({"item": next(self.items)}),
                ),
            )
        )


def _run_answer(client, items):
    adapter = _adapter(client)
    answer = _ScriptedAnswer(items)
    result = AnswerAgentSubtaskRunner(
        client=answer,
        executor=adapter,
        config=AnswerAgentSubtaskRunnerConfig(
            max_inference_steps=4,
            max_tool_calls_per_inference=1,
            max_consecutive_tool_failures=2,
            interaction_mode=AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL,
        ),
    ).run(
        AnswerAgentSubtaskInput(task_prompt="Click a visible button.", subtask_input="Description page"),
        tool_definitions=_definitions(),
    )
    return result, answer


def test_rejected_click_reaches_answer_and_counts_toward_failure_limit():
    client = _Client(available_actions={"clickables": ["< prev", "back to search"]})
    result, answer = _run_answer(client, ["Buy Now", "Buy Now"])
    assert result.error_code.value == "CONSECUTIVE_TOOL_FAILURE_LIMIT"
    assert result.inference_step_count == 2
    feedback = answer.conversations[1].tool_exchanges[-1].tool_observations[0]
    assert feedback.status is ToolObservationStatus.ERROR and "< prev" in feedback.content
    assert client.step_calls == []


def test_answer_can_return_to_product_page_and_buy_after_error():
    class RecoveringClient(_Client):
        def step(self, action):
            result = super().step(action)
            self.available_actions = {"clickables": ["buy now", "description"]}
            return result

    client = RecoveringClient(
        available_actions={"clickables": ["< prev", "back to search"]},
        step_outputs=[_StepOutput("Product [SEP] Buy Now", False), _StepOutput("WEBSHOPPLUS TASK COMPLETED", True)],
    )
    result, answer = _run_answer(client, ["Buy Now", "< Prev", "Buy Now"])
    assert result.status.value == "COMPLETED"
    assert result.inference_step_count == 3
    assert client.step_calls == ["click[< Prev]", "click[Buy Now]"]
    assert len(client.get_calls) == 3
    assert [step.tool_outcomes[0].tool_observation.status.value for step in result.steps] == [
        "error",
        "success",
        "success",
    ]
    assert "not clickable" in answer.conversations[1].tool_exchanges[-1].tool_observations[0].content
