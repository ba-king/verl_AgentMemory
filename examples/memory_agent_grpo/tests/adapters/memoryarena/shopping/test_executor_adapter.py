from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping import tool_definitions
from examples.memory_agent_grpo.adapters.memoryarena.shopping.environment_session import (
    ShoppingEnvironmentSession,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.executor_adapter import (
    ShoppingToolExecutorAdapter,
)
from examples.memory_agent_grpo.answer_agent.response import AnswerAgentToolCall
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

    def reset(self, seed: object) -> object:
        return None

    def observe(self) -> str:
        return "WebShop [SEP] Search [SEP] search"

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
    client = _Client(step_outputs=[_StepOutput(state="public search results", done=False)])

    result = _adapter(client).execute(_validated_call())

    assert client.step_calls == ["search[face cleanser]"]
    assert result.is_terminal is False
    assert result.tool_observation.status is ToolObservationStatus.SUCCESS
    assert result.tool_observation.content == "public search results"
    assert "SEALED" not in repr(result.to_dict())


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
