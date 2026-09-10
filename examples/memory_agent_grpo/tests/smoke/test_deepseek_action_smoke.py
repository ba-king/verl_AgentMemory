import pytest

from examples.memory_agent_grpo.memory_agent.action.protocol import MemoryDecisionPhase
from examples.memory_agent_grpo.memory_agent.action.validation import ValidationErrorCode, ValidationStatus
from examples.memory_agent_grpo.memory_agent.decision.model_input import MemoryAgentInput, MemoryAgentMessage
from examples.memory_agent_grpo.smoke.deepseek_action_smoke import (
    DEEPSEEK_CONTEXT_WINDOW_TOKENS,
    DEEPSEEK_MODEL,
    DeepSeekV4ConservativeTokenCounter,
    build_deepseek_smoke_client,
    build_deepseek_smoke_context,
    run_deepseek_action_smoke,
)


class _FakeClient:
    def __init__(self, raw_output: str) -> None:
        self.raw_output = raw_output
        self.calls: list[MemoryAgentInput] = []

    def generate(self, model_input: MemoryAgentInput) -> str:
        self.calls.append(model_input)
        return self.raw_output


def test_smoke_context_is_pre_task_and_motivates_retrieval() -> None:
    context = build_deepseek_smoke_context()

    assert context.phase is MemoryDecisionPhase.PRE_TASK
    assert context.bank_record_count > 0
    assert "saved" in context.task_input
    assert context.visible_memory_ids == ()


def test_conservative_counter_uses_utf8_bytes_and_large_template_reserve() -> None:
    counter = DeepSeekV4ConservativeTokenCounter(
        tokens_per_utf8_byte=4,
        chat_template_token_reserve=100,
    )
    model_input = (
        MemoryAgentMessage(role="system", content="a"),
        MemoryAgentMessage(role="user", content="旅"),
    )

    count = counter.count_input_tokens(model_input, model=DEEPSEEK_MODEL)

    assert count == 100 + 4 * len("a旅".encode("utf-8"))


def test_conservative_counter_rejects_an_unreviewed_model() -> None:
    counter = DeepSeekV4ConservativeTokenCounter()

    with pytest.raises(ValueError, match="Unsupported model"):
        counter.count_input_tokens(
            (
                MemoryAgentMessage(role="system", content="system"),
                MemoryAgentMessage(role="user", content="user"),
            ),
            model="another-model",
        )


@pytest.mark.parametrize(
    ("multiplier", "reserve"),
    [(0, 1), (True, 1), ("4", 1), (1, -1), (1, True), (1, "100")],
)
def test_conservative_counter_rejects_invalid_configuration(
    multiplier: object,
    reserve: object,
) -> None:
    with pytest.raises(ValueError):
        DeepSeekV4ConservativeTokenCounter(
            tokens_per_utf8_byte=multiplier,  # type: ignore[arg-type]
            chat_template_token_reserve=reserve,  # type: ignore[arg-type]
        )


def test_real_client_configuration_is_frozen_without_reading_a_key() -> None:
    client = build_deepseek_smoke_client(api_key_env="TEST_DEEPSEEK_KEY")

    assert client.config.base_url == "https://api.deepseek.com"
    assert client.config.model == DEEPSEEK_MODEL
    assert client.config.api_key_env == "TEST_DEEPSEEK_KEY"
    assert client.config.context_window_tokens == DEEPSEEK_CONTEXT_WINDOW_TOKENS
    assert client.config.max_tokens == 4096


def test_valid_retrieve_action_passes_without_execution() -> None:
    client = _FakeClient(
        '{"memory_action":{"action":"RETRIEVE","arguments":{"query":"travel preferences"}}}'
    )

    result = run_deepseek_action_smoke(client)

    assert result.emitted_executable_action
    assert result.validation.status is ValidationStatus.VALID
    assert result.to_dict()["decision"] == {
        "memory_action": {
            "action": "RETRIEVE",
            "arguments": {"query": "travel preferences"},
        }
    }
    assert len(client.calls) == 1
    assert tuple(message.role for message in client.calls[0]) == ("system", "user")


def test_valid_null_is_reported_but_does_not_pass_action_smoke() -> None:
    client = _FakeClient('{"memory_action":null}')

    result = run_deepseek_action_smoke(client)

    assert result.validation.status is ValidationStatus.VALID
    assert not result.emitted_executable_action
    assert result.to_dict()["decision"] == {"memory_action": None}
    assert len(client.calls) == 1


def test_invalid_output_is_reported_without_repair_or_second_call() -> None:
    client = _FakeClient("```json\n{}\n```")

    result = run_deepseek_action_smoke(client)

    assert result.validation.status is ValidationStatus.INVALID_OUTPUT
    assert result.validation.error_code is ValidationErrorCode.INVALID_JSON
    assert not result.emitted_executable_action
    assert result.raw_output == "```json\n{}\n```"
    assert len(client.calls) == 1
