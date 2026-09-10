"""Run one DeepSeek request through input building and action validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from examples.memory_agent_grpo.memory_agent.action.protocol import MemoryDecisionPhase
from examples.memory_agent_grpo.memory_agent.action.validation import (
    ValidationResult,
    ValidationStatus,
    validate_memory_decision_output,
)
from examples.memory_agent_grpo.memory_agent.decision.context import MemoryDecisionContext
from examples.memory_agent_grpo.memory_agent.decision.model_client import (
    MemoryAgentClient,
    MemoryAgentClientConfig,
    MemoryAgentClientError,
    MemoryAgentResponseFormat,
    MemoryAgentThinkingMode,
    OpenAICompatibleMemoryAgentClient,
)
from examples.memory_agent_grpo.memory_agent.decision.model_input import (
    MemoryAgentInput,
)
from examples.memory_agent_grpo.memory_agent.prompts.decision import (
    build_memory_agent_input,
)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_CONTEXT_WINDOW_TOKENS = 1_000_000
DEFAULT_API_KEY_ENV = "P2R_ANSWER_API_KEY"


@dataclass(frozen=True)
class DeepSeekV4ConservativeTokenCounter:
    """A deliberately loose upper bound for the two-message smoke prompt.

    DeepSeek uses a byte-level tokenizer. Multiplying every UTF-8 byte by four
    and adding a large fixed chat-template reserve is intentionally much more
    conservative than the actual count for this small test. This counter is
    smoke-test-specific; training code must use the deployed tokenizer.
    """

    tokens_per_utf8_byte: int = 4
    chat_template_token_reserve: int = 65_536

    def __post_init__(self) -> None:
        if (
            isinstance(self.tokens_per_utf8_byte, bool)
            or not isinstance(self.tokens_per_utf8_byte, int)
            or self.tokens_per_utf8_byte < 1
        ):
            raise ValueError("tokens_per_utf8_byte must be a positive integer.")
        if (
            isinstance(self.chat_template_token_reserve, bool)
            or not isinstance(self.chat_template_token_reserve, int)
            or self.chat_template_token_reserve < 0
        ):
            raise ValueError("chat_template_token_reserve must be a non-negative integer.")

    def count_input_tokens(self, model_input: MemoryAgentInput, *, model: str) -> int:
        if model != DEEPSEEK_MODEL:
            raise ValueError(f"Unsupported model for this conservative counter: {model!r}.")
        content_bytes = sum(len(message.content.encode("utf-8")) for message in model_input)
        return self.chat_template_token_reserve + self.tokens_per_utf8_byte * content_bytes


@dataclass(frozen=True)
class DeepSeekActionSmokeResult:
    raw_output: str
    validation: ValidationResult

    @property
    def emitted_executable_action(self) -> bool:
        decision = self.validation.decision
        return (
            self.validation.status is ValidationStatus.VALID
            and decision is not None
            and decision.memory_action is not None
        )

    def to_dict(self) -> dict[str, object | None]:
        decision = self.validation.decision
        error_code = self.validation.error_code
        return {
            "smoke_pass": self.emitted_executable_action,
            "phase": MemoryDecisionPhase.PRE_TASK.value,
            "raw_output": self.raw_output,
            "validation_status": self.validation.status.value,
            "validation_error_code": None if error_code is None else error_code.value,
            "decision": None if decision is None else decision.to_dict(),
        }


def build_deepseek_smoke_context() -> MemoryDecisionContext:
    """Create a public PRE_TASK case that should motivate retrieval."""

    return MemoryDecisionContext(
        phase=MemoryDecisionPhase.PRE_TASK,
        task_input=(
            "Plan a three-day Boston trip for me while honoring my usual saved "
            "travel, lodging, budget, and dietary preferences."
        ),
        bank_record_count=3,
    )


def build_deepseek_smoke_client(*, api_key_env: str) -> OpenAICompatibleMemoryAgentClient:
    """Build the one-request DeepSeek client without reading the key yet."""

    return OpenAICompatibleMemoryAgentClient(
        MemoryAgentClientConfig(
            base_url=DEEPSEEK_BASE_URL,
            model=DEEPSEEK_MODEL,
            api_key_env=api_key_env,
            context_window_tokens=DEEPSEEK_CONTEXT_WINDOW_TOKENS,
            timeout_seconds=120.0,
            temperature=0.0,
            max_tokens=4096,
            response_format=MemoryAgentResponseFormat.JSON_OBJECT,
            thinking_mode=MemoryAgentThinkingMode.DISABLED,
        ),
        token_counter=DeepSeekV4ConservativeTokenCounter(),
    )


def run_deepseek_action_smoke(client: MemoryAgentClient) -> DeepSeekActionSmokeResult:
    """Make one model call, then validate its raw PRE_TASK decision."""

    context = build_deepseek_smoke_context()
    model_input = build_memory_agent_input(context)
    raw_output = client.generate(model_input)
    validation = validate_memory_decision_output(
        raw_output,
        phase=context.phase,
        visible_memory_ids=context.visible_memory_ids,
    )
    return DeepSeekActionSmokeResult(raw_output=raw_output, validation=validation)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="Environment variable containing the DeepSeek API key.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = run_deepseek_action_smoke(
            build_deepseek_smoke_client(api_key_env=args.api_key_env)
        )
    except MemoryAgentClientError as exc:
        print(
            json.dumps(
                {"smoke_pass": False, "client_error_code": exc.code.value, "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.emitted_executable_action else 2


if __name__ == "__main__":
    raise SystemExit(main())
