"""Make one initial DeepSeek Answer Agent call without executing its tool."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from examples.memory_agent_grpo.answer_agent.input import (
    AnswerAgentInitialMessages,
    AnswerAgentSubtaskInput,
)
from examples.memory_agent_grpo.answer_agent.model_client import (
    AnswerAgentClient,
    AnswerAgentClientError,
)
from examples.memory_agent_grpo.answer_agent.prompts.builder import (
    build_answer_agent_initial_messages,
)
from examples.memory_agent_grpo.answer_agent.providers.deepseek import (
    DeepSeekAnswerAgentClient,
    DeepSeekAnswerAgentClientConfig,
)
from examples.memory_agent_grpo.answer_agent.response import AnswerAgentResponse
from examples.memory_agent_grpo.answer_agent.runtime.conversation import (
    AnswerAgentConversation,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.definition import (
    ToolDefinition,
)

DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_CONTEXT_WINDOW_TOKENS = 1_000_000
DEFAULT_API_KEY_ENV = "P2R_ANSWER_API_KEY"
SMOKE_TOOL_NAME = "lookup_public_route"


@dataclass(frozen=True)
class DeepSeekAnswerConservativeTokenCounter:
    """A loose upper bound used only for this small fixed API smoke case."""

    tokens_per_utf8_byte: int = 4
    chat_and_tool_template_reserve: int = 65_536

    def __post_init__(self) -> None:
        if (
            isinstance(self.tokens_per_utf8_byte, bool)
            or not isinstance(self.tokens_per_utf8_byte, int)
            or self.tokens_per_utf8_byte < 1
        ):
            raise ValueError("tokens_per_utf8_byte must be a positive integer.")
        if (
            isinstance(self.chat_and_tool_template_reserve, bool)
            or not isinstance(self.chat_and_tool_template_reserve, int)
            or self.chat_and_tool_template_reserve < 0
        ):
            raise ValueError(
                "chat_and_tool_template_reserve must be a non-negative integer."
            )

    def count_initial_input_tokens(
        self,
        messages: AnswerAgentInitialMessages,
        tool_definitions: tuple[ToolDefinition, ...],
        *,
        model: str,
    ) -> int:
        if model != DEEPSEEK_MODEL:
            raise ValueError(f"Unsupported smoke model {model!r}.")
        message_bytes = sum(
            len(message.content.encode("utf-8")) for message in messages
        )
        tool_bytes = sum(
            len(definition.name.encode("utf-8"))
            + len(definition.description.encode("utf-8"))
            + len(definition.parameters_json_schema.encode("utf-8"))
            for definition in tool_definitions
        )
        return self.chat_and_tool_template_reserve + self.tokens_per_utf8_byte * (
            message_bytes + tool_bytes
        )

    def count_conversation_input_tokens(
        self,
        conversation: AnswerAgentConversation,
        tool_definitions: tuple[ToolDefinition, ...],
        *,
        model: str,
    ) -> int:
        base_count = self.count_initial_input_tokens(
            conversation.initial_messages,
            tool_definitions,
            model=model,
        )
        exchange_bytes = 0
        for exchange in conversation.tool_exchanges:
            response = exchange.assistant_response
            if response.text is not None:
                exchange_bytes += len(response.text.encode("utf-8"))
            for tool_call in response.tool_calls:
                exchange_bytes += len(tool_call.tool_call_id.encode("utf-8"))
                exchange_bytes += len(tool_call.name.encode("utf-8"))
                exchange_bytes += len(tool_call.arguments_json.encode("utf-8"))
            for observation in exchange.tool_observations:
                exchange_bytes += len(observation.tool_call_id.encode("utf-8"))
                exchange_bytes += len(observation.content.encode("utf-8"))
        return base_count + self.tokens_per_utf8_byte * exchange_bytes


@dataclass(frozen=True)
class DeepSeekAnswerInitialSmokeResult:
    response: AnswerAgentResponse

    @property
    def emitted_one_expected_tool_call(self) -> bool:
        return (
            len(self.response.tool_calls) == 1
            and self.response.tool_calls[0].name == SMOKE_TOOL_NAME
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "smoke_pass": self.emitted_one_expected_tool_call,
            "tool_was_executed": False,
            "response": self.response.to_dict(),
        }


def build_deepseek_answer_smoke_input() -> AnswerAgentSubtaskInput:
    """Create a public request whose missing fact requires the supplied tool."""

    return AnswerAgentSubtaskInput(
        task_prompt=(
            "Complete the current request using only public evidence. The requested "
            "route fact is not present in the prompt or memory. Use the available "
            "lookup tool instead of inventing a result."
        ),
        subtask_input=(
            "Find the current public transit route for smoke-case location ALPHA to "
            "smoke-case location BETA."
        ),
    )


def build_deepseek_answer_smoke_tool() -> ToolDefinition:
    """Return one smoke-only model-facing tool definition, with no executor."""

    return ToolDefinition(
        name=SMOKE_TOOL_NAME,
        description="Look up a public transit route between two named locations.",
        parameters_json_schema=(
            '{"type":"object","properties":{'
            '"origin":{"type":"string"},"destination":{"type":"string"}},'
            '"required":["origin","destination"],"additionalProperties":false}'
        ),
    )


def build_deepseek_answer_smoke_client(
    *,
    api_key_env: str,
) -> DeepSeekAnswerAgentClient:
    """Build the real provider without reading the API key yet."""

    return DeepSeekAnswerAgentClient(
        DeepSeekAnswerAgentClientConfig(
            model=DEEPSEEK_MODEL,
            api_key_env=api_key_env,
            context_window_tokens=DEEPSEEK_CONTEXT_WINDOW_TOKENS,
            timeout_seconds=120.0,
            temperature=0.0,
            max_tokens=1024,
        ),
        token_counter=DeepSeekAnswerConservativeTokenCounter(),
    )


def run_deepseek_answer_initial_smoke(
    client: AnswerAgentClient,
) -> DeepSeekAnswerInitialSmokeResult:
    """Generate once and return the normalized response without dispatching it."""

    messages = build_answer_agent_initial_messages(build_deepseek_answer_smoke_input())
    response = client.generate_initial(messages, (build_deepseek_answer_smoke_tool(),))
    return DeepSeekAnswerInitialSmokeResult(response=response)


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
        result = run_deepseek_answer_initial_smoke(
            build_deepseek_answer_smoke_client(api_key_env=args.api_key_env)
        )
    except AnswerAgentClientError as exc:
        print(
            json.dumps(
                {
                    "smoke_pass": False,
                    "client_error_code": exc.code.value,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.emitted_one_expected_tool_call else 2


if __name__ == "__main__":
    raise SystemExit(main())
