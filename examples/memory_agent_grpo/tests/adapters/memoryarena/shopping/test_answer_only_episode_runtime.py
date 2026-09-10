from __future__ import annotations

import json
import re
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping import (
    answer_only_episode_runtime as episode_runtime,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.answer_only_episode_runtime import (
    SHOPPING_ANSWER_ONLY_PRODUCT_COUNT,
    ShoppingAnswerOnlyEpisodeEvaluationPolicy,
    ShoppingAnswerOnlyEpisodeFailureCode,
    ShoppingAnswerOnlyEpisodeInputError,
    ShoppingAnswerOnlyEpisodeInputErrorCode,
    ShoppingAnswerOnlyEpisodeResult,
    ShoppingAnswerOnlyEpisodeRunner,
    ShoppingAnswerOnlyEpisodeRunStatus,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.environment_session import (
    ShoppingEnvironmentSession,
    ShoppingTerminationKind,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.task_input_adapter import (
    ShoppingTaskInputAdapter,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.task_splitter import (
    ShoppingTaskSplitterError,
    ShoppingTaskSplitterErrorCode,
)
from examples.memory_agent_grpo.answer_agent.response import (
    AnswerAgentResponse,
    AnswerAgentToolCall,
)
from examples.memory_agent_grpo.answer_agent.runtime import (
    AnswerAgentConversation,
    AnswerAgentInteractionMode,
    AnswerAgentSubtaskRunnerConfig,
)
from examples.memory_agent_grpo.answer_agent.tool_protocol.definition import (
    ToolDefinition,
)


def _full_task() -> dict[str, object]:
    product_sections = "".join(
        (
            "\n\n"
            + "-" * 64
            + f"\nProduct {ordinal}:\n"
            + f"Choose the public Product {ordinal} option matching its requirements."
        )
        for ordinal in range(1, SHOPPING_ANSWER_ONLY_PRODUCT_COUNT + 1)
    )
    targets = [f"SEALED-TARGET-{ordinal}" for ordinal in range(1, SHOPPING_ANSWER_ONLY_PRODUCT_COUNT + 1)]
    return {
        "task_id": "six_product_fixture",
        "task_type": "bundled_shopping",
        "task_description": "Choose six public products.",
        "agent_instruction": ("Use only public store evidence and respect the public bundle rules." + product_sections),
        "steps": [
            {
                "step": ordinal,
                "target_asin": targets[ordinal - 1],
                "step_description": f"Product {ordinal}",
                "requirements": {"public_requirement": ordinal},
            }
            for ordinal in range(1, SHOPPING_ANSWER_ONLY_PRODUCT_COUNT + 1)
        ],
        "target_products": targets,
        "global_constraints": {
            "max_steps": SHOPPING_ANSWER_ONLY_PRODUCT_COUNT,
            "purchase_order": "sequence",
        },
        "evaluation_criteria": {
            **{
                f"step_{ordinal}_success": {"sealed": ordinal}
                for ordinal in range(
                    1,
                    SHOPPING_ANSWER_ONLY_PRODUCT_COUNT + 1,
                )
            },
            "overall_success": {"all_steps_completed": True},
        },
    }


def _tool_definitions() -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="search",
            description="Search the public store.",
            parameters_json_schema=(
                '{"additionalProperties":false,"properties":{"keywords":'
                '{"type":"string"}},"required":["keywords"],"type":"object"}'
            ),
        ),
        ToolDefinition(
            name="click",
            description="Click one public page value.",
            parameters_json_schema=(
                '{"additionalProperties":false,"properties":{"item":'
                '{"type":"string"}},"required":["item"],"type":"object"}'
            ),
        ),
    )


def _runner_config() -> AnswerAgentSubtaskRunnerConfig:
    return AnswerAgentSubtaskRunnerConfig(
        max_inference_steps=3,
        max_tool_calls_per_inference=1,
        max_consecutive_tool_failures=2,
        interaction_mode=(AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL),
    )


@dataclass(frozen=True)
class _AnswerCall:
    product_ordinal: int
    conversation: AnswerAgentConversation
    tool_definitions: tuple[ToolDefinition, ...]


class _FakeAnswerClient:
    """One shared stateless client; repeated call IDs expose history leakage."""

    def __init__(
        self,
        *,
        text_only_products: set[int] | None = None,
        raised_products: set[int] | None = None,
    ) -> None:
        self.text_only_products = set() if text_only_products is None else text_only_products
        self.raised_products = set() if raised_products is None else raised_products
        self.calls: list[_AnswerCall] = []

    def generate(
        self,
        conversation: AnswerAgentConversation,
        tool_definitions: tuple[ToolDefinition, ...] = (),
    ) -> AnswerAgentResponse:
        initial_user_content = conversation.initial_messages[1].content
        match = re.search(r"PUBLIC_PRODUCT_(\d+)_INITIAL", initial_user_content)
        if match is None:
            raise AssertionError("current Product marker is missing")
        product_ordinal = int(match.group(1))
        self.calls.append(
            _AnswerCall(
                product_ordinal=product_ordinal,
                conversation=conversation,
                tool_definitions=tool_definitions,
            )
        )

        if product_ordinal in self.raised_products:
            raise RuntimeError("private provider failure")
        if product_ordinal in self.text_only_products and not conversation.tool_exchanges:
            return AnswerAgentResponse(text="Invalid text-only completion.")
        if not conversation.tool_exchanges:
            return AnswerAgentResponse(
                tool_calls=(
                    AnswerAgentToolCall(
                        tool_call_id="reused-search-call-id",
                        name="search",
                        arguments_json='{"keywords":"public query"}',
                    ),
                )
            )
        if len(conversation.tool_exchanges) == 1:
            return AnswerAgentResponse(
                tool_calls=(
                    AnswerAgentToolCall(
                        tool_call_id="reused-buy-call-id",
                        name="click",
                        arguments_json='{"item":"Buy Now"}',
                    ),
                )
            )
        raise AssertionError("unexpected extra inference")


class _FakeUpstreamClient:
    def __init__(
        self,
        *,
        single_product_task: dict[str, object],
        outcome: str,
        reset_raises: bool = False,
    ) -> None:
        self.enable_feedback = False
        self.task_def = single_product_task
        steps = single_product_task["steps"]
        assert type(steps) is list and len(steps) == 1
        step = steps[0]
        assert type(step) is dict
        self.product_ordinal = step["step"]
        self.target_asin = step["target_asin"]
        assert type(self.product_ordinal) is int
        assert type(self.target_asin) is str
        self.outcome = outcome
        self.reset_raises = reset_raises
        self.episode_done = False
        self.purchased_asins: list[str] = []
        self.reset_calls: list[int] = []
        self.step_calls: list[str] = []
        self.close_calls = 0

    def reset(self, seed: int) -> object:
        self.reset_calls.append(seed)
        if self.reset_raises:
            raise RuntimeError("private reset failure")
        self.episode_done = False
        self.purchased_asins = []
        return {"sealed_reset_payload": "MUST_NOT_ESCAPE"}

    def observe(self) -> str:
        instruction = self.task_def["agent_instruction"]
        assert type(instruction) is str
        return f"PUBLIC_PRODUCT_{self.product_ordinal}_INITIAL\n{instruction}\nSearch [SEP] search"

    def _get(self, path: str) -> dict[str, object]:
        assert path == "available_actions"
        return {"has_search_bar": not self.step_calls, "clickables": ["buy now"] if self.step_calls else ["search"]}

    def step(self, action: str) -> object:
        self.step_calls.append(action)
        if action.startswith("search["):
            return SimpleNamespace(
                state=(f"PUBLIC_PRODUCT_{self.product_ordinal}_SEARCH_RESULT [SEP] Buy Now"),
                done=False,
            )
        if action != "click[Buy Now]":
            raise AssertionError("unexpected fake Shopping action")

        self.episode_done = True
        if self.outcome == "match":
            self.purchased_asins = [self.target_asin]
            state = f"WEBSHOPPLUS TASK COMPLETED\nPUBLIC_PRODUCT_{self.product_ordinal}_PURCHASE"
        elif self.outcome == "mismatch":
            self.purchased_asins = ["WRONG-PUBLIC-ASIN"]
            state = f"WEBSHOPPLUS TASK COMPLETED\nPUBLIC_PRODUCT_{self.product_ordinal}_PURCHASE"
        elif self.outcome == "soft_exit":
            self.purchased_asins = []
            state = f"WEBSHOPPLUS SOFT EXIT\nPUBLIC_PRODUCT_{self.product_ordinal}_STOP"
        elif self.outcome == "other_terminal":
            self.purchased_asins = [self.target_asin]
            state = f"PUBLIC_PRODUCT_{self.product_ordinal}_OTHER_TERMINAL"
        else:
            raise AssertionError("unknown fake outcome")
        return SimpleNamespace(state=state, done=True)

    def close(self) -> None:
        self.close_calls += 1


class _FakeSessionFactory:
    def __init__(
        self,
        *,
        outcomes: dict[int, str] | None = None,
        fail_open_at: int | None = None,
        fail_reset_at: int | None = None,
    ) -> None:
        self.outcomes = {} if outcomes is None else outcomes
        self.fail_open_at = fail_open_at
        self.fail_reset_at = fail_reset_at
        self.open_calls: list[int] = []
        self.single_product_tasks: list[dict[str, object]] = []
        self.clients: list[_FakeUpstreamClient] = []

    def open_session(
        self,
        *,
        single_product_task: dict[str, object],
        product_ordinal: int,
    ) -> ShoppingEnvironmentSession:
        self.open_calls.append(product_ordinal)
        self.single_product_tasks.append(single_product_task)
        if self.fail_open_at == product_ordinal:
            raise RuntimeError("private session setup failure")
        client = _FakeUpstreamClient(
            single_product_task=single_product_task,
            outcome=self.outcomes.get(product_ordinal, "match"),
            reset_raises=self.fail_reset_at == product_ordinal,
        )
        self.clients.append(client)
        return ShoppingEnvironmentSession(
            upstream_client=client,
            max_observation_chars=16_000,
        )


def _episode_runner(
    *,
    answer_client: _FakeAnswerClient | None = None,
    session_factory: _FakeSessionFactory | None = None,
    evaluation_policy: ShoppingAnswerOnlyEpisodeEvaluationPolicy = (
        ShoppingAnswerOnlyEpisodeEvaluationPolicy.FAIL_FAST
    ),
) -> tuple[
    ShoppingAnswerOnlyEpisodeRunner,
    _FakeAnswerClient,
    _FakeSessionFactory,
]:
    client = _FakeAnswerClient() if answer_client is None else answer_client
    factory = _FakeSessionFactory() if session_factory is None else session_factory
    return (
        ShoppingAnswerOnlyEpisodeRunner(
            answer_client=client,
            session_factory=factory,
            task_input_adapter=ShoppingTaskInputAdapter(max_observation_chars=16_000),
            tool_definitions=_tool_definitions(),
            config=_runner_config(),
            evaluation_policy=evaluation_policy,
        ),
        client,
        factory,
    )


def test_six_products_complete_with_fresh_raw_conversation_per_product() -> None:
    runner, answer_client, session_factory = _episode_runner()

    result = runner.run(_full_task())

    assert result.status is ShoppingAnswerOnlyEpisodeRunStatus.COMPLETED
    assert result.passed is True
    assert result.completed_product_count == 6
    assert result.attempted_product_count == 6
    assert result.failed_product_ordinal is None
    assert result.failure_code is None
    assert session_factory.open_calls == [1, 2, 3, 4, 5, 6]
    assert len(session_factory.clients) == 6
    assert all(client.reset_calls == [0] for client in session_factory.clients)
    assert all(client.close_calls == 1 for client in session_factory.clients)
    assert len(answer_client.calls) == 12

    for product_ordinal in range(1, 7):
        product_calls = [call for call in answer_client.calls if call.product_ordinal == product_ordinal]
        assert len(product_calls) == 2
        initial_call, continuation_call = product_calls
        assert initial_call.conversation.tool_exchanges == ()
        assert len(continuation_call.conversation.tool_exchanges) == 1
        assert initial_call.tool_definitions == _tool_definitions()
        assert continuation_call.tool_definitions == _tool_definitions()

        initial_serialized = json.dumps(
            initial_call.conversation.to_dict(),
            ensure_ascii=False,
        )
        continuation_serialized = json.dumps(
            continuation_call.conversation.to_dict(),
            ensure_ascii=False,
        )
        assert f"PUBLIC_PRODUCT_{product_ordinal}_INITIAL" in initial_serialized
        assert '"retrieved_memories": []' in initial_call.conversation.initial_messages[1].content
        assert "SEALED-TARGET" not in initial_serialized
        assert f"PUBLIC_PRODUCT_{product_ordinal}_SEARCH_RESULT" in continuation_serialized
        for other_ordinal in set(range(1, 7)) - {product_ordinal}:
            assert f"PUBLIC_PRODUCT_{other_ordinal}_" not in initial_serialized
            assert f"PUBLIC_PRODUCT_{other_ordinal}_" not in continuation_serialized

    assert [product.passed for product in result.product_results] == [True] * 6
    assert all(product.terminal_tool_execution_count == 1 for product in result.product_results)
    assert all(
        product.termination_evidence.kind is ShoppingTerminationKind.PURCHASE_COMPLETED
        for product in result.product_results
    )
    safe_serialized = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "SEALED-TARGET" not in safe_serialized
    assert "PUBLIC_PRODUCT" not in safe_serialized
    assert all(
        product["emitted_tool_call_count"] == 2
        and product["successful_tool_observation_count"] == 2
        and product["error_tool_observation_count"] == 0
        for product in result.to_dict()["product_results"]
    )


def test_all_six_products_are_split_before_first_session_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, answer_client, session_factory = _episode_runner()
    split_calls: list[int] = []
    original_split = episode_runtime.build_single_product_task

    def fail_on_sixth(
        full_task_definition: object,
        *,
        product_ordinal: int,
    ) -> dict[str, object]:
        split_calls.append(product_ordinal)
        if product_ordinal == 6:
            raise ShoppingTaskSplitterError(
                ShoppingTaskSplitterErrorCode.STEPS_INVALID,
                "safe test failure",
            )
        return original_split(
            full_task_definition,
            product_ordinal=product_ordinal,
        )

    monkeypatch.setattr(
        episode_runtime,
        "build_single_product_task",
        fail_on_sixth,
    )

    with pytest.raises(ShoppingAnswerOnlyEpisodeInputError) as caught:
        runner.run(_full_task())

    assert caught.value.code is (ShoppingAnswerOnlyEpisodeInputErrorCode.TASK_SPLIT_FAILED)
    assert split_calls == [1, 2, 3, 4, 5, 6]
    assert session_factory.open_calls == []
    assert answer_client.calls == []


def test_invalid_future_target_fails_preflight_without_session_or_model_call() -> None:
    runner, answer_client, session_factory = _episode_runner()
    task = _full_task()
    steps = task["steps"]
    assert type(steps) is list and type(steps[5]) is dict
    steps[5]["target_asin"] = ""

    with pytest.raises(ShoppingAnswerOnlyEpisodeInputError) as caught:
        runner.run(task)

    assert caught.value.code is (ShoppingAnswerOnlyEpisodeInputErrorCode.TASK_SPLIT_FAILED)
    assert session_factory.open_calls == []
    assert answer_client.calls == []


def test_product_count_must_be_exactly_six_before_any_side_effect() -> None:
    runner, answer_client, session_factory = _episode_runner()
    task = _full_task()
    steps = task["steps"]
    assert type(steps) is list
    task["steps"] = steps[:-1]

    with pytest.raises(ShoppingAnswerOnlyEpisodeInputError) as caught:
        runner.run(task)

    assert caught.value.code is (ShoppingAnswerOnlyEpisodeInputErrorCode.PRODUCT_COUNT_INVALID)
    assert session_factory.open_calls == []
    assert answer_client.calls == []


def test_target_mismatch_fails_current_product_and_stops_future_products() -> None:
    factory = _FakeSessionFactory(outcomes={3: "mismatch"})
    runner, answer_client, _ = _episode_runner(session_factory=factory)

    result = runner.run(_full_task())

    assert result.status is ShoppingAnswerOnlyEpisodeRunStatus.FAILED
    assert result.completed_product_count == 2
    assert result.failed_product_ordinal == 3
    assert result.failure_code is (ShoppingAnswerOnlyEpisodeFailureCode.TARGET_MISMATCH)
    assert [product.product_ordinal for product in result.product_results] == [
        1,
        2,
        3,
    ]
    assert result.product_results[-1].passed is False
    assert result.product_results[-1].termination_evidence.purchased_target_matched is False
    assert result.stopped_product_ordinal is None
    assert result.stop_failure_code is None
    assert runner.evaluation_policy is ShoppingAnswerOnlyEpisodeEvaluationPolicy.FAIL_FAST
    assert factory.open_calls == [1, 2, 3]
    assert len(answer_client.calls) == 6
    assert all(client.close_calls == 1 for client in factory.clients)


def test_continue_policy_collects_products_after_target_mismatch() -> None:
    factory = _FakeSessionFactory(outcomes={3: "mismatch"})
    runner, answer_client, _ = _episode_runner(
        session_factory=factory,
        evaluation_policy=(ShoppingAnswerOnlyEpisodeEvaluationPolicy.CONTINUE_EVALUATION_FAILURES),
    )

    result = runner.run(_full_task())

    assert result.status is ShoppingAnswerOnlyEpisodeRunStatus.FAILED
    assert result.completed_product_count == 5
    assert result.attempted_product_count == 6
    assert result.failed_product_ordinal == 3
    assert result.failure_code is ShoppingAnswerOnlyEpisodeFailureCode.TARGET_MISMATCH
    assert result.stopped_product_ordinal is None
    assert result.stop_failure_code is None
    assert [product.passed for product in result.product_results] == [
        True,
        True,
        False,
        True,
        True,
        True,
    ]
    assert factory.open_calls == [1, 2, 3, 4, 5, 6]
    assert len(answer_client.calls) == 12


def test_continue_policy_collects_multiple_evaluation_failures() -> None:
    factory = _FakeSessionFactory(outcomes={2: "soft_exit", 4: "mismatch"})
    runner, _, _ = _episode_runner(
        session_factory=factory,
        evaluation_policy=(ShoppingAnswerOnlyEpisodeEvaluationPolicy.CONTINUE_EVALUATION_FAILURES),
    )

    result = runner.run(_full_task())

    assert result.status is ShoppingAnswerOnlyEpisodeRunStatus.FAILED
    assert result.completed_product_count == 4
    assert result.attempted_product_count == 6
    assert result.failed_product_ordinal == 2
    assert result.failure_code is (ShoppingAnswerOnlyEpisodeFailureCode.PURCHASE_NOT_COMPLETED)
    assert [product.failure_code for product in result.product_results] == [
        None,
        ShoppingAnswerOnlyEpisodeFailureCode.PURCHASE_NOT_COMPLETED,
        None,
        ShoppingAnswerOnlyEpisodeFailureCode.TARGET_MISMATCH,
        None,
        None,
    ]
    assert factory.open_calls == [1, 2, 3, 4, 5, 6]


def test_continue_policy_stops_on_retained_runner_failure_after_mismatch() -> None:
    factory = _FakeSessionFactory(outcomes={3: "mismatch"})
    answer_client = _FakeAnswerClient(text_only_products={5})
    runner, _, _ = _episode_runner(
        answer_client=answer_client,
        session_factory=factory,
        evaluation_policy=(ShoppingAnswerOnlyEpisodeEvaluationPolicy.CONTINUE_EVALUATION_FAILURES),
    )

    result = runner.run(_full_task())

    assert result.status is ShoppingAnswerOnlyEpisodeRunStatus.FAILED
    assert result.completed_product_count == 3
    assert result.attempted_product_count == 5
    assert result.failed_product_ordinal == 3
    assert result.failure_code is ShoppingAnswerOnlyEpisodeFailureCode.TARGET_MISMATCH
    assert result.stopped_product_ordinal is None
    assert result.stop_failure_code is None
    assert [product.product_ordinal for product in result.product_results] == [1, 2, 3, 4, 5]
    assert result.product_results[-1].failure_code is (ShoppingAnswerOnlyEpisodeFailureCode.ANSWER_RUNNER_FAILED)
    assert factory.open_calls == [1, 2, 3, 4, 5]


def test_continue_policy_preserves_later_boundary_stop_after_mismatch() -> None:
    factory = _FakeSessionFactory(outcomes={3: "mismatch"})
    answer_client = _FakeAnswerClient(raised_products={5})
    runner, _, _ = _episode_runner(
        answer_client=answer_client,
        session_factory=factory,
        evaluation_policy=(ShoppingAnswerOnlyEpisodeEvaluationPolicy.CONTINUE_EVALUATION_FAILURES),
    )

    result = runner.run(_full_task())

    assert result.completed_product_count == 3
    assert result.attempted_product_count == 5
    assert result.failed_product_ordinal == 3
    assert result.failure_code is ShoppingAnswerOnlyEpisodeFailureCode.TARGET_MISMATCH
    assert result.stopped_product_ordinal == 5
    assert result.stop_failure_code is (ShoppingAnswerOnlyEpisodeFailureCode.ANSWER_RUNNER_EXCEPTION)
    assert [product.product_ordinal for product in result.product_results] == [1, 2, 3, 4]
    assert factory.open_calls == [1, 2, 3, 4, 5]
    safe_result = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "private provider failure" not in safe_result


@pytest.mark.parametrize("outcome", ["soft_exit", "other_terminal"])
def test_non_purchase_terminal_fails_and_stops_future_products(
    outcome: str,
) -> None:
    factory = _FakeSessionFactory(outcomes={2: outcome})
    runner, answer_client, _ = _episode_runner(session_factory=factory)

    result = runner.run(_full_task())

    assert result.completed_product_count == 1
    assert result.failed_product_ordinal == 2
    assert result.failure_code is (ShoppingAnswerOnlyEpisodeFailureCode.PURCHASE_NOT_COMPLETED)
    assert factory.open_calls == [1, 2]
    assert len(answer_client.calls) == 4


def test_text_only_answer_failure_is_retained_and_stops_future_products() -> None:
    answer_client = _FakeAnswerClient(text_only_products={4})
    runner, _, session_factory = _episode_runner(answer_client=answer_client)

    result = runner.run(_full_task())

    assert result.completed_product_count == 3
    assert result.failed_product_ordinal == 4
    assert result.failure_code is (ShoppingAnswerOnlyEpisodeFailureCode.ANSWER_RUNNER_FAILED)
    assert len(result.product_results) == 4
    assert result.product_results[-1].answer_result.status.value == "FAILED"
    assert session_factory.open_calls == [1, 2, 3, 4]
    assert [call.product_ordinal for call in answer_client.calls].count(4) == 1


def test_answer_client_exception_is_safe_and_stops_future_products() -> None:
    answer_client = _FakeAnswerClient(raised_products={3})
    runner, _, session_factory = _episode_runner(answer_client=answer_client)

    result = runner.run(_full_task())

    assert result.completed_product_count == 2
    assert result.failed_product_ordinal == 3
    assert result.failure_code is (ShoppingAnswerOnlyEpisodeFailureCode.ANSWER_RUNNER_EXCEPTION)
    assert len(result.product_results) == 2
    assert session_factory.open_calls == [1, 2, 3]
    assert all(client.close_calls == 1 for client in session_factory.clients)
    assert "private provider failure" not in json.dumps(result.to_dict())


def test_session_open_failure_returns_only_completed_prefix() -> None:
    factory = _FakeSessionFactory(fail_open_at=5)
    runner, answer_client, _ = _episode_runner(session_factory=factory)

    result = runner.run(_full_task())

    assert result.completed_product_count == 4
    assert result.failed_product_ordinal == 5
    assert result.failure_code is (ShoppingAnswerOnlyEpisodeFailureCode.SESSION_FAILED)
    assert len(result.product_results) == 4
    assert factory.open_calls == [1, 2, 3, 4, 5]
    assert len(answer_client.calls) == 8
    assert all(client.close_calls == 1 for client in factory.clients)


def test_session_reset_failure_is_classified_as_session_failure() -> None:
    factory = _FakeSessionFactory(fail_reset_at=3)
    runner, answer_client, _ = _episode_runner(session_factory=factory)

    result = runner.run(_full_task())

    assert result.completed_product_count == 2
    assert result.failed_product_ordinal == 3
    assert result.failure_code is ShoppingAnswerOnlyEpisodeFailureCode.SESSION_FAILED
    assert len(result.product_results) == 2
    assert factory.open_calls == [1, 2, 3]
    assert len(answer_client.calls) == 4
    assert factory.clients[-1].reset_calls == [0]
    assert factory.clients[-1].close_calls == 1
    assert "private reset failure" not in json.dumps(result.to_dict())


def test_result_invariant_rejects_incomplete_completed_episode() -> None:
    runner, _, _ = _episode_runner()
    valid = runner.run(_full_task())

    with pytest.raises(ValueError, match="six passed Products"):
        ShoppingAnswerOnlyEpisodeResult(
            status=ShoppingAnswerOnlyEpisodeRunStatus.COMPLETED,
            product_results=valid.product_results[:-1],
            completed_product_count=5,
        )


def test_constructor_requires_strict_environment_terminated_single_tool_mode() -> None:
    with pytest.raises(ValueError, match="environment-terminated single-tool"):
        ShoppingAnswerOnlyEpisodeRunner(
            answer_client=_FakeAnswerClient(),
            session_factory=_FakeSessionFactory(),
            task_input_adapter=ShoppingTaskInputAdapter(max_observation_chars=16_000),
            tool_definitions=_tool_definitions(),
            config=AnswerAgentSubtaskRunnerConfig(
                max_inference_steps=3,
                max_tool_calls_per_inference=1,
                max_consecutive_tool_failures=2,
            ),
        )
