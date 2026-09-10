from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping.memory_episode_contract import (
    ShoppingMemoryEpisodeFailureCode,
    ShoppingMemoryEpisodeFailureStage,
    ShoppingMemoryEpisodeInputError,
    ShoppingMemoryEpisodeInputErrorCode,
    ShoppingMemoryEpisodeRunStatus,
    ShoppingMemoryProductTrace,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.memory_episode_runtime import (
    ShoppingMemoryEpisodeRunner,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.task_input_adapter import (
    ShoppingTaskInputAdapter,
)
from examples.memory_agent_grpo.answer_agent.interaction_mode import (
    AnswerAgentInteractionMode,
)
from examples.memory_agent_grpo.answer_agent.runtime import (
    AnswerAgentSubtaskRunnerConfig,
)
from examples.memory_agent_grpo.episode.trace import MemoryDecisionTrace
from examples.memory_agent_grpo.memory_agent.action.protocol import (
    MemoryDecisionPhase,
)
from examples.memory_agent_grpo.memory_agent.action.validation import (
    ValidationErrorCode,
)
from examples.memory_agent_grpo.memory_agent.decision.model_client import (
    MemoryAgentClientError,
    MemoryAgentClientErrorCode,
)
from examples.memory_agent_grpo.tests.episode.runtime._support import (
    NULL_ACTION,
    ScriptedMemoryClient,
    add_action,
    retrieve_action,
)

from .test_answer_only_episode_runtime import (
    _FakeAnswerClient,
    _FakeSessionFactory,
    _full_task,
    _runner_config,
    _tool_definitions,
)


def _memory_runner(
    *,
    memory_client: ScriptedMemoryClient,
    answer_client: _FakeAnswerClient | None = None,
    session_factory: _FakeSessionFactory | None = None,
    memory_output_correction_budget: int = 0,
    memory_decision_observer: Callable[[MemoryDecisionTrace], None] | None = None,
    product_observer: Callable[[ShoppingMemoryProductTrace], None] | None = None,
) -> tuple[
    ShoppingMemoryEpisodeRunner,
    _FakeAnswerClient,
    _FakeSessionFactory,
]:
    resolved_answer_client = (
        _FakeAnswerClient() if answer_client is None else answer_client
    )
    resolved_session_factory = (
        _FakeSessionFactory() if session_factory is None else session_factory
    )
    return (
        ShoppingMemoryEpisodeRunner(
            memory_client=memory_client,
            answer_client=resolved_answer_client,
            session_factory=resolved_session_factory,
            task_input_adapter=ShoppingTaskInputAdapter(
                max_observation_chars=16_000
            ),
            tool_definitions=_tool_definitions(),
            config=_runner_config(),
            memory_output_correction_budget=(
                memory_output_correction_budget
            ),
            memory_decision_observer=memory_decision_observer,
            product_observer=product_observer,
        ),
        resolved_answer_client,
        resolved_session_factory,
    )


def test_six_products_share_bank_and_inject_exact_pre_task_retrieval() -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[
            NULL_ACTION,
            add_action("PERSISTENT_PRODUCT_PREFERENCE"),
            retrieve_action("PERSISTENT_PRODUCT_PREFERENCE"),
            NULL_ACTION,
            *([NULL_ACTION] * 8),
        ]
    )
    runner, answer_client, session_factory = _memory_runner(
        memory_client=memory_client
    )

    result = runner.run(_full_task())

    assert result.status is ShoppingMemoryEpisodeRunStatus.COMPLETED
    assert result.pipeline_completed is True
    assert result.memory_agent_invocation_count == 12
    assert result.memory_decision_count == 12
    assert result.final_bank_state.record_count == 1
    assert result.memory_effect_observed is True
    assert [trace.retrieved_memory_count for trace in result.product_traces] == [
        0,
        1,
        0,
        0,
        0,
        0,
    ]
    assert result.product_traces[1].answer_input.retrieved_memories == (
        "PERSISTENT_PRODUCT_PREFERENCE",
    )
    assert tuple(trace.bank_before for trace in result.product_traces[1:]) == tuple(
        trace.bank_after for trace in result.product_traces[:-1]
    )
    assert session_factory.open_calls == [1, 2, 3, 4, 5, 6]
    assert all(client.reset_calls == [0] for client in session_factory.clients)
    assert all(client.close_calls == 1 for client in session_factory.clients)
    assert len(answer_client.calls) == 12

    product_two_initial = answer_client.calls[2].conversation.initial_messages[1].content
    assert "PERSISTENT_PRODUCT_PREFERENCE" in product_two_initial
    assert answer_client.calls[0].conversation.tool_exchanges == ()
    assert answer_client.calls[2].conversation.tool_exchanges == ()


def test_post_task_retrieval_creates_exactly_one_follow_up() -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[
            NULL_ACTION,
            retrieve_action("ABSENT_QUERY"),
            add_action("FOLLOW_UP_MEMORY"),
            retrieve_action("FOLLOW_UP_MEMORY"),
            NULL_ACTION,
            *([NULL_ACTION] * 8),
        ]
    )
    runner, _, _ = _memory_runner(memory_client=memory_client)

    result = runner.run(_full_task())

    assert result.status is ShoppingMemoryEpisodeRunStatus.COMPLETED
    assert result.memory_agent_invocation_count == 13
    assert len(result.product_traces[0].post_task_decisions) == 2
    assert result.product_traces[1].retrieved_memory_count == 1
    assert memory_client.phases[:5] == (
        MemoryDecisionPhase.PRE_TASK,
        MemoryDecisionPhase.POST_TASK,
        MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
        MemoryDecisionPhase.PRE_TASK,
        MemoryDecisionPhase.POST_TASK,
    )


def test_evaluation_failures_do_not_stop_structural_trajectory() -> None:
    memory_client = ScriptedMemoryClient(outcomes=[NULL_ACTION] * 12)
    session_factory = _FakeSessionFactory(
        outcomes={3: "mismatch", 4: "soft_exit"}
    )
    runner, _, _ = _memory_runner(
        memory_client=memory_client,
        session_factory=session_factory,
    )

    result = runner.run(_full_task())

    assert result.status is ShoppingMemoryEpisodeRunStatus.COMPLETED
    assert len(result.product_traces) == 6
    assert result.purchase_completed_count == 5
    assert result.target_match_count == 4
    assert result.to_safe_dict()["evaluation_all_targets_matched"] is False
    assert memory_client.phases == tuple(
        phase
        for _ in range(6)
        for phase in (
            MemoryDecisionPhase.PRE_TASK,
            MemoryDecisionPhase.POST_TASK,
        )
    )


def test_invalid_memory_json_stops_at_pre_task_and_keeps_committed_prefix() -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[
            NULL_ACTION,
            add_action("COMMITTED_MEMORY"),
            "not valid JSON",
        ]
    )
    runner, answer_client, session_factory = _memory_runner(
        memory_client=memory_client
    )

    result = runner.run(_full_task())

    assert result.status is ShoppingMemoryEpisodeRunStatus.FAILED
    assert result.failed_product_ordinal == 2
    assert result.failure_stage is ShoppingMemoryEpisodeFailureStage.PRE_TASK
    assert result.failure_code is (
        ShoppingMemoryEpisodeFailureCode.MEMORY_OUTPUT_VALIDATION_FAILED
    )
    assert result.memory_validation_error_code is ValidationErrorCode.INVALID_JSON
    assert result.to_safe_dict()["memory_validation_error_code"] == "INVALID_JSON"
    assert len(result.product_traces) == 1
    assert result.final_bank_state.record_count == 1
    assert result.memory_agent_invocation_count == 3
    assert session_factory.open_calls == [1, 2]
    assert all(client.close_calls == 1 for client in session_factory.clients)
    assert len(answer_client.calls) == 2


def test_memory_provider_error_code_is_propagated_without_private_detail() -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[
            MemoryAgentClientError(
                MemoryAgentClientErrorCode.REQUEST_TIMEOUT,
                "PRIVATE_PROVIDER_DETAIL",
            )
        ]
    )
    runner, answer_client, session_factory = _memory_runner(
        memory_client=memory_client
    )

    result = runner.run(_full_task())
    serialized = json.dumps(result.to_safe_dict(), ensure_ascii=False)

    assert result.status is ShoppingMemoryEpisodeRunStatus.FAILED
    assert result.failed_product_ordinal == 1
    assert result.failure_stage is ShoppingMemoryEpisodeFailureStage.PRE_TASK
    assert result.failure_code is (
        ShoppingMemoryEpisodeFailureCode.MEMORY_MODEL_CALL_FAILED
    )
    assert result.memory_model_error_code is (
        MemoryAgentClientErrorCode.REQUEST_TIMEOUT
    )
    assert result.memory_agent_invocation_count == 1
    assert answer_client.calls == []
    assert session_factory.open_calls == [1]
    assert "PRIVATE_PROVIDER_DETAIL" not in serialized


def test_one_episode_wide_output_correction_replays_data_but_keeps_summary_safe() -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[
            NULL_ACTION,
            "PRIVATE_BAD_OUTPUT",
            add_action("ONE_CORRECTED_MEMORY"),
            *([NULL_ACTION] * 10),
        ]
    )
    runner, _, _ = _memory_runner(
        memory_client=memory_client,
        memory_output_correction_budget=1,
    )

    result = runner.run(_full_task())

    assert result.status is ShoppingMemoryEpisodeRunStatus.COMPLETED
    assert result.memory_agent_invocation_count == 13
    assert result.memory_decision_count == 12
    assert result.final_bank_state.record_count == 1
    assert result.to_safe_dict()["committed_memory_action_counts"] == {"ADD": 1}
    assert result.memory_output_correction is not None
    assert result.memory_output_correction.product_ordinal == 1
    assert result.memory_output_correction.phase is MemoryDecisionPhase.POST_TASK
    assert result.memory_output_correction.trigger_error_code is (
        ValidationErrorCode.INVALID_JSON
    )
    assert result.memory_output_correction.recovered is True
    assert "INVALID_JSON" in memory_client.calls[2][1].content
    correction = memory_client.calls[2][1].content
    data = json.loads(correction.split("<correction_data_json>\n")[1].split("\n</correction_data_json>")[0])
    assert data["rejected_output"] == "PRIVATE_BAD_OUTPUT"
    assert "line 1" in data["validation_error_message"]
    assert "PRIVATE_BAD_OUTPUT" not in json.dumps(result.to_safe_dict())


def test_unrecovered_output_correction_fails_after_exactly_two_calls() -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=["PRIVATE_BAD_FIRST", "PRIVATE_BAD_SECOND"]
    )
    runner, answer_client, _ = _memory_runner(
        memory_client=memory_client,
        memory_output_correction_budget=1,
    )

    result = runner.run(_full_task())

    assert result.status is ShoppingMemoryEpisodeRunStatus.FAILED
    assert result.failure_code is (
        ShoppingMemoryEpisodeFailureCode.MEMORY_OUTPUT_VALIDATION_FAILED
    )
    assert result.memory_validation_error_code is ValidationErrorCode.INVALID_JSON
    assert result.memory_agent_invocation_count == 2
    assert result.memory_output_correction is not None
    assert result.memory_output_correction.recovered is False
    assert answer_client.calls == []
    serialized = json.dumps(result.to_safe_dict(), ensure_ascii=False)
    assert "PRIVATE_BAD_FIRST" not in serialized
    assert "PRIVATE_BAD_SECOND" not in serialized


def test_three_different_decisions_can_recover_and_write_once_each() -> None:
    outcomes = []
    for n in range(3):
        repaired = add_action(f"PUBLIC_MEMORY_{n}")
        # Two missing braces remain outside the deterministic repair rule.
        outcomes.extend([NULL_ACTION, repaired[:-2], repaired])
    outcomes.extend([NULL_ACTION] * 6)
    client = ScriptedMemoryClient(outcomes=outcomes)
    runner, _, _ = _memory_runner(memory_client=client, memory_output_correction_budget=3)
    result = runner.run(_full_task())
    assert result.pipeline_completed
    assert result.memory_agent_invocation_count == 15
    assert result.final_bank_state.record_count == 3
    assert [c.product_ordinal for c in result.memory_output_corrections] == [1, 2, 3]
    assert all(c.recovered for c in result.memory_output_corrections)
    summary = result.to_safe_dict()
    assert summary["memory_output_correction_count"] == 3
    assert len(summary["memory_output_corrections"]) == 3
    assert summary["memory_output_correction_limit_per_decision"] == 1
    assert result.memory_output_correction == result.memory_output_corrections[0]
    with pytest.raises(ValueError):
        replace(result, memory_output_corrections=(result.memory_output_corrections[0],) * 3)


def test_six_missing_add_braces_do_not_spend_retry_budget_and_memory_is_injected() -> None:
    outcomes = []
    for n in range(6):
        outcomes.extend([
            NULL_ACTION if n == 0 else retrieve_action(f"EXACT_MEMORY_{n - 1}"),
            add_action(f"EXACT_MEMORY_{n}")[:-1],
        ])
    observed = []
    client = ScriptedMemoryClient(outcomes=outcomes)
    runner, _, _ = _memory_runner(
        memory_client=client, memory_output_correction_budget=0,
        memory_decision_observer=observed.append,
    )
    result = runner.run(_full_task())
    assert result.pipeline_completed
    assert result.memory_agent_invocation_count == len(client.calls) == 12
    assert result.memory_output_corrections == ()
    assert result.final_bank_state.record_count == 6
    assert sum(trace.output_repair is not None for trace in observed) == 6
    for n, product in enumerate(result.product_traces[1:], start=1):
        assert f"EXACT_MEMORY_{n - 1}" in product.answer_input.retrieved_memories
    summary = result.to_safe_dict()
    assert summary["committed_memory_output_repair_count"] == 6
    assert summary["memory_output_correction_count"] == 0
    assert "EXACT_MEMORY" not in json.dumps(summary)
    repaired = summary["product_results"][0]["memory_decisions"][1]
    assert repaired["native_output_valid"] is False
    assert repaired["raw_validation_error_code"] == "INVALID_JSON"
    assert repaired["execution_code"] == "ADDED"


def test_fourth_invalid_decision_stops_when_three_episode_corrections_are_used() -> None:
    outcomes = []
    for n in range(3):
        repaired = add_action(f"MEMORY_{n}")
        outcomes.extend([NULL_ACTION, repaired[:-2], repaired])
    outcomes.extend([NULL_ACTION, "FOURTH_BAD_OUTPUT", add_action("MUST_NOT_EXECUTE")])
    client = ScriptedMemoryClient(outcomes=outcomes)
    runner, _, _ = _memory_runner(memory_client=client, memory_output_correction_budget=3)
    result = runner.run(_full_task())
    assert result.failed_product_ordinal == 4
    assert result.failure_code is ShoppingMemoryEpisodeFailureCode.MEMORY_OUTPUT_VALIDATION_FAILED
    assert result.memory_agent_invocation_count == len(client.calls) == 11
    assert len(result.memory_output_corrections) == 3
    assert result.final_bank_state.record_count == 3


def test_one_decision_cannot_spend_remaining_episode_budget_on_second_retry() -> None:
    client = ScriptedMemoryClient(outcomes=["BAD_FIRST", "BAD_CORRECTION", NULL_ACTION])
    runner, answer, _ = _memory_runner(memory_client=client, memory_output_correction_budget=3)
    result = runner.run(_full_task())
    assert len(client.calls) == 2
    assert len(result.memory_output_corrections) == 1
    assert result.memory_output_corrections[0].recovered is False
    assert not result.pipeline_completed and answer.calls == []


def test_distinct_pre_post_and_after_retrieval_decisions_each_allow_one_correction() -> None:
    client = ScriptedMemoryClient(outcomes=[
        "BAD_PRE", NULL_ACTION,
        "BAD_POST", retrieve_action("prior facts"),
        "BAD_AFTER_RETRIEVAL", add_action("ONE_MEMORY"),
        *([NULL_ACTION] * 10),
    ])
    runner, _, _ = _memory_runner(memory_client=client, memory_output_correction_budget=3)
    result = runner.run(_full_task())
    assert result.pipeline_completed and result.memory_agent_invocation_count == 16
    assert [c.phase for c in result.memory_output_corrections] == list(MemoryDecisionPhase)
    assert result.final_bank_state.record_count == 1


@pytest.mark.parametrize("budget", [-1, 4, True, 1.5])
def test_correction_budget_rejects_values_outside_supported_range(budget) -> None:
    with pytest.raises(ValueError):
        _memory_runner(memory_client=ScriptedMemoryClient(outcomes=[]), memory_output_correction_budget=budget)


def test_observers_receive_rejected_attempt_then_correction_and_each_product() -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[
            NULL_ACTION,
            "REJECTED_RAW_OUTPUT",
            add_action("CORRECTED_MEMORY"),
            *([NULL_ACTION] * 10),
        ]
    )
    memory_traces: list[MemoryDecisionTrace] = []
    products: list[ShoppingMemoryProductTrace] = []
    calls_at_product_observation: list[int] = []

    def observe_product(trace: ShoppingMemoryProductTrace) -> None:
        products.append(trace)
        calls_at_product_observation.append(len(memory_client.calls))

    runner, _, _ = _memory_runner(
        memory_client=memory_client,
        memory_output_correction_budget=1,
        memory_decision_observer=memory_traces.append,
        product_observer=observe_product,
    )

    result = runner.run(_full_task())

    assert result.status is ShoppingMemoryEpisodeRunStatus.COMPLETED
    assert len(memory_traces) == 13
    rejected = memory_traces[1]
    assert rejected.raw_output == "REJECTED_RAW_OUTPUT"
    assert rejected.gateway_result.validation.error_code is ValidationErrorCode.INVALID_JSON
    assert rejected.bank_before == rejected.bank_after
    assert memory_traces[2].bank_before == rejected.bank_before
    assert memory_traces[2].bank_after.record_count == 1
    assert tuple(products) == result.product_traces
    assert calls_at_product_observation == [3, 5, 7, 9, 11, 13]


def test_memory_observer_receives_post_retrieval_follow_up() -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[NULL_ACTION, retrieve_action("absent"), add_action("memory"), *([NULL_ACTION] * 10)]
    )
    traces: list[MemoryDecisionTrace] = []
    runner, _, _ = _memory_runner(
        memory_client=memory_client,
        memory_decision_observer=traces.append,
    )

    result = runner.run(_full_task())

    assert result.status is ShoppingMemoryEpisodeRunStatus.COMPLETED
    assert len(traces) == 13
    assert traces[2].context.phase is MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL
    assert traces[2] is result.product_traces[0].post_task_decisions[1]


def test_rejected_output_is_observable_when_no_correction_is_available() -> None:
    memory_client = ScriptedMemoryClient(outcomes=["not JSON"])
    traces: list[MemoryDecisionTrace] = []
    runner, answer_client, _ = _memory_runner(
        memory_client=memory_client,
        memory_decision_observer=traces.append,
    )

    result = runner.run(_full_task())

    assert result.status is ShoppingMemoryEpisodeRunStatus.FAILED
    assert result.memory_validation_error_code is ValidationErrorCode.INVALID_JSON
    assert len(traces) == 1
    assert traces[0].raw_output == "not JSON"
    assert traces[0].bank_before == traces[0].bank_after
    assert answer_client.calls == []


@pytest.mark.parametrize("raw_output", ["not JSON", add_action("memory")])
def test_memory_observer_failure_stops_without_retrying_action(raw_output: str) -> None:
    memory_client = ScriptedMemoryClient(outcomes=[NULL_ACTION, raw_output])
    traces: list[MemoryDecisionTrace] = []

    def fail_on_post_task(trace: MemoryDecisionTrace) -> None:
        traces.append(trace)
        if trace.context.phase is MemoryDecisionPhase.POST_TASK:
            raise OSError("PRIVATE_FILESYSTEM_DETAIL")

    runner, _, _ = _memory_runner(
        memory_client=memory_client,
        memory_output_correction_budget=1,
        memory_decision_observer=fail_on_post_task,
    )

    result = runner.run(_full_task())

    assert result.status is ShoppingMemoryEpisodeRunStatus.FAILED
    assert result.failure_stage is ShoppingMemoryEpisodeFailureStage.INTERNAL
    assert result.failure_code is ShoppingMemoryEpisodeFailureCode.MEMORY_INTERNAL_CONTRACT_ERROR
    assert result.memory_agent_invocation_count == 2
    assert len(traces) == 2
    assert traces[-1].raw_output == raw_output
    assert result.memory_output_correction is None
    assert "PRIVATE_FILESYSTEM_DETAIL" not in json.dumps(result.to_safe_dict())


def test_product_observer_failure_stops_before_opening_next_session() -> None:
    memory_client = ScriptedMemoryClient(outcomes=[NULL_ACTION, add_action("memory")])
    products: list[ShoppingMemoryProductTrace] = []

    def fail_to_persist(trace: ShoppingMemoryProductTrace) -> None:
        products.append(trace)
        raise OSError("PRIVATE_FILESYSTEM_DETAIL")

    runner, _, sessions = _memory_runner(
        memory_client=memory_client,
        product_observer=fail_to_persist,
    )

    result = runner.run(_full_task())

    assert result.status is ShoppingMemoryEpisodeRunStatus.FAILED
    assert result.failure_stage is ShoppingMemoryEpisodeFailureStage.INTERNAL
    assert result.failure_code is ShoppingMemoryEpisodeFailureCode.TRACE_CONTRACT_ERROR
    assert result.failed_product_ordinal == 1
    assert result.memory_agent_invocation_count == 2
    assert sessions.open_calls == [1]
    assert len(products) == 1
    assert products[0].bank_after.record_count == 1
    assert result.product_traces == ()
    assert "PRIVATE_FILESYSTEM_DETAIL" not in json.dumps(result.to_safe_dict())


def test_target_and_evaluator_evidence_never_reach_either_model() -> None:
    memory_client = ScriptedMemoryClient(outcomes=[NULL_ACTION] * 12)
    runner, answer_client, _ = _memory_runner(memory_client=memory_client)

    result = runner.run(_full_task())

    assert result.status is ShoppingMemoryEpisodeRunStatus.COMPLETED
    memory_inputs = "\n".join(memory_client.serialized_inputs)
    answer_inputs = json.dumps(
        [call.conversation.to_dict() for call in answer_client.calls],
        ensure_ascii=False,
    )
    for rendered in (memory_inputs, answer_inputs):
        assert "SEALED-TARGET" not in rendered
        assert "purchased_target_matched" not in rendered
        assert "target_asin" not in rendered
    assert all(
        trace.task_result.reward is None and trace.task_result.feedback is None
        for trace in result.product_traces
    )


def test_safe_summary_omits_raw_trajectory_and_memory_content() -> None:
    memory_client = ScriptedMemoryClient(
        outcomes=[
            NULL_ACTION,
            add_action("PRIVATE_MEMORY_CONTENT"),
            *([NULL_ACTION] * 10),
        ]
    )
    runner, _, _ = _memory_runner(memory_client=memory_client)

    result = runner.run(_full_task())
    safe_payload = result.to_safe_dict()
    serialized = json.dumps(safe_payload, ensure_ascii=False)

    assert safe_payload["pipeline_completed"] is True
    assert safe_payload["committed_memory_action_counts"] == {"ADD": 1}
    assert "PRIVATE_MEMORY_CONTENT" not in serialized
    assert "PUBLIC_PRODUCT" not in serialized
    assert "SEALED-TARGET" not in serialized
    assert "raw_output" not in serialized
    assert "interaction_events" not in serialized


def test_task_preflight_fails_before_session_or_model_calls() -> None:
    memory_client = ScriptedMemoryClient(outcomes=[NULL_ACTION])
    runner, answer_client, session_factory = _memory_runner(
        memory_client=memory_client
    )
    task = _full_task()
    steps = task["steps"]
    assert type(steps) is list
    task["steps"] = steps[:-1]

    with pytest.raises(ShoppingMemoryEpisodeInputError) as caught:
        runner.run(task)

    assert caught.value.code is (
        ShoppingMemoryEpisodeInputErrorCode.PRODUCT_COUNT_INVALID
    )
    assert memory_client.calls == []
    assert answer_client.calls == []
    assert session_factory.open_calls == []


def test_constructor_requires_environment_terminated_single_tool_mode() -> None:
    with pytest.raises(ValueError, match="environment-terminated single-tool"):
        ShoppingMemoryEpisodeRunner(
            memory_client=ScriptedMemoryClient(outcomes=[NULL_ACTION]),
            answer_client=_FakeAnswerClient(),
            session_factory=_FakeSessionFactory(),
            task_input_adapter=ShoppingTaskInputAdapter(
                max_observation_chars=16_000
            ),
            tool_definitions=_tool_definitions(),
            config=AnswerAgentSubtaskRunnerConfig(
                max_inference_steps=3,
                max_tool_calls_per_inference=1,
                max_consecutive_tool_failures=2,
                interaction_mode=AnswerAgentInteractionMode.TOOL_OR_FINAL,
            ),
        )
