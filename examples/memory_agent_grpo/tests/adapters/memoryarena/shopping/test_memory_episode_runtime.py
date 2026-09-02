from __future__ import annotations

import json

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping.memory_episode_contract import (
    ShoppingMemoryEpisodeFailureCode,
    ShoppingMemoryEpisodeFailureStage,
    ShoppingMemoryEpisodeInputError,
    ShoppingMemoryEpisodeInputErrorCode,
    ShoppingMemoryEpisodeRunStatus,
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


def test_one_episode_wide_output_correction_can_recover_without_raw_replay() -> None:
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
    assert "PRIVATE_BAD_OUTPUT" not in memory_client.calls[2][1].content


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
