"""Leakage-safe summaries for Memory-aware Shopping episode traces."""

from __future__ import annotations

from ....episode.trace import MemoryDecisionTrace
from .memory_episode_contract import (
    SHOPPING_MEMORY_EPISODE_PRODUCT_COUNT,
    ShoppingMemoryEpisodeResult,
    ShoppingMemoryProductTrace,
)
from .memory_episode_validation import memory_decisions, retrieval_from_trace


def memory_decision_to_safe_dict(
    trace: MemoryDecisionTrace,
) -> dict[str, object | None]:
    """Summarize one decision without arguments, content, or raw output."""

    validation = trace.gateway_result.validation
    decision = validation.decision
    action = None if decision is None else decision.memory_action
    execution = trace.gateway_result.execution
    retrieval = None if execution is None else execution.retrieval
    return {
        "phase": trace.context.phase.value,
        "action": None if action is None else action.action.value,
        "gateway_status": trace.gateway_result.status.value,
        "execution_status": (
            None if execution is None else execution.status.value
        ),
        "execution_code": None if execution is None else execution.code.value,
        "retrieval_match_count": (
            None if retrieval is None else retrieval.match_count
        ),
        "bank_record_count_before": trace.bank_before.record_count,
        "bank_record_count_after": trace.bank_after.record_count,
    }


def memory_mutation_observed(trace: ShoppingMemoryProductTrace) -> bool:
    return any(
        decision.bank_before != decision.bank_after
        for decision in memory_decisions(trace)
    )


def retrieval_injected_into_answer(trace: ShoppingMemoryProductTrace) -> bool:
    retrieval = retrieval_from_trace(trace.pre_task_decision)
    return (
        retrieval is not None
        and retrieval.match_count > 0
        and trace.answer_input.retrieved_memories
        == tuple(match.content for match in retrieval.matches)
    )


def memory_effect_observed(
    product_traces: tuple[ShoppingMemoryProductTrace, ...],
) -> bool:
    """Require an earlier mutation and a later non-empty Answer injection."""

    mutation_seen = False
    for trace in product_traces:
        if mutation_seen and retrieval_injected_into_answer(trace):
            return True
        mutation_seen = mutation_seen or memory_mutation_observed(trace)
    return False


def memory_product_trace_to_safe_dict(
    trace: ShoppingMemoryProductTrace,
) -> dict[str, object | None]:
    payload = trace.product_result.to_dict()
    payload.update(
        {
            "memory_decision_count": trace.memory_decision_count,
            "retrieved_memory_count": trace.retrieved_memory_count,
            "memory_mutation_observed": memory_mutation_observed(trace),
            "retrieval_injected_into_answer": (
                retrieval_injected_into_answer(trace)
            ),
            "bank_record_count_before": trace.bank_before.record_count,
            "bank_record_count_after": trace.bank_after.record_count,
            "memory_decisions": [
                memory_decision_to_safe_dict(decision)
                for decision in memory_decisions(trace)
            ],
        }
    )
    return payload


def memory_episode_result_to_safe_dict(
    result: ShoppingMemoryEpisodeResult,
) -> dict[str, object | None]:
    action_counts: dict[str, int] = {}
    null_count = 0
    for product_trace in result.product_traces:
        for decision_trace in memory_decisions(product_trace):
            decision = decision_trace.gateway_result.validation.decision
            action = None if decision is None else decision.memory_action
            if action is None:
                null_count += 1
            else:
                name = action.action.value
                action_counts[name] = action_counts.get(name, 0) + 1
    correction = result.memory_output_correction
    return {
        "status": result.status.value,
        "pipeline_completed": result.pipeline_completed,
        "expected_product_count": SHOPPING_MEMORY_EPISODE_PRODUCT_COUNT,
        "completed_product_count": len(result.product_traces),
        "attempted_product_count": (
            len(result.product_traces)
            if result.pipeline_completed
            else len(result.product_traces) + 1
        ),
        "failed_product_ordinal": result.failed_product_ordinal,
        "failure_stage": (
            None if result.failure_stage is None else result.failure_stage.value
        ),
        "failure_code": (
            None if result.failure_code is None else result.failure_code.value
        ),
        "memory_validation_error_code": (
            None
            if result.memory_validation_error_code is None
            else result.memory_validation_error_code.value
        ),
        "memory_model_error_code": (
            None
            if result.memory_model_error_code is None
            else result.memory_model_error_code.value
        ),
        "answer_error_code": (
            None
            if result.answer_error_code is None
            else result.answer_error_code.value
        ),
        "answer_provider_error_code": (
            None
            if result.answer_provider_error_code is None
            else result.answer_provider_error_code.value
        ),
        "memory_agent_invocation_count": result.memory_agent_invocation_count,
        "memory_output_correction_budget": (
            result.memory_output_correction_budget
        ),
        "memory_output_correction_count": 0 if correction is None else 1,
        "memory_output_correction": (
            None
            if correction is None
            else {
                "product_ordinal": correction.product_ordinal,
                "phase": correction.phase.value,
                "trigger_error_code": correction.trigger_error_code.value,
                "recovered": correction.recovered,
            }
        ),
        "committed_memory_decision_count": result.memory_decision_count,
        "committed_null_memory_decision_count": null_count,
        "committed_memory_action_counts": dict(sorted(action_counts.items())),
        "final_bank_record_count": result.final_bank_state.record_count,
        "memory_effect_observed": result.memory_effect_observed,
        "purchase_completed_count": result.purchase_completed_count,
        "target_match_count": result.target_match_count,
        "evaluation_all_targets_matched": (
            result.pipeline_completed
            and result.target_match_count == SHOPPING_MEMORY_EPISODE_PRODUCT_COUNT
        ),
        "product_results": [
            trace.to_safe_dict() for trace in result.product_traces
        ],
    }


__all__ = [
    "memory_decision_to_safe_dict",
    "memory_effect_observed",
    "memory_episode_result_to_safe_dict",
    "memory_mutation_observed",
    "memory_product_trace_to_safe_dict",
    "retrieval_injected_into_answer",
]
