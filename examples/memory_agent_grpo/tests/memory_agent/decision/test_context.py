from dataclasses import FrozenInstanceError

import pytest

from examples.memory_agent_grpo.memory_agent.action.protocol import MemoryDecisionPhase
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryRecord
from examples.memory_agent_grpo.memory_agent.decision.context import (
    InteractionEvent,
    InteractionEventType,
    MemoryDecisionContext,
    TaskResult,
)
from examples.memory_agent_grpo.memory_agent.retrieval.result import RetrievalMatch, RetrievalResult


def _retrieval(query: str, *records: MemoryRecord) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        matches=tuple(RetrievalMatch(record=record, score=1.0) for record in records),
    )


def test_pre_task_context_serializes_only_public_contract_fields() -> None:
    context = MemoryDecisionContext(
        phase=MemoryDecisionPhase.PRE_TASK,
        task_input="Plan a trip",
        bank_record_count=4,
    )

    assert context.visible_memory_ids == ()
    assert context.to_dict() == {
        "phase": "PRE_TASK",
        "task_input": "Plan a trip",
        "bank_record_count": 4,
        "interaction_events": [],
        "task_result": None,
        "pre_task_retrieval": None,
        "post_task_retrieval": None,
    }


def test_visible_ids_come_only_from_presented_retrievals_in_first_visible_order() -> None:
    first = MemoryRecord("M001", "first")
    second = MemoryRecord("M002", "second")
    context = MemoryDecisionContext(
        phase=MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
        task_input="Plan a trip",
        bank_record_count=2,
        task_result=TaskResult(final_answer="Done"),
        pre_task_retrieval=_retrieval("trip", first),
        post_task_retrieval=_retrieval("failure", second, first),
    )

    assert context.visible_memory_ids == ("M001", "M002")


def test_post_task_serialization_preserves_interaction_order_and_result() -> None:
    context = MemoryDecisionContext(
        phase=MemoryDecisionPhase.POST_TASK,
        task_input="Find a hotel",
        bank_record_count=1,
        interaction_events=(
            InteractionEvent(InteractionEventType.TASK_AGENT_ACTION, "search hotels"),
            InteractionEvent(InteractionEventType.ENVIRONMENT_OBSERVATION, "three results"),
        ),
        task_result=TaskResult(final_answer="Hotel A", reward=1.0, feedback="accepted"),
    )

    payload = context.to_dict()

    assert payload["interaction_events"] == [
        {"event_type": "TASK_AGENT_ACTION", "content": "search hotels"},
        {"event_type": "ENVIRONMENT_OBSERVATION", "content": "three results"},
    ]
    assert payload["task_result"] == {
        "status": "COMPLETED",
        "final_answer": "Hotel A",
        "reward": 1.0,
        "feedback": "accepted",
    }


def test_context_and_nested_models_are_immutable() -> None:
    event = InteractionEvent(InteractionEventType.TASK_AGENT_ACTION, "search")
    result = TaskResult(final_answer="answer")
    context = MemoryDecisionContext(
        phase=MemoryDecisionPhase.POST_TASK,
        task_input="task",
        bank_record_count=0,
        interaction_events=(event,),
        task_result=result,
    )

    with pytest.raises(FrozenInstanceError):
        event.content = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.final_answer = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        context.bank_record_count = 1  # type: ignore[misc]
