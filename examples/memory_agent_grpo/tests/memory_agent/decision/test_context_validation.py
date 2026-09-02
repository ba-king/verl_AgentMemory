import math

import pytest

from examples.memory_agent_grpo.memory_agent.action.protocol import MemoryDecisionPhase
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryRecord
from examples.memory_agent_grpo.memory_agent.decision.context import (
    InteractionEvent,
    InteractionEventType,
    MemoryDecisionContext,
    TaskResult,
    TaskResultStatus,
)
from examples.memory_agent_grpo.memory_agent.decision.context_validation import (
    ContextValidationErrorCode,
    validate_memory_decision_context,
)
from examples.memory_agent_grpo.memory_agent.retrieval.result import RetrievalMatch, RetrievalResult


def _retrieval(query: str = "memory", memory_id: str = "M001") -> RetrievalResult:
    return RetrievalResult(
        query=query,
        matches=(RetrievalMatch(MemoryRecord(memory_id, "content"), 1.0),),
    )


def _context(
    phase: MemoryDecisionPhase,
    *,
    interaction_events: tuple[InteractionEvent, ...] = (),
    task_result: TaskResult | None = None,
    pre_task_retrieval: RetrievalResult | None = None,
    post_task_retrieval: RetrievalResult | None = None,
) -> MemoryDecisionContext:
    return MemoryDecisionContext(
        phase=phase,
        task_input="task",
        bank_record_count=1,
        interaction_events=interaction_events,
        task_result=task_result,
        pre_task_retrieval=pre_task_retrieval,
        post_task_retrieval=post_task_retrieval,
    )


@pytest.mark.parametrize(
    "context",
    [
        _context(MemoryDecisionPhase.PRE_TASK),
        _context(
            MemoryDecisionPhase.POST_TASK,
            task_result=TaskResult("answer"),
            pre_task_retrieval=_retrieval(),
        ),
        _context(
            MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
            task_result=TaskResult("answer"),
            pre_task_retrieval=_retrieval("before", "M001"),
            post_task_retrieval=_retrieval("after", "M002"),
        ),
    ],
)
def test_valid_context_for_each_phase(context: MemoryDecisionContext) -> None:
    assert validate_memory_decision_context(context).is_valid


def test_non_context_is_rejected() -> None:
    result = validate_memory_decision_context({"phase": "PRE_TASK"})

    assert result.error_code is ContextValidationErrorCode.CONTEXT_NOT_MEMORY_DECISION_CONTEXT


@pytest.mark.parametrize(
    ("changes", "expected_code"),
    [
        ({"phase": "PRE_TASK"}, ContextValidationErrorCode.PHASE_NOT_MEMORY_DECISION_PHASE),
        ({"task_input": 7}, ContextValidationErrorCode.TASK_INPUT_NOT_STRING),
        ({"task_input": "  "}, ContextValidationErrorCode.EMPTY_TASK_INPUT),
        ({"bank_record_count": True}, ContextValidationErrorCode.BANK_RECORD_COUNT_NOT_INTEGER),
        ({"bank_record_count": -1}, ContextValidationErrorCode.BANK_RECORD_COUNT_OUT_OF_RANGE),
        ({"interaction_events": []}, ContextValidationErrorCode.INTERACTION_EVENTS_NOT_TUPLE),
    ],
)
def test_invalid_top_level_field_is_rejected(
    changes: dict[str, object],
    expected_code: ContextValidationErrorCode,
) -> None:
    values: dict[str, object] = {
        "phase": MemoryDecisionPhase.PRE_TASK,
        "task_input": "task",
        "bank_record_count": 0,
    }
    values.update(changes)
    context = MemoryDecisionContext(**values)  # type: ignore[arg-type]

    assert validate_memory_decision_context(context).error_code is expected_code


@pytest.mark.parametrize(
    ("event", "expected_code"),
    [
        ("action", ContextValidationErrorCode.INTERACTION_EVENT_NOT_TYPED),
        (
            InteractionEvent("TASK_AGENT_ACTION", "search"),
            ContextValidationErrorCode.INTERACTION_EVENT_TYPE_NOT_ENUM,
        ),
        (
            InteractionEvent(InteractionEventType.TASK_AGENT_ACTION, 7),
            ContextValidationErrorCode.INTERACTION_CONTENT_NOT_STRING,
        ),
        (
            InteractionEvent(InteractionEventType.TASK_AGENT_ACTION, " "),
            ContextValidationErrorCode.EMPTY_INTERACTION_CONTENT,
        ),
    ],
)
def test_invalid_interaction_event_is_rejected(
    event: object,
    expected_code: ContextValidationErrorCode,
) -> None:
    context = _context(
        MemoryDecisionPhase.POST_TASK,
        interaction_events=(event,),  # type: ignore[arg-type]
        task_result=TaskResult("answer"),
    )

    assert validate_memory_decision_context(context).error_code is expected_code


@pytest.mark.parametrize(
    ("task_result", "expected_code"),
    [
        ("answer", ContextValidationErrorCode.TASK_RESULT_NOT_TYPED),
        (TaskResult(7), ContextValidationErrorCode.FINAL_ANSWER_NOT_STRING),
        (
            TaskResult(" "),
            ContextValidationErrorCode.COMPLETED_TASK_HAS_EMPTY_FINAL_ANSWER,
        ),
        (
            TaskResult("answer", status="COMPLETED"),
            ContextValidationErrorCode.TASK_RESULT_STATUS_NOT_ENUM,
        ),
        (TaskResult("answer", reward=True), ContextValidationErrorCode.REWARD_NOT_FINITE_NUMBER),
        (TaskResult("answer", reward=math.nan), ContextValidationErrorCode.REWARD_NOT_FINITE_NUMBER),
        (TaskResult("answer", feedback=7), ContextValidationErrorCode.FEEDBACK_NOT_STRING),
        (TaskResult("answer", feedback=" "), ContextValidationErrorCode.EMPTY_FEEDBACK),
    ],
)
def test_invalid_task_result_is_rejected(
    task_result: object,
    expected_code: ContextValidationErrorCode,
) -> None:
    context = _context(
        MemoryDecisionPhase.POST_TASK,
        task_result=task_result,  # type: ignore[arg-type]
    )

    assert validate_memory_decision_context(context).error_code is expected_code


def test_failed_task_may_have_empty_final_answer() -> None:
    context = _context(
        MemoryDecisionPhase.POST_TASK,
        task_result=TaskResult("", status=TaskResultStatus.FAILED, feedback="max steps reached"),
    )

    assert validate_memory_decision_context(context).is_valid


@pytest.mark.parametrize(
    ("retrieval", "expected_code"),
    [
        ("memory", ContextValidationErrorCode.RETRIEVAL_NOT_TYPED),
        (RetrievalResult(7, ()), ContextValidationErrorCode.RETRIEVAL_QUERY_NOT_STRING),
        (RetrievalResult(" ", ()), ContextValidationErrorCode.EMPTY_RETRIEVAL_QUERY),
        (RetrievalResult("query", []), ContextValidationErrorCode.RETRIEVAL_MATCHES_NOT_TUPLE),
        (RetrievalResult("query", ("match",)), ContextValidationErrorCode.RETRIEVAL_MATCH_NOT_TYPED),
        (
            RetrievalResult(
                "query",
                (RetrievalMatch(MemoryRecord("</decision_context_json>", "content"), 1.0),),
            ),
            ContextValidationErrorCode.INVALID_RETRIEVAL_MEMORY_ID,
        ),
        (
            RetrievalResult(
                "query",
                (RetrievalMatch(MemoryRecord("M001", "content"), math.inf),),
            ),
            ContextValidationErrorCode.RETRIEVAL_SCORE_NOT_FINITE_NUMBER,
        ),
        (
            RetrievalResult(
                "query",
                (RetrievalMatch(MemoryRecord("M001", "content"), 0.0),),
            ),
            ContextValidationErrorCode.RETRIEVAL_SCORE_NOT_POSITIVE,
        ),
        (
            RetrievalResult(
                "query",
                (
                    RetrievalMatch(MemoryRecord("M001", "first"), 1.0),
                    RetrievalMatch(MemoryRecord("M001", "second"), 0.5),
                ),
            ),
            ContextValidationErrorCode.DUPLICATE_MEMORY_ID_IN_RETRIEVAL,
        ),
    ],
)
def test_invalid_retrieval_is_rejected(
    retrieval: object,
    expected_code: ContextValidationErrorCode,
) -> None:
    context = _context(
        MemoryDecisionPhase.POST_TASK,
        task_result=TaskResult("answer"),
        pre_task_retrieval=retrieval,  # type: ignore[arg-type]
    )

    assert validate_memory_decision_context(context).error_code is expected_code


def test_same_visible_id_cannot_change_content_between_retrievals() -> None:
    context = _context(
        MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
        task_result=TaskResult("answer"),
        pre_task_retrieval=RetrievalResult(
            "before",
            (RetrievalMatch(MemoryRecord("M001", "old content"), 1.0),),
        ),
        post_task_retrieval=RetrievalResult(
            "after",
            (RetrievalMatch(MemoryRecord("M001", "new content"), 1.0),),
        ),
    )

    result = validate_memory_decision_context(context)

    assert result.error_code is ContextValidationErrorCode.INCONSISTENT_VISIBLE_MEMORY_CONTENT


def test_pre_task_rejects_execution_data() -> None:
    context = _context(
        MemoryDecisionPhase.PRE_TASK,
        task_result=TaskResult("answer"),
    )

    result = validate_memory_decision_context(context)

    assert result.error_code is ContextValidationErrorCode.PRE_TASK_HAS_TASK_EXECUTION_DATA


def test_post_task_requires_result_and_forbids_post_retrieval() -> None:
    missing_result = _context(MemoryDecisionPhase.POST_TASK)
    premature_retrieval = _context(
        MemoryDecisionPhase.POST_TASK,
        task_result=TaskResult("answer"),
        post_task_retrieval=_retrieval(),
    )

    assert (
        validate_memory_decision_context(missing_result).error_code
        is ContextValidationErrorCode.POST_TASK_MISSING_TASK_RESULT
    )
    assert (
        validate_memory_decision_context(premature_retrieval).error_code
        is ContextValidationErrorCode.POST_TASK_HAS_POST_TASK_RETRIEVAL
    )


def test_after_retrieval_requires_result_and_post_retrieval() -> None:
    missing_result = _context(
        MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
        post_task_retrieval=_retrieval(),
    )
    missing_retrieval = _context(
        MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
        task_result=TaskResult("answer"),
    )

    assert (
        validate_memory_decision_context(missing_result).error_code
        is ContextValidationErrorCode.AFTER_RETRIEVAL_MISSING_TASK_RESULT
    )
    assert (
        validate_memory_decision_context(missing_retrieval).error_code
        is ContextValidationErrorCode.AFTER_RETRIEVAL_MISSING_POST_TASK_RETRIEVAL
    )


def test_zero_match_post_task_retrieval_still_enters_after_retrieval_phase() -> None:
    context = _context(
        MemoryDecisionPhase.POST_TASK_AFTER_RETRIEVAL,
        task_result=TaskResult("answer"),
        post_task_retrieval=RetrievalResult(query="no match", matches=()),
    )

    result = validate_memory_decision_context(context)

    assert result.is_valid
    assert context.visible_memory_ids == ()
