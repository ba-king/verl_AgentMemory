from dataclasses import FrozenInstanceError

import pytest

from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryRecord
from examples.memory_agent_grpo.memory_agent.retrieval.result import RetrievalMatch, RetrievalResult


def test_match_exposes_record_fields_and_serializes_score() -> None:
    match = RetrievalMatch(record=MemoryRecord("M001", "memory"), score=1.25)

    assert match.memory_id == "M001"
    assert match.content == "memory"
    assert match.to_dict() == {
        "memory_id": "M001",
        "content": "memory",
        "score": 1.25,
    }


def test_result_preserves_ranked_order_and_derives_metadata() -> None:
    first = RetrievalMatch(record=MemoryRecord("M003", "best"), score=2.0)
    second = RetrievalMatch(record=MemoryRecord("M001", "next"), score=1.0)
    result = RetrievalResult(query="hotel", matches=(first, second))

    assert result.match_count == 2
    assert result.memory_ids == ("M003", "M001")
    assert result.to_dict() == {
        "query": "hotel",
        "matches": [first.to_dict(), second.to_dict()],
    }


def test_retrieval_models_are_immutable() -> None:
    match = RetrievalMatch(record=MemoryRecord("M001", "memory"), score=1.0)
    result = RetrievalResult(query="memory", matches=(match,))

    with pytest.raises(FrozenInstanceError):
        match.score = 2.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.query = "changed"  # type: ignore[misc]
