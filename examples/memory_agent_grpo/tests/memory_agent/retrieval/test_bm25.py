import math

import pytest

from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState, MemoryRecord
from examples.memory_agent_grpo.memory_agent.bank.validation import BankValidationErrorCode
from examples.memory_agent_grpo.memory_agent.retrieval.bm25 import (
    BM25_TOP_K,
    BM25Retriever,
    InvalidRetrievalStateError,
)


def _state(*contents: str) -> MemoryBankState:
    records = tuple(
        MemoryRecord(memory_id=f"M{position:03d}", content=content)
        for position, content in enumerate(contents, start=1)
    )
    return MemoryBankState(records=records, next_id=len(records) + 1)


def test_empty_bank_returns_an_empty_result() -> None:
    result = BM25Retriever().retrieve("hotel", state=MemoryBankState.empty())

    assert result.query == "hotel"
    assert result.matches == ()


def test_query_without_lexical_overlap_returns_no_zero_score_fillers() -> None:
    state = _state("Paris hotel", "Tokyo flight")

    result = BM25Retriever().retrieve("museum", state=state)

    assert result.matches == ()


def test_single_term_score_matches_the_frozen_bm25_formula() -> None:
    result = BM25Retriever().retrieve("hotel", state=_state("hotel"))

    assert result.matches[0].score == pytest.approx(math.log(4.0 / 3.0))


def test_term_frequency_and_length_normalization_match_the_frozen_formula() -> None:
    result = BM25Retriever().retrieve(
        "hotel",
        state=_state("hotel hotel paris", "hotel tokyo"),
    )
    inverse_document_frequency = math.log1p(0.5 / 2.5)
    expected_first = inverse_document_frequency * (2.0 * 2.5) / (
        2.0 + 1.5 * (0.25 + 0.75 * 3.0 / 2.5)
    )
    expected_second = inverse_document_frequency * 2.5 / (
        1.0 + 1.5 * (0.25 + 0.75 * 2.0 / 2.5)
    )

    assert result.matches[0].score == pytest.approx(expected_first)
    assert result.matches[1].score == pytest.approx(expected_second)


def test_retrieval_returns_at_most_the_fixed_top_three() -> None:
    state = _state(
        "hotel alpha",
        "hotel bravo",
        "hotel charlie",
        "hotel delta",
        "hotel echo",
    )

    result = BM25Retriever().retrieve("hotel", state=state)

    assert BM25_TOP_K == 3
    assert result.match_count == 3
    assert result.memory_ids == ("M001", "M002", "M003")


def test_higher_term_frequency_can_rank_a_document_first() -> None:
    state = _state(
        "hotel hotel hotel paris",
        "hotel paris",
        "flight tokyo",
    )

    result = BM25Retriever().retrieve("hotel", state=state)

    assert result.memory_ids == ("M001", "M002")
    assert result.matches[0].score > result.matches[1].score > 0.0


def test_exact_score_ties_preserve_creation_order() -> None:
    state = _state("hotel alpha", "hotel bravo", "flight charlie")

    result = BM25Retriever().retrieve("hotel", state=state)

    assert result.memory_ids == ("M001", "M002")
    assert result.matches[0].score == pytest.approx(result.matches[1].score)


def test_tokenization_is_case_insensitive_and_keeps_numeric_terms() -> None:
    state = _state("Hotel A costs $815.", "Hotel B costs $900.")

    result = BM25Retriever().retrieve("HOTEL-a $815", state=state)

    assert result.memory_ids[0] == "M001"
    assert all(match.score > 0.0 for match in result.matches)


def test_tokenization_uses_unicode_casefold_and_splits_underscores() -> None:
    state = _state("Straße foo_bar", "unrelated memory")

    result = BM25Retriever().retrieve("STRASSE foo bar", state=state)

    assert result.memory_ids == ("M001",)


def test_repeated_query_terms_do_not_change_ranking_or_scores() -> None:
    state = _state("hotel hotel paris", "hotel tokyo")
    retriever = BM25Retriever()

    once = retriever.retrieve("hotel", state=state)
    repeated = retriever.retrieve("hotel hotel hotel", state=state)

    assert repeated.matches == once.matches


def test_query_with_no_searchable_tokens_returns_empty_result() -> None:
    state = _state("hotel paris")

    result = BM25Retriever().retrieve("___---$$$", state=state)

    assert result.matches == ()


def test_corpus_with_no_searchable_tokens_returns_empty_result() -> None:
    state = _state("___", "---")

    result = BM25Retriever().retrieve("hotel", state=state)

    assert result.matches == ()


def test_retrieval_preserves_the_original_query_for_audit() -> None:
    state = _state("hotel paris")

    result = BM25Retriever().retrieve("  HOTEL  ", state=state)

    assert result.query == "  HOTEL  "
    assert result.memory_ids == ("M001",)


def test_every_returned_score_is_positive_and_finite() -> None:
    state = _state("hotel paris", "hotel tokyo")

    result = BM25Retriever().retrieve("hotel", state=state)

    assert result.matches
    assert all(match.score > 0.0 and math.isfinite(match.score) for match in result.matches)


def test_retrieval_does_not_mutate_the_bank_snapshot() -> None:
    state = _state("hotel paris", "flight tokyo")
    serialized_before = state.to_dict()

    BM25Retriever().retrieve("hotel", state=state)

    assert state.to_dict() == serialized_before


@pytest.mark.parametrize("query", [None, 1, True])
def test_query_must_be_a_string(query: object) -> None:
    with pytest.raises(TypeError, match="query must be a string"):
        BM25Retriever().retrieve(query, state=MemoryBankState.empty())  # type: ignore[arg-type]


@pytest.mark.parametrize("query", ["", "  \n\t"])
def test_query_must_not_be_empty(query: str) -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        BM25Retriever().retrieve(query, state=MemoryBankState.empty())


def test_invalid_bank_state_is_rejected_before_scoring() -> None:
    invalid_state = MemoryBankState(
        records=(MemoryRecord("M000", "hotel"),),
        next_id=1,
    )

    with pytest.raises(InvalidRetrievalStateError) as exc_info:
        BM25Retriever().retrieve("hotel", state=invalid_state)

    assert exc_info.value.validation_result.error_code is BankValidationErrorCode.INVALID_MEMORY_ID
