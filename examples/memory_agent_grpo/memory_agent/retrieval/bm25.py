"""Dependency-free, deterministic BM25 retrieval over one Bank snapshot."""

from __future__ import annotations

import math
import re
from collections import Counter

from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState, MemoryRecord
from examples.memory_agent_grpo.memory_agent.bank.validation import BankValidationResult, validate_memory_bank_state

from .result import RetrievalMatch, RetrievalResult


BM25_TOP_K = 3
BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


class InvalidRetrievalStateError(ValueError):
    """Raised when retrieval receives an invalid Memory Bank snapshot."""

    def __init__(self, result: BankValidationResult) -> None:
        self.validation_result = result
        code = result.error_code.value if result.error_code is not None else "UNKNOWN"
        super().__init__(f"Invalid retrieval state ({code}): {result.error_message}")


class BM25Retriever:
    """Rank positive lexical matches and return at most the fixed top three."""

    def retrieve(self, query: str, *, state: MemoryBankState) -> RetrievalResult:
        query = _require_query(query)
        validation_result = validate_memory_bank_state(state)
        if not validation_result.is_valid:
            raise InvalidRetrievalStateError(validation_result)

        query_terms = tuple(sorted(set(_tokenize(query))))
        if not query_terms or not state.records:
            return RetrievalResult(query=query, matches=())

        tokenized_documents = tuple(_tokenize(record.content) for record in state.records)
        document_lengths = tuple(len(tokens) for tokens in tokenized_documents)
        total_document_length = sum(document_lengths)
        if total_document_length == 0:
            return RetrievalResult(query=query, matches=())

        document_count = len(state.records)
        average_document_length = total_document_length / document_count
        document_frequencies: Counter[str] = Counter()
        for tokens in tokenized_documents:
            document_frequencies.update(set(tokens))

        ranked: list[tuple[float, int, MemoryRecord]] = []
        for position, (record, tokens, document_length) in enumerate(
            zip(state.records, tokenized_documents, document_lengths, strict=True)
        ):
            term_frequencies = Counter(tokens)
            score = _score_document(
                query_terms=query_terms,
                term_frequencies=term_frequencies,
                document_frequencies=document_frequencies,
                document_count=document_count,
                document_length=document_length,
                average_document_length=average_document_length,
            )
            if score > 0.0:
                ranked.append((score, position, record))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        matches = tuple(
            RetrievalMatch(record=record, score=score)
            for score, _, record in ranked[:BM25_TOP_K]
        )
        return RetrievalResult(query=query, matches=matches)


def _score_document(
    *,
    query_terms: tuple[str, ...],
    term_frequencies: Counter[str],
    document_frequencies: Counter[str],
    document_count: int,
    document_length: int,
    average_document_length: float,
) -> float:
    score = 0.0
    length_normalization = 1.0 - BM25_B + BM25_B * document_length / average_document_length
    for term in query_terms:
        term_frequency = term_frequencies.get(term, 0)
        if term_frequency == 0:
            continue
        document_frequency = document_frequencies[term]
        inverse_document_frequency = math.log1p(
            (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        saturation = term_frequency * (BM25_K1 + 1.0) / (
            term_frequency + BM25_K1 * length_normalization
        )
        score += inverse_document_frequency * saturation
    return score


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_PATTERN.findall(text))


def _require_query(query: object) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query must not be empty")
    return query


__all__ = [
    "BM25_B",
    "BM25_K1",
    "BM25_TOP_K",
    "BM25Retriever",
    "InvalidRetrievalStateError",
]
