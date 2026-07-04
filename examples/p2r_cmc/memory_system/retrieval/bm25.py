from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from ..schemas import MemoryEntry


def bm25_tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", text.lower())
        if len(token) > 1
    ]


def entry_bm25_document(entry: MemoryEntry) -> list[str]:
    """Build a lexical document for sparse memory retrieval.

    Keywords are intentionally repeated once because they are curator/model
    selected terms and should carry slightly more sparse weight than raw text.
    """

    keyword_text = " ".join(entry.keywords)
    entity_text = " ".join(entry.entities)
    return bm25_tokenize(
        f"{keyword_text} {keyword_text} {entity_text} {entry.summary} {entry.content}"
    )


@dataclass(frozen=True)
class BM25Document:
    memory_id: str
    tokens: list[str]


class BM25Index:
    """Small Okapi BM25 index for in-memory candidate sets."""

    def __init__(
        self,
        documents: Iterable[BM25Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.doc_freq: Counter[str] = Counter()
        self.term_freqs: dict[str, Counter[str]] = {}
        self.doc_lengths: dict[str, int] = {}
        for document in self.documents:
            term_freq = Counter(document.tokens)
            self.term_freqs[document.memory_id] = term_freq
            self.doc_lengths[document.memory_id] = len(document.tokens)
            self.doc_freq.update(term_freq.keys())
        self.avg_doc_len = (
            sum(self.doc_lengths.values()) / len(self.doc_lengths)
            if self.doc_lengths
            else 0.0
        )

    @classmethod
    def from_entries(cls, entries: Iterable[MemoryEntry]) -> "BM25Index":
        documents = [
            BM25Document(memory_id=entry.memory_id, tokens=entry_bm25_document(entry))
            for entry in entries
        ]
        return cls(documents)

    def score(self, query: str) -> dict[str, float]:
        query_terms = bm25_tokenize(query)
        if not query_terms or not self.documents or self.avg_doc_len <= 0.0:
            return {document.memory_id: 0.0 for document in self.documents}

        num_docs = len(self.documents)
        scores: dict[str, float] = {}
        for document in self.documents:
            score = 0.0
            term_freq = self.term_freqs[document.memory_id]
            doc_len = self.doc_lengths[document.memory_id]
            for term in query_terms:
                frequency = term_freq.get(term, 0)
                if frequency == 0:
                    continue
                idf = math.log(1 + (num_docs - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += idf * (frequency * (self.k1 + 1)) / denominator
            scores[document.memory_id] = score
        return scores


def normalized_bm25_scores(query: str, entries: Iterable[MemoryEntry]) -> dict[str, float]:
    active_entries = list(entries)
    raw_scores = BM25Index.from_entries(active_entries).score(query)
    return {
        memory_id: score / (score + 1.0) if score > 0.0 else 0.0
        for memory_id, score in raw_scores.items()
    }
