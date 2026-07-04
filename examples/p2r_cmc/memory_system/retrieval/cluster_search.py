from __future__ import annotations

import asyncio

from ..clustering.metrics import query_entry_score
from ..schemas import EvidenceItem, MemoryEntry
from .bm25 import normalized_bm25_scores


def to_evidence(entry: MemoryEntry, score: float, cluster_id: str, rationale: str) -> EvidenceItem:
    return EvidenceItem(
        memory_id=entry.memory_id,
        content=entry.content,
        summary=entry.summary,
        score=score,
        cluster_id=cluster_id,
        facets=entry.facets,
        status=entry.status,
        source_refs=entry.source_refs,
        rationale=rationale,
    )


async def simple_dense_search(
    query: str,
    cluster_id: str,
    entries: list[MemoryEntry],
    *,
    top_k: int = 5,
    delay_s: float = 0.0,
    query_embedding: list[float] | None = None,
) -> list[EvidenceItem]:
    if delay_s:
        await asyncio.sleep(delay_s)
    if query_embedding is None:
        return []
    scored = [
        to_evidence(
            entry,
            query_entry_score(query, entry, query_embedding=query_embedding),
            cluster_id,
            "dense_embedding",
        )
        for entry in entries
        if entry.status != "deprecated" and entry.embedding is not None
    ]
    return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


async def simple_sparse_search(
    query: str,
    cluster_id: str,
    entries: list[MemoryEntry],
    *,
    top_k: int = 5,
    delay_s: float = 0.0,
) -> list[EvidenceItem]:
    if delay_s:
        await asyncio.sleep(delay_s)
    active_entries = [entry for entry in entries if entry.status != "deprecated"]
    bm25_scores = normalized_bm25_scores(query, active_entries)
    scored = [
        to_evidence(entry, bm25_scores.get(entry.memory_id, 0.0), cluster_id, "bm25_sparse")
        for entry in active_entries
    ]
    return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


async def simple_temporal_search(
    query: str,
    cluster_id: str,
    entries: list[MemoryEntry],
    *,
    top_k: int = 3,
    delay_s: float = 0.0,
) -> list[EvidenceItem]:
    if delay_s:
        await asyncio.sleep(delay_s)
    recent = sorted([entry for entry in entries if entry.status == "active"], key=lambda entry: entry.updated_at, reverse=True)
    return [to_evidence(entry, 0.18 + 0.05 * entry.importance, cluster_id, "recent_memory") for entry in recent[:top_k]]


async def simple_graph_search(
    query: str,
    cluster_id: str,
    entries: list[MemoryEntry],
    *,
    top_k: int = 3,
    delay_s: float = 0.0,
) -> list[EvidenceItem]:
    if delay_s:
        await asyncio.sleep(delay_s)
    related_ids = {
        related_id
        for entry in entries
        for related_id in entry.metadata.get("related_ids", [])
        if isinstance(related_id, str)
    }
    related = [entry for entry in entries if entry.memory_id in related_ids and entry.status == "active"]
    return [to_evidence(entry, 0.2, cluster_id, "related_memory") for entry in related[:top_k]]


async def search_one_cluster_hybrid(
    query: str,
    cluster_id: str,
    entries: list[MemoryEntry],
    *,
    top_k: int = 6,
    delay_s: float = 0.0,
    query_embedding: list[float] | None = None,
) -> list[EvidenceItem]:
    results = await asyncio.gather(
        simple_dense_search(query, cluster_id, entries, top_k=top_k, delay_s=delay_s, query_embedding=query_embedding),
        simple_sparse_search(query, cluster_id, entries, top_k=top_k, delay_s=delay_s),
        simple_temporal_search(query, cluster_id, entries, top_k=min(3, top_k), delay_s=delay_s),
        simple_graph_search(query, cluster_id, entries, top_k=min(3, top_k), delay_s=delay_s),
    )
    flattened = [item for result in results for item in result]
    return flattened
