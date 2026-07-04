from __future__ import annotations

import asyncio
from typing import Any

from ..schemas import EvidenceItem, MemoryEntry
from .cluster_search import search_one_cluster_hybrid
from .merger import global_merge


def _entry_map(entries: list[MemoryEntry] | dict[str, MemoryEntry]) -> dict[str, MemoryEntry]:
    if isinstance(entries, dict):
        return entries
    return {entry.memory_id: entry for entry in entries}


async def expand_clusters_parallel(
    query: str,
    cluster_plans: list[dict[str, Any]],
    entries: list[MemoryEntry] | dict[str, MemoryEntry],
    *,
    max_concurrency: int = 8,
    top_k_per_cluster: int = 6,
    delay_s: float = 0.0,
    query_embedding: list[float] | None = None,
) -> list[EvidenceItem]:
    semaphore = asyncio.Semaphore(max_concurrency)
    by_id = _entry_map(entries)

    async def run_one(plan: dict[str, Any]) -> list[EvidenceItem]:
        cluster_id = plan["cluster_id"]
        memory_ids = plan.get("memory_ids", [])
        cluster_entries = [by_id[memory_id] for memory_id in memory_ids if memory_id in by_id]
        async with semaphore:
            return await search_one_cluster_hybrid(
                query,
                cluster_id,
                cluster_entries,
                top_k=top_k_per_cluster,
                delay_s=delay_s,
                query_embedding=query_embedding,
            )

    results = await asyncio.gather(*(run_one(plan) for plan in cluster_plans))
    flattened = [item for result in results for item in result]
    return global_merge(flattened, top_k=top_k_per_cluster * max(1, len(cluster_plans)))
