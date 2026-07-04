import asyncio
import time

from examples.p2r_cmc.memory_system.retrieval.parallel_retriever import expand_clusters_parallel
from examples.p2r_cmc.memory_system.schemas import MemoryEntry


def test_parallel_cluster_expansion_is_faster_than_serial_delay():
    entries = [
        MemoryEntry(memory_id=f"m{i}", user_id="u1", project_id="p1", content=f"P2R-CMC MemoryArena note {i}", summary=f"P2R-CMC MemoryArena note {i}", keywords=["p2r", "memoryarena"])
        for i in range(3)
    ]
    plans = [{"cluster_id": f"c{i}", "memory_ids": [entry.memory_id]} for i, entry in enumerate(entries)]
    started = time.perf_counter()
    result = asyncio.run(expand_clusters_parallel("P2R-CMC MemoryArena", plans, entries, max_concurrency=3, delay_s=0.1))
    elapsed = time.perf_counter() - started
    assert elapsed < 0.25
    assert result
