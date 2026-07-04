import asyncio

from examples.p2r_cmc.memory_system.retrieval.bm25 import normalized_bm25_scores
from examples.p2r_cmc.memory_system.retrieval.cluster_search import simple_sparse_search
from examples.p2r_cmc.memory_system.schemas import MemoryEntry


def make_entry(memory_id: str, content: str, keywords: list[str]) -> MemoryEntry:
    return MemoryEntry(
        memory_id=memory_id,
        user_id="u1",
        project_id="p1",
        content=content,
        summary=content,
        keywords=keywords,
        facets=["project_state"],
    )


def test_bm25_scores_exact_project_terms_above_unrelated_memory():
    entries = [
        make_entry(
            "target",
            "P2R-CMC Stage 1 should be evaluated with MemoryArena before RL.",
            ["p2r-cmc", "memoryarena", "stage"],
        ),
        make_entry(
            "other",
            "The user prefers iced latte before noon and avoids caffeine later.",
            ["latte", "caffeine"],
        ),
    ]
    scores = normalized_bm25_scores("How should P2R-CMC use MemoryArena?", entries)
    assert scores["target"] > scores["other"]
    assert 0.0 < scores["target"] < 1.0


def test_sparse_cluster_search_uses_bm25_rationale():
    entries = [
        make_entry("target", "MemoryArena tests P2R-CMC prompt memory.", ["memoryarena", "p2r-cmc"]),
        make_entry("other", "Dinner preference includes noodles and eggs.", ["dinner"]),
    ]
    result = asyncio.run(simple_sparse_search("P2R-CMC MemoryArena", "c1", entries, top_k=1))
    assert result[0].memory_id == "target"
    assert result[0].rationale == "bm25_sparse"
