from examples.p2r_cmc.memory_system.retrieval.merger import global_merge, rrf_fuse
from examples.p2r_cmc.memory_system.schemas import EvidenceItem


def item(memory_id: str, score: float, rationale: str) -> EvidenceItem:
    return EvidenceItem(
        memory_id=memory_id,
        content=f"content {memory_id}",
        summary=f"summary {memory_id}",
        score=score,
        cluster_id="c1",
        status="active",
        rationale=rationale,
    )


def test_rrf_fusion_rewards_cross_retriever_agreement():
    fused = rrf_fuse(
        [
            item("shared", 0.72, "dense_embedding"),
            item("dense_only", 0.95, "dense_embedding"),
            item("shared", 0.70, "bm25_sparse"),
            item("bm25_only", 0.92, "bm25_sparse"),
        ],
        top_k=3,
    )
    assert fused[0].memory_id == "shared"
    assert fused[0].rationale == "rrf:bm25_sparse,dense_embedding"


def test_global_merge_uses_rrf_by_default_and_can_fallback_to_max_score():
    items = [
        item("shared", 0.6, "dense_embedding"),
        item("single", 0.99, "dense_embedding"),
        item("shared", 0.6, "bm25_sparse"),
    ]
    assert global_merge(items, top_k=1)[0].memory_id == "shared"
    assert global_merge(items, top_k=1, fusion="max_score")[0].memory_id == "single"
