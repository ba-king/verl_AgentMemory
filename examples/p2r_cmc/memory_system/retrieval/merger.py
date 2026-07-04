from __future__ import annotations

from collections import defaultdict

from ..schemas import EvidenceItem, EvidencePacket
from .aspect_extractor import extract_query_aspects
from .coverage_selector import select_evidence_with_coverage
from .reranker import rerank_evidence

RRF_K = 60
RRF_WEIGHTS = {
    "dense_embedding": 1.0,
    "bm25_sparse": 1.0,
    "recent_memory": 0.35,
    "related_memory": 0.25,
    "global_dense_embedding_topk": 1.0,
    "global_dense_topk": 1.0,
    "bm25_topk": 1.0,
}


def global_merge(
    items: list[EvidenceItem],
    *,
    top_k: int = 8,
    allow_non_active: bool = False,
    fusion: str = "rrf",
) -> list[EvidenceItem]:
    candidates = [
        item for item in items if allow_non_active or item.status == "active"
    ]
    if fusion == "max_score":
        return max_score_merge(candidates, top_k=top_k)
    return rrf_fuse(candidates, top_k=top_k)


def max_score_merge(items: list[EvidenceItem], *, top_k: int = 8) -> list[EvidenceItem]:
    best_by_memory: dict[str, EvidenceItem] = {}
    for item in items:
        current = best_by_memory.get(item.memory_id)
        if current is None or item.score > current.score:
            best_by_memory[item.memory_id] = item
    return rerank_evidence(list(best_by_memory.values()), top_k=top_k)


def rrf_fuse(
    items: list[EvidenceItem],
    *,
    top_k: int = 8,
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> list[EvidenceItem]:
    if not items:
        return []
    weights = weights or RRF_WEIGHTS
    by_retriever: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in items:
        by_retriever[item.rationale or "unknown"].append(item)

    raw_scores: dict[str, float] = defaultdict(float)
    best_item: dict[str, EvidenceItem] = {}
    best_original_score: dict[str, float] = defaultdict(float)
    rationales: dict[str, set[str]] = defaultdict(set)
    for retriever, retriever_items in by_retriever.items():
        ranked = sorted(retriever_items, key=lambda item: item.score, reverse=True)
        weight = weights.get(retriever, 0.5)
        seen_in_retriever: set[str] = set()
        for rank, item in enumerate(ranked, start=1):
            if item.memory_id in seen_in_retriever:
                continue
            seen_in_retriever.add(item.memory_id)
            raw_scores[item.memory_id] += weight / (k + rank)
            rationales[item.memory_id].add(retriever)
            if item.score > best_original_score[item.memory_id]:
                best_original_score[item.memory_id] = item.score
                best_item[item.memory_id] = item

    max_raw = max(raw_scores.values(), default=0.0)
    fused_items: list[EvidenceItem] = []
    for memory_id, raw_score in raw_scores.items():
        item = best_item[memory_id].model_copy(deep=True)
        rrf_score = raw_score / max_raw if max_raw > 0 else 0.0
        item.score = min(1.0, 0.85 * rrf_score + 0.15 * best_original_score[memory_id])
        item.rationale = "rrf:" + ",".join(sorted(rationales[memory_id]))
        fused_items.append(item)
    return rerank_evidence(fused_items, top_k=top_k)


def estimate_tokens(items: list[EvidenceItem]) -> int:
    return sum(max(1, len(item.content) // 4) for item in items)


def build_evidence_packet(
    query: str,
    items: list[EvidenceItem],
    *,
    max_items: int = 6,
    token_budget: int = 1200,
) -> EvidencePacket:
    aspects = extract_query_aspects(query)
    selection = select_evidence_with_coverage(
        items,
        aspects,
        max_items=max_items,
        token_budget=token_budget,
    )
    selected = selection.selected

    profile_hints = [
        item.summary for item in selected if "preference" in item.facets and item.status == "active"
    ][:3]
    task_state = [
        item.summary for item in selected if "project_state" in item.facets or "task_state" in item.facets
    ][:4]
    conflicts = [item.summary for item in selected if "conflict" in item.facets]
    return EvidencePacket(
        query=query,
        evidence_items=selected,
        profile_hints=profile_hints,
        task_state=task_state,
        conflicts=conflicts,
        required_aspects=aspects,
        coverage=selection.coverage,
        missing_aspects=selection.missing_aspects,
        token_budget=token_budget,
        token_estimate=selection.token_estimate,
        omitted_memory_ids=selection.omitted_memory_ids,
    )
