from __future__ import annotations

from typing import Any

from ..schemas import ClusterCard, ClusterNode, MemoryEntry
from .metrics import MAX_CLUSTER_SIZE, cosine_similarity, jaccard, keyword_set, overlap_coefficient, tokenize


def build_cluster_card(cluster: ClusterNode, entries: list[MemoryEntry]) -> ClusterCard:
    active_entries = [entry for entry in entries if entry.status == "active"]
    representative = ""
    if active_entries:
        best = max(active_entries, key=lambda entry: (entry.utility, entry.importance, entry.updated_at))
        representative = best.summary or best.content

    risk_flags: list[str] = []
    if any(entry.status in {"superseded", "deprecated"} for entry in entries):
        risk_flags.append("contains_non_active_memory")
    if any("conflict" in entry.facets for entry in entries):
        risk_flags.append("contains_conflict")
    if cluster.dirty:
        risk_flags.append("stale_summary")

    health_flags: list[str] = []
    if cluster.size > MAX_CLUSTER_SIZE:
        health_flags.append("too_large")
    if 0 < cluster.size < 3:
        health_flags.append("too_small")
    if cluster.cohesion < 0.55:
        health_flags.append("low_cohesion")

    return ClusterCard(
        cluster_id=cluster.cluster_id,
        title=cluster.title,
        summary=cluster.summary,
        representative_memory=representative,
        facet_distribution=cluster.facet_distribution,
        size=cluster.size,
        time_range=cluster.time_range,
        keywords=cluster.keywords,
        risk_flags=risk_flags,
        health_flags=health_flags,
        retrieval_score=0.0,
        utility_score=cluster.avg_utility,
        shard_id=cluster.shard_id,
        user_id=cluster.user_id,
        project_id=cluster.project_id,
    )


def cluster_card_score(
    query: str,
    card: ClusterCard,
    *,
    query_embedding: list[float] | None = None,
    cluster_embedding: list[float] | None = None,
) -> float:
    query_terms = tokenize(query)
    card_terms = keyword_set(card.keywords, f"{card.title} {card.summary} {card.representative_memory}")
    lexical = max(jaccard(query_terms, card_terms), overlap_coefficient(query_terms, card_terms))
    semantic = cosine_similarity(query_embedding, cluster_embedding)
    utility = card.utility_score
    health_penalty = 0.08 if "low_cohesion" in card.health_flags else 0.0
    risk_penalty = 0.05 if "stale_summary" in card.risk_flags else 0.0
    if semantic > 0.0:
        return max(0.0, min(1.0, 0.70 * semantic + 0.18 * lexical + 0.12 * utility - health_penalty - risk_penalty))
    return max(0.0, min(1.0, 0.74 * lexical + 0.26 * utility - health_penalty - risk_penalty))


def _passes_filters(card: ClusterCard, filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    if (user_id := filters.get("user_id")) and card.user_id != user_id:
        return False
    if "project_id" in filters and card.project_id != filters["project_id"]:
        return False
    required_facets = set(filters.get("facets") or [])
    if required_facets and not required_facets.intersection(card.facet_distribution):
        return False
    return True


def search_cluster_cards(
    query: str,
    cluster_cards: list[ClusterCard],
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    *,
    query_embedding: list[float] | None = None,
    cluster_embeddings: dict[str, list[float] | None] | None = None,
) -> list[ClusterCard]:
    scored: list[ClusterCard] = []
    for card in cluster_cards:
        if not _passes_filters(card, filters):
            continue
        updated = card.model_copy(deep=True)
        updated.retrieval_score = cluster_card_score(
            query,
            updated,
            query_embedding=query_embedding,
            cluster_embedding=(cluster_embeddings or {}).get(card.cluster_id),
        )
        scored.append(updated)
    return sorted(scored, key=lambda card: (card.retrieval_score, card.utility_score), reverse=True)[:top_k]
