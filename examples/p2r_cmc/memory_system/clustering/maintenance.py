from __future__ import annotations

from ..schemas import ClusterHealthReport, ClusterNode, MemoryEntry
from .metrics import MAX_CLUSTER_SIZE, jaccard, keyword_set, refresh_cluster_stats


def cluster_similarity(left: ClusterNode, right: ClusterNode) -> float:
    if left.user_id != right.user_id or left.project_id != right.project_id or left.scope != right.scope:
        return 0.0
    left_terms = keyword_set(left.keywords, f"{left.title} {left.summary}")
    right_terms = keyword_set(right.keywords, f"{right.title} {right.summary}")
    facet_score = jaccard(set(left.facet_distribution), set(right.facet_distribution))
    return 0.8 * jaccard(left_terms, right_terms) + 0.2 * facet_score


def maintain_clusters(
    clusters: list[ClusterNode],
    entries: list[MemoryEntry],
    *,
    max_cluster_size: int = MAX_CLUSTER_SIZE,
    merge_threshold: float = 0.88,
) -> ClusterHealthReport:
    by_id = {entry.memory_id: entry for entry in entries}
    refreshed: list[ClusterNode] = []
    for cluster in clusters:
        members = [by_id[memory_id] for memory_id in cluster.member_memory_ids if memory_id in by_id]
        refreshed.append(refresh_cluster_stats(cluster.model_copy(deep=True), members))

    active_clusters = [cluster for cluster in refreshed if cluster.status == "active"]
    sizes = [cluster.size for cluster in active_clusters]
    num_clusters = len(active_clusters)
    avg_size = sum(sizes) / num_clusters if num_clusters else 0.0
    small_ratio = sum(1 for size in sizes if size < 3) / num_clusters if num_clusters else 0.0
    large_ratio = sum(1 for size in sizes if size > max_cluster_size) / num_clusters if num_clusters else 0.0
    avg_radius = sum(cluster.radius for cluster in active_clusters) / num_clusters if num_clusters else 0.0
    avg_cohesion = sum(cluster.cohesion for cluster in active_clusters) / num_clusters if num_clusters else 0.0

    split_candidates: list[str] = []
    refresh_candidates: list[str] = []
    proposals: list[dict[str, object]] = []
    for cluster in active_clusters:
        if cluster.size > max_cluster_size or cluster.radius > 0.38 or cluster.cohesion < 0.55:
            split_candidates.append(cluster.cluster_id)
            proposals.append(
                {
                    "action": "SPLIT_CLUSTER",
                    "cluster_id": cluster.cluster_id,
                    "reason": "large_or_low_cohesion",
                    "size": cluster.size,
                    "radius": cluster.radius,
                    "cohesion": cluster.cohesion,
                }
            )
        if cluster.dirty:
            refresh_candidates.append(cluster.cluster_id)
            proposals.append({"action": "REFRESH_CLUSTER_CARD", "cluster_id": cluster.cluster_id, "reason": "dirty"})

    merge_candidates: list[tuple[str, str]] = []
    for idx, left in enumerate(active_clusters):
        for right in active_clusters[idx + 1 :]:
            score = cluster_similarity(left, right)
            if score >= merge_threshold:
                merge_candidates.append((left.cluster_id, right.cluster_id))
                proposals.append(
                    {
                        "action": "MERGE_CLUSTER",
                        "cluster_ids": [left.cluster_id, right.cluster_id],
                        "reason": "high_cluster_similarity",
                        "score": score,
                    }
                )

    return ClusterHealthReport(
        num_clusters=num_clusters,
        avg_size=avg_size,
        small_ratio=small_ratio,
        large_ratio=large_ratio,
        avg_radius=avg_radius,
        avg_cohesion=avg_cohesion,
        split_candidates=split_candidates,
        merge_candidates=merge_candidates,
        refresh_candidates=refresh_candidates,
        proposals=proposals,
    )
