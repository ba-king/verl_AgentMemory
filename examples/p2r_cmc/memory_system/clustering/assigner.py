from __future__ import annotations

from dataclasses import dataclass

from ..schemas import ClusterMembership, ClusterNode, MemoryEntry, utc_now
from ..store import SQLiteMemoryStore
from .metrics import (
    ASSIGN_THRESHOLD,
    MAX_CLUSTER_SIZE,
    MAX_SECONDARY_CLUSTERS,
    SECONDARY_THRESHOLD,
    entry_cluster_similarity,
    refresh_cluster_stats,
)


@dataclass
class AssignmentResult:
    entry: MemoryEntry
    primary_cluster: ClusterNode
    secondary_clusters: list[ClusterNode]
    created_new_cluster: bool
    scores: dict[str, float]


def create_cluster_from_entry(entry: MemoryEntry) -> ClusterNode:
    title = entry.summary[:80] or entry.content[:80]
    return ClusterNode(
        title=title,
        summary=entry.summary or entry.content,
        keywords=entry.keywords,
        entities=entry.entities,
        facet_distribution={facet: 1 for facet in entry.facets},
        member_memory_ids=[entry.memory_id],
        medoid_memory_id=entry.memory_id,
        size=1,
        user_id=entry.user_id,
        project_id=entry.project_id,
        scope=entry.scope,
        shard_id=f"shard:{entry.project_id or entry.user_id}",
        dirty=True,
    )


def assign_entry_to_cluster(
    entry: MemoryEntry,
    clusters: list[ClusterNode],
    *,
    assign_threshold: float = ASSIGN_THRESHOLD,
    max_cluster_size: int = MAX_CLUSTER_SIZE,
) -> AssignmentResult:
    eligible_clusters = [
        cluster
        for cluster in clusters
        if cluster.status == "active"
        and cluster.user_id == entry.user_id
        and cluster.project_id == entry.project_id
        and cluster.scope == entry.scope
    ]
    scored = sorted(
        ((cluster, entry_cluster_similarity(entry, cluster)) for cluster in eligible_clusters),
        key=lambda item: item[1],
        reverse=True,
    )
    scores = {cluster.cluster_id: score for cluster, score in scored}
    if scored and scored[0][1] >= assign_threshold and scored[0][0].size < max_cluster_size:
        primary = scored[0][0]
        secondary = [
            cluster
            for cluster, score in scored[1 : 1 + MAX_SECONDARY_CLUSTERS]
            if score >= SECONDARY_THRESHOLD and cluster.size < max_cluster_size
        ]
        entry.primary_cluster_id = primary.cluster_id
        entry.secondary_cluster_ids = [cluster.cluster_id for cluster in secondary]
        return AssignmentResult(entry, primary, secondary, False, scores)

    primary = create_cluster_from_entry(entry)
    entry.primary_cluster_id = primary.cluster_id
    entry.secondary_cluster_ids = []
    return AssignmentResult(entry, primary, [], True, scores)


def persist_assignment(store: SQLiteMemoryStore, result: AssignmentResult) -> AssignmentResult:
    entry = result.entry
    primary = result.primary_cluster
    if entry.memory_id not in primary.member_memory_ids:
        primary.member_memory_ids.append(entry.memory_id)
    primary.dirty = True
    primary.updated_at = utc_now()
    primary_entries = store.get_memories(primary.member_memory_ids)
    primary = refresh_cluster_stats(primary, primary_entries or [entry])
    store.upsert_cluster(primary)
    store.upsert_membership(
        ClusterMembership(
            cluster_id=primary.cluster_id,
            memory_id=entry.memory_id,
            role="primary",
            score=result.scores.get(primary.cluster_id, 1.0),
        )
    )

    for secondary in result.secondary_clusters:
        if entry.memory_id not in secondary.member_memory_ids:
            secondary.member_memory_ids.append(entry.memory_id)
        secondary.dirty = True
        secondary.updated_at = utc_now()
        secondary_entries = store.get_memories(secondary.member_memory_ids)
        secondary = refresh_cluster_stats(secondary, secondary_entries or [entry])
        store.upsert_cluster(secondary)
        store.upsert_membership(
            ClusterMembership(
                cluster_id=secondary.cluster_id,
                memory_id=entry.memory_id,
                role="secondary",
                score=result.scores.get(secondary.cluster_id, SECONDARY_THRESHOLD),
            )
        )

    store.upsert_memory(entry)
    return result


def assign_and_persist_entry(store: SQLiteMemoryStore, entry: MemoryEntry) -> AssignmentResult:
    clusters = store.list_clusters(user_id=entry.user_id, project_id=entry.project_id, statuses={"active"})
    result = assign_entry_to_cluster(entry, clusters)
    return persist_assignment(store, result)
