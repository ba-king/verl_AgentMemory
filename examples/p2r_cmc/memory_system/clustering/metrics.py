from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from typing import Iterable

from ..schemas import ClusterNode, MemoryEntry, utc_now

ASSIGN_THRESHOLD = 0.72
SECONDARY_THRESHOLD = 0.55
MAX_CLUSTER_SIZE = 100
MAX_SECONDARY_CLUSTERS = 2


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", text.lower())
    return {token for token in tokens if len(token) > 1}


def keyword_set(keywords: Iterable[str], text: str = "") -> set[str]:
    return {item.lower() for item in keywords if item} | tokenize(text)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def overlap_coefficient(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    # Normalize from [-1, 1] to [0, 1] so it can mix with lexical scores.
    return max(0.0, min(1.0, (dot / (left_norm * right_norm) + 1.0) / 2.0))


def average_embedding(entries: Iterable[MemoryEntry]) -> list[float] | None:
    vectors = [entry.embedding for entry in entries if entry.embedding]
    if not vectors:
        return None
    dim = len(vectors[0])
    compatible = [vector for vector in vectors if len(vector) == dim]
    if not compatible:
        return None
    return [sum(vector[idx] for vector in compatible) / len(compatible) for idx in range(dim)]


def lexical_similarity(entry: MemoryEntry, cluster: ClusterNode) -> float:
    entry_terms = keyword_set(entry.keywords, f"{entry.summary} {entry.content}")
    cluster_terms = keyword_set(cluster.keywords, f"{cluster.title} {cluster.summary}")
    keyword_score = max(jaccard(entry_terms, cluster_terms), overlap_coefficient(set(entry.keywords), set(cluster.keywords)))
    facet_score = 0.0
    if entry.facets and cluster.facet_distribution:
        cluster_facets = set(cluster.facet_distribution)
        facet_score = len(set(entry.facets) & cluster_facets) / len(set(entry.facets) | cluster_facets)
    scope_score = 1.0 if entry.scope == cluster.scope else 0.0
    project_score = 1.0 if entry.project_id == cluster.project_id else 0.0
    return min(1.0, 0.58 * keyword_score + 0.18 * facet_score + 0.12 * scope_score + 0.12 * project_score)


def entry_cluster_similarity(entry: MemoryEntry, cluster: ClusterNode) -> float:
    lexical = lexical_similarity(entry, cluster)
    semantic = cosine_similarity(entry.embedding, cluster.centroid_embedding)
    if semantic > 0.0:
        return min(1.0, 0.72 * semantic + 0.28 * lexical)
    return lexical


def entry_similarity(left: MemoryEntry, right: MemoryEntry) -> float:
    left_terms = keyword_set(left.keywords, f"{left.summary} {left.content}")
    right_terms = keyword_set(right.keywords, f"{right.summary} {right.content}")
    facet_score = jaccard(set(left.facets), set(right.facets))
    lexical = min(1.0, 0.75 * jaccard(left_terms, right_terms) + 0.25 * facet_score)
    semantic = cosine_similarity(left.embedding, right.embedding)
    if semantic > 0.0:
        return min(1.0, 0.72 * semantic + 0.28 * lexical)
    return lexical


def query_entry_score(query: str, entry: MemoryEntry, query_embedding: list[float] | None = None) -> float:
    query_terms = tokenize(query)
    entry_terms = keyword_set(entry.keywords, f"{entry.summary} {entry.content}")
    lexical = max(jaccard(query_terms, entry_terms), overlap_coefficient(query_terms, entry_terms))
    semantic = cosine_similarity(query_embedding, entry.embedding)
    utility = entry.utility
    importance = entry.importance
    recency = 0.0
    if entry.updated_at:
        age_days = max(0.0, (utc_now() - entry.updated_at).total_seconds() / 86400)
        recency = 1 / (1 + age_days)
    if semantic > 0.0:
        return min(1.0, 0.68 * semantic + 0.16 * lexical + 0.08 * utility + 0.05 * importance + 0.03 * recency)
    return min(1.0, 0.62 * lexical + 0.16 * utility + 0.14 * importance + 0.08 * recency)


def facet_distribution(entries: Iterable[MemoryEntry]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for entry in entries:
        counter.update(entry.facets)
    return dict(counter)


def merged_keywords(entries: Iterable[MemoryEntry], limit: int = 12) -> list[str]:
    counter: Counter[str] = Counter()
    for entry in entries:
        counter.update(keyword_set(entry.keywords, entry.summary))
    return [item for item, _ in counter.most_common(limit)]


def time_range(entries: list[MemoryEntry]) -> tuple[datetime, datetime] | None:
    if not entries:
        return None
    created = [entry.created_at for entry in entries]
    return (min(created), max(created))


def entropy(distribution: dict[str, int]) -> float:
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log(count / total, 2) for count in distribution.values() if count)


def cluster_cohesion(entries: list[MemoryEntry]) -> float:
    if len(entries) <= 1:
        return 1.0
    scores: list[float] = []
    for idx, left in enumerate(entries):
        for right in entries[idx + 1 :]:
            scores.append(entry_similarity(left, right))
    return sum(scores) / len(scores) if scores else 1.0


def cluster_radius(entries: list[MemoryEntry]) -> float:
    return max(0.0, 1.0 - cluster_cohesion(entries))


def refresh_cluster_stats(cluster: ClusterNode, entries: list[MemoryEntry]) -> ClusterNode:
    active_entries = [entry for entry in entries if entry.status == "active"]
    cluster.size = len(active_entries)
    cluster.facet_distribution = facet_distribution(active_entries)
    cluster.keywords = merged_keywords(active_entries)
    cluster.time_range = time_range(active_entries)
    cluster.avg_utility = (
        sum(entry.utility for entry in active_entries) / len(active_entries) if active_entries else 0.0
    )
    cluster.centroid_embedding = average_embedding(active_entries)
    cluster.cohesion = cluster_cohesion(active_entries)
    cluster.radius = cluster_radius(active_entries)
    cluster.entropy = entropy(cluster.facet_distribution)
    cluster.updated_at = utc_now()
    return cluster
