from __future__ import annotations

from typing import Any

from .store import SQLiteMemoryStore


def _compact_embedding(payload: dict[str, Any], *, include_embeddings: bool) -> dict[str, Any]:
    if include_embeddings:
        return payload
    embedding = payload.get("embedding")
    if isinstance(embedding, list):
        payload["embedding"] = f"<omitted dim={len(embedding)}>"
    centroid = payload.get("centroid_embedding")
    if isinstance(centroid, list):
        payload["centroid_embedding"] = f"<omitted dim={len(centroid)}>"
    return payload


def export_store_json(store: SQLiteMemoryStore, *, include_embeddings: bool = False) -> dict[str, Any]:
    memories = [
        _compact_embedding(memory.model_dump(mode="json"), include_embeddings=include_embeddings)
        for memory in store.list_memories()
    ]
    clusters = [
        _compact_embedding(cluster.model_dump(mode="json"), include_embeddings=include_embeddings)
        for cluster in store.list_clusters()
    ]
    memberships = [membership.model_dump(mode="json") for membership in store.list_memberships()]
    edges = [edge.model_dump(mode="json") for edge in store.list_edges()]
    action_traces = [trace.model_dump(mode="json") for trace in store.list_action_traces()]
    exposure_events = [event.model_dump(mode="json") for event in store.list_exposure_events()]
    return {
        "counts": {
            "memories": len(memories),
            "clusters": len(clusters),
            "memberships": len(memberships),
            "edges": len(edges),
            "action_traces": len(action_traces),
            "exposure_events": len(exposure_events),
        },
        "memories": memories,
        "clusters": clusters,
        "memberships": memberships,
        "edges": edges,
        "action_traces": action_traces,
        "exposure_events": exposure_events,
    }
