from __future__ import annotations

from typing import Any

from ..schemas import EvidencePacket


def compute_memory_metrics(
    packet: EvidencePacket,
    *,
    gold_memory_ids: set[str] | None = None,
    gold_cluster_ids: set[str] | None = None,
    latency_ms: float = 0.0,
    tool_calls: int = 0,
) -> dict[str, Any]:
    gold_memory_ids = gold_memory_ids or set()
    gold_cluster_ids = gold_cluster_ids or set()
    selected_memory_ids = {item.memory_id for item in packet.evidence_items}
    selected_cluster_ids = {item.cluster_id for item in packet.evidence_items if item.cluster_id}
    hits = selected_memory_ids & gold_memory_ids
    cluster_hits = selected_cluster_ids & gold_cluster_ids
    precision = len(hits) / len(selected_memory_ids) if selected_memory_ids else 0.0
    recall = len(hits) / len(gold_memory_ids) if gold_memory_ids else 0.0
    cluster_hit_rate = len(cluster_hits) / len(gold_cluster_ids) if gold_cluster_ids else 0.0
    stale_error = int(any(item.status in {"deprecated", "superseded"} for item in packet.evidence_items))
    return {
        "task_success": 1.0 if recall > 0 and stale_error == 0 else 0.0,
        "memory_hit_rate": recall,
        "evidence_precision": precision,
        "evidence_recall": recall,
        "cluster_hit_rate": cluster_hit_rate,
        "routing_miss_rate": 1.0 - cluster_hit_rate if gold_cluster_ids else 0.0,
        "exposed_tokens": packet.token_estimate,
        "latency_ms": latency_ms,
        "tool_calls": tool_calls,
        "stale_memory_error": stale_error,
    }
