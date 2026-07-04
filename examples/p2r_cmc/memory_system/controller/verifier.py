from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import ActionTrace, EvidenceItem, EvidencePacket, MemoryEdge
from ..store import SQLiteMemoryStore
from ..retrieval.merger import build_evidence_packet

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b[A-Za-z0-9_]*API[_-]?KEY[A-Za-z0-9_]*\s*[:=]", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9_]*TOKEN[A-Za-z0-9_]*\s*[:=]", re.IGNORECASE),
]


@dataclass
class VerificationResult:
    memory_id: str
    verdict: str
    reason: str
    keep: bool


class MemoryVerifier:
    """Lightweight query-time verifier for retrieved memories."""

    def __init__(self, store: SQLiteMemoryStore) -> None:
        self.store = store

    def verify_packet(self, packet: EvidencePacket) -> tuple[EvidencePacket, list[VerificationResult]]:
        results = [self.verify_item(item, packet.evidence_items) for item in packet.evidence_items]
        kept_ids = {result.memory_id for result in results if result.keep}
        verified_items = [item for item in packet.evidence_items if item.memory_id in kept_ids]
        conflicts = list(packet.conflicts)
        for result in results:
            if not result.keep:
                conflicts.append(f"{result.memory_id}: {result.verdict} - {result.reason}")
        verified = build_evidence_packet(
            packet.query,
            verified_items,
            max_items=len(verified_items),
            token_budget=packet.token_budget,
        )
        verified.conflicts = conflicts
        verified.omitted_memory_ids = packet.omitted_memory_ids
        self.store.append_action_trace(
            ActionTrace(
                phase="verify",
                action="CHECK_CONFLICT",
                observation_json={"query": packet.query, "memory_ids": [item.memory_id for item in packet.evidence_items]},
                action_json={"verifier": "rule_based"},
                tool_result_json={"results": [result.__dict__ for result in results]},
                valid_format=True,
            )
        )
        return verified, results

    def verify_item(self, item: EvidenceItem, selected_items: list[EvidenceItem]) -> VerificationResult:
        entry = self.store.get_memory(item.memory_id)
        if entry is None:
            return VerificationResult(item.memory_id, "unverifiable", "memory entry not found", False)
        if entry.status != "active":
            return VerificationResult(item.memory_id, "stale", f"memory status is {entry.status}", False)
        if entry.confidence < 0.45:
            return VerificationResult(item.memory_id, "unverifiable", "confidence below threshold", False)
        if any(pattern.search(entry.content) for pattern in SECRET_PATTERNS):
            return VerificationResult(item.memory_id, "unsafe", "memory appears to contain a secret or token", False)
        contradiction = self._selected_contradiction(entry.memory_id, {item.memory_id for item in selected_items})
        if contradiction is not None:
            return VerificationResult(item.memory_id, "contradicted", contradiction, False)
        return VerificationResult(item.memory_id, "valid", "memory is active and passed lightweight checks", True)

    def _selected_contradiction(self, memory_id: str, selected_ids: set[str]) -> str | None:
        for edge in self.store.list_edges(memory_id=memory_id):
            if edge.relation_type != "CONFLICTS_WITH":
                continue
            other_id = edge.dst_memory_id if edge.src_memory_id == memory_id else edge.src_memory_id
            if other_id in selected_ids:
                if self._is_older(memory_id, other_id):
                    return f"conflicts with newer selected memory {other_id}"
        return None

    def _is_older(self, left_id: str, right_id: str) -> bool:
        left = self.store.get_memory(left_id)
        right = self.store.get_memory(right_id)
        if left is None or right is None:
            return True
        return left.updated_at <= right.updated_at
