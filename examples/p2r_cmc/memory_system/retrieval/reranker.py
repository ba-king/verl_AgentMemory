from __future__ import annotations

from ..schemas import EvidenceItem


def rerank_evidence(items: list[EvidenceItem], top_k: int = 8) -> list[EvidenceItem]:
    return sorted(items, key=lambda item: item.score, reverse=True)[:top_k]
