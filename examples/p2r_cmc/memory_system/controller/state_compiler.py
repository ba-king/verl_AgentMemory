from __future__ import annotations

from typing import Any

from ..clustering.router import build_cluster_card, search_cluster_cards
from ..model_clients import EmbeddingClient
from ..schemas import PolicyObservation
from ..store import SQLiteMemoryStore


class StateCompiler:
    """Compiles bounded policy observations from the full store."""

    def __init__(self, store: SQLiteMemoryStore, *, embedder: EmbeddingClient | None = None) -> None:
        self.store = store
        self.embedder = embedder

    def compile(
        self,
        *,
        query: str,
        budget: dict[str, Any] | None = None,
        phase: str = "use",
        user_id: str | None = None,
        project_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> PolicyObservation:
        budget = budget or {"max_clusters": 3, "max_items": 6, "token_budget": 1200}
        filters = filters or {}
        if user_id:
            filters["user_id"] = user_id
        if project_id is not None:
            filters["project_id"] = project_id

        clusters = self.store.list_clusters(user_id=user_id, project_id=project_id, statuses={"active"})
        cards = []
        for cluster in clusters:
            entries = self.store.get_memories(cluster.member_memory_ids)
            cards.append(build_cluster_card(cluster, entries))
        query_embedding = self.embedder.embed_text(query) if self.embedder is not None else None
        cluster_embeddings = {cluster.cluster_id: cluster.centroid_embedding for cluster in clusters}
        top_cards = search_cluster_cards(
            query,
            cards,
            top_k=int(budget.get("top_m", 12)),
            filters=filters,
            query_embedding=query_embedding,
            cluster_embeddings=cluster_embeddings,
        )
        entries_count = len(self.store.list_memories(user_id=user_id, project_id=project_id))
        risk_flags = []
        if any("stale_summary" in card.risk_flags for card in top_cards):
            risk_flags.append("some_cluster_cards_stale")
        return PolicyObservation(
            query=query,
            phase=phase,  # type: ignore[arg-type]
            budget=budget,
            cluster_cards=top_cards,
            evidence_items=[],
            bank_stats={
                "num_clusters": len(clusters),
                "num_memories": entries_count,
                "visible_cluster_cards": len(top_cards),
            },
            risk_flags=risk_flags,
        )
