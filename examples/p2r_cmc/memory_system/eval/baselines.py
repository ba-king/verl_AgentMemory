from __future__ import annotations

import time

from ..clustering.metrics import query_entry_score
from ..controller.prompt_policy import RuleBasedPromptPolicy
from ..controller.use_pipeline import UsePipeline
from ..model_clients import EmbeddingClient
from ..retrieval.cluster_search import to_evidence
from ..retrieval.bm25 import normalized_bm25_scores
from ..retrieval.merger import build_evidence_packet
from ..schemas import EvidencePacket
from ..store import SQLiteMemoryStore


def no_memory_baseline(query: str) -> EvidencePacket:
    return EvidencePacket(query=query)


def full_context_baseline(store: SQLiteMemoryStore, query: str, *, user_id: str, project_id: str | None) -> EvidencePacket:
    entries = store.list_memories(user_id=user_id, project_id=project_id, statuses={"active"})
    items = [to_evidence(entry, 1.0, entry.primary_cluster_id or "", "full_context") for entry in entries]
    return build_evidence_packet(query, items, max_items=100, token_budget=100000)


def dense_topk_baseline(
    store: SQLiteMemoryStore,
    query: str,
    *,
    user_id: str,
    project_id: str | None,
    top_k: int = 6,
    embedder: EmbeddingClient | None = None,
) -> EvidencePacket:
    entries = store.list_memories(user_id=user_id, project_id=project_id, statuses={"active"})
    query_embedding = embedder.embed_text(query) if embedder is not None else None
    items = [
        to_evidence(
            entry,
            query_entry_score(query, entry, query_embedding=query_embedding),
            entry.primary_cluster_id or "",
            "global_dense_embedding_topk" if query_embedding is not None and entry.embedding is not None else "global_dense_topk",
        )
        for entry in entries
    ]
    return build_evidence_packet(query, items, max_items=top_k)


def bm25_topk_baseline(
    store: SQLiteMemoryStore,
    query: str,
    *,
    user_id: str,
    project_id: str | None,
    top_k: int = 6,
) -> EvidencePacket:
    entries = store.list_memories(user_id=user_id, project_id=project_id, statuses={"active"})
    bm25_scores = normalized_bm25_scores(query, entries)
    items = [
        to_evidence(entry, bm25_scores.get(entry.memory_id, 0.0), entry.primary_cluster_id or "", "bm25_topk")
        for entry in entries
    ]
    return build_evidence_packet(query, items, max_items=top_k)


def p2r_cmc_prompt_baseline(
    store: SQLiteMemoryStore,
    query: str,
    *,
    user_id: str,
    project_id: str | None,
    embedder: EmbeddingClient | None = None,
    policy: RuleBasedPromptPolicy | None = None,
) -> tuple[EvidencePacket, float]:
    started = time.perf_counter()
    packet = UsePipeline(store, policy=policy, embedder=embedder).run_sync(
        query, user_id=user_id, project_id=project_id
    )
    return packet, (time.perf_counter() - started) * 1000
