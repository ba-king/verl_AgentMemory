from __future__ import annotations

import asyncio
import time
from typing import Any

from .prompt_policy import RuleBasedPromptPolicy
from .state_compiler import StateCompiler
from .verifier import MemoryVerifier
from ..model_clients import EmbeddingClient
from ..retrieval.merger import build_evidence_packet
from ..retrieval.parallel_retriever import expand_clusters_parallel
from ..schemas import ActionTrace, EvidencePacket, ExposureEvent, MemoryAction, utc_now
from ..store import SQLiteMemoryStore


class UsePipeline:
    """Prompt-based use-stage flow: cards -> action -> expansion -> exposure."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        *,
        policy: RuleBasedPromptPolicy | None = None,
        state_compiler: StateCompiler | None = None,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or RuleBasedPromptPolicy()
        self.embedder = embedder
        self.state_compiler = state_compiler or StateCompiler(store, embedder=embedder)
        self.verifier = MemoryVerifier(store)

    async def run(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str | None = None,
        budget: dict[str, Any] | None = None,
        max_concurrency: int = 8,
    ) -> EvidencePacket:
        started = time.perf_counter()
        observation = self.state_compiler.compile(
            query=query,
            budget=budget,
            phase="use",
            user_id=user_id,
            project_id=project_id,
        )
        action = self.policy.decide_use(observation)
        self.store.append_action_trace(
            ActionTrace(
                phase="use",
                action=action.action,
                observation_json=observation.model_dump(mode="json"),
                action_json=action.model_dump(mode="json"),
                tool_result_json={"selected_clusters": action.cluster_ids},
                valid_format=True,
            )
        )

        if action.action == MemoryAction.ANSWER_NO_MEMORY:
            packet = EvidencePacket(query=query, token_budget=int(observation.budget.get("token_budget", 1200)))
            self._log_exposure(query, packet, [], time.perf_counter() - started)
            return packet

        selected_clusters = [self.store.get_cluster(cluster_id) for cluster_id in action.cluster_ids]
        selected_clusters = [cluster for cluster in selected_clusters if cluster is not None]
        cluster_plans = [
            {"cluster_id": cluster.cluster_id, "memory_ids": cluster.member_memory_ids}
            for cluster in selected_clusters
        ]
        memory_ids = sorted({memory_id for plan in cluster_plans for memory_id in plan["memory_ids"]})
        entries = self.store.get_memories(memory_ids)
        query_embedding = self.embedder.embed_text(query) if self.embedder is not None else None
        evidence = await expand_clusters_parallel(
            query,
            cluster_plans,
            entries,
            max_concurrency=max_concurrency,
            top_k_per_cluster=int(observation.budget.get("top_k_per_cluster", 6)),
            query_embedding=query_embedding,
        )
        packet = build_evidence_packet(
            query,
            evidence,
            max_items=int(observation.budget.get("max_items", 6)),
            token_budget=int(observation.budget.get("token_budget", 1200)),
        )
        if observation.budget.get("verify", True):
            packet, _ = self.verifier.verify_packet(packet)
        for item in packet.evidence_items:
            entry = self.store.get_memory(item.memory_id)
            if entry is None:
                continue
            entry.use_count += 1
            entry.last_used_at = utc_now()
            entry.utility = min(1.0, entry.utility + 0.05)
            self.store.upsert_memory(entry)
        self._log_exposure(query, packet, [cluster.cluster_id for cluster in selected_clusters], time.perf_counter() - started)
        return packet

    def run_sync(self, *args: Any, **kwargs: Any) -> EvidencePacket:
        return asyncio.run(self.run(*args, **kwargs))

    def _log_exposure(
        self,
        query: str,
        packet: EvidencePacket,
        cluster_ids: list[str],
        elapsed_s: float,
    ) -> None:
        self.store.append_exposure_event(
            ExposureEvent(
                query=query,
                selected_memory_ids=[item.memory_id for item in packet.evidence_items],
                cluster_ids=cluster_ids,
                token_count=packet.token_estimate,
                latency_ms=elapsed_s * 1000,
                metadata={
                    "evidence_count": len(packet.evidence_items),
                    "required_aspect_count": len(packet.required_aspects),
                    "missing_aspects": packet.missing_aspects,
                    "conflicts": packet.conflicts,
                    "omitted_memory_ids": packet.omitted_memory_ids,
                },
            )
        )
