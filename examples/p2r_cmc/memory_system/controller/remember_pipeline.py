from __future__ import annotations

from typing import Any

from .prompt_policy import RuleBasedPromptPolicy
from ..clustering.assigner import assign_and_persist_entry, assign_entry_to_cluster
from ..clustering.metrics import ASSIGN_THRESHOLD, entry_similarity, refresh_cluster_stats
from ..executor import MemoryExecutor
from ..model_clients import EmbeddingClient
from ..schemas import MemoryAction, MemoryEntry, RawEpisode
from ..store import SQLiteMemoryStore

DUPLICATE_THRESHOLD = 0.9
UPDATE_THRESHOLD = 0.72
PROMPT_UPDATE_FIELDS = {
    "content",
    "summary",
    "facets",
    "keywords",
    "entities",
    "confidence",
    "importance",
    "status",
    "metadata",
    "embedding",
}


class RememberPipeline:
    """Prompt-based remember-stage flow with deterministic mutations."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        *,
        executor: MemoryExecutor | None = None,
        policy: RuleBasedPromptPolicy | None = None,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        self.store = store
        self.executor = executor or MemoryExecutor(store)
        self.policy = policy or RuleBasedPromptPolicy()
        self.embedder = embedder

    def ingest_episode(self, episode: RawEpisode) -> list[dict[str, Any]]:
        candidates = self.policy.extract_candidates(
            episode.content,
            user_id=episode.user_id,
            project_id=episode.project_id,
            source_ref=episode.episode_id,
        )
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            entry = candidate.to_entry()
            results.append(self.remember_entry(entry, observation={"episode_id": episode.episode_id}))
        return results

    def remember_text(
        self,
        text: str,
        *,
        user_id: str,
        project_id: str | None = None,
        source_ref: str | None = None,
    ) -> list[dict[str, Any]]:
        candidates = self.policy.extract_candidates(
            text,
            user_id=user_id,
            project_id=project_id,
            source_ref=source_ref,
        )
        return [
            self.remember_entry(candidate.to_entry(), observation={"source_ref": source_ref})
            for candidate in candidates
        ]

    def remember_entry(self, entry: MemoryEntry, *, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        self._ensure_entry_embedding(entry)
        clusters = self.store.list_clusters(user_id=entry.user_id, project_id=entry.project_id, statuses={"active"})
        assignment_preview = assign_entry_to_cluster(entry.model_copy(deep=True), clusters)
        candidate_cluster_ids = [
            cluster_id
            for cluster_id, score in assignment_preview.scores.items()
            if score >= ASSIGN_THRESHOLD * 0.6
        ]
        related_clusters = [
            cluster
            for cluster_id in candidate_cluster_ids
            if (cluster := self.store.get_cluster(cluster_id))
        ]
        candidate_entries: list[MemoryEntry] = []
        for cluster in related_clusters:
            candidate_entries.extend(self.store.get_memories(cluster.member_memory_ids))

        best_existing: MemoryEntry | None = None
        best_score = 0.0
        for existing in candidate_entries:
            if existing.status != "active":
                continue
            score = entry_similarity(entry, existing)
            if score > best_score:
                best_existing = existing
                best_score = score

        prompt_action = self.policy.decide_remember(
            candidate=entry,
            related_memories=candidate_entries,
            related_clusters=related_clusters,
            assignment_scores=assignment_preview.scores,
            thresholds={
                "duplicate": DUPLICATE_THRESHOLD,
                "update": UPDATE_THRESHOLD,
                "cluster_assignment": ASSIGN_THRESHOLD,
            },
            observation={
                **(observation or {}),
                "best_existing_memory_id": best_existing.memory_id if best_existing else None,
                "best_existing_score": best_score,
            },
        )
        if prompt_action is not None:
            return self._execute_prompt_action(
                prompt_action,
                entry,
                best_existing=best_existing,
                best_score=best_score,
                observation=observation,
            )

        if best_existing and best_score >= DUPLICATE_THRESHOLD:
            return self.executor.execute(
                MemoryAction.NO_OP,
                {"reason": "duplicate_memory", "matched_memory_id": best_existing.memory_id, "score": best_score},
                phase="remember",
                observation=observation,
            )

        if best_existing and best_score >= UPDATE_THRESHOLD:
            updates = {
                "content": entry.content,
                "summary": entry.summary,
                "facets": sorted(set(best_existing.facets) | set(entry.facets)),
                "keywords": sorted(set(best_existing.keywords) | set(entry.keywords)),
                "confidence": max(best_existing.confidence, entry.confidence),
                "importance": max(best_existing.importance, entry.importance),
            }
            if entry.embedding:
                updates["embedding"] = entry.embedding
            result = self.executor.execute(
                MemoryAction.UPDATE_EXISTING,
                {
                    "memory_id": best_existing.memory_id,
                    "updates": updates,
                },
                phase="remember",
                observation=observation,
            )
            self._mark_related_clusters_dirty(best_existing.memory_id)
            return result

        return self._write_new(entry, observation=observation)

    def _mark_related_clusters_dirty(self, memory_id: str) -> None:
        memberships = self.store.list_memberships(memory_id=memory_id)
        for membership in memberships:
            cluster = self.store.get_cluster(membership.cluster_id)
            if cluster is None:
                continue
            cluster.dirty = True
            entries = self.store.get_memories(cluster.member_memory_ids)
            self.store.upsert_cluster(refresh_cluster_stats(cluster, entries))

    def _ensure_entry_embedding(self, entry: MemoryEntry) -> None:
        if entry.embedding is not None or self.embedder is None:
            return
        entry.embedding = self.embedder.embed_text(entry.summary or entry.content)

    def _execute_prompt_action(
        self,
        action: Any,
        entry: MemoryEntry,
        *,
        best_existing: MemoryEntry | None,
        best_score: float,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        action_value = MemoryAction(action.action)
        params = action.params or {}
        prompt_observation = {
            **(observation or {}),
            "policy_reason": action.reason,
            "policy_action": action.model_dump(mode="json"),
            "best_existing_memory_id": best_existing.memory_id if best_existing else None,
            "best_existing_score": best_score,
        }

        if action_value == MemoryAction.NO_OP:
            return self.executor.execute(
                MemoryAction.NO_OP,
                {"reason": action.reason or params.get("reason", "prompt_policy_no_op")},
                phase="remember",
                observation=prompt_observation,
            )

        if action_value == MemoryAction.WRITE_NEW:
            return self._write_new(entry, observation=prompt_observation)

        if action_value == MemoryAction.UPDATE_EXISTING:
            memory_id = action.memory_ids[0] if action.memory_ids else (
                best_existing.memory_id if best_existing else None
            )
            if not memory_id:
                return self.executor.execute(
                    MemoryAction.NO_OP,
                    {"reason": "prompt_update_missing_memory_id"},
                    phase="remember",
                    observation=prompt_observation,
                )
            updates = self._sanitize_updates(params.get("updates") or self._default_updates_from_entry(entry))
            result = self.executor.execute(
                MemoryAction.UPDATE_EXISTING,
                {"memory_id": memory_id, "updates": updates},
                phase="remember",
                observation=prompt_observation,
            )
            if result.get("ok"):
                self._mark_related_clusters_dirty(memory_id)
            return result

        if action_value == MemoryAction.DEPRECATE:
            memory_id = action.memory_ids[0] if action.memory_ids else (
                best_existing.memory_id if best_existing else None
            )
            if not memory_id:
                return self.executor.execute(
                    MemoryAction.NO_OP,
                    {"reason": "prompt_deprecate_missing_memory_id"},
                    phase="remember",
                    observation=prompt_observation,
                )
            return self.executor.execute(
                MemoryAction.DEPRECATE,
                {"memory_id": memory_id, "reason": params.get("reason") or action.reason},
                phase="remember",
                observation=prompt_observation,
            )

        if action_value == MemoryAction.MERGE:
            source_ids = params.get("source_ids") or action.memory_ids
            if not isinstance(source_ids, list) or len(source_ids) < 2:
                return self.executor.execute(
                    MemoryAction.NO_OP,
                    {"reason": "prompt_merge_requires_two_existing_memories"},
                    phase="remember",
                    observation=prompt_observation,
                )
            merge_payload: dict[str, Any] = {"source_memory_ids": source_ids}
            if isinstance(params.get("entry"), dict):
                merge_payload["entry"] = params["entry"]
            elif isinstance(params.get("updates"), dict):
                existing_sources = self.store.get_memories(source_ids)
                if existing_sources:
                    merge_payload["entry"] = {
                        **self._default_updates_from_entry(entry),
                        "user_id": existing_sources[0].user_id,
                        "project_id": existing_sources[0].project_id,
                        "scope": existing_sources[0].scope,
                        **self._sanitize_updates(params["updates"]),
                    }
            return self.executor.execute(
                MemoryAction.MERGE,
                merge_payload,
                phase="remember",
                observation=prompt_observation,
            )

        return self.executor.execute(
            MemoryAction.NO_OP,
            {"reason": f"unsupported_prompt_action:{action_value.value}"},
            phase="remember",
            observation=prompt_observation,
        )

    def _write_new(self, entry: MemoryEntry, *, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.executor.execute(
            MemoryAction.WRITE_NEW,
            {"entry": entry.model_dump(mode="json")},
            phase="remember",
            observation=observation,
        )
        stored_entry = self.store.get_memory(result["memory_id"]) if result.get("ok") else None
        if stored_entry:
            assign_and_persist_entry(self.store, stored_entry)
        return result

    @staticmethod
    def _default_updates_from_entry(entry: MemoryEntry) -> dict[str, Any]:
        updates = {
            "content": entry.content,
            "summary": entry.summary,
            "facets": entry.facets,
            "keywords": entry.keywords,
            "entities": entry.entities,
            "confidence": entry.confidence,
            "importance": entry.importance,
        }
        if entry.embedding:
            updates["embedding"] = entry.embedding
        return updates

    @staticmethod
    def _sanitize_updates(updates: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(updates, dict):
            return {}
        return {key: value for key, value in updates.items() if key in PROMPT_UPDATE_FIELDS}
