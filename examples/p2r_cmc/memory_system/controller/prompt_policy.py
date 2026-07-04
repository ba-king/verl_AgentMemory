from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .action_parser import MAINTENANCE_ACTIONS, REMEMBER_ACTIONS, USE_ACTIONS, parse_controller_action
from ..model_clients import OpenAICompatibleChatClient
from ..clustering.metrics import tokenize
from ..schemas import (
    ClusterNode,
    ControllerAction,
    MemoryAction,
    MemoryCandidate,
    MemoryEntry,
    PolicyObservation,
)


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


class RuleBasedPromptPolicy:
    """Prompt-policy stand-in used before connecting a real LLM."""

    def extract_candidates(
        self,
        text: str,
        *,
        user_id: str,
        project_id: str | None = None,
        source_ref: str | None = None,
    ) -> list[MemoryCandidate]:
        stripped = text.strip()
        lowered = stripped.lower()
        remember_prefixes = ("remember:", "memory:", "记住：", "记住:", "记忆：", "记忆:")
        content = ""
        for prefix in remember_prefixes:
            if lowered.startswith(prefix.lower()) or stripped.startswith(prefix):
                content = stripped[len(prefix) :].strip()
                break
        if not content:
            durable_markers = ["prefer", "usually", "avoid", "project", "trying to", "我通常", "我喜欢", "项目"]
            if any(marker in lowered or marker in stripped for marker in durable_markers):
                content = stripped
        if not content:
            return []

        facets = self._infer_facets(content)
        keywords = sorted(tokenize(content))[:12]
        return [
            MemoryCandidate(
                content=content,
                summary=content[:160],
                user_id=user_id,
                project_id=project_id,
                scope="project" if project_id else "global",
                facets=facets,
                keywords=keywords,
                source_refs=[source_ref] if source_ref else [],
                confidence=0.76,
                importance=0.62 if "project" in facets or "task_state" in facets else 0.52,
            )
        ]

    def decide_use(self, observation: PolicyObservation) -> ControllerAction:
        max_clusters = int(observation.budget.get("max_clusters", 3))
        cards = [card for card in observation.cluster_cards if card.retrieval_score > 0]
        if not cards:
            return parse_controller_action(
                {
                    "action": MemoryAction.ANSWER_NO_MEMORY.value,
                    "cluster_ids": [],
                    "reason": "No cluster card has positive retrieval score.",
                },
                allowed_actions=USE_ACTIONS,
            )
        selected = cards[:max_clusters]
        return parse_controller_action(
            {
                "action": MemoryAction.EXPAND_CLUSTER.value,
                "cluster_ids": [card.cluster_id for card in selected],
                "reason": "Select the highest-scoring cluster cards within budget.",
                "params": {"max_items": observation.budget.get("max_items", 6)},
            },
            allowed_actions=USE_ACTIONS,
        )

    def decide_remember(
        self,
        *,
        candidate: MemoryEntry,
        related_memories: list[MemoryEntry],
        related_clusters: list[ClusterNode],
        assignment_scores: dict[str, float],
        thresholds: dict[str, float],
        observation: dict[str, Any] | None = None,
    ) -> ControllerAction | None:
        return None

    def decide(self, observation: PolicyObservation) -> ControllerAction:
        if observation.phase == "use":
            return self.decide_use(observation)
        return parse_controller_action({"action": MemoryAction.NO_OP.value, "reason": "Rule policy only handles use."})

    @staticmethod
    def _infer_facets(content: str) -> list[str]:
        lowered = content.lower()
        facets: list[str] = []
        if any(term in lowered for term in ["prefer", "like", "avoid", "usually", "喜欢", "避免", "通常"]):
            facets.append("preference")
        if any(term in lowered for term in ["project", "benchmark", "grpo", "memoryarena", "codex", "verl", "项目"]):
            facets.append("project_state")
        if any(term in lowered for term in ["deadline", "todo", "next", "tomorrow", "任务", "明天"]):
            facets.append("task_state")
        if any(term in lowered for term in ["error", "failed", "bug", "wrong", "失败", "错误"]):
            facets.append("error")
        return facets or ["fact"]


class MockLLMPolicy(RuleBasedPromptPolicy):
    """Deterministic mock with the same interface as an LLM-backed policy."""


class OpenAICompatiblePolicy(RuleBasedPromptPolicy):
    """Memory-agent policy backed by an OpenAI-compatible chat model.

    This is intended for DeepSeek/GPT/local-Qwen memory-controller calls. If no
    client is attached, methods fall back to the rule-based policy so tests do
    not require an API key.
    """

    def __init__(
        self,
        *,
        chat_client: OpenAICompatibleChatClient | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.chat_client = chat_client
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    @classmethod
    def from_env(cls) -> "OpenAICompatiblePolicy":
        return cls(chat_client=OpenAICompatibleChatClient.from_env(prefix="P2R_MEMORY"))

    def call_model(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if self.chat_client is None:
            raise RuntimeError("OpenAICompatiblePolicy requires chat_client or use from_env().")
        return self.chat_client.chat_json(messages, temperature=0.0)

    def extract_candidates(
        self,
        text: str,
        *,
        user_id: str,
        project_id: str | None = None,
        source_ref: str | None = None,
    ) -> list[MemoryCandidate]:
        if self.chat_client is None:
            return super().extract_candidates(text, user_id=user_id, project_id=project_id, source_ref=source_ref)

        messages = [
            {
                "role": "system",
                "content": load_prompt("candidate_extractor.md"),
            },
            {
                "role": "user",
                "content": (
                    f"user_id={user_id}\nproject_id={project_id}\nsource_ref={source_ref}\n\n"
                    f"Turn:\n{text}"
                ),
            },
        ]
        payload = self.call_model(messages)
        candidates = []
        for item in payload.get("candidates", []):
            candidates.append(
                MemoryCandidate(
                    content=item["content"],
                    summary=item.get("summary") or item["content"][:160],
                    user_id=user_id,
                    project_id=project_id,
                    scope=item.get("scope") or ("project" if project_id else "global"),
                    facets=item.get("facets") or ["fact"],
                    keywords=item.get("keywords") or sorted(tokenize(item["content"]))[:12],
                    entities=item.get("entities") or [],
                    source_refs=[source_ref] if source_ref else [],
                    confidence=float(item.get("confidence", 0.75)),
                    importance=float(item.get("importance", 0.5)),
                )
            )
        return candidates

    def decide_remember(
        self,
        *,
        candidate: MemoryEntry,
        related_memories: list[MemoryEntry],
        related_clusters: list[ClusterNode],
        assignment_scores: dict[str, float],
        thresholds: dict[str, float],
        observation: dict[str, Any] | None = None,
    ) -> ControllerAction | None:
        if self.chat_client is None:
            return None

        compact_candidate = _compact_memory(candidate)
        compact_memories = [_compact_memory(memory) for memory in related_memories[:12]]
        compact_clusters = [_compact_cluster(cluster) for cluster in related_clusters[:8]]
        messages = [
            {
                "role": "system",
                "content": load_prompt("remember_action.md"),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "candidate": compact_candidate,
                        "related_memories": compact_memories,
                        "related_clusters": compact_clusters,
                        "assignment_scores": assignment_scores,
                        "thresholds": thresholds,
                        "observation": observation or {},
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        payload = self.call_model(messages)
        return parse_controller_action(payload, allowed_actions=REMEMBER_ACTIONS)

    def decide_use(self, observation: PolicyObservation) -> ControllerAction:
        if self.chat_client is None:
            return super().decide_use(observation)

        messages = [
            {
                "role": "system",
                "content": load_prompt("memory_controller.md"),
            },
            {
                "role": "user",
                "content": json.dumps(observation.model_dump(mode="json"), ensure_ascii=False),
            },
        ]
        payload = self.call_model(messages)
        return parse_controller_action(payload, allowed_actions=USE_ACTIONS)

    def propose_cluster_maintenance(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        if self.chat_client is None:
            return []

        messages = [
            {
                "role": "system",
                "content": load_prompt("cluster_maintenance.md"),
            },
            {
                "role": "user",
                "content": json.dumps(observation, ensure_ascii=False),
            },
        ]
        payload = self.call_model(messages)
        proposals = payload.get("proposals", [])
        if not isinstance(proposals, list):
            return []
        valid_proposals: list[dict[str, Any]] = []
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            try:
                action = parse_controller_action(
                    {
                        "action": proposal.get("action"),
                        "cluster_ids": proposal.get("cluster_ids") or [],
                        "memory_ids": proposal.get("memory_ids") or [],
                        "reason": proposal.get("reason") or "",
                        "params": proposal.get("params") or {},
                    },
                    allowed_actions=MAINTENANCE_ACTIONS,
                )
            except ValueError:
                continue
            action_value = action.action.value if isinstance(action.action, MemoryAction) else str(action.action)
            valid_proposals.append(
                {
                    "action": action_value,
                    "cluster_ids": action.cluster_ids,
                    "memory_ids": action.memory_ids,
                    "reason": action.reason,
                    "params": action.params,
                    "confidence": float(proposal.get("confidence", 0.0)),
                }
            )
        return valid_proposals


def _compact_memory(entry: MemoryEntry) -> dict[str, Any]:
    data = entry.model_dump(mode="json")
    data.pop("embedding", None)
    data.pop("history", None)
    return {
        key: data.get(key)
        for key in (
            "memory_id",
            "content",
            "summary",
            "scope",
            "facets",
            "keywords",
            "entities",
            "confidence",
            "importance",
            "utility",
            "status",
            "created_at",
            "updated_at",
            "use_count",
            "primary_cluster_id",
            "secondary_cluster_ids",
            "metadata",
        )
    }


def _compact_cluster(cluster: ClusterNode) -> dict[str, Any]:
    data = cluster.model_dump(mode="json")
    data.pop("centroid_embedding", None)
    return {
        key: data.get(key)
        for key in (
            "cluster_id",
            "title",
            "summary",
            "keywords",
            "entities",
            "facet_distribution",
            "size",
            "radius",
            "cohesion",
            "query_hits",
            "avg_utility",
            "status",
            "member_memory_ids",
            "shard_id",
        )
    }
