from __future__ import annotations

import json
import time
from typing import Iterable

from .baselines import (
    bm25_topk_baseline,
    dense_topk_baseline,
    full_context_baseline,
    no_memory_baseline,
    p2r_cmc_prompt_baseline,
)
from .memory_arena_adapter import MemoryArenaAdapter, MemoryArenaResult, MemoryArenaTask
from .metrics import compute_memory_metrics
from ..controller.prompt_policy import OpenAICompatiblePolicy, RuleBasedPromptPolicy
from ..controller.remember_pipeline import RememberPipeline
from ..model_clients import EmbeddingClient, LocalQwenEmbeddingClient
from ..schemas import RawEpisode
from ..store import SQLiteMemoryStore


def build_toy_task() -> MemoryArenaTask:
    user_id = "toy_user"
    project_id = "p2r-cmc"
    sessions = [
        [
            RawEpisode(
                user_id=user_id,
                project_id=project_id,
                session_id="s1",
                content="Remember: I prefer concise Chinese, structured bullet points, and material that I can show to my advisor.",
            ),
            RawEpisode(
                user_id=user_id,
                project_id=project_id,
                session_id="s1",
                content="Remember: My project is P2R-CMC, using a unified memory pool, soft facets, adaptive cluster index, and MemoryArena evaluation.",
            ),
        ],
        [
            RawEpisode(
                user_id=user_id,
                project_id=project_id,
                session_id="s2",
                content="Remember: The first stage should be prompt-based only; GRPO and RL are reserved for the second stage.",
            ),
            RawEpisode(
                user_id=user_id,
                project_id=project_id,
                session_id="s2",
                content="Remember: Previous artificial memory levels are not discarded, but downgraded into hard boundaries and soft facet priors.",
            ),
            RawEpisode(
                user_id=user_id,
                project_id=project_id,
                session_id="s2",
                content="This is a distractor: I bought a green notebook yesterday.",
            ),
        ],
    ]
    return MemoryArenaTask(
        task_id="toy_p2r_cmc_report",
        sessions=sessions,
        final_query=(
            "请根据我之前的项目记忆，生成一个给老师汇报 P2R-CMC 第一阶段 prompt-based "
            "实现和 MemoryArena 测试计划的方案。"
        ),
        user_id=user_id,
        project_id=project_id,
        gold_keywords={"P2R-CMC", "prompt-based", "MemoryArena", "advisor", "Chinese"},
    )


class ToyMemoryArenaAdapter(MemoryArenaAdapter):
    def __init__(
        self,
        *,
        embedder: EmbeddingClient | None = None,
        policy: RuleBasedPromptPolicy | None = None,
    ) -> None:
        self.embedder = embedder
        self.policy = policy

    def load_tasks(self) -> list[MemoryArenaTask]:
        return [build_toy_task()]

    def build_store(self, task: MemoryArenaTask) -> tuple[SQLiteMemoryStore, set[str], set[str]]:
        store = SQLiteMemoryStore()
        remember = RememberPipeline(store, policy=self.policy, embedder=self.embedder)
        for session in task.sessions:
            for episode in session:
                remember.ingest_episode(episode)
        gold_memory_ids: set[str] = set()
        gold_cluster_ids: set[str] = set()
        for entry in store.list_memories(user_id=task.user_id, project_id=task.project_id, statuses={"active"}):
            haystack = f"{entry.content} {entry.summary}"
            if any(keyword.lower() in haystack.lower() for keyword in task.gold_keywords):
                gold_memory_ids.add(entry.memory_id)
                if entry.primary_cluster_id:
                    gold_cluster_ids.add(entry.primary_cluster_id)
        return store, gold_memory_ids, gold_cluster_ids

    def run(self, baseline: str = "p2r_cmc_prompt") -> list[MemoryArenaResult]:
        results: list[MemoryArenaResult] = []
        for task in self.load_tasks():
            store, gold_memory_ids, gold_cluster_ids = self.build_store(task)
            started = time.perf_counter()
            if baseline == "no_memory":
                packet = no_memory_baseline(task.final_query)
                latency_ms = (time.perf_counter() - started) * 1000
            elif baseline == "full_context":
                packet = full_context_baseline(store, task.final_query, user_id=task.user_id, project_id=task.project_id)
                latency_ms = (time.perf_counter() - started) * 1000
            elif baseline == "dense_topk":
                packet = dense_topk_baseline(
                    store,
                    task.final_query,
                    user_id=task.user_id,
                    project_id=task.project_id,
                    embedder=self.embedder,
                    policy=self.policy,
                )
                latency_ms = (time.perf_counter() - started) * 1000
            elif baseline == "bm25_topk":
                packet = bm25_topk_baseline(store, task.final_query, user_id=task.user_id, project_id=task.project_id)
                latency_ms = (time.perf_counter() - started) * 1000
            elif baseline in {"cluster_prompt_only", "p2r_cmc_prompt"}:
                packet, latency_ms = p2r_cmc_prompt_baseline(
                    store,
                    task.final_query,
                    user_id=task.user_id,
                    project_id=task.project_id,
                    embedder=self.embedder,
                )
            else:
                raise ValueError(f"Unknown baseline: {baseline}")
            metrics = compute_memory_metrics(
                packet,
                gold_memory_ids=gold_memory_ids,
                gold_cluster_ids=gold_cluster_ids,
                latency_ms=latency_ms,
                tool_calls=len(store.list_action_traces()),
            )
            results.append(MemoryArenaResult(task.task_id, baseline, packet, metrics))
            store.close()
        return results


def run_baselines(
    baselines: Iterable[str],
    *,
    embedding: str = "local-qwen",
    memory_agent: str = "rule",
) -> list[dict[str, object]]:
    embedder = LocalQwenEmbeddingClient.from_env() if embedding == "local-qwen" else None
    policy = OpenAICompatiblePolicy.from_env() if memory_agent == "openai" else None
    adapter = ToyMemoryArenaAdapter(embedder=embedder, policy=policy)
    output: list[dict[str, object]] = []
    for baseline in baselines:
        for result in adapter.run(baseline):
            output.append(
                {
                    "task_id": result.task_id,
                    "baseline": result.baseline,
                    "metrics": result.metrics,
                    "evidence": [item.model_dump(mode="json") for item in result.packet.evidence_items],
                }
            )
    return output


def main() -> None:
    print(json.dumps(run_baselines(["p2r_cmc_prompt"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
