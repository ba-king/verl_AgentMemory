from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..schemas import ActionTrace, ClusterMembership, ClusterNode, MemoryAction, MemoryEntry, utc_now
from ..store import SQLiteMemoryStore
from .assigner import create_cluster_from_entry
from .maintenance import maintain_clusters
from .metrics import refresh_cluster_stats


def run_cluster_maintenance(
    store: SQLiteMemoryStore,
    *,
    user_id: str,
    project_id: str | None = None,
    max_proposals: int = 8,
    policy: Any | None = None,
    min_policy_confidence: float = 0.45,
) -> dict[str, Any]:
    clusters = store.list_clusters(user_id=user_id, project_id=project_id, statuses={"active"})
    entries = store.list_memories(user_id=user_id, project_id=project_id)
    report = maintain_clusters(clusters, entries)
    observation = _build_maintenance_observation(
        store,
        user_id=user_id,
        project_id=project_id,
        report=report,
        clusters=clusters,
        entries=entries,
        max_proposals=max_proposals,
    )
    proposal_source = "heuristic"
    raw_proposals: list[dict[str, Any]] = list(report.proposals)
    if policy is not None and hasattr(policy, "propose_cluster_maintenance"):
        try:
            policy_proposals = policy.propose_cluster_maintenance(observation)
        except Exception as exc:
            policy_proposals = [
                {
                    "action": MemoryAction.NO_OP.value,
                    "reason": f"policy_proposal_failed:{type(exc).__name__}",
                    "params": {"error": str(exc)},
                }
            ]
        sanitized_policy_proposals = _sanitize_proposals(
            store,
            policy_proposals,
            source="policy",
            min_confidence=min_policy_confidence,
        )
        if sanitized_policy_proposals:
            raw_proposals = sanitized_policy_proposals
            proposal_source = "policy"

    proposals = _sanitize_proposals(
        store,
        raw_proposals,
        source=proposal_source,
        min_confidence=0.0,
    )
    results: list[dict[str, Any]] = []
    retired_clusters: set[str] = set()
    for proposal in proposals[:max_proposals]:
        action = proposal["action"]
        if action == MemoryAction.REFRESH_CLUSTER_CARD.value:
            cluster_id = str(proposal["cluster_ids"][0])
            if cluster_id not in retired_clusters:
                results.append(_refresh_cluster_card(store, cluster_id, params=proposal.get("params")))
        elif action == MemoryAction.SPLIT_CLUSTER.value:
            cluster_id = str(proposal["cluster_ids"][0])
            if cluster_id not in retired_clusters:
                result = _split_cluster(store, cluster_id, params=proposal.get("params"))
                results.append(result)
                if result.get("ok"):
                    retired_clusters.add(cluster_id)
        elif action == MemoryAction.MERGE_CLUSTER.value:
            cluster_ids = [str(item) for item in proposal["cluster_ids"]]
            if not retired_clusters.intersection(cluster_ids):
                result = _merge_clusters(store, cluster_ids, params=proposal.get("params"))
                results.append(result)
                if result.get("ok"):
                    retired_clusters.update(cluster_ids[1:])
        elif action == MemoryAction.NO_OP.value:
            results.append({"ok": True, "action": MemoryAction.NO_OP.value, "reason": proposal.get("reason", "")})
    if not results:
        results.append({"ok": True, "action": MemoryAction.NO_OP.value, "reason": "no maintenance proposals"})
    store.append_action_trace(
        ActionTrace(
            phase="maintenance",
            action="CLUSTER_MAINTENANCE",
            observation_json=observation,
            action_json={
                "max_proposals": max_proposals,
                "proposal_source": proposal_source,
                "proposals": proposals[:max_proposals],
            },
            tool_result_json={"results": results},
            valid_format=True,
        )
    )
    return {"report": report, "results": results, "proposal_source": proposal_source, "proposals": proposals[:max_proposals]}


def _refresh_cluster_card(store: SQLiteMemoryStore, cluster_id: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    cluster = store.get_cluster(cluster_id)
    if cluster is None:
        return {"ok": False, "action": MemoryAction.REFRESH_CLUSTER_CARD.value, "error": "cluster_not_found"}
    params = params or {}
    entries = store.get_memories(cluster.member_memory_ids)
    refreshed = refresh_cluster_stats(cluster, entries)
    medoid = _choose_medoid(entries)
    if medoid is not None:
        refreshed.medoid_memory_id = medoid.memory_id
        refreshed.title = _compact_title(medoid.summary or medoid.content)
        refreshed.summary = _summarize_cluster(entries, fallback=medoid.summary or medoid.content)
    _apply_cluster_card_params(refreshed, params, entries)
    refreshed.dirty = False
    refreshed.version += 1
    refreshed.updated_at = utc_now()
    store.upsert_cluster(refreshed)
    return {"ok": True, "action": MemoryAction.REFRESH_CLUSTER_CARD.value, "cluster_id": cluster_id}


def _split_cluster(store: SQLiteMemoryStore, cluster_id: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    cluster = store.get_cluster(cluster_id)
    if cluster is None:
        return {"ok": False, "action": MemoryAction.SPLIT_CLUSTER.value, "error": "cluster_not_found"}
    params = params or {}
    entries = store.get_memories(cluster.member_memory_ids)
    grouped_entries = _groups_from_split_params(params, entries) or [
        {"entries": group_entries, "params": {"title": key, "summary": ""}}
        for key, group_entries in _group_entries_for_split(entries).items()
    ]
    grouped_entries = [group for group in grouped_entries if group["entries"]]
    if len(grouped_entries) < 2:
        return _refresh_cluster_card(store, cluster_id)

    new_clusters: list[ClusterNode] = []
    for group in grouped_entries:
        group_entries = group["entries"]
        group_params = group["params"]
        seed = _choose_medoid(group_entries)
        if seed is None:
            continue
        new_cluster = create_cluster_from_entry(seed)
        new_cluster.member_memory_ids = [entry.memory_id for entry in group_entries]
        new_cluster = refresh_cluster_stats(new_cluster, group_entries)
        new_cluster.title = _compact_title(seed.summary or seed.content)
        new_cluster.summary = _summarize_cluster(group_entries, fallback=seed.summary or seed.content)
        _apply_cluster_card_params(new_cluster, group_params, group_entries)
        new_cluster.dirty = False
        store.upsert_cluster(new_cluster)
        for entry in group_entries:
            entry.primary_cluster_id = new_cluster.cluster_id
            entry.secondary_cluster_ids = [
                cluster_id for cluster_id in entry.secondary_cluster_ids if cluster_id != cluster.cluster_id
            ]
            store.upsert_memory(entry)
            store.upsert_membership(
                ClusterMembership(
                    cluster_id=new_cluster.cluster_id,
                    memory_id=entry.memory_id,
                    role="primary",
                    score=1.0,
                )
            )
        new_clusters.append(new_cluster)

    _delete_cluster_memberships(store, cluster.cluster_id)
    cluster.status = "split"
    cluster.redirect_to = [new_cluster.cluster_id for new_cluster in new_clusters]
    cluster.updated_at = utc_now()
    store.upsert_cluster(cluster)
    return {
        "ok": True,
        "action": MemoryAction.SPLIT_CLUSTER.value,
        "cluster_id": cluster_id,
        "new_cluster_ids": cluster.redirect_to,
    }


def _merge_clusters(store: SQLiteMemoryStore, cluster_ids: list[str], *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    clusters = [cluster for cluster_id in cluster_ids if (cluster := store.get_cluster(cluster_id)) is not None]
    if len(clusters) < 2:
        return {"ok": False, "action": MemoryAction.MERGE_CLUSTER.value, "error": "need_at_least_two_clusters"}
    params = params or {}
    target = clusters[0]
    member_ids = sorted({memory_id for cluster in clusters for memory_id in cluster.member_memory_ids})
    entries = store.get_memories(member_ids)
    target.member_memory_ids = member_ids
    target = refresh_cluster_stats(target, entries)
    medoid = _choose_medoid(entries)
    if medoid is not None:
        target.medoid_memory_id = medoid.memory_id
        target.title = _compact_title(medoid.summary or medoid.content)
        target.summary = _summarize_cluster(entries, fallback=medoid.summary or medoid.content)
    _apply_cluster_card_params(target, params, entries)
    target.dirty = False
    target.version += 1
    target.updated_at = utc_now()
    store.upsert_cluster(target)
    for entry in entries:
        entry.primary_cluster_id = target.cluster_id
        entry.secondary_cluster_ids = [
            cluster_id for cluster_id in entry.secondary_cluster_ids if cluster_id not in cluster_ids
        ]
        store.upsert_memory(entry)
        store.upsert_membership(
            ClusterMembership(cluster_id=target.cluster_id, memory_id=entry.memory_id, role="primary", score=1.0)
        )
    for source in clusters[1:]:
        _delete_cluster_memberships(store, source.cluster_id)
        source.status = "merged"
        source.redirect_to = [target.cluster_id]
        source.updated_at = utc_now()
        store.upsert_cluster(source)
    return {
        "ok": True,
        "action": MemoryAction.MERGE_CLUSTER.value,
        "cluster_id": target.cluster_id,
        "merged_cluster_ids": cluster_ids[1:],
    }


def _group_entries_for_split(entries: list[MemoryEntry]) -> dict[str, list[MemoryEntry]]:
    groups: dict[str, list[MemoryEntry]] = defaultdict(list)
    for entry in entries:
        key = entry.facets[0] if entry.facets else (entry.keywords[0] if entry.keywords else "fact")
        groups[key].append(entry)
    large_groups = {key: value for key, value in groups.items() if value}
    if len(large_groups) >= 2:
        return large_groups
    midpoint = max(1, len(entries) // 2)
    return {"left": entries[:midpoint], "right": entries[midpoint:]}


def _groups_from_split_params(params: dict[str, Any], entries: list[MemoryEntry]) -> list[dict[str, Any]] | None:
    groups = params.get("groups")
    if not isinstance(groups, list):
        return None
    entries_by_id = {entry.memory_id: entry for entry in entries}
    used_ids: set[str] = set()
    result: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        member_ids = [str(memory_id) for memory_id in group.get("member_memory_ids") or []]
        group_entries = []
        for memory_id in member_ids:
            if memory_id in entries_by_id and memory_id not in used_ids:
                used_ids.add(memory_id)
                group_entries.append(entries_by_id[memory_id])
        if group_entries:
            result.append({"entries": group_entries, "params": group})
    remaining = [entry for entry in entries if entry.memory_id not in used_ids]
    if remaining:
        result.append({"entries": remaining, "params": {"title": "Remaining memories", "summary": "Members not assigned by the maintenance proposal."}})
    return result if len(result) >= 2 else None


def _choose_medoid(entries: list[MemoryEntry]) -> MemoryEntry | None:
    if not entries:
        return None
    return sorted(entries, key=lambda entry: (entry.importance, entry.utility, entry.updated_at), reverse=True)[0]


def _compact_title(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text[:limit]


def _summarize_cluster(entries: list[MemoryEntry], *, fallback: str) -> str:
    if not entries:
        return fallback
    facets = sorted({facet for entry in entries for facet in entry.facets})
    keywords = sorted({keyword for entry in entries for keyword in entry.keywords})[:8]
    parts = []
    if facets:
        parts.append("facets=" + ",".join(facets))
    if keywords:
        parts.append("keywords=" + ",".join(keywords))
    return f"{fallback} ({'; '.join(parts)})" if parts else fallback


def _delete_cluster_memberships(store: SQLiteMemoryStore, cluster_id: str) -> None:
    store.conn.execute("DELETE FROM memberships WHERE cluster_id = ?", (cluster_id,))
    store.conn.commit()


def _apply_cluster_card_params(cluster: ClusterNode, params: dict[str, Any], entries: list[MemoryEntry]) -> None:
    title = params.get("title")
    if isinstance(title, str) and title.strip():
        cluster.title = _compact_title(title)
    summary = params.get("summary")
    if isinstance(summary, str) and summary.strip():
        cluster.summary = " ".join(summary.split())[:500]
    keywords = _string_list(params.get("keywords"), limit=12)
    if keywords:
        cluster.keywords = keywords
    entities = _string_list(params.get("entities"), limit=12)
    if entities:
        cluster.entities = entities
    medoid_memory_id = params.get("medoid_memory_id")
    if isinstance(medoid_memory_id, str) and medoid_memory_id in {entry.memory_id for entry in entries}:
        cluster.medoid_memory_id = medoid_memory_id


def _string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text[:80])
        if len(result) >= limit:
            break
    return result


def _build_maintenance_observation(
    store: SQLiteMemoryStore,
    *,
    user_id: str,
    project_id: str | None,
    report: Any,
    clusters: list[ClusterNode],
    entries: list[MemoryEntry],
    max_proposals: int,
) -> dict[str, Any]:
    candidate_cluster_ids = _candidate_cluster_ids(report)
    candidate_clusters = [
        cluster for cluster in clusters if not candidate_cluster_ids or cluster.cluster_id in candidate_cluster_ids
    ][:12]
    entries_by_id = {entry.memory_id: entry for entry in entries}
    sample_memories: dict[str, list[dict[str, Any]]] = {}
    for cluster in candidate_clusters:
        members = [entries_by_id[memory_id] for memory_id in cluster.member_memory_ids if memory_id in entries_by_id]
        ranked_members = sorted(members, key=lambda entry: (entry.importance, entry.utility, entry.updated_at), reverse=True)
        sample_memories[cluster.cluster_id] = [_compact_memory_for_maintenance(entry) for entry in ranked_members[:10]]
    return {
        "mode": "cluster_index_only",
        "user_id": user_id,
        "project_id": project_id,
        "policy_constraints": {
            "allowed_actions": [
                MemoryAction.REFRESH_CLUSTER_CARD.value,
                MemoryAction.SPLIT_CLUSTER.value,
                MemoryAction.MERGE_CLUSTER.value,
                MemoryAction.NO_OP.value,
            ],
            "disallowed_actions": [MemoryAction.DEPRECATE.value, MemoryAction.UPDATE_EXISTING.value, MemoryAction.MERGE.value],
            "mutation_rule": "Do not change MemoryEntry content/status in cluster maintenance; only update ClusterNode cards, memberships, and redirects.",
        },
        "report": report.model_dump(mode="json"),
        "clusters": [_compact_cluster_for_maintenance(cluster) for cluster in candidate_clusters],
        "sample_memories": sample_memories,
        "recent_action_traces": [trace.model_dump(mode="json") for trace in store.list_action_traces()[-12:]],
        "recent_exposure_events": [event.model_dump(mode="json") for event in store.list_exposure_events()[-8:]],
        "budget": {
            "max_proposals": max_proposals,
            "max_clusters": len(candidate_clusters),
            "max_sample_memories_per_cluster": 10,
        },
    }


def _candidate_cluster_ids(report: Any) -> set[str]:
    cluster_ids: set[str] = set()
    for cluster_id in report.refresh_candidates:
        cluster_ids.add(cluster_id)
    for cluster_id in report.split_candidates:
        cluster_ids.add(cluster_id)
    for left, right in report.merge_candidates:
        cluster_ids.add(left)
        cluster_ids.add(right)
    for proposal in report.proposals:
        cluster_ids.update(_proposal_cluster_ids(proposal))
    return cluster_ids


def _sanitize_proposals(
    store: SQLiteMemoryStore,
    proposals: list[dict[str, Any]],
    *,
    source: str,
    min_confidence: float,
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    allowed = {
        MemoryAction.REFRESH_CLUSTER_CARD.value,
        MemoryAction.SPLIT_CLUSTER.value,
        MemoryAction.MERGE_CLUSTER.value,
        MemoryAction.NO_OP.value,
    }
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        action = str(proposal.get("action"))
        if action not in allowed:
            continue
        confidence = float(proposal.get("confidence", 1.0 if source == "heuristic" else 0.0))
        if source == "policy" and action != MemoryAction.NO_OP.value and confidence < min_confidence:
            continue
        cluster_ids = _proposal_cluster_ids(proposal)
        if action == MemoryAction.NO_OP.value:
            sanitized.append(_normalized_proposal(proposal, action, [], confidence, source))
            continue
        if action in {MemoryAction.REFRESH_CLUSTER_CARD.value, MemoryAction.SPLIT_CLUSTER.value}:
            if len(cluster_ids) != 1 or store.get_cluster(cluster_ids[0]) is None:
                continue
        if action == MemoryAction.MERGE_CLUSTER.value:
            if len(cluster_ids) < 2 or any(store.get_cluster(cluster_id) is None for cluster_id in cluster_ids):
                continue
        sanitized.append(_normalized_proposal(proposal, action, cluster_ids, confidence, source))
    return sanitized


def _normalized_proposal(
    proposal: dict[str, Any],
    action: str,
    cluster_ids: list[str],
    confidence: float,
    source: str,
) -> dict[str, Any]:
    params = proposal.get("params")
    raw_memory_ids = proposal.get("memory_ids") if isinstance(proposal.get("memory_ids"), list) else []
    return {
        "action": action,
        "cluster_ids": cluster_ids,
        "memory_ids": [str(memory_id) for memory_id in raw_memory_ids if memory_id],
        "reason": str(proposal.get("reason", "")),
        "params": params if isinstance(params, dict) else {},
        "confidence": confidence,
        "source": source,
    }


def _proposal_cluster_ids(proposal: dict[str, Any]) -> list[str]:
    if isinstance(proposal.get("cluster_ids"), list):
        return [str(cluster_id) for cluster_id in proposal["cluster_ids"] if cluster_id]
    if proposal.get("cluster_id"):
        return [str(proposal["cluster_id"])]
    return []


def _compact_cluster_for_maintenance(cluster: ClusterNode) -> dict[str, Any]:
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
            "member_memory_ids",
            "medoid_memory_id",
            "size",
            "radius",
            "cohesion",
            "entropy",
            "dirty",
            "version",
            "query_hits",
            "avg_utility",
            "shard_id",
        )
    }


def _compact_memory_for_maintenance(entry: MemoryEntry) -> dict[str, Any]:
    return {
        "memory_id": entry.memory_id,
        "summary": entry.summary,
        "content": entry.content[:400],
        "facets": entry.facets,
        "keywords": entry.keywords,
        "entities": entry.entities,
        "confidence": entry.confidence,
        "importance": entry.importance,
        "utility": entry.utility,
        "status": entry.status,
        "updated_at": entry.updated_at.isoformat(),
        "source_refs": entry.source_refs,
    }
