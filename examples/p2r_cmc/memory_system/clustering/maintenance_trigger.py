from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..schemas import ActionTrace, MaintenanceTriggerDecision, MemoryAction, utc_now
from ..store import SQLiteMemoryStore
from .maintenance import maintain_clusters
from .metrics import MAX_CLUSTER_SIZE


@dataclass(frozen=True)
class MaintenanceTriggerPolicy:
    """Cheap trigger policy for deciding when to run heavier cluster maintenance."""

    min_new_or_updated_memories: int = 3
    dirty_cluster_threshold: int = 1
    split_candidate_threshold: int = 1
    merge_candidate_threshold: int = 1
    verifier_issue_threshold: int = 1
    missing_aspect_event_threshold: int = 1
    recent_event_limit: int = 16
    max_cluster_size: int = MAX_CLUSTER_SIZE
    merge_threshold: float = 0.88
    min_seconds_between_runs: float = 0.0

    def decide(
        self,
        store: SQLiteMemoryStore,
        *,
        user_id: str,
        project_id: str | None = None,
        force: bool = False,
    ) -> MaintenanceTriggerDecision:
        clusters = store.list_clusters(user_id=user_id, project_id=project_id, statuses={"active"})
        entries = store.list_memories(user_id=user_id, project_id=project_id)
        report = maintain_clusters(
            clusters,
            entries,
            max_cluster_size=self.max_cluster_size,
            merge_threshold=self.merge_threshold,
        )
        last_maintenance_at = self._latest_cluster_maintenance_at(store, user_id=user_id, project_id=project_id)
        new_or_updated = self._count_new_or_updated_entries(entries, since=last_maintenance_at)
        verifier_issues = self._count_recent_verifier_issues(store, since=last_maintenance_at)
        missing_aspect_events = self._count_recent_missing_aspect_events(store, since=last_maintenance_at)

        reasons: list[str] = []
        skipped_reasons: list[str] = []
        if force:
            reasons.append("force")
        if not clusters and not entries:
            skipped_reasons.append("empty_memory_bank")
        if new_or_updated >= self.min_new_or_updated_memories:
            reasons.append(f"new_or_updated_memories>={self.min_new_or_updated_memories}")
        if len(report.refresh_candidates) >= self.dirty_cluster_threshold:
            reasons.append(f"dirty_clusters>={self.dirty_cluster_threshold}")
        if len(report.split_candidates) >= self.split_candidate_threshold:
            reasons.append(f"split_candidates>={self.split_candidate_threshold}")
        if len(report.merge_candidates) >= self.merge_candidate_threshold:
            reasons.append(f"merge_candidates>={self.merge_candidate_threshold}")
        if verifier_issues >= self.verifier_issue_threshold:
            reasons.append(f"recent_verifier_issues>={self.verifier_issue_threshold}")
        if missing_aspect_events >= self.missing_aspect_event_threshold:
            reasons.append(f"recent_missing_aspect_events>={self.missing_aspect_event_threshold}")

        should_run = bool(reasons) and bool(clusters or entries)
        if should_run and self.min_seconds_between_runs > 0 and last_maintenance_at is not None and not force:
            elapsed_s = (utc_now() - last_maintenance_at).total_seconds()
            high_priority = bool(report.split_candidates or report.merge_candidates or verifier_issues)
            if elapsed_s < self.min_seconds_between_runs and not high_priority:
                should_run = False
                skipped_reasons.append(f"cooldown:{elapsed_s:.1f}s<{self.min_seconds_between_runs:.1f}s")

        return MaintenanceTriggerDecision(
            should_run=should_run,
            reasons=reasons,
            skipped_reasons=skipped_reasons,
            last_maintenance_at=last_maintenance_at,
            stats={
                "num_clusters": report.num_clusters,
                "num_memories": len(entries),
                "new_or_updated_memories": new_or_updated,
                "dirty_clusters": len(report.refresh_candidates),
                "split_candidates": len(report.split_candidates),
                "merge_candidates": len(report.merge_candidates),
                "recent_verifier_issues": verifier_issues,
                "recent_missing_aspect_events": missing_aspect_events,
                "health_report": report.model_dump(mode="json"),
            },
        )

    def log_decision(
        self,
        store: SQLiteMemoryStore,
        decision: MaintenanceTriggerDecision,
        *,
        user_id: str,
        project_id: str | None = None,
    ) -> None:
        store.append_action_trace(
            ActionTrace(
                phase="maintenance",
                action="MAINTENANCE_TRIGGER_CHECK",
                observation_json={"user_id": user_id, "project_id": project_id, "stats": decision.stats},
                action_json=decision.model_dump(mode="json"),
                tool_result_json={"should_run": decision.should_run, "reasons": decision.reasons},
                valid_format=True,
            )
        )

    def _latest_cluster_maintenance_at(
        self,
        store: SQLiteMemoryStore,
        *,
        user_id: str,
        project_id: str | None,
    ) -> datetime | None:
        for trace in reversed(store.list_action_traces()):
            if trace.phase != "maintenance" or str(trace.action) != "CLUSTER_MAINTENANCE":
                continue
            if self._trace_matches_scope(trace, user_id=user_id, project_id=project_id):
                return trace.created_at
        return None

    def _trace_matches_scope(self, trace: ActionTrace, *, user_id: str, project_id: str | None) -> bool:
        observed_user = trace.observation_json.get("user_id")
        observed_project = trace.observation_json.get("project_id")
        if observed_user is None and observed_project is None:
            return True
        return observed_user == user_id and observed_project == project_id

    def _count_new_or_updated_entries(self, entries: list[Any], *, since: datetime | None) -> int:
        if since is None:
            return len(entries)
        return sum(1 for entry in entries if entry.updated_at > since)

    def _count_recent_verifier_issues(self, store: SQLiteMemoryStore, *, since: datetime | None) -> int:
        issue_count = 0
        recent_traces = store.list_action_traces()[-self.recent_event_limit :]
        for trace in recent_traces:
            if trace.phase != "verify":
                continue
            if since is not None and trace.created_at <= since:
                continue
            for result in trace.tool_result_json.get("results", []):
                if not isinstance(result, dict):
                    continue
                verdict = str(result.get("verdict", ""))
                if result.get("keep") is False or verdict not in {"", "valid"}:
                    issue_count += 1
        return issue_count

    def _count_recent_missing_aspect_events(self, store: SQLiteMemoryStore, *, since: datetime | None) -> int:
        event_count = 0
        recent_events = store.list_exposure_events()[-self.recent_event_limit :]
        for event in recent_events:
            if since is not None and event.created_at <= since:
                continue
            missing_aspects = event.metadata.get("missing_aspects", [])
            if isinstance(missing_aspects, list) and missing_aspects:
                event_count += 1
        return event_count


def run_cluster_maintenance_if_needed(
    store: SQLiteMemoryStore,
    *,
    user_id: str,
    project_id: str | None = None,
    policy: Any | None = None,
    trigger_policy: MaintenanceTriggerPolicy | None = None,
    force: bool = False,
    max_proposals: int = 8,
    min_policy_confidence: float = 0.45,
    log_decision: bool = True,
) -> dict[str, Any]:
    trigger = trigger_policy or MaintenanceTriggerPolicy()
    decision = trigger.decide(store, user_id=user_id, project_id=project_id, force=force)
    if log_decision:
        trigger.log_decision(store, decision, user_id=user_id, project_id=project_id)
    if not decision.should_run:
        return {
            "triggered": False,
            "decision": decision.model_dump(mode="json"),
            "maintenance": None,
        }

    from .maintenance_loop import run_cluster_maintenance

    maintenance = run_cluster_maintenance(
        store,
        user_id=user_id,
        project_id=project_id,
        max_proposals=max_proposals,
        policy=policy,
        min_policy_confidence=min_policy_confidence,
    )
    return {
        "triggered": True,
        "decision": decision.model_dump(mode="json"),
        "maintenance": maintenance,
    }
