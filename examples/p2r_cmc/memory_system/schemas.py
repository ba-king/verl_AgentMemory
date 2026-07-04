from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class MemoryAction(str, Enum):
    WRITE_NEW = "WRITE_NEW"
    UPDATE_EXISTING = "UPDATE_EXISTING"
    DEPRECATE = "DEPRECATE"
    MERGE = "MERGE"
    NO_OP = "NO_OP"
    EXPAND_CLUSTER = "EXPAND_CLUSTER"
    SEARCH_GLOBAL = "SEARCH_GLOBAL"
    CHECK_CONFLICT = "CHECK_CONFLICT"
    ANSWER_NO_MEMORY = "ANSWER_NO_MEMORY"
    REFRESH_CLUSTER_CARD = "REFRESH_CLUSTER_CARD"
    SPLIT_CLUSTER = "SPLIT_CLUSTER"
    MERGE_CLUSTER = "MERGE_CLUSTER"


class P2RBaseModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)


class RawEpisode(P2RBaseModel):
    episode_id: str = Field(default_factory=lambda: new_id("episode"))
    user_id: str
    project_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    role: Literal["user", "assistant", "system", "tool"] = "user"
    content: str
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryEntry(P2RBaseModel):
    memory_id: str = Field(default_factory=lambda: new_id("mem"))
    content: str
    summary: str
    user_id: str
    project_id: str | None = None
    scope: Literal["global", "project", "session"] = "project"
    facets: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    utility: float = Field(default=0.0, ge=0.0, le=1.0)
    status: Literal["active", "superseded", "deprecated", "uncertain"] = "active"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_used_at: datetime | None = None
    use_count: int = Field(default=0, ge=0)
    version: int = Field(default=1, ge=1)
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    primary_cluster_id: str | None = None
    secondary_cluster_ids: list[str] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("facets", "keywords", "entities", "secondary_cluster_ids", mode="before")
    @classmethod
    def _dedupe_strings(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        seen: set[str] = set()
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result


class ClusterNode(P2RBaseModel):
    cluster_id: str = Field(default_factory=lambda: new_id("cluster"))
    member_memory_ids: list[str] = Field(default_factory=list)
    centroid_embedding: list[float] | None = None
    medoid_memory_id: str | None = None
    title: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    facet_distribution: dict[str, int] = Field(default_factory=dict)
    time_range: tuple[datetime, datetime] | None = None
    size: int = 0
    radius: float = 0.0
    cohesion: float = 1.0
    entropy: float = 0.0
    query_hits: int = 0
    avg_utility: float = 0.0
    shard_id: str = "default"
    dirty: bool = False
    version: int = 1
    status: Literal["active", "split", "merged", "archived"] = "active"
    redirect_to: list[str] = Field(default_factory=list)
    user_id: str
    project_id: str | None = None
    scope: Literal["global", "project", "session"] = "project"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ClusterCard(P2RBaseModel):
    cluster_id: str
    title: str
    summary: str
    representative_memory: str
    facet_distribution: dict[str, int] = Field(default_factory=dict)
    size: int
    time_range: tuple[datetime, datetime] | None = None
    keywords: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    health_flags: list[str] = Field(default_factory=list)
    retrieval_score: float = 0.0
    utility_score: float = 0.0
    shard_id: str = "default"
    user_id: str
    project_id: str | None = None


class ClusterMembership(P2RBaseModel):
    membership_id: str = Field(default_factory=lambda: new_id("membership"))
    cluster_id: str
    memory_id: str
    role: Literal["primary", "secondary"] = "primary"
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)


class MemoryEdge(P2RBaseModel):
    edge_id: str = Field(default_factory=lambda: new_id("edge"))
    src_memory_id: str
    dst_memory_id: str
    relation_type: Literal["SUPPORTS", "UPDATES", "CONFLICTS_WITH", "DUPLICATE_OF", "DERIVED_FROM"]
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class EvidenceItem(P2RBaseModel):
    memory_id: str
    content: str
    summary: str
    score: float = 0.0
    cluster_id: str | None = None
    facets: list[str] = Field(default_factory=list)
    status: str = "active"
    source_refs: list[str] = Field(default_factory=list)
    rationale: str | None = None


class QueryAspect(P2RBaseModel):
    aspect_id: str = Field(default_factory=lambda: new_id("aspect"))
    text: str
    kind: Literal[
        "topic",
        "constraint",
        "cardinality",
        "exclusive_alternative",
        "format",
        "unknown",
    ] = "topic"
    required: bool = True
    requires_evidence: bool = True
    keywords: list[str] = Field(default_factory=list)
    source: Literal["rule", "llm"] = "rule"


class EvidencePacket(P2RBaseModel):
    query: str
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    profile_hints: list[str] = Field(default_factory=list)
    task_state: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    required_aspects: list[QueryAspect] = Field(default_factory=list)
    coverage: dict[str, list[str]] = Field(default_factory=dict)
    missing_aspects: list[str] = Field(default_factory=list)
    token_budget: int = 1200
    token_estimate: int = 0
    omitted_memory_ids: list[str] = Field(default_factory=list)


class ActionTrace(P2RBaseModel):
    trace_id: str = Field(default_factory=lambda: new_id("trace"))
    phase: Literal["remember", "use", "verify", "maintenance", "improve", "test"] = "use"
    action: MemoryAction | str
    observation_json: dict[str, Any] = Field(default_factory=dict)
    action_json: dict[str, Any] = Field(default_factory=dict)
    tool_result_json: dict[str, Any] = Field(default_factory=dict)
    valid_format: bool = True
    error_code: str | None = None
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=utc_now)


class ExposureEvent(P2RBaseModel):
    exposure_id: str = Field(default_factory=lambda: new_id("exposure"))
    query_id: str = Field(default_factory=lambda: new_id("query"))
    query: str
    selected_memory_ids: list[str] = Field(default_factory=list)
    cluster_ids: list[str] = Field(default_factory=list)
    exposure_format: str = "evidence_packet"
    token_count: int = 0
    answer_score: float | None = None
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardTrace(P2RBaseModel):
    reward_id: str = Field(default_factory=lambda: new_id("reward"))
    query_id: str | None = None
    task_score: float = 0.0
    memory_recall: float = 0.0
    evidence_precision: float = 0.0
    token_cost: float = 0.0
    latency: float = 0.0
    conflict_penalty: float = 0.0
    total_reward: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ClusterHealthReport(P2RBaseModel):
    report_id: str = Field(default_factory=lambda: new_id("health"))
    num_clusters: int = 0
    avg_size: float = 0.0
    small_ratio: float = 0.0
    large_ratio: float = 0.0
    avg_radius: float = 0.0
    avg_cohesion: float = 0.0
    split_candidates: list[str] = Field(default_factory=list)
    merge_candidates: list[tuple[str, str]] = Field(default_factory=list)
    refresh_candidates: list[str] = Field(default_factory=list)
    deprecate_candidates: list[str] = Field(default_factory=list)
    proposals: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class MaintenanceTriggerDecision(P2RBaseModel):
    should_run: bool = False
    reasons: list[str] = Field(default_factory=list)
    skipped_reasons: list[str] = Field(default_factory=list)
    last_maintenance_at: datetime | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class MemoryCandidate(P2RBaseModel):
    content: str
    summary: str
    user_id: str
    project_id: str | None = None
    scope: Literal["global", "project", "session"] = "project"
    facets: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_entry(self) -> MemoryEntry:
        return MemoryEntry(**self.model_dump())


class PolicyObservation(P2RBaseModel):
    query: str
    phase: Literal["remember", "use", "improve"] = "use"
    budget: dict[str, Any] = Field(default_factory=dict)
    cluster_cards: list[ClusterCard] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    bank_stats: dict[str, Any] = Field(default_factory=dict)
    risk_flags: list[str] = Field(default_factory=list)


class ControllerAction(P2RBaseModel):
    action: MemoryAction
    cluster_ids: list[str] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
