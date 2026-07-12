from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Literal, TypeVar
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass
class Event:
    event_id: str
    task_id: str
    step_id: int
    raw_chunk: str
    timestamp: str
    session_id: str | None = None
    subtask_id: str | None = None
    action: str | None = None
    observation: str | None = None
    reward: str | None = None
    feedback: str | None = None
    is_final_action: bool = False
    is_success: bool = False
    is_failure: bool = False
    boundary_signal: bool = False
    boundary_reason: str | None = None
    status: Literal["active", "failed", "ignored"] = "active"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    checkpoint_id: str
    task_id: str
    subtask_id: str | None
    start_event_id: str
    end_event_id: str
    summary: str
    outcome: Literal["success", "partial", "failed", "uncertain"]
    covered_delta_ids: list[str]
    active_state_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskContractCard:
    contract_id: str
    task_id: str
    version: int
    source_hash: str
    task_role_summary: str
    allowed_action_patterns: list[dict[str, Any]]
    irreversible_action_patterns: list[dict[str, Any]]
    final_action_patterns: list[dict[str, Any]]
    grounding_rules: list[dict[str, Any]]
    evidence_requirements: list[dict[str, Any]]
    invariant_constraints: list[dict[str, Any]]
    completion_conditions: list[dict[str, Any]]
    forbidden_shortcuts: list[dict[str, Any]]
    state_commitment_signals: list[dict[str, Any]]
    values_to_preserve_exactly: list[str]
    uncertainty_rules: list[dict[str, Any]]
    token_policy: dict[str, Any]
    confidence: float
    created_at: str
    updated_at: str | None = None


@dataclass
class TaskSchemaCard:
    schema_id: str
    task_id: str
    version: int
    task_family_description: str
    entity_types: list[dict[str, Any]]
    state_variables: list[dict[str, Any]]
    constraint_types: list[dict[str, Any]]
    dependency_types: list[dict[str, Any]]
    action_outcome_types: list[dict[str, Any]]
    status_rules: dict[str, Any]
    provenance_requirements: list[str]
    update_rules: list[dict[str, Any]]
    examples_from_trace: list[dict[str, Any]]
    confidence: float
    created_at: str
    updated_at: str | None = None


StateDeltaOp = Literal[
    "SET",
    "FINALIZE",
    "REJECT",
    "CONSTRAIN",
    "DERIVE",
    "DEPEND",
    "SUPERSEDE",
    "INVALIDATE",
    "BACKFILL",
]

StateDeltaStatus = Literal["final", "candidate", "rejected", "failed", "superseded", "uncertain"]


@dataclass
class StateDelta:
    delta_id: str
    task_id: str
    op: StateDeltaOp
    subject: str
    value: Any
    status: StateDeltaStatus
    confidence: float
    source_event_ids: list[str]
    source_span_ids: list[str]
    session_id: str | None = None
    checkpoint_id: str | None = None
    contract_id: str | None = None
    schema_id: str | None = None
    variable: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    publishable: bool = True
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyEdge:
    edge_id: str
    task_id: str
    source_delta_id: str
    target_delta_id: str
    relation: Literal[
        "depends_on",
        "derived_from",
        "same_as",
        "constrains",
        "avoids",
        "supersedes",
        "contradicts",
        "invalidates",
        "evidence_for",
        "descendant_of",
        "repair_successor_of",
    ]
    rule: str
    status: Literal["active", "inactive"] = "active"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterializedState:
    state_id: str
    task_id: str
    subject: str
    variable: str
    value: Any
    attributes: dict[str, Any]
    status: str
    source_delta_ids: list[str]
    dependency_delta_ids: list[str]
    updated_at: str
    contract_id: str | None = None
    schema_id: str | None = None
    valid: bool = True
    publishable: bool = True


@dataclass
class StateLaneResult:
    lane_name: str
    entries: list[dict[str, Any]]
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionSignature:
    decision_id: str
    task_id: str
    current_instruction: str
    expected_action_type: str
    irreversible_risk: Literal["none", "low", "medium", "high"]
    needed_contract_rules: list[str]
    needed_subjects: list[str]
    needed_variables: list[str]
    needed_final_states: list[str]
    needed_constraints: list[str]
    needed_dependencies: list[str]
    needed_negative_memory: list[str]
    needed_evidence: list[str]
    missing_field_risks: list[str]
    must_have: list[str]
    nice_to_have: list[str]
    rationale: str


@dataclass
class DecisionSupportCapsule:
    capsule_id: str
    task_id: str
    decision_id: str
    contract_rules: list[dict[str, Any]]
    final_states: list[dict[str, Any]]
    active_constraints: list[dict[str, Any]]
    derived_requirements: list[dict[str, Any]]
    dependencies: list[dict[str, Any]]
    negative_states_to_avoid: list[dict[str, Any]]
    uncertainty_items: list[dict[str, Any]]
    evidence_pointers: list[dict[str, Any]]
    missing_fields: list[dict[str, Any]]
    before_final_action_checklist: list[str]
    token_estimate: int


T = TypeVar("T")


def to_dict(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Cannot serialize object of type {type(obj)!r}")


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    field_names = {item.name for item in fields(cls)}
    filtered = {key: value for key, value in data.items() if key in field_names}
    return cls(**filtered)  # type: ignore[arg-type]


def event_from_chunk(
    *,
    task_id: str,
    step_id: int,
    chunk: str,
    action: str | None,
    observation: str | None,
    reward: str | None,
    feedback: str | None = None,
    is_final_action: bool = False,
    is_success: bool = False,
    is_failure: bool = False,
    boundary_signal: bool = False,
    boundary_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    subtask_id: str | None = None,
) -> Event:
    return Event(
        event_id=new_id("event"),
        task_id=task_id,
        session_id=session_id,
        subtask_id=subtask_id,
        step_id=step_id,
        raw_chunk=chunk,
        action=action,
        observation=observation,
        reward=reward,
        feedback=feedback,
        is_final_action=is_final_action,
        is_success=is_success,
        is_failure=is_failure,
        boundary_signal=boundary_signal,
        boundary_reason=boundary_reason,
        timestamp=utc_now(),
        metadata=metadata or {},
    )
