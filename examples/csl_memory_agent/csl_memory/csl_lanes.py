from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .csl_contract import relevant_contract_rules
from .csl_schema import DependencyEdge, Event, MaterializedState, StateDelta, StateLaneResult, TaskContractCard, new_id, utc_now


def materialize_state(deltas: list[StateDelta], edges: list[DependencyEdge]) -> list[MaterializedState]:
    invalid_delta_ids = _invalidated_delta_ids(deltas, edges)
    valid_deltas = [
        delta
        for delta in deltas
        if delta.valid
        and delta.publishable
        and delta.delta_id not in invalid_delta_ids
        and delta.status not in {"rejected", "failed", "superseded"}
    ]
    dependency_map = _dependency_map(edges)
    by_key: dict[tuple[str | None, str, str], StateDelta] = {}
    conflicts: defaultdict[tuple[str | None, str, str], list[str]] = defaultdict(list)

    for delta in valid_deltas:
        variable = delta.variable or delta.op.lower()
        if delta.op == "CONSTRAIN":
            variable = "constraint"
        elif delta.op in {"DEPEND", "DERIVE"}:
            variable = "dependency"
        key = (delta.session_id, delta.subject, variable)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = delta
            continue
        if delta.status == "final" or existing.status != "final":
            conflicts[key].append(existing.delta_id)
            by_key[key] = delta
        else:
            conflicts[key].append(delta.delta_id)

    states: list[MaterializedState] = []
    for (scope_id, subject, variable), delta in by_key.items():
        dependency_ids = sorted(
            set(dependency_map.get(delta.delta_id, []))
            | set(conflicts.get((scope_id, subject, variable), []))
        )
        states.append(
            MaterializedState(
                state_id=new_id("state"),
                task_id=delta.task_id,
                subject=subject,
                variable=variable,
                value=delta.value,
                attributes={
                    **delta.attributes,
                    "op": delta.op,
                    "confidence": delta.confidence,
                    "scope_id": scope_id,
                    "conflict_delta_ids": conflicts.get((scope_id, subject, variable), []),
                },
                status=delta.status,
                source_delta_ids=[delta.delta_id],
                dependency_delta_ids=dependency_ids,
                updated_at=delta.created_at or utc_now(),
                contract_id=delta.contract_id,
                schema_id=delta.schema_id,
                valid=delta.valid,
                publishable=delta.publishable,
            )
        )
    return states


def derive_state_lanes(
    *,
    materialized_state: list[MaterializedState],
    deltas: list[StateDelta],
    edges: list[DependencyEdge],
    recent_events: list[Event],
    contract: TaskContractCard | None = None,
    decision_signature: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, StateLaneResult]:
    config = config or {}
    decision_signature = decision_signature or {}
    lanes: dict[str, StateLaneResult] = {}
    if _enabled(config, "enable_contract_lane", True):
        rules = relevant_contract_rules(
            contract,
            str(decision_signature.get("current_instruction") or decision_signature.get("question") or ""),
            limit=int(config.get("max_contract_rules", 8)),
        )
        lanes["contract"] = StateLaneResult(
            lane_name="Contract Lane",
            entries=[{**rule, "score": rule.get("score", 0.0)} for rule in rules],
            token_estimate=_estimate_entries_tokens(rules),
        )
    if _enabled(config, "enable_decision_lane", True):
        entries = [_state_entry(state) for state in materialized_state if _is_decision_state(state)]
        lanes["decision"] = StateLaneResult(
            lane_name="Decision Lane",
            entries=entries,
            token_estimate=_estimate_entries_tokens(entries),
        )
    if _enabled(config, "enable_constraint_lane", True):
        entries = [_state_entry(state) for state in materialized_state if state.variable == "constraint"]
        if contract is not None:
            entries.extend(
                {
                    "subject": "task_contract",
                    "variable": "constraint",
                    "value": item.get("constraint") or item.get("rule") or str(item),
                    "status": "final",
                    "source_delta_ids": [],
                    "dependency_delta_ids": [],
                    "confidence": contract.confidence,
                    "evidence": f"TaskContractCard v{contract.version}",
                    "publishable": True,
                    "valid": True,
                    "score": 0.0,
                }
                for item in contract.invariant_constraints
            )
        lanes["constraint"] = StateLaneResult(
            lane_name="Constraint Lane",
            entries=entries,
            token_estimate=_estimate_entries_tokens(entries),
        )
    if _enabled(config, "enable_dependency_lane", True):
        dependency_delta_ids = {edge.source_delta_id for edge in edges if edge.relation in {"depends_on", "derived_from", "constrains"}}
        entries = [
            _state_entry(state)
            for state in materialized_state
            if state.variable == "dependency" or bool(set(state.source_delta_ids) & dependency_delta_ids)
        ]
        lanes["dependency"] = StateLaneResult(
            lane_name="Dependency Lane",
            entries=entries,
            token_estimate=_estimate_entries_tokens(entries),
            metadata={"edge_count": len(edges)},
        )
    if _enabled(config, "enable_negative_lane", True):
        entries = [
            _delta_entry(delta)
            for delta in deltas
            if delta.status in {"rejected", "failed", "superseded"} or delta.op in {"REJECT", "INVALIDATE"} or not delta.publishable
        ]
        lanes["negative"] = StateLaneResult(
            lane_name="Negative Lane",
            entries=entries,
            token_estimate=_estimate_entries_tokens(entries),
        )
    if _enabled(config, "enable_evidence_lane", True):
        entries = [_evidence_entry(delta) for delta in deltas if delta.valid and delta.publishable and delta.source_event_ids]
        lanes["evidence"] = StateLaneResult(
            lane_name="Evidence Lane",
            entries=entries,
            token_estimate=_estimate_entries_tokens(entries),
        )
    if _enabled(config, "enable_recent_lane", True):
        recent_count = int(config.get("recent_event_count", 3))
        entries = [
            {
                "event_id": event.event_id,
                "step_id": event.step_id,
                "text": event.raw_chunk,
                "status": event.status,
                "score": 0.0,
            }
            for event in recent_events[-recent_count:]
        ]
        lanes["recent"] = StateLaneResult(
            lane_name="Recent Lane",
            entries=entries,
            token_estimate=_estimate_entries_tokens(entries),
        )
    if _enabled(config, "enable_uncertainty_lane", True):
        entries = [
            _state_entry(state)
            for state in materialized_state
            if state.status == "uncertain" or float(state.attributes.get("confidence", 1.0)) < 0.6
        ] + [_delta_entry(delta) for delta in deltas if delta.status == "uncertain" and delta.valid and delta.publishable]
        lanes["uncertainty"] = StateLaneResult(
            lane_name="Uncertainty Lane",
            entries=entries,
            token_estimate=_estimate_entries_tokens(entries),
        )
    if _enabled(config, "enable_repair_lane", True):
        invalidated = _invalidated_delta_ids(deltas, edges)
        entries = [
            _delta_entry(delta)
            for delta in deltas
            if delta.delta_id in invalidated
            or not delta.publishable
            or delta.op in {"INVALIDATE", "SUPERSEDE", "BACKFILL"}
            or delta.status == "superseded"
        ]
        lanes["repair"] = StateLaneResult(
            lane_name="Repair Lane",
            entries=entries,
            token_estimate=_estimate_entries_tokens(entries),
        )
    return lanes


def retrieve_state_for_need(
    *,
    state_need: dict[str, Any],
    lanes: dict[str, StateLaneResult],
    max_entries_per_lane: int = 8,
) -> dict[str, StateLaneResult]:
    query_terms = _need_terms(state_need)
    retrieved: dict[str, StateLaneResult] = {}
    for lane_key, lane in lanes.items():
        scored = []
        for entry in lane.entries:
            score = _entry_score(entry, query_terms)
            if lane_key in {"decision", "constraint", "negative", "uncertainty"}:
                score += 0.15
            scored.append((score, entry))
        ranked = [entry for score, entry in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0]
        if not ranked and lane_key in {"decision", "constraint", "dependency", "negative", "recent"}:
            ranked = [entry for _score, entry in sorted(scored, key=lambda item: item[0], reverse=True)[:2]]
        retrieved[lane_key] = StateLaneResult(
            lane_name=lane.lane_name,
            entries=ranked[:max_entries_per_lane],
            token_estimate=_estimate_entries_tokens(ranked[:max_entries_per_lane]),
            metadata={**lane.metadata, "retrieval_terms": sorted(query_terms)},
        )
    return retrieved


def lanes_to_dict(lanes: dict[str, StateLaneResult]) -> dict[str, Any]:
    return {
        key: {
            "lane_name": lane.lane_name,
            "entries": lane.entries,
            "token_estimate": lane.token_estimate,
            "metadata": lane.metadata,
        }
        for key, lane in lanes.items()
    }


def _invalidated_delta_ids(deltas: list[StateDelta], edges: list[DependencyEdge]) -> set[str]:
    index = {delta.delta_id: idx for idx, delta in enumerate(deltas)}
    invalidated = {
        edge.target_delta_id
        for edge in edges
        if edge.status == "active"
        and edge.relation in {"invalidates", "supersedes", "descendant_of"}
        and edge.confidence >= 0.5
    }
    for delta in deltas:
        if delta.op == "INVALIDATE":
            targets = [str(target) for target in delta.metadata.get("target_delta_ids", [])]
            if not targets:
                targets = [
                    prior.delta_id
                    for prior in deltas
                    if prior.delta_id != delta.delta_id
                    and index.get(prior.delta_id, 0) < index.get(delta.delta_id, 0)
                    and (prior.subject == delta.subject or delta.subject == "current_execution")
                ]
            invalidated.update(targets)
        if delta.op == "SUPERSEDE":
            for target in delta.metadata.get("target_delta_ids", []):
                invalidated.add(str(target))
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.status != "active":
                continue
            if edge.relation in {"derived_from", "depends_on", "descendant_of"} and edge.source_delta_id in invalidated:
                if edge.target_delta_id not in invalidated:
                    invalidated.add(edge.target_delta_id)
                    changed = True
    return invalidated


def _dependency_map(edges: list[DependencyEdge]) -> dict[str, list[str]]:
    mapping: defaultdict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.status != "active":
            continue
        mapping[edge.target_delta_id].append(edge.source_delta_id)
    return dict(mapping)


def _state_entry(state: MaterializedState) -> dict[str, Any]:
    return {
        "subject": state.subject,
        "variable": state.variable,
        "value": state.value,
        "status": state.status,
        "source_delta_ids": state.source_delta_ids,
        "dependency_delta_ids": state.dependency_delta_ids,
        "confidence": state.attributes.get("confidence"),
        "evidence": state.attributes.get("short_evidence"),
        "publishable": state.publishable,
        "valid": state.valid,
        "score": 0.0,
    }


def _delta_entry(delta: StateDelta) -> dict[str, Any]:
    return {
        "delta_id": delta.delta_id,
        "op": delta.op,
        "subject": delta.subject,
        "variable": delta.variable,
        "value": delta.value,
        "status": delta.status,
        "confidence": delta.confidence,
        "source_event_ids": delta.source_event_ids,
        "evidence": delta.attributes.get("short_evidence"),
        "publishable": delta.publishable,
        "valid": delta.valid,
        "score": 0.0,
    }


def _evidence_entry(delta: StateDelta) -> dict[str, Any]:
    entry = _delta_entry(delta)
    entry["provenance"] = {
        "source_event_ids": delta.source_event_ids,
        "source_span_ids": delta.source_span_ids,
    }
    return entry


def _is_decision_state(state: MaterializedState) -> bool:
    return state.status == "final" or state.variable in {"decision", "status"} or state.attributes.get("op") == "FINALIZE"


def _need_terms(state_need: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for key in [
        "question",
        "needed_subjects",
        "needed_variables",
        "needed_constraints",
        "needed_dependencies",
        "needed_negative_memory",
        "needed_evidence",
    ]:
        value = state_need.get(key)
        if isinstance(value, list):
            for item in value:
                terms.update(_tokens(str(item)))
        elif value:
            terms.update(_tokens(str(value)))
    return terms


def _entry_score(entry: dict[str, Any], query_terms: set[str]) -> float:
    haystack = " ".join(str(value) for value in entry.values())
    entry_terms = _tokens(haystack)
    if not query_terms:
        return 0.1
    overlap = len(query_terms & entry_terms)
    score = overlap / max(len(query_terms), 1)
    if entry.get("status") == "final":
        score += 0.2
    if entry.get("confidence") and float(entry["confidence"]) >= 0.75:
        score += 0.1
    return score


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 1}


def _estimate_entries_tokens(entries: list[dict[str, Any]]) -> int:
    text = " ".join(str(entry) for entry in entries)
    return max(1, len(text) // 4) if text else 0


def _enabled(config: dict[str, Any], key: str, default: bool) -> bool:
    return bool(config.get(key, default))
