from __future__ import annotations

import json
import re
from typing import Any

from .csl_schema import Checkpoint, DependencyEdge, Event, StateDelta, TaskContractCard, TaskSchemaCard, new_id, utc_now
from .csl_visible_entities import parse_visible_catalog_entities


def parse_event_chunk(chunk: str) -> dict[str, Any]:
    structured = _parse_structured_event(chunk)
    if structured is not None:
        action = str(structured.get("action") or "").strip() or None
        observation = str(structured.get("observation") or "").strip() or None
        reward = structured.get("reward")
        feedback = str(structured.get("feedback") or "").strip() or None
        event_type = str(structured.get("event_type") or "").strip().lower()
        is_success = bool(structured.get("is_success")) or event_type in {"purchase_accepted", "completed"}
        is_failure = bool(structured.get("is_failure")) or event_type in {"error", "rejected", "failed"}
        is_final = bool(structured.get("is_final_action")) or event_type in {
            "selection_committed",
            "purchase_accepted",
            "completed",
        }
        boundary = bool(structured.get("boundary_signal")) or is_success or is_failure or is_final
        entities = structured.get("entities") if isinstance(structured.get("entities"), list) else []
        return {
            "action": action,
            "observation": observation,
            "reward": reward,
            "feedback": feedback,
            "is_final_action": is_final,
            "is_success": is_success,
            "is_failure": is_failure,
            "boundary_signal": boundary,
            "boundary_reason": str(structured.get("boundary_reason") or _boundary_reason(is_success, is_failure, is_final, boundary)),
            "explicit_entities": [
                str(entity.get("subject")) for entity in entities if isinstance(entity, dict) and entity.get("subject")
            ],
            "explicit_values": [],
            "raw_action_type": _raw_action_type(action or event_type),
            "structured_event": structured,
        }

    action = _extract_labeled_value(chunk, ["action", "act", "agent action"])
    observation = _extract_labeled_value(chunk, ["observation", "obs", "result", "env observation"])
    reward = _extract_labeled_value(chunk, ["reward", "score"])
    feedback = _extract_labeled_value(chunk, ["feedback", "judgement", "judge"])
    done_text = _extract_labeled_value(chunk, ["done"])
    explicit_done = None
    if done_text is not None:
        explicit_done = done_text.strip().lower() in {"true", "1", "yes"}
    visible_result = "\n".join(item for item in [action, observation, feedback, reward] if item)
    lowered = visible_result.lower()
    guard_blocked = "guard blocked" in lowered
    explicit_terminal_failure = bool(
        re.search(
            r"webshopplus soft exit|\bsoft exit\b|\bno purchase\b|"
            r"repeat_action\s*>=|\bmax(?:imum)? rounds?\b|\btimeout\b",
            lowered,
        )
    )
    is_success = bool(
        not explicit_terminal_failure
        and (
            explicit_done is True
            or re.search(r"\b(thank you for shopping|purchase accepted|task completed|successfully purchased)\b", lowered)
        )
    )
    is_failure = bool(
        guard_blocked
        or explicit_terminal_failure
        or re.search(r"\b(failed|failure|incorrect|rejected|fatal error|invalid action)\b", lowered)
    )
    is_final = bool(
        is_success
        or (
            explicit_done is not False
            and not guard_blocked
            and re.search(r"\b(finalize|committed|finish|booked|purchased|submitted)\b", lowered)
        )
    )
    boundary = is_success or is_failure or is_final or explicit_done is True
    return {
        "action": action,
        "observation": observation,
        "reward": reward,
        "feedback": feedback,
        "is_final_action": is_final,
        "is_success": is_success,
        "is_failure": is_failure,
        "boundary_signal": boundary,
        "boundary_reason": _boundary_reason(is_success, is_failure, is_final, boundary),
        "explicit_entities": extract_subjects(observation or visible_result),
        "explicit_values": extract_values(observation or visible_result),
        "raw_action_type": _raw_action_type(action or chunk),
        "source_format": "labeled_action_observation" if action is not None and observation is not None else "generic_text",
        "explicit_done": explicit_done,
    }


def summarize_checkpoint(events: list[Event]) -> dict[str, Any]:
    chunks = [event.raw_chunk for event in events]
    text = "\n".join(chunks)
    failed = [chunk for chunk in chunks if re.search(r"\b(failed|failure|incorrect|rejected|invalid)\b", chunk, re.I)]
    finals = [chunk for chunk in chunks if re.search(r"\b(final|selected|accepted|committed|completed)\b", chunk, re.I)]
    constraints = [chunk for chunk in chunks if re.search(r"\b(must|should|required|cannot|avoid|prefer|compatible)\b", chunk, re.I)]
    observations = [event.observation or event.raw_chunk for event in events[:6]]
    if finals and not failed:
        outcome = "success"
    elif failed:
        outcome = "failed"
    elif observations:
        outcome = "partial"
    else:
        outcome = "uncertain"
    return {
        "summary": _compact_text(text, 1200),
        "outcome": outcome,
        "final_decisions": finals[:6],
        "important_observations": observations[:8],
        "rejected_or_failed_attempts": failed[:6],
        "discovered_constraints": constraints[:6],
        "unresolved_uncertainties": [],
        "evidence_event_ids": [event.event_id for event in events],
    }


def extract_state_deltas(
    *,
    task_id: str,
    checkpoint: Checkpoint,
    schema_card: TaskSchemaCard | None,
    events: list[Event],
    contract: TaskContractCard | None = None,
) -> list[StateDelta]:
    deltas: list[StateDelta] = []
    schema_id = schema_card.schema_id if schema_card else None
    contract_id = contract.contract_id if contract else None
    for event in events:
        structured = event.metadata.get("structured_event")
        if isinstance(structured, dict):
            deltas.extend(
                _extract_structured_event_deltas(
                    task_id=task_id,
                    checkpoint_id=checkpoint.checkpoint_id,
                    contract_id=contract_id,
                    schema_id=schema_id,
                    event=event,
                    structured=structured,
                )
            )
            continue
        if event.metadata.get("source_format") == "labeled_action_observation":
            deltas.extend(
                _extract_labeled_visible_deltas(
                    task_id=task_id,
                    checkpoint_id=checkpoint.checkpoint_id,
                    contract_id=contract_id,
                    schema_id=schema_id,
                    event=event,
                )
            )
            continue
        text = event.raw_chunk
        subjects = extract_subjects(text)
        subject = subjects[0] if subjects else "current_execution"
        deltas.extend(_extract_property_deltas(task_id, checkpoint.checkpoint_id, contract_id, schema_id, event, subject))
        deltas.extend(_extract_constraint_deltas(task_id, checkpoint.checkpoint_id, contract_id, schema_id, event, subject))
        deltas.extend(_extract_dependency_deltas(task_id, checkpoint.checkpoint_id, contract_id, schema_id, event, subject))
        deltas.extend(_extract_repair_deltas(task_id, checkpoint.checkpoint_id, contract_id, schema_id, event, subject))
        final_delta = _extract_final_delta(task_id, checkpoint.checkpoint_id, contract_id, schema_id, event, subject)
        if final_delta is not None:
            deltas.append(final_delta)
        rejection_delta = _extract_rejection_delta(task_id, checkpoint.checkpoint_id, contract_id, schema_id, event, subject)
        if rejection_delta is not None:
            deltas.append(rejection_delta)
    return _dedupe_deltas(deltas)


def _extract_labeled_visible_deltas(
    *,
    task_id: str,
    checkpoint_id: str,
    contract_id: str | None,
    schema_id: str | None,
    event: Event,
) -> list[StateDelta]:
    """Extract only visible environment evidence, never the repeated task prompt."""
    entities = parse_visible_catalog_entities(event.action, event.observation)
    deltas: list[StateDelta] = []
    if entities:
        deltas.extend(
            _extract_structured_event_deltas(
                task_id=task_id,
                checkpoint_id=checkpoint_id,
                contract_id=contract_id,
                schema_id=schema_id,
                event=event,
                structured={"event_type": "webshop_observation", "entities": entities},
            )
        )

    observation = event.observation or ""
    if event.is_failure:
        deltas.append(
            _new_delta(
                task_id=task_id,
                checkpoint_id=checkpoint_id,
                contract_id=contract_id,
                schema_id=schema_id,
                op="REJECT",
                subject="current_execution",
                variable="negative_memory",
                value=observation or event.action or "visible action failure",
                status="failed",
                confidence=0.98,
                event=event,
                reason="visible action failure or guard rejection",
            )
        )
    if event.is_success:
        deltas.append(
            _new_delta(
                task_id=task_id,
                checkpoint_id=checkpoint_id,
                contract_id=contract_id,
                schema_id=schema_id,
                op="FINALIZE",
                subject="current_selection",
                variable="purchase_status",
                value="accepted",
                status="final",
                confidence=0.99,
                event=event,
                reason="visible purchase completion",
            )
        )
    return deltas


def _extract_structured_event_deltas(
    *,
    task_id: str,
    checkpoint_id: str,
    contract_id: str | None,
    schema_id: str | None,
    event: Event,
    structured: dict[str, Any],
) -> list[StateDelta]:
    deltas: list[StateDelta] = []
    entities = structured.get("entities")
    if isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            subject = _canonical_name(str(entity.get("subject") or "current_execution"))
            properties = entity.get("properties")
            if not isinstance(properties, dict):
                continue
            status = str(entity.get("status") or "candidate").lower()
            confidence = float(entity.get("confidence", 0.98))
            for variable, value in properties.items():
                deltas.append(
                    _new_delta(
                        task_id=task_id,
                        checkpoint_id=checkpoint_id,
                        contract_id=contract_id,
                        schema_id=schema_id,
                        op="SET",
                        subject=subject,
                        variable=_canonical_variable(str(variable)),
                        value=value,
                        status=status,
                        confidence=confidence,
                        event=event,
                        reason="entity-bound property from a structured visible event",
                    )
                )

    constraints = structured.get("constraints")
    if isinstance(constraints, list):
        for item in constraints:
            if isinstance(item, dict):
                subject = _canonical_name(str(item.get("subject") or "current_execution"))
                value = item.get("value") or item.get("constraint")
            else:
                subject = "current_execution"
                value = item
            if value is None or value == "":
                continue
            deltas.append(
                _new_delta(
                    task_id=task_id,
                    checkpoint_id=checkpoint_id,
                    contract_id=contract_id,
                    schema_id=schema_id,
                    op="CONSTRAIN",
                    subject=subject,
                    variable="constraint",
                    value=value,
                    status="final",
                    confidence=0.95,
                    event=event,
                    reason="scoped constraint from a structured visible event",
                )
            )

    dependencies = structured.get("dependencies")
    if isinstance(dependencies, list):
        for item in dependencies:
            if not isinstance(item, dict) or not item.get("value"):
                continue
            deltas.append(
                _new_delta(
                    task_id=task_id,
                    checkpoint_id=checkpoint_id,
                    contract_id=contract_id,
                    schema_id=schema_id,
                    op="DEPEND",
                    subject=_canonical_name(str(item.get("subject") or "current_execution")),
                    variable="dependency",
                    value=item["value"],
                    status="candidate",
                    confidence=0.9,
                    event=event,
                    reason="dependency from a structured visible event",
                )
            )

    decision = structured.get("decision")
    if isinstance(decision, dict) and decision.get("value") is not None:
        deltas.append(
            _new_delta(
                task_id=task_id,
                checkpoint_id=checkpoint_id,
                contract_id=contract_id,
                schema_id=schema_id,
                op="FINALIZE",
                subject=_canonical_name(str(decision.get("subject") or "current_execution")),
                variable=str(decision.get("variable") or "decision"),
                value=decision["value"],
                status="final",
                confidence=float(decision.get("confidence", 0.98)),
                event=event,
                reason="visible structured commitment",
            )
        )

    if str(structured.get("event_type") or "").lower() in {"error", "rejected", "failed"}:
        deltas.append(
            _new_delta(
                task_id=task_id,
                checkpoint_id=checkpoint_id,
                contract_id=contract_id,
                schema_id=schema_id,
                op="REJECT",
                subject="current_execution",
                variable="negative_memory",
                value=structured.get("observation") or structured.get("feedback") or "structured action failure",
                status="failed",
                confidence=0.98,
                event=event,
                reason="visible structured failure or rejection",
            )
        )

    return deltas


def canonicalize_and_validate_deltas(deltas: list[StateDelta]) -> list[StateDelta]:
    canonicalized = []
    for delta in deltas:
        delta.subject = _canonical_name(delta.subject)
        if delta.variable:
            delta.variable = _canonical_variable(delta.variable)
        if not delta.source_event_ids:
            delta.valid = False
            delta.status = "uncertain"
            delta.confidence = min(delta.confidence, 0.4)
        if delta.op == "FINALIZE" and delta.status != "final":
            delta.status = "final"
        if delta.op == "REJECT" and delta.status not in {"rejected", "failed"}:
            delta.status = "rejected"
        canonicalized.append(delta)
    return canonicalized


def build_dependency_edges(
    *,
    task_id: str,
    deltas: list[StateDelta],
    existing_delta_ids_by_key: dict[tuple[str | None, str, str], str] | None = None,
) -> list[DependencyEdge]:
    edges: list[DependencyEdge] = []
    by_key: dict[tuple[str | None, str, str], StateDelta] = {}
    existing = existing_delta_ids_by_key or {}
    for delta in deltas:
        variable = delta.variable or delta.op.lower()
        key = (delta.session_id, delta.subject.lower(), variable.lower())
        if key in by_key and delta.delta_id != by_key[key].delta_id:
            relation = "supersedes" if delta.status == "final" else "same_as"
            edges.append(
                DependencyEdge(
                    edge_id=new_id("edge"),
                    task_id=task_id,
                    source_delta_id=delta.delta_id,
                    target_delta_id=by_key[key].delta_id,
                    relation=relation,
                    rule="same subject-variable observed in the same extraction span",
                    confidence=0.75,
                )
            )
        elif key in existing:
            edges.append(
                DependencyEdge(
                    edge_id=new_id("edge"),
                    task_id=task_id,
                    source_delta_id=delta.delta_id,
                    target_delta_id=existing[key],
                    relation="supersedes" if delta.status == "final" else "same_as",
                    rule="same subject-variable observed in ledger",
                    confidence=0.7,
                )
            )
        by_key[key] = delta

    for source in deltas:
        if source.op not in {"CONSTRAIN", "DEPEND", "DERIVE", "REJECT"}:
            continue
        for target in deltas:
            if source.delta_id == target.delta_id:
                continue
            if source.subject.lower() == target.subject.lower() or source.subject == "current_execution":
                relation = {
                    "CONSTRAIN": "constrains",
                    "DEPEND": "depends_on",
                    "DERIVE": "derived_from",
                    "REJECT": "avoids",
                }[source.op]
                edges.append(
                    DependencyEdge(
                        edge_id=new_id("edge"),
                        task_id=task_id,
                        source_delta_id=source.delta_id,
                        target_delta_id=target.delta_id,
                        relation=relation,  # type: ignore[arg-type]
                        rule=f"{source.op} state is relevant to target state",
                        confidence=0.6,
                    )
                )
    return edges


def extract_subjects(text: str) -> list[str]:
    subjects: list[str] = []
    for pattern in [
        r"\b(?:option|candidate|choice|item|result|answer|plan|case|entity)\s+[A-Z0-9][A-Za-z0-9_-]*\b",
        r"\bid\s+([A-Z0-9][A-Za-z0-9_-]*)\b",
    ]:
        for match in re.finditer(pattern, text, flags=re.I):
            subjects.append(match.group(0).strip())
    final_match = re.search(r"\bselect(?:ed)?\s+(.+?)\s+as\s+(?:the\s+)?final", text, flags=re.I)
    if final_match:
        subjects.append(final_match.group(1).strip())
    return _dedupe_strings([_canonical_name(subject) for subject in subjects])[:8]


def extract_values(text: str) -> list[str]:
    values = []
    values.extend(match.group(0) for match in re.finditer(r"\b\d+(?:\.\d+)?\b", text))
    values.extend(match.group(1).strip() for match in re.finditer(r"\bid\s*[:=]?\s*([A-Za-z0-9_-]+)", text, re.I))
    return _dedupe_strings(values)


def _extract_property_deltas(
    task_id: str,
    checkpoint_id: str,
    contract_id: str | None,
    schema_id: str | None,
    event: Event,
    subject: str,
) -> list[StateDelta]:
    text = event.raw_chunk
    deltas: list[StateDelta] = []
    property_patterns = [
        (r"\battribute\s+([A-Za-z0-9_.-]+)", "attribute"),
        (r"\bscore\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)", "score"),
        (r"\bid\s*(?:is|=|:)?\s*([A-Za-z0-9_-]+)", "id"),
        (r"\bstatus\s*(?:is|=|:)?\s*([A-Za-z0-9_.-]+)", "status"),
    ]
    for pattern, variable in property_patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            value: Any = match.group(1).strip()
            if variable == "score":
                value = float(value)
            deltas.append(
                _new_delta(
                    task_id=task_id,
                    checkpoint_id=checkpoint_id,
                    contract_id=contract_id,
                    schema_id=schema_id,
                    op="SET",
                    subject=subject,
                    variable=variable,
                    value=value,
                    status="candidate",
                    confidence=0.8,
                    event=event,
                    reason=f"visible {variable} value",
                )
            )
    for match in re.finditer(r"\b([a-z][a-z0-9_-]{1,30})\s*[:=]\s*([^\n,;]+)", text):
        variable = match.group(1).strip().lower()
        if variable in {"action", "observation", "obs", "reward", "feedback"}:
            continue
        value = match.group(2).strip()
        deltas.append(
            _new_delta(
                task_id=task_id,
                checkpoint_id=checkpoint_id,
                contract_id=contract_id,
                schema_id=schema_id,
                op="SET",
                subject=subject,
                variable=variable,
                value=value,
                status="candidate",
                confidence=0.72,
                event=event,
                reason="visible key-value state",
            )
        )
    return deltas


def _extract_constraint_deltas(
    task_id: str,
    checkpoint_id: str,
    contract_id: str | None,
    schema_id: str | None,
    event: Event,
    subject: str,
) -> list[StateDelta]:
    sentences = _sentences(event.raw_chunk)
    deltas = []
    for sentence in sentences:
        if not re.search(r"\b(must|should|required|cannot|avoid|prefer|compatible|constraint)\b", sentence, re.I):
            continue
        deltas.append(
            _new_delta(
                task_id=task_id,
                checkpoint_id=checkpoint_id,
                contract_id=contract_id,
                schema_id=schema_id,
                op="CONSTRAIN",
                subject=subject,
                variable="constraint",
                value=sentence.strip(),
                status="final",
                confidence=0.75,
                event=event,
                reason="visible constraint or preference",
            )
        )
    return deltas


def _extract_dependency_deltas(
    task_id: str,
    checkpoint_id: str,
    contract_id: str | None,
    schema_id: str | None,
    event: Event,
    subject: str,
) -> list[StateDelta]:
    deltas = []
    for sentence in _sentences(event.raw_chunk):
        if re.search(r"\b(depends on|requires|because|derived from|based on)\b", sentence, re.I):
            op = "DERIVE" if re.search(r"\b(derived from|because|based on)\b", sentence, re.I) else "DEPEND"
            deltas.append(
                _new_delta(
                    task_id=task_id,
                    checkpoint_id=checkpoint_id,
                    contract_id=contract_id,
                    schema_id=schema_id,
                    op=op,
                    subject=subject,
                    variable="dependency",
                    value=sentence.strip(),
                    status="candidate",
                    confidence=0.68,
                    event=event,
                    reason="visible dependency expression",
                )
            )
    return deltas


def _extract_repair_deltas(
    task_id: str,
    checkpoint_id: str,
    contract_id: str | None,
    schema_id: str | None,
    event: Event,
    subject: str,
) -> list[StateDelta]:
    text = event.raw_chunk
    if not re.search(r"\b(wrong|incorrect|invalid|correct|correction|revise|repair)\b", text, re.I):
        return []
    deltas: list[StateDelta] = [
        _new_delta(
            task_id=task_id,
            checkpoint_id=checkpoint_id,
            contract_id=contract_id,
            schema_id=schema_id,
            op="INVALIDATE",
            subject=subject,
            variable="repair_barrier",
            value=_compact_text(text, 400),
            status="uncertain",
            confidence=0.72,
            event=event,
            reason="visible feedback indicates a prior state may be wrong or invalid",
            publishable=False,
        )
    ]
    correction_match = re.search(r"\bcorrect\s+([A-Za-z0-9_.-]+)\s*(?:is|=|:)\s*([^\n.;]+)", text, re.I)
    if correction_match:
        deltas.append(
            _new_delta(
                task_id=task_id,
                checkpoint_id=checkpoint_id,
                contract_id=contract_id,
                schema_id=schema_id,
                op="BACKFILL",
                subject=subject,
                variable=correction_match.group(1).strip().lower(),
                value=correction_match.group(2).strip(),
                status="candidate",
                confidence=0.7,
                event=event,
                reason="visible correction supplies a replacement value",
            )
        )
    return deltas


def _extract_final_delta(
    task_id: str,
    checkpoint_id: str,
    contract_id: str | None,
    schema_id: str | None,
    event: Event,
    subject: str,
) -> StateDelta | None:
    text = event.raw_chunk
    if not re.search(r"\b(final|finalize|selected|choose|chosen|commit|accepted|completed|finish|booked|purchased|submitted)\b", text, re.I):
        return None
    return _new_delta(
        task_id=task_id,
        checkpoint_id=checkpoint_id,
        contract_id=contract_id,
        schema_id=schema_id,
        op="FINALIZE",
        subject=subject,
        variable="decision",
        value=_compact_text(text, 400),
        status="final",
        confidence=0.82,
        event=event,
        reason="visible finalization or acceptance signal",
    )


def _extract_rejection_delta(
    task_id: str,
    checkpoint_id: str,
    contract_id: str | None,
    schema_id: str | None,
    event: Event,
    subject: str,
) -> StateDelta | None:
    text = event.raw_chunk
    if not re.search(r"\b(reject|rejected|failed|failure|invalid|avoid|incorrect|superseded)\b", text, re.I):
        return None
    status = "failed" if re.search(r"\b(failed|failure|incorrect)\b", text, re.I) else "rejected"
    return _new_delta(
        task_id=task_id,
        checkpoint_id=checkpoint_id,
        contract_id=contract_id,
        schema_id=schema_id,
        op="REJECT",
        subject=subject,
        variable="negative_memory",
        value=_compact_text(text, 400),
        status=status,
        confidence=0.78,
        event=event,
        reason="visible rejection or failure signal",
    )


def _new_delta(
    *,
    task_id: str,
    checkpoint_id: str,
    contract_id: str | None,
    schema_id: str | None,
    op: str,
    subject: str,
    variable: str | None,
    value: Any,
    status: str,
    confidence: float,
    event: Event,
    reason: str,
    publishable: bool = True,
) -> StateDelta:
    return StateDelta(
        delta_id=new_id("delta"),
        task_id=task_id,
        contract_id=contract_id,
        checkpoint_id=checkpoint_id,
        schema_id=schema_id,
        op=op,  # type: ignore[arg-type]
        subject=subject,
        variable=variable,
        value=value,
        attributes={"reason": reason, "short_evidence": _compact_text(event.raw_chunk, 240)},
        status=status,  # type: ignore[arg-type]
        confidence=confidence,
        source_event_ids=[event.event_id],
        source_span_ids=[f"{event.event_id}:raw_chunk"],
        session_id=event.session_id,
        valid=True,
        publishable=publishable,
        created_at=utc_now(),
    )


def _extract_labeled_value(chunk: str, labels: list[str]) -> str | None:
    labels_re = "|".join(re.escape(label) for label in labels)
    pattern = rf"(?is)(?:^|\n)\s*(?:{labels_re})\s*:\s*(.+?)(?=\n\s*(?:user instruction|agent action|env observation|action|act|observation|obs|result|reward|score|feedback|judgement|judge|done)\s*:|\Z)"
    match = re.search(pattern, chunk)
    if match:
        return match.group(1).strip()
    return None


def _parse_structured_event(chunk: str) -> dict[str, Any] | None:
    stripped = chunk.strip()
    marker = "DART_EVENT_JSON:"
    if stripped.startswith(marker):
        stripped = stripped[len(marker) :].strip()
    elif not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("event_type"):
        return parsed
    if "final_action" not in parsed and "final_observation" not in parsed:
        return None

    observation = parsed.get("final_observation")
    if isinstance(observation, dict):
        observation_text = str(observation.get("state") or json.dumps(observation, ensure_ascii=False))
    else:
        observation_text = str(observation or "")
    task_text = _compact_text(str(parsed.get("task") or "current execution"), 160)
    done = bool(parsed.get("done"))
    status = "final" if done else "candidate"
    properties = {
        "last_action": parsed.get("final_action"),
        "last_observation": observation_text,
    }
    if parsed.get("reward") is not None:
        properties["reward"] = parsed["reward"]
    synthesized = {
        "event_type": "completed" if done else "interaction_turn",
        "action": parsed.get("final_action"),
        "observation": observation_text,
        "reward": parsed.get("reward"),
        "is_final_action": done,
        "is_success": done and not parsed.get("error"),
        "is_failure": bool(parsed.get("error")),
        "boundary_signal": done,
        "entities": [
            {
                "subject": task_text,
                "status": status,
                "confidence": 0.98,
                "properties": properties,
            }
        ],
    }
    if done:
        synthesized["decision"] = {
            "subject": task_text,
            "variable": "decision",
            "value": {
                "action": parsed.get("final_action"),
                "observation": observation_text,
                "reward": parsed.get("reward"),
            },
        }
    return synthesized


def _boundary_reason(is_success: bool, is_failure: bool, is_final: bool, boundary: bool) -> str:
    if is_success:
        return "visible success signal"
    if is_failure:
        return "visible failure signal"
    if is_final:
        return "visible finalization signal"
    if boundary:
        return "visible boundary signal"
    return "none"


def _raw_action_type(text: str) -> str:
    match = re.search(r"\b([A-Za-z][A-Za-z0-9_]*)\s*(?:\[|\()", text)
    if match:
        return match.group(1)
    match = re.search(r"\b(action|search|read|click|select|finish|answer|book|buy|submit|think|revise)\b", text, re.I)
    return match.group(1).lower() if match else "unknown"


def _canonical_name(text: str) -> str:
    stripped = re.sub(r"\s+", " ", text.strip())
    stripped = re.sub(r"^(id)\s+", "", stripped, flags=re.I)
    if re.match(r"^(option|candidate|choice|item|result|answer|plan|case|entity)\s+", stripped, flags=re.I):
        first, rest = stripped.split(" ", 1)
        return f"{first.lower()} {rest.strip()}"
    return stripped


def _canonical_variable(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.strip().lower()).strip("_") or "state"


def _dedupe_deltas(deltas: list[StateDelta]) -> list[StateDelta]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[StateDelta] = []
    for delta in deltas:
        key = (delta.op, delta.subject.lower(), str(delta.variable).lower(), str(delta.value))
        if key in seen:
            continue
        seen.add(key)
        result.append(delta)
    return result


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = re.sub(r"\s+", " ", str(item).strip())
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", text) if item.strip()]


def _compact_text(text: str, limit: int) -> str:
    stripped = re.sub(r"\s+", " ", text.strip())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3].rstrip() + "..."
