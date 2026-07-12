from __future__ import annotations

import re
from typing import Any

from .csl_schema import Event, StateDelta, TaskSchemaCard, new_id, utc_now


def detect_missing_fields(
    *,
    state_need: dict[str, Any],
    retrieved_lanes: dict[str, Any],
    schema_card: TaskSchemaCard | None,
) -> list[dict[str, Any]]:
    needed_variables = [str(item).lower() for item in state_need.get("needed_variables", [])]
    if not needed_variables:
        return []
    retrieved_text = str(retrieved_lanes).lower()
    missing = []
    for variable in needed_variables:
        if variable in {"state", "evidence"}:
            continue
        if variable not in retrieved_text:
            missing.append(
                {
                    "subject": "current_need",
                    "variable": variable,
                    "why_needed": f"The current subtask appears to need {variable}, but no retrieved state supplied it.",
                    "suggested_trace_scope": "full_event_path",
                    "priority": "medium",
                }
            )
    return missing[:8]


def backfill_missing_fields(
    *,
    task_id: str,
    schema_card: TaskSchemaCard | None,
    missing_fields: list[dict[str, Any]],
    events: list[Event],
    question: str,
) -> list[StateDelta]:
    deltas: list[StateDelta] = []
    schema_id = schema_card.schema_id if schema_card else None
    for field in missing_fields:
        variable = str(field.get("variable", "")).strip().lower()
        if not variable:
            continue
        for event in reversed(events):
            value = _find_variable_value(event.raw_chunk, variable)
            if value is None:
                continue
            deltas.append(
                StateDelta(
                    delta_id=new_id("delta"),
                    task_id=task_id,
                    checkpoint_id=None,
                    schema_id=schema_id,
                    op="BACKFILL",
                    subject=str(field.get("subject") or "current_need"),
                    variable=variable,
                    value=value,
                    attributes={
                        "rationale": f"Backfilled {variable} for current subtask from historical trace.",
                        "question": question,
                        "short_evidence": event.raw_chunk[:300],
                    },
                    status="candidate",
                    confidence=0.65,
                    source_event_ids=[event.event_id],
                    source_span_ids=[f"{event.event_id}:raw_chunk"],
                    valid=True,
                    created_at=utc_now(),
                )
            )
            break
    return deltas


def _find_variable_value(text: str, variable: str) -> str | float | None:
    patterns = [
        rf"\b{re.escape(variable)}\s*(?:is|=|:)?\s*([^\n,;.]+)",
        rf"\b{re.escape(variable)}\s+([A-Za-z0-9_.-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        raw = match.group(1).strip()
        if re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
            return float(raw)
        return raw
    return None
