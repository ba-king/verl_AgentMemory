from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any

from .csl_schema import TaskSchemaCard, new_id, utc_now


GENERIC_STATUS_RULES = {
    "final": "Visible commitment, acceptance, completion, or finalization signal.",
    "candidate": "Exploratory or observed state without a finalization signal.",
    "rejected": "Visible rejection, avoidance, contradiction, or invalid result.",
    "failed": "Visible failure signal or unsuccessful attempt.",
    "superseded": "Visible replacement by a newer state for the same subject and variable.",
    "uncertain": "Ambiguous, weakly supported, stale, or incomplete evidence.",
}


def induce_schema_card(
    *,
    task_id: str,
    task_instruction: str = "",
    environment_description: str = "",
    recent_trace: str = "",
    existing_schema: TaskSchemaCard | None = None,
    existing_state_summary: str = "",
    llm_json: dict[str, Any] | None = None,
) -> TaskSchemaCard:
    visible_text = "\n".join(
        text for text in [task_instruction, environment_description, recent_trace, existing_state_summary] if text
    )
    inferred = _infer_schema_parts(visible_text)
    if llm_json:
        inferred = _merge_inferred_dicts(inferred, llm_json)

    now = utc_now()
    if existing_schema is None:
        return TaskSchemaCard(
            schema_id=new_id("schema"),
            task_id=task_id,
            version=1,
            task_family_description=inferred["task_family_description"],
            entity_types=inferred["entity_types"],
            state_variables=inferred["state_variables"],
            constraint_types=inferred["constraint_types"],
            dependency_types=inferred["dependency_types"],
            action_outcome_types=inferred["action_outcome_types"],
            status_rules=inferred["status_rules"],
            provenance_requirements=inferred["provenance_requirements"],
            update_rules=inferred["update_rules"],
            examples_from_trace=inferred["examples_from_trace"],
            confidence=float(inferred["confidence"]),
            created_at=now,
        )
    return merge_schema_card(existing_schema, inferred)


def merge_schema_card(existing_schema: TaskSchemaCard, new_schema: TaskSchemaCard | dict[str, Any]) -> TaskSchemaCard:
    new_data = _schema_to_dict(new_schema)
    merged_values = {
        "task_family_description": _prefer_longer(
            existing_schema.task_family_description,
            str(new_data.get("task_family_description", "")),
        ),
        "entity_types": _dedupe_named(existing_schema.entity_types, new_data.get("entity_types", [])),
        "state_variables": _dedupe_named(existing_schema.state_variables, new_data.get("state_variables", [])),
        "constraint_types": _dedupe_named(existing_schema.constraint_types, new_data.get("constraint_types", [])),
        "dependency_types": _dedupe_named(existing_schema.dependency_types, new_data.get("dependency_types", [])),
        "action_outcome_types": _dedupe_named(
            existing_schema.action_outcome_types,
            new_data.get("action_outcome_types", []),
        ),
        "status_rules": {**GENERIC_STATUS_RULES, **existing_schema.status_rules, **new_data.get("status_rules", {})},
        "provenance_requirements": _dedupe_strings(
            existing_schema.provenance_requirements + list(new_data.get("provenance_requirements", []))
        ),
        "update_rules": _dedupe_named(existing_schema.update_rules, new_data.get("update_rules", []), key="condition"),
        "examples_from_trace": _dedupe_examples(
            existing_schema.examples_from_trace + list(new_data.get("examples_from_trace", []))
        )[:24],
        "confidence": max(existing_schema.confidence, float(new_data.get("confidence", 0.0))),
    }
    if all(getattr(existing_schema, key) == value for key, value in merged_values.items()):
        return existing_schema

    return TaskSchemaCard(
        schema_id=existing_schema.schema_id,
        task_id=existing_schema.task_id,
        version=existing_schema.version + 1,
        **merged_values,
        created_at=existing_schema.created_at,
        updated_at=utc_now(),
    )


def summarize_schema_card(schema_card: TaskSchemaCard | None) -> str:
    if schema_card is None:
        return "None"
    variables = ", ".join(item.get("name", "") for item in schema_card.state_variables[:10])
    constraints = ", ".join(item.get("name", "") for item in schema_card.constraint_types[:8])
    return (
        f"TaskSchemaCard(version={schema_card.version}, "
        f"description={schema_card.task_family_description}, "
        f"variables=[{variables}], constraints=[{constraints}])"
    )


def _infer_schema_parts(visible_text: str) -> dict[str, Any]:
    explicit_entities = _extract_entity_mentions(visible_text)
    explicit_variables = _extract_variable_mentions(visible_text)
    examples = [{"text": item} for item in (explicit_entities[:5] + explicit_variables[:5])]
    entity_types = [
        {
            "name": "visible_subject",
            "description": "A subject explicitly named in task instructions, actions, or observations.",
            "examples_from_trace": explicit_entities[:8],
        }
    ]
    if not explicit_entities:
        entity_types[0]["examples_from_trace"] = []

    state_variables = [
        {
            "name": variable,
            "applies_to": "visible_subject",
            "why_future_relevant": "The value was visible in prior trace and may affect future decisions.",
            "examples_from_trace": [variable],
        }
        for variable in explicit_variables
    ]
    for fallback in ["decision", "status", "constraint", "evidence"]:
        if fallback not in explicit_variables:
            state_variables.append(
                {
                    "name": fallback,
                    "applies_to": "visible_subject",
                    "why_future_relevant": "Generic execution-state field used across visible traces.",
                    "examples_from_trace": [],
                }
            )

    return {
        "task_family_description": "Execution-state schema induced from visible instructions and trace.",
        "entity_types": entity_types,
        "state_variables": state_variables,
        "constraint_types": [
            {
                "name": "active_constraint",
                "description": "A visible rule, requirement, preference, incompatibility, or condition to preserve.",
                "trigger_patterns_visible_in_task": [],
                "examples_from_trace": _extract_constraint_sentences(visible_text)[:5],
            }
        ],
        "dependency_types": [
            {
                "name": "state_dependency",
                "source": "prior visible state",
                "target": "future decision or derived state",
                "description": "A prior state that future action depends on or is derived from.",
            }
        ],
        "action_outcome_types": [
            {"name": "visible_outcome", "description": "Observed result, acceptance, failure, or feedback."}
        ],
        "status_rules": GENERIC_STATUS_RULES.copy(),
        "provenance_requirements": ["source_event_ids", "source_span_ids", "short_evidence"],
        "update_rules": [
            {
                "condition": "A newer valid final state for the same subject and variable is observed.",
                "state_update": "Keep the newest final state active and retain older conflicting evidence.",
            }
        ],
        "examples_from_trace": examples,
        "confidence": 0.6 if visible_text.strip() else 0.45,
    }


def _extract_entity_mentions(text: str) -> list[str]:
    candidates = []
    patterns = [
        r"\b(?:option|candidate|choice|item|result|answer|plan|case|entity)\s+[A-Z0-9][A-Za-z0-9_-]*\b",
        r"\b[A-Z][A-Za-z0-9_-]{1,30}\b",
    ]
    for pattern in patterns:
        candidates.extend(match.group(0).strip() for match in re.finditer(pattern, text))
    stop = {"Action", "Observation", "Reward", "Feedback", "User", "Assistant", "The", "This"}
    return _dedupe_strings([item for item in candidates if item not in stop])[:20]


def _extract_variable_mentions(text: str) -> list[str]:
    variables = []
    for match in re.finditer(r"\b([a-z][a-z0-9_-]{1,30})\s*[:=]\s*([^\n,;]+)", text):
        variables.append(match.group(1).lower())
    for phrase in ["score", "id", "attribute", "status", "decision", "constraint", "reward", "feedback"]:
        if re.search(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE):
            variables.append(phrase)
    return _dedupe_strings(variables)[:20]


def _extract_constraint_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [
        sentence.strip()
        for sentence in sentences
        if re.search(r"\b(must|should|required|cannot|avoid|prefer|compatible|constraint)\b", sentence, re.I)
    ]


def _merge_inferred_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key in ["entity_types", "state_variables", "constraint_types", "dependency_types", "update_rules"]:
        merged[key] = _dedupe_named(base.get(key, []), overlay.get(key, []))
    for key in ["status_rules"]:
        merged[key] = {**base.get(key, {}), **overlay.get(key, {})}
    for key in ["provenance_requirements"]:
        merged[key] = _dedupe_strings(base.get(key, []) + overlay.get(key, []))
    if overlay.get("task_family_description"):
        merged["task_family_description"] = str(overlay["task_family_description"])
    if overlay.get("confidence") is not None:
        merged["confidence"] = max(float(base.get("confidence", 0)), float(overlay.get("confidence", 0)))
    return merged


def _schema_to_dict(schema: TaskSchemaCard | dict[str, Any]) -> dict[str, Any]:
    if isinstance(schema, TaskSchemaCard):
        return {
            "task_family_description": schema.task_family_description,
            "entity_types": schema.entity_types,
            "state_variables": schema.state_variables,
            "constraint_types": schema.constraint_types,
            "dependency_types": schema.dependency_types,
            "action_outcome_types": schema.action_outcome_types,
            "status_rules": schema.status_rules,
            "provenance_requirements": schema.provenance_requirements,
            "update_rules": schema.update_rules,
            "examples_from_trace": schema.examples_from_trace,
            "confidence": schema.confidence,
        }
    return schema


def _dedupe_named(first: list[dict[str, Any]], second: list[dict[str, Any]], key: str = "name") -> list[dict[str, Any]]:
    by_name: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for item in first + list(second):
        if not isinstance(item, dict):
            continue
        name = str(item.get(key) or item.get("name") or item).strip().lower()
        if not name:
            continue
        if name not in by_name:
            by_name[name] = dict(item)
        else:
            by_name[name] = {**by_name[name], **item}
    return list(by_name.values())


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _dedupe_examples(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = repr(sorted(item.items()))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _prefer_longer(left: str, right: str) -> str:
    return right if len(right.strip()) > len(left.strip()) else left
