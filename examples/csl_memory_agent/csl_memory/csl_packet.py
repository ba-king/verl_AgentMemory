from __future__ import annotations

from typing import Any

from .csl_lanes import lanes_to_dict
from .csl_schema import DecisionSignature, DecisionSupportCapsule, StateLaneResult, TaskContractCard, TaskSchemaCard, new_id


def build_state_packet(
    *,
    question: str,
    schema_card: TaskSchemaCard | None,
    state_need: dict[str, Any],
    retrieved: dict[str, StateLaneResult],
    missing_fields: list[dict[str, Any]],
    max_chars: int = 7000,
) -> str:
    lines = ["[CSL STATE PACKET]"]
    if schema_card is not None:
        lines.append(f"Induced schema: {schema_card.task_family_description} (v{schema_card.version})")
    lines.append(f"Current subtask: {_focus_excerpt(question, 700)}")

    lane_order = [
        ("decision", "Relevant final states"),
        ("constraint", "Active constraints"),
        ("dependency", "Dependencies and derived requirements"),
        ("negative", "Rejected, failed, invalid, or superseded states"),
        ("evidence", "Useful provenance"),
        ("recent", "Recent raw events"),
        ("uncertainty", "Missing or uncertain information"),
    ]
    for lane_key, title in lane_order:
        lane = retrieved.get(lane_key)
        if lane is None or not lane.entries:
            continue
        lines.append("")
        lines.append(f"{title}:")
        for entry in lane.entries[:8]:
            lines.append(f"- {_format_entry(entry)}")

    if missing_fields:
        lines.append("")
        lines.append("Missing fields to resolve:")
        for field in missing_fields[:8]:
            subject = field.get("subject", "unknown")
            variable = field.get("variable", "unknown")
            why = field.get("why_needed", "")
            lines.append(f"- {subject}.{variable}: {_clip(str(why), 240)}")

    must_have = state_need.get("priority", {}).get("must_have", [])
    if must_have:
        lines.append("")
        lines.append("Memory priorities:")
        for item in must_have[:6]:
            lines.append(f"- {_clip(str(item), 220)}")

    lines.append("")
    lines.append("Before-final-action checklist:")
    lines.append("1. Preserve final states unless visible evidence supersedes them.")
    lines.append("2. Check active constraints and dependencies before committing.")
    lines.append("3. Do not reuse rejected or failed states as active answers.")
    lines.append("4. Treat missing or uncertain fields as unresolved, not assumed.")

    packet = "\n".join(lines)
    if len(packet) > max_chars:
        packet = packet[: max_chars - 32].rstrip() + "\n[truncated for packet budget]"
    return packet


def build_decision_support_capsule(
    *,
    task_id: str,
    decision_signature: DecisionSignature,
    contract: TaskContractCard | None,
    retrieved: dict[str, StateLaneResult],
    missing_fields: list[dict[str, Any]],
    max_items: dict[str, int] | None = None,
) -> DecisionSupportCapsule:
    max_items = max_items or {}
    contract_entries = retrieved.get("contract", StateLaneResult("Contract Lane", [])).entries[
        : int(max_items.get("max_contract_rules", 8))
    ]
    decision_entries = retrieved.get("decision", StateLaneResult("Decision Lane", [])).entries[
        : int(max_items.get("max_final_states", 12))
    ]
    constraint_entries = retrieved.get("constraint", StateLaneResult("Constraint Lane", [])).entries[
        : int(max_items.get("max_constraints", 12))
    ]
    dependency_entries = retrieved.get("dependency", StateLaneResult("Dependency Lane", [])).entries[
        : int(max_items.get("max_dependencies", 8))
    ]
    negative_entries = retrieved.get("negative", StateLaneResult("Negative Lane", [])).entries[
        : int(max_items.get("max_negative_states", 8))
    ]
    evidence_entries = retrieved.get("evidence", StateLaneResult("Evidence Lane", [])).entries[
        : int(max_items.get("max_evidence_pointers", 8))
    ]
    uncertainty_entries = retrieved.get("uncertainty", StateLaneResult("Uncertainty Lane", [])).entries[:8]
    checklist = [
        "Check relevant public task rules before committing.",
        "Preserve final states unless visible evidence supersedes them.",
        "Check active constraints and dependency ancestors.",
        "Do not reuse rejected, failed, invalid, or blocked states.",
        "Resolve high-priority missing or uncertain fields before final action.",
    ]
    if contract is not None and contract.completion_conditions:
        for condition in contract.completion_conditions[:3]:
            text = condition.get("condition") or str(condition)
            checklist.append(f"Completion condition: {text}")
    token_estimate = sum(
        lane.token_estimate
        for lane in retrieved.values()
    ) + sum(len(item) for item in checklist) // 4
    return DecisionSupportCapsule(
        capsule_id=new_id("capsule"),
        task_id=task_id,
        decision_id=decision_signature.decision_id,
        contract_rules=[{"rule": _entry_text(entry), "why_relevant": "matched current decision signature"} for entry in contract_entries],
        final_states=[
            {"state": _format_entry(entry), "why_relevant": "final or committed state to preserve", "evidence": entry.get("evidence")}
            for entry in decision_entries
        ],
        active_constraints=[
            {"constraint": _format_entry(entry), "scope": "current decision", "evidence": entry.get("evidence")}
            for entry in constraint_entries
        ],
        derived_requirements=[
            {"requirement": _format_entry(entry), "derived_from": entry.get("dependency_delta_ids", [])}
            for entry in dependency_entries
        ],
        dependencies=[
            {"needed_prior_state": _format_entry(entry), "source": entry.get("source_delta_ids", entry.get("delta_id"))}
            for entry in dependency_entries
        ],
        negative_states_to_avoid=[
            {"state": _format_entry(entry), "reason": entry.get("evidence") or "negative memory"}
            for entry in negative_entries
        ],
        uncertainty_items=[
            {"item": _format_entry(entry), "required_behavior": "treat as uncertain or gather evidence"}
            for entry in uncertainty_entries
        ],
        evidence_pointers=[
            {"pointer": str(entry.get("source_event_ids") or entry.get("provenance") or entry.get("delta_id")), "supports": _format_entry(entry)}
            for entry in evidence_entries
        ],
        missing_fields=missing_fields,
        before_final_action_checklist=checklist,
        token_estimate=token_estimate,
    )


def build_dart_state_packet(
    *,
    question: str,
    capsule: DecisionSupportCapsule,
    decision_signature: DecisionSignature,
    active_path: list[str] | None = None,
    candidate_ledger: list[str] | None = None,
    candidate_guidance: str | None = None,
    action_directive: dict[str, Any] | None = None,
    max_chars: int = 4800,
) -> str:
    lines = ["[DART-CSL STATE PACKET]"]
    lines.append(f"Current decision focus: {_focus_excerpt(question, 700)}")
    lines.append(f"Expected action type: {decision_signature.expected_action_type}")
    lines.append(f"Irreversible risk: {decision_signature.irreversible_risk}")
    required_directive = bool(
        action_directive
        and action_directive.get("mode") in {"required", "required_state_change"}
    )
    if required_directive:
        lines.append("")
        state_change_mode = action_directive.get("mode") == "required_state_change"
        lines.append(
            "Required state change (machine-validated before environment execution):"
            if state_change_mode
            else "Required evidence transition (machine-validated before environment execution):"
        )
        lines.append(f"- obligation: {action_directive.get('obligation_id') or 'current evidence gap'}")
        state_delta = action_directive.get("required_state_delta") or []
        if state_delta:
            lines.append(f"- evidence goal: {'; '.join(str(item) for item in state_delta)}")
        forbidden = action_directive.get("forbidden_actions") or []
        if forbidden:
            lines.append(f"- prohibited until the goal changes: {'; '.join(str(item) for item in forbidden)}")
        actions = []
        if action_directive.get("action"):
            actions.append(str(action_directive["action"]))
        actions.extend(
            str(item.get("action"))
            for item in action_directive.get("recovery_actions") or []
            if isinstance(item, dict) and item.get("action")
        )
        if actions:
            lines.append("- admissible recovery actions (choose any currently legal option):")
            lines.extend(f"  - {item}" for item in dict.fromkeys(actions))
        lines.append(f"- reason: {action_directive.get('reason')}")
        if action_directive.get("lease_id"):
            lines.append("- lease: current environment transition only; recompute after the next observation")
    if active_path:
        lines.append("")
        lines.append("Active execution path (recent transitions):")
        lines.extend(f"- {item}" for item in active_path)
    if candidate_ledger:
        lines.append("")
        lines.append(
            "Candidate evidence ledger (list price and detail-only fields retain their visible evidence level):"
        )
        lines.extend(f"- {item}" for item in candidate_ledger)
    if not required_directive and candidate_guidance:
        lines.append(
            "Non-binding decision support (verify legality against the final current page): "
            f"{candidate_guidance}"
        )
    _append_capsule_section(lines, "Relevant task rules", capsule.contract_rules, "rule")
    _append_capsule_section(lines, "Final states to preserve", capsule.final_states, "state")
    _append_capsule_section(lines, "Active constraints", capsule.active_constraints, "constraint")
    _append_capsule_section(lines, "Dependencies and derived requirements", capsule.derived_requirements, "requirement")
    _append_capsule_section(lines, "Do-not-reuse / negative states", capsule.negative_states_to_avoid, "state")
    _append_capsule_section(lines, "Evidence pointers", capsule.evidence_pointers, "pointer")
    if capsule.missing_fields or capsule.uncertainty_items:
        lines.append("")
        lines.append("Missing or uncertain information:")
        for item in capsule.missing_fields[:8]:
            lines.append(f"- {item.get('subject', 'unknown')}.{item.get('variable', 'unknown')}: {_clip(str(item.get('why_needed', 'needed for current decision')), 240)}")
        for item in capsule.uncertainty_items[:8]:
            lines.append(f"- {_clip(str(item.get('item')), 300)}")
    lines.append("")
    lines.append("Before-final-action checklist:")
    for idx, check in enumerate(capsule.before_final_action_checklist[:8], start=1):
        lines.append(f"{idx}. {_clip(check, 260)}")
    packet = "\n".join(lines)
    if len(packet) > max_chars:
        packet = packet[: max_chars - 34].rstrip() + "\n[truncated for DART-CSL budget]"
    return packet


def build_path_summary_packet(
    *,
    question: str,
    path_summaries: list[str],
    recent_events: list[str],
    failed_hints: list[str],
) -> str:
    lines = ["[PATH-SUMMARY EXECUTION STATE]", f"Current subtask: {_clip(question, 500)}", ""]
    lines.append("Current active path summary:")
    lines.extend(f"- {_clip(item, 500)}" for item in path_summaries[-8:])
    lines.append("")
    lines.append("Recent raw trace:")
    lines.extend(f"- {_clip(item, 500)}" for item in recent_events[-5:])
    lines.append("")
    lines.append("Execution hints / warnings:")
    lines.extend(f"- {_clip(item, 500)}" for item in failed_hints[-5:])
    return "\n".join(lines)


def build_soft_verifier(*, question: str, state_packet: str, schema_card: TaskSchemaCard | None) -> str:
    del question, state_packet, schema_card
    return (
        "[DART-CSL SOFT VERIFIER]\n"
        "Before final action or irreversible action, verify:\n"
        "1. The proposed action is consistent with final states in the CSL packet.\n"
        "2. All relevant public task rules, active constraints, and dependencies are satisfied.\n"
        "3. Rejected, failed, invalid, or superseded states are not reused as active facts.\n"
        "4. Missing or uncertain fields are resolved, backfilled, or explicitly treated as uncertain.\n"
        "5. Evidence pointers exist for states used in a final or irreversible action."
    )


def retrieved_debug_dict(retrieved: dict[str, StateLaneResult]) -> dict[str, Any]:
    return lanes_to_dict(retrieved)


def _format_entry(entry: dict[str, Any]) -> str:
    if "text" in entry:
        return _clip(str(entry["text"]), 500)
    subject = entry.get("subject", "unknown")
    variable = entry.get("variable") or entry.get("op") or "state"
    value = entry.get("value")
    status = entry.get("status", "unknown")
    confidence = entry.get("confidence")
    evidence = entry.get("evidence")
    base = f"{subject}.{variable} = {_clip(str(value), 320)} [{status}"
    if confidence is not None:
        base += f", conf={float(confidence):.2f}"
    base += "]"
    if evidence:
        base += f" evidence: {_clip(str(evidence), 240)}"
    return base


def _append_capsule_section(lines: list[str], title: str, entries: list[dict[str, Any]], primary_key: str) -> None:
    if not entries:
        return
    lines.append("")
    lines.append(f"{title}:")
    for entry in entries[:8]:
        main = entry.get(primary_key) or _entry_text(entry)
        detail = entry.get("why_relevant") or entry.get("evidence") or entry.get("reason") or entry.get("supports")
        text = _clip(str(main), 420)
        if detail:
            text += f" ({_clip(str(detail), 180)})"
        lines.append(f"- {text}")


def _entry_text(entry: dict[str, Any]) -> str:
    for key in ["rule", "constraint", "condition", "requirement", "pattern", "state", "value", "text"]:
        if entry.get(key):
            return str(entry[key])
    return str(entry)


def _clip(text: str, limit: int) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3].rstrip() + "..."


def _focus_excerpt(text: str, limit: int) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    head = min(180, max(80, limit // 4))
    tail = limit - head - 5
    return f"{stripped[:head].rstrip()} ... {stripped[-tail:].lstrip()}"
