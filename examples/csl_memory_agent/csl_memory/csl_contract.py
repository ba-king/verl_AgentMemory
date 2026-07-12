from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Any

from .csl_schema import TaskContractCard, new_id, utc_now


DEFAULT_VALUES_TO_PRESERVE = ["ids", "names", "numbers", "dates", "formulas", "titles", "labels"]


def compile_task_contract(
    *,
    task_id: str,
    visible_task_prompt: str | None = None,
    tool_description: str | None = None,
    existing_contract: TaskContractCard | None = None,
    llm_json: dict[str, Any] | None = None,
) -> TaskContractCard:
    prompt = visible_task_prompt or ""
    tools = tool_description or ""
    source_text = f"{prompt}\n\n{tools}".strip()
    source_hash = _source_hash(source_text)
    if existing_contract is not None and existing_contract.source_hash == source_hash:
        return existing_contract

    inferred = _infer_contract(source_text)
    if llm_json:
        inferred = _merge_contract_dicts(inferred, llm_json)

    now = utc_now()
    new_contract = TaskContractCard(
        contract_id=existing_contract.contract_id if existing_contract else new_id("contract"),
        task_id=task_id,
        version=(existing_contract.version + 1) if existing_contract else 1,
        source_hash=source_hash,
        task_role_summary=inferred["task_role_summary"],
        allowed_action_patterns=inferred["allowed_action_patterns"],
        irreversible_action_patterns=inferred["irreversible_action_patterns"],
        final_action_patterns=inferred["final_action_patterns"],
        grounding_rules=inferred["grounding_rules"],
        evidence_requirements=inferred["evidence_requirements"],
        invariant_constraints=inferred["invariant_constraints"],
        completion_conditions=inferred["completion_conditions"],
        forbidden_shortcuts=inferred["forbidden_shortcuts"],
        state_commitment_signals=inferred["state_commitment_signals"],
        values_to_preserve_exactly=inferred["values_to_preserve_exactly"],
        uncertainty_rules=inferred["uncertainty_rules"],
        token_policy=inferred["token_policy"],
        confidence=float(inferred["confidence"]),
        created_at=existing_contract.created_at if existing_contract else now,
        updated_at=now if existing_contract else None,
    )
    return new_contract


def merge_task_contract(existing: TaskContractCard, new: TaskContractCard | dict[str, Any]) -> TaskContractCard:
    new_data = _contract_to_dict(new)
    merged = TaskContractCard(
        contract_id=existing.contract_id,
        task_id=existing.task_id,
        version=existing.version + 1,
        source_hash=str(new_data.get("source_hash") or existing.source_hash),
        task_role_summary=_prefer_longer(existing.task_role_summary, str(new_data.get("task_role_summary", ""))),
        allowed_action_patterns=_dedupe_rule_dicts(existing.allowed_action_patterns, new_data.get("allowed_action_patterns", [])),
        irreversible_action_patterns=_dedupe_rule_dicts(
            existing.irreversible_action_patterns,
            new_data.get("irreversible_action_patterns", []),
        ),
        final_action_patterns=_dedupe_rule_dicts(existing.final_action_patterns, new_data.get("final_action_patterns", [])),
        grounding_rules=_dedupe_rule_dicts(existing.grounding_rules, new_data.get("grounding_rules", [])),
        evidence_requirements=_dedupe_rule_dicts(
            existing.evidence_requirements,
            new_data.get("evidence_requirements", []),
        ),
        invariant_constraints=_dedupe_rule_dicts(
            existing.invariant_constraints,
            new_data.get("invariant_constraints", []),
        ),
        completion_conditions=_dedupe_rule_dicts(
            existing.completion_conditions,
            new_data.get("completion_conditions", []),
        ),
        forbidden_shortcuts=_dedupe_rule_dicts(existing.forbidden_shortcuts, new_data.get("forbidden_shortcuts", [])),
        state_commitment_signals=_dedupe_rule_dicts(
            existing.state_commitment_signals,
            new_data.get("state_commitment_signals", []),
        ),
        values_to_preserve_exactly=_dedupe_strings(
            existing.values_to_preserve_exactly + list(new_data.get("values_to_preserve_exactly", []))
        ),
        uncertainty_rules=_dedupe_rule_dicts(existing.uncertainty_rules, new_data.get("uncertainty_rules", [])),
        token_policy={**existing.token_policy, **new_data.get("token_policy", {})},
        confidence=max(existing.confidence, float(new_data.get("confidence", 0))),
        created_at=existing.created_at,
        updated_at=utc_now(),
    )
    return merged


def summarize_task_contract(contract: TaskContractCard | None, max_chars: int = 1200) -> str:
    if contract is None:
        return "None"
    rules = []
    for item in contract.grounding_rules[:4]:
        rules.append(item.get("rule") or item.get("what_counts_as_grounding") or str(item))
    for item in contract.invariant_constraints[:4]:
        rules.append(item.get("constraint") or item.get("rule") or str(item))
    for item in contract.completion_conditions[:3]:
        rules.append(item.get("condition") or str(item))
    text = (
        f"TaskContractCard(version={contract.version}, summary={contract.task_role_summary}, "
        f"rules={rules}, final_patterns={contract.final_action_patterns[:3]}, "
        f"irreversible_patterns={contract.irreversible_action_patterns[:3]})"
    )
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def relevant_contract_rules(contract: TaskContractCard | None, query: str, limit: int = 8) -> list[dict[str, Any]]:
    if contract is None:
        return []
    query_terms = _tokens(query)
    candidates: list[dict[str, Any]] = []
    for group, items in [
        ("grounding", contract.grounding_rules),
        ("evidence", contract.evidence_requirements),
        ("invariant", contract.invariant_constraints),
        ("completion", contract.completion_conditions),
        ("forbidden", contract.forbidden_shortcuts),
        ("final", contract.final_action_patterns),
        ("irreversible", contract.irreversible_action_patterns),
        ("uncertainty", contract.uncertainty_rules),
    ]:
        for item in items:
            text = " ".join(str(value) for value in item.values())
            score = len(query_terms & _tokens(text))
            if group in {"grounding", "invariant", "completion"}:
                score += 1
            rule = dict(item)
            rule.setdefault("rule_type", group)
            rule["score"] = score
            candidates.append(rule)
    ranked = sorted(candidates, key=lambda item: item.get("score", 0), reverse=True)
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ranked:
        primary = _primary_rule_text(item)
        key = re.sub(r"[^a-z0-9]+", " ", primary.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
        if len(deduplicated) >= limit:
            break
    return deduplicated


def _infer_contract(text: str) -> dict[str, Any]:
    sentences = _sentences(text)
    action_patterns = _extract_action_patterns(text)
    final_patterns = _rule_items(
        sentences,
        r"\b(finish|final|answer|submit|buy|purchase|book|select|commit|complete|done)\b",
        "final_action",
    )
    irreversible_patterns = _rule_items(
        sentences,
        r"\b(buy|purchase|book|submit|finish|final|irreversible|cannot undo)\b",
        "irreversible_action",
    )
    grounding = _rule_items(
        sentences,
        r"\b(search|read|tool|returned|evidence|document|id|observation|verify|support)\b",
        "grounding_rule",
    )
    evidence = _rule_items(
        sentences,
        r"\b(evidence|document|source|id|observation|returned|support|exact)\b",
        "evidence_requirement",
    )
    constraints = _rule_items(
        sentences,
        r"\b(must|should|required|cannot|do not|never|only|exact|budget|compatible|valid|avoid)\b",
        "invariant_constraint",
    )
    forbidden = _rule_items(
        sentences,
        r"\b(do not|never|cannot|avoid|without|forbidden|not invent|do not fabricate)\b",
        "forbidden_shortcut",
    )
    completion = _rule_items(
        sentences,
        r"\b(before|finish|final|complete|done|when every|only when)\b",
        "completion_condition",
    )
    uncertainty = _rule_items(
        sentences,
        r"\b(uncertain|if no|if none|missing|ambiguous|whenever uncertain)\b",
        "uncertainty_rule",
    )
    commitment = [
        {"signal": signal, "meaning": meaning}
        for signal, meaning in [
            ("accepted", "final"),
            ("selected", "final"),
            ("booked", "final"),
            ("purchased", "final"),
            ("Finish", "final"),
            ("failed", "failed"),
            ("rejected", "rejected"),
            ("invalid", "failed"),
        ]
        if re.search(rf"\b{re.escape(signal)}\b", text, re.I) or signal in {"accepted", "failed", "rejected", "invalid"}
    ]
    return {
        "task_role_summary": _compact(sentences[0] if sentences else "Visible task contract compiled from public instructions.", 220),
        "allowed_action_patterns": action_patterns,
        "irreversible_action_patterns": irreversible_patterns,
        "final_action_patterns": final_patterns,
        "grounding_rules": grounding,
        "evidence_requirements": evidence,
        "invariant_constraints": constraints,
        "completion_conditions": completion,
        "forbidden_shortcuts": forbidden,
        "state_commitment_signals": commitment,
        "values_to_preserve_exactly": DEFAULT_VALUES_TO_PRESERVE.copy(),
        "uncertainty_rules": uncertainty
        or [{"condition": "needed state is missing or ambiguous", "required_behavior": "treat it as uncertain and gather evidence"}],
        "token_policy": {
            "cache_contract": True,
            "inject_only_relevant_rules": True,
            "prefer_pointers_for_evidence": True,
        },
        "confidence": 0.65 if text.strip() else 0.45,
    }


def _extract_action_patterns(text: str) -> list[dict[str, Any]]:
    patterns = []
    for action in sorted(set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\[[^\]]*\]", text))):
        patterns.append({"pattern": action, "description": "Visible action syntax from task/tool text", "source_quote_or_summary": action})
    if not patterns:
        for word in ["search", "read", "click", "select", "finish", "answer", "book", "buy", "submit"]:
            if re.search(rf"\b{word}\b", text, re.I):
                patterns.append(
                    {
                        "pattern": word,
                        "description": "Visible action verb from task/tool text",
                        "source_quote_or_summary": word,
                    }
                )
    return patterns[:12]


def _rule_items(sentences: list[str], pattern: str, kind: str) -> list[dict[str, Any]]:
    items = []
    for sentence in sentences:
        if not re.search(pattern, sentence, re.I):
            continue
        if kind == "final_action":
            items.append({"pattern": _compact(sentence, 180), "completion_meaning": "visible final or committed action", "preconditions": []})
        elif kind == "irreversible_action":
            items.append({"pattern": _compact(sentence, 180), "why_irreversible": "visible text marks this as final or committed", "preconditions": []})
        elif kind == "grounding_rule":
            items.append({"rule": _compact(sentence, 220), "what_counts_as_grounding": "visible observation, returned id, tool result, or document evidence", "violation_example": "using unsupported memory as active fact"})
        elif kind == "evidence_requirement":
            items.append({"requirement": _compact(sentence, 220), "required_evidence_type": "visible provenance", "when_needed": "before using the state in a decision"})
        elif kind == "invariant_constraint":
            items.append({"constraint": _compact(sentence, 220), "scope": "subtask", "priority": "hard" if re.search(r"\b(must|cannot|never|only|required)\b", sentence, re.I) else "soft"})
        elif kind == "completion_condition":
            items.append({"condition": _compact(sentence, 220), "must_check_before_finish": bool(re.search(r"\b(before|only|must|required)\b", sentence, re.I))})
        elif kind == "forbidden_shortcut":
            items.append({"shortcut": _compact(sentence, 220), "why_forbidden": "visible task text disallows this behavior"})
        elif kind == "uncertainty_rule":
            items.append({"condition": _compact(sentence, 220), "required_behavior": "gather evidence or mark state uncertain"})
    return items[:10]


def _merge_contract_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key in [
        "allowed_action_patterns",
        "irreversible_action_patterns",
        "final_action_patterns",
        "grounding_rules",
        "evidence_requirements",
        "invariant_constraints",
        "completion_conditions",
        "forbidden_shortcuts",
        "state_commitment_signals",
        "uncertainty_rules",
    ]:
        merged[key] = _dedupe_rule_dicts(base.get(key, []), overlay.get(key, []))
    merged["values_to_preserve_exactly"] = _dedupe_strings(
        base.get("values_to_preserve_exactly", []) + overlay.get("values_to_preserve_exactly", [])
    )
    merged["token_policy"] = {**base.get("token_policy", {}), **overlay.get("token_policy", {})}
    if overlay.get("task_role_summary"):
        merged["task_role_summary"] = str(overlay["task_role_summary"])
    if overlay.get("confidence") is not None:
        merged["confidence"] = max(float(base.get("confidence", 0)), float(overlay.get("confidence", 0)))
    return merged


def _contract_to_dict(contract: TaskContractCard | dict[str, Any]) -> dict[str, Any]:
    if isinstance(contract, dict):
        return contract
    return {
        "source_hash": contract.source_hash,
        "task_role_summary": contract.task_role_summary,
        "allowed_action_patterns": contract.allowed_action_patterns,
        "irreversible_action_patterns": contract.irreversible_action_patterns,
        "final_action_patterns": contract.final_action_patterns,
        "grounding_rules": contract.grounding_rules,
        "evidence_requirements": contract.evidence_requirements,
        "invariant_constraints": contract.invariant_constraints,
        "completion_conditions": contract.completion_conditions,
        "forbidden_shortcuts": contract.forbidden_shortcuts,
        "state_commitment_signals": contract.state_commitment_signals,
        "values_to_preserve_exactly": contract.values_to_preserve_exactly,
        "uncertainty_rules": contract.uncertainty_rules,
        "token_policy": contract.token_policy,
        "confidence": contract.confidence,
    }


def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _dedupe_rule_dicts(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_text: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for item in first + list(second):
        if not isinstance(item, dict):
            continue
        text = " ".join(str(value) for value in item.values()).strip().lower()
        if not text:
            continue
        by_text.setdefault(text, dict(item))
    return list(by_text.values())[:20]


def _dedupe_strings(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        text = str(item).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _sentences(text: str) -> list[str]:
    return [item.strip(" -\t") for item in re.split(r"(?<=[.!?])\s+|\n+", text) if item.strip(" -\t")]


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 1}


def _primary_rule_text(item: dict[str, Any]) -> str:
    for key in ["rule", "constraint", "condition", "requirement", "shortcut", "pattern"]:
        if item.get(key):
            return str(item[key])
    return " ".join(str(value) for key, value in item.items() if key not in {"score", "rule_type"})


def _prefer_longer(left: str, right: str) -> str:
    return right if len(right.strip()) > len(left.strip()) else left


def _compact(text: str, limit: int) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3].rstrip() + "..."
