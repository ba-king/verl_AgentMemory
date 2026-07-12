from __future__ import annotations

import json
import re
from typing import Any

from .csl_audit import parse_decision_certificate
from .csl_schema import TaskContractCard


FINAL_ACTION_RE = re.compile(
    r"\b(finish|final|submit|answer|buy|purchase|book|select|commit|done|complete)\b|\bFinish\s*\[",
    re.I,
)


def is_irreversible_or_final(action: str, contract: TaskContractCard | None = None) -> bool:
    if FINAL_ACTION_RE.search(action):
        return True
    if contract is None:
        return False
    haystack = action.lower()
    for item in contract.final_action_patterns + contract.irreversible_action_patterns:
        pattern = str(item.get("pattern") or item.get("rule") or item.get("condition") or "").lower()
        if not pattern:
            continue
        tokens = [token for token in re.findall(r"[a-z0-9_]+", pattern) if len(token) > 2]
        if tokens and any(token in haystack for token in tokens[:6]):
            return True
    return False


def deterministic_guard_precheck(
    *,
    action: str,
    contract: TaskContractCard | None,
    state_packet: str,
) -> dict[str, Any]:
    if not is_irreversible_or_final(action, contract):
        return {"allow": True, "violations": [], "repair_instruction": None, "needs_backfill": False, "missing_evidence": []}

    violations = []
    missing_evidence = []
    packet_lower = state_packet.lower()
    certificate_validation = _validate_decision_certificate(action)

    if re.match(r"\s*select\s*\[", action, flags=re.I) and certificate_validation is None:
        missing_evidence.append("DecisionCertificate for the proposed selection")
        violations.append(
            {
                "constraint": "A final selection requires a complete DecisionCertificate.",
                "why_violated": "The proposed select action did not include a DecisionCertificate.",
                "evidence": "No DecisionCertificate marker was present in the proposed action.",
            }
        )
    elif (
        re.match(r"\s*(?:click\[\s*buy now\s*\]|buy\[|purchase\[|checkout\[)", action, flags=re.I)
        and certificate_validation is None
        and ("candidate evidence ledger" in packet_lower or "evaluate all" in packet_lower)
    ):
        missing_evidence.append("DecisionCertificate covering every listed candidate before purchase")
        violations.append(
            {
                "constraint": "A comparison-based purchase requires a complete DecisionCertificate.",
                "why_violated": "The proposed purchase had no certificate covering all listed candidates.",
                "evidence": "No DecisionCertificate marker was present in the proposed action.",
            }
        )
    elif certificate_validation is not None and not certificate_validation["valid"]:
        for reason in certificate_validation["errors"]:
            violations.append(
                {
                    "constraint": "DecisionCertificate must be complete, grounded, and optimal.",
                    "why_violated": reason,
                    "evidence": "Proposed action DecisionCertificate",
                }
            )

    if "missing or uncertain information" in packet_lower and re.search(r"\bmissing|uncertain|unresolved\b", packet_lower):
        violations.append(
            {
                "constraint": "Resolve missing or uncertain memory before final action.",
                "why_violated": "The DART-CSL packet reports missing or uncertain fields.",
                "evidence": "StatePacket missing/uncertain section",
            }
        )

    if contract is not None:
        for requirement in contract.evidence_requirements[:8]:
            req_text = str(requirement.get("requirement") or requirement)
            if re.search(r"\b(evidence|document|source|support|returned|id|observation)\b", req_text, re.I):
                has_evidence_pointer = bool(re.search(r"evidence pointers:\s*\n\s*-", packet_lower))
                has_provenance = "source_event" in packet_lower or "provenance" in packet_lower
                if not has_evidence_pointer and not has_provenance:
                    missing_evidence.append(req_text)
        for shortcut in contract.forbidden_shortcuts[:8]:
            shortcut_text = str(shortcut.get("shortcut") or shortcut)
            if _roughly_matches(action, shortcut_text):
                violations.append(
                    {
                        "constraint": shortcut_text,
                        "why_violated": "The proposed action resembles a visible forbidden shortcut.",
                        "evidence": shortcut_text,
                    }
                )

    if missing_evidence:
        violations.append(
            {
                "constraint": "Final or irreversible action requires visible supporting evidence.",
                "why_violated": "No evidence pointer was found in the current StatePacket.",
                "evidence": "; ".join(missing_evidence[:3]),
            }
        )

    if violations:
        result = {
            "allow": False,
            "violations": violations,
            "repair_instruction": "Repair the decision certificate or gather the missing evidence, then rebuild the state packet before the final action.",
            "needs_backfill": bool(missing_evidence),
            "missing_evidence": missing_evidence,
        }
        if certificate_validation is not None:
            result["decision_certificate_validation"] = certificate_validation
        return result
    result = {"allow": True, "violations": [], "repair_instruction": None, "needs_backfill": False, "missing_evidence": []}
    if certificate_validation is not None:
        result["decision_certificate_validation"] = certificate_validation
    return result


def llm_hard_action_guard(
    *,
    question: str,
    contract: TaskContractCard | None,
    state_packet: str,
    action: str,
    llm: Any,
) -> dict[str, Any]:
    deterministic = deterministic_guard_precheck(action=action, contract=contract, state_packet=state_packet)
    if deterministic["allow"] is False:
        return deterministic
    if str(getattr(llm, "provider", "mock")).lower() == "mock":
        return deterministic

    prompt = f"""\
You are the DART-CSL memory hard-action guard.

Audit the proposed final/irreversible action using only:
1. the visible current question,
2. the visible task contract,
3. the current DART-CSL state packet,
4. the proposed action and its DecisionCertificate, when present.

Follow this order exactly:
1. Identify hard constraints and completion conditions.
2. Check that every expected candidate was evaluated and bound to its own evidence.
3. Filter infeasible candidates before applying preferences.
4. Verify that the selected candidate is the argmax/argmin required by the primary objective.
5. Apply tie breakers only after the primary objective.
6. Verify budget, provenance, and prior-state dependencies.

Do not confuse ids or values belonging to different candidates. Do not treat the
mere presence of comparison text as proof that the comparison is complete. If
evidence is missing or backfill is needed, allow must be false.

When the DecisionCertificate contains a deterministic GroundingReceipt with
validated_against_environment=true, treat its observed_facts as visible grounded
evidence. The receipt means the controller matched those fields against the
actual search observations before calling this guard. Do not report a receipted
field as missing merely because the token-bounded StatePacket omitted a duplicate
evidence pointer.

Return exactly one JSON object:
{{
  "allow": true or false,
  "violations": [
    {{"constraint": "short rule", "why_violated": "short reason", "evidence": "visible evidence"}}
  ],
  "repair_instruction": null or "short repair instruction",
  "needs_backfill": true or false,
  "missing_evidence": ["field or evidence needed"],
  "decision_audit": {{
    "hard_constraints_satisfied": true or false,
    "all_candidates_evaluated": true or false,
    "selected_is_objective_optimal": true or false,
    "budget_satisfied": true or false
  }}
}}

Current question:
{question}

Proposed action:
{action}

Deterministic certificate precheck:
{json.dumps(deterministic.get("decision_certificate_validation"), ensure_ascii=False, indent=2)}

Visible task contract:
{_contract_for_prompt(contract)}

DART-CSL state packet:
{state_packet}
"""
    try:
        llm_result = llm.call_json_prompt(prompt)
    except Exception as exc:  # pragma: no cover - external API resilience
        result = dict(deterministic)
        result["memory_llm_used"] = False
        result["memory_llm_error"] = str(exc)
        return result

    result = _coerce_guard_json(llm_result)
    result["memory_llm_used"] = True
    if deterministic.get("decision_certificate_validation") is not None:
        result["decision_certificate_validation"] = deterministic["decision_certificate_validation"]
    return result


def _roughly_matches(action: str, rule_text: str) -> bool:
    action_command = re.split(r"DecisionCertificate\s*:", action, maxsplit=1, flags=re.I)[0]
    action_terms = {token for token in re.findall(r"[a-z0-9_]+", action_command.lower()) if len(token) > 3}
    rule_terms = {token for token in re.findall(r"[a-z0-9_]+", rule_text.lower()) if len(token) > 3}
    if not action_terms or not rule_terms:
        return False
    return len(action_terms & rule_terms) >= min(2, len(rule_terms))


def _contract_for_prompt(contract: TaskContractCard | None) -> str:
    if contract is None:
        return "None"
    data = {
        "task_role_summary": contract.task_role_summary,
        "grounding_rules": contract.grounding_rules[:12],
        "invariant_constraints": contract.invariant_constraints[:12],
        "completion_conditions": contract.completion_conditions[:8],
        "evidence_requirements": contract.evidence_requirements[:8],
        "forbidden_shortcuts": contract.forbidden_shortcuts[:8],
        "allowed_action_patterns": contract.allowed_action_patterns[:8],
        "final_action_patterns": contract.final_action_patterns[:8],
        "irreversible_action_patterns": contract.irreversible_action_patterns[:8],
        "uncertainty_rules": contract.uncertainty_rules[:8],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def _coerce_guard_json(data: dict[str, Any]) -> dict[str, Any]:
    raw_allow = data.get("allow")
    if isinstance(raw_allow, bool):
        allow = raw_allow
    elif isinstance(raw_allow, str):
        allow = raw_allow.strip().lower() in {"true", "yes", "allow", "allowed"}
    else:
        allow = bool(raw_allow)
    violations = data.get("violations")
    if not isinstance(violations, list):
        violations = []
    missing = data.get("missing_evidence")
    if not isinstance(missing, list):
        missing = []
    repair = data.get("repair_instruction")
    if repair is not None:
        repair = str(repair)
    needs_backfill = bool(data.get("needs_backfill"))
    normalized_fail_closed = bool(allow and (violations or missing or needs_backfill))
    if normalized_fail_closed:
        allow = False
        if repair is None:
            repair = "Resolve all reported violations and missing evidence before retrying the final action."
    result = {
        "allow": allow,
        "violations": violations,
        "repair_instruction": repair,
        "needs_backfill": needs_backfill,
        "missing_evidence": missing,
    }
    if isinstance(data.get("decision_audit"), dict):
        result["decision_audit"] = data["decision_audit"]
    if normalized_fail_closed:
        result["guard_output_normalized_fail_closed"] = True
    return result


def _validate_decision_certificate(action: str) -> dict[str, Any] | None:
    present, certificate, parse_error = parse_decision_certificate(action)
    if not present:
        return None
    if certificate is None:
        return {"present": True, "valid": False, "errors": [parse_error or "DecisionCertificate could not be parsed."]}

    errors: list[str] = []
    expected = _string_list(certificate.get("expected_option_ids"))
    evaluated = _string_list(certificate.get("evaluated_option_ids"))
    selected = str(certificate.get("selected_option_id") or "").strip()
    candidates = certificate.get("candidates")
    objective = certificate.get("objective")
    uncertainties = certificate.get("uncertainties")

    if not expected:
        errors.append("expected_option_ids is empty.")
    if len(expected) != len(set(expected)):
        errors.append("expected_option_ids contains duplicates.")
    if set(evaluated) != set(expected) or len(evaluated) != len(expected):
        errors.append("evaluated_option_ids does not exactly cover expected_option_ids.")

    action_selection = re.match(r"\s*select\s*\[\s*([^\]]+)\s*\]", action, flags=re.I)
    if action_selection is not None and selected.upper() != action_selection.group(1).strip().upper():
        errors.append("selected_option_id does not match the select action.")

    if not isinstance(candidates, list) or not candidates:
        errors.append("candidates must contain one row for every expected option.")
        candidate_rows: list[dict[str, Any]] = []
    else:
        candidate_rows = [row for row in candidates if isinstance(row, dict)]
        candidate_ids = [str(row.get("option_id") or "").strip() for row in candidate_rows]
        if set(candidate_ids) != set(expected) or len(candidate_ids) != len(expected):
            errors.append("candidate rows do not exactly cover expected_option_ids.")

    selected_row = next((row for row in candidate_rows if str(row.get("option_id") or "").strip() == selected), None)
    if selected_row is None:
        errors.append("selected_option_id is missing from candidates.")
    elif selected_row.get("eligible") is not True:
        errors.append("The selected candidate is not marked eligible.")

    if not isinstance(objective, dict):
        errors.append("objective must specify metric and direction.")
        direction = ""
        metric = ""
    else:
        direction = str(objective.get("direction") or "").strip().lower()
        metric = str(objective.get("metric") or "").strip()
        if direction not in {"maximize", "minimize"}:
            errors.append("objective.direction must be maximize or minimize.")
        if not metric:
            errors.append("objective.metric is empty.")

    eligible_values: list[tuple[str, float]] = []
    for row in candidate_rows:
        if row.get("eligible") is not True:
            continue
        option_id = str(row.get("option_id") or "").strip()
        value = row.get("objective_value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"Eligible candidate {option_id} has no numeric objective_value.")
            continue
        eligible_values.append((option_id, float(value)))
    if eligible_values and selected:
        optimum = max(value for _option_id, value in eligible_values) if direction == "maximize" else min(
            value for _option_id, value in eligible_values
        )
        selected_values = [value for option_id, value in eligible_values if option_id == selected]
        if selected_values and abs(selected_values[0] - optimum) > 1e-9:
            errors.append(
                f"Selected candidate is not objective-optimal for {direction} {metric}: "
                f"selected={selected_values[0]}, optimum={optimum}."
            )

    if isinstance(uncertainties, list) and any(str(item).strip() for item in uncertainties):
        errors.append("DecisionCertificate contains unresolved uncertainties.")

    budget = certificate.get("budget")
    if isinstance(budget, dict):
        before = budget.get("before")
        after = budget.get("after")
        limit = budget.get("limit")
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in [before, after, limit]):
            if float(after) < float(before):
                errors.append("budget.after is lower than budget.before.")
            if float(after) > float(limit) + 1e-9:
                errors.append(f"budget.after={after} exceeds budget.limit={limit}.")
        else:
            errors.append("budget before, after, and limit must be numeric.")

    receipt = certificate.get("grounding_receipt")
    receipt_valid = False
    if isinstance(receipt, dict):
        receipt_ids = _string_list(receipt.get("validated_option_ids"))
        receipt_valid = bool(receipt.get("validated_against_environment")) and set(receipt_ids) == set(expected)
        if not receipt_valid:
            errors.append("GroundingReceipt does not validate every expected option.")

    return {
        "present": True,
        "valid": not errors,
        "errors": errors,
        "selected_option_id": selected,
        "expected_option_count": len(expected),
        "evaluated_option_count": len(evaluated),
        "eligible_option_count": len(eligible_values),
        "objective": {"metric": metric, "direction": direction},
        "grounding_receipt_valid": receipt_valid,
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
