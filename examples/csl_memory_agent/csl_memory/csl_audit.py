from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_POLICY_VERSION = "selection-proof-v1"


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_decision_certificate(action: str) -> tuple[bool, dict[str, Any] | None, str | None]:
    marker = "DecisionCertificate:"
    marker_index = action.lower().find(marker.lower())
    if marker_index < 0:
        return False, None, None
    raw = action[marker_index + len(marker) :].lstrip()
    try:
        certificate, _end = json.JSONDecoder().raw_decode(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        return True, None, f"DecisionCertificate is not valid JSON: {exc}"
    if not isinstance(certificate, dict):
        return True, None, "DecisionCertificate must be a JSON object."
    return True, certificate, None


def extract_decision_certificate(action: str) -> dict[str, Any] | None:
    _present, certificate, _error = parse_decision_certificate(action)
    return certificate


def certificate_hash(certificate_or_action: dict[str, Any] | str) -> str | None:
    certificate = (
        extract_decision_certificate(certificate_or_action)
        if isinstance(certificate_or_action, str)
        else certificate_or_action
    )
    if not isinstance(certificate, dict):
        return None
    return canonical_json_hash(_semantic_certificate(certificate))


def certificate_from_structured_event(parsed_event: dict[str, Any]) -> dict[str, Any] | None:
    structured = parsed_event.get("structured_event")
    if not isinstance(structured, dict):
        return None
    direct = structured.get("decision_certificate")
    if isinstance(direct, dict):
        return direct
    for entity in structured.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        properties = entity.get("properties")
        if isinstance(properties, dict) and isinstance(properties.get("decision_certificate"), dict):
            return properties["decision_certificate"]
    return None


def is_selection_action(action: str) -> bool:
    command = action.split("DecisionCertificate:", 1)[0].strip().lower()
    return command.startswith("select[")


class GuardAuditCache:
    """Persistent selection approvals that can authorize an unchanged commit."""

    def __init__(self, state_dir: str | Path, policy_version: str = DEFAULT_AUDIT_POLICY_VERSION):
        self.path = Path(state_dir) / "guard_audits.json"
        self.denial_path = Path(state_dir) / "guard_denials.json"
        self.policy_version = policy_version

    def reset(self) -> None:
        for path in (self.path, self.denial_path):
            if path.exists():
                path.unlink()

    def record_selection(
        self,
        *,
        certificate: dict[str, Any],
        state_version: str,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not _result_is_reusable(result, certificate):
            return None
        digest = certificate_hash(certificate)
        if digest is None:
            return None
        record = {
            "audit_id": f"audit_{digest[:16]}",
            "certificate_hash": digest,
            "selected_option_id": str(certificate.get("selected_option_id") or ""),
            "policy_version": self.policy_version,
            "status": "pending_selection_commit",
            "selection_state_version": state_version,
            "bound_state_version": None,
            "result": result,
            "created_at": _utc_now(),
            "bound_at": None,
        }
        records = [
            item
            for item in self._load()
            if not (
                item.get("certificate_hash") == digest
                and item.get("policy_version") == self.policy_version
            )
        ]
        records.append(record)
        self._save(records)
        return record

    def bind_selection_commit(
        self,
        *,
        certificate: dict[str, Any],
        state_version: str,
    ) -> dict[str, Any] | None:
        digest = certificate_hash(certificate)
        if digest is None:
            return None
        records = self._load()
        matched: dict[str, Any] | None = None
        for record in reversed(records):
            if record.get("certificate_hash") != digest:
                continue
            if record.get("policy_version") != self.policy_version:
                continue
            if record.get("status") != "pending_selection_commit":
                continue
            record["status"] = "bound"
            record["bound_state_version"] = state_version
            record["bound_at"] = _utc_now()
            matched = record
            break
        if matched is not None:
            self._save(records)
        return matched

    def lookup_commit(
        self,
        *,
        certificate: dict[str, Any],
        state_version: str,
    ) -> tuple[dict[str, Any] | None, str]:
        digest = certificate_hash(certificate)
        if digest is None:
            return None, "missing_certificate_hash"
        matching = [
            item
            for item in self._load()
            if item.get("certificate_hash") == digest
            and item.get("policy_version") == self.policy_version
        ]
        if not matching:
            return None, "no_selection_audit"
        record = matching[-1]
        if record.get("status") != "bound":
            return None, "selection_not_committed"
        if record.get("bound_state_version") != state_version:
            return None, "state_version_changed"
        return record, "hit"

    def record_denial(
        self,
        *,
        action: str,
        state_version: str,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if result.get("allow") is not False:
            return None
        digest = _guard_transition_hash(
            action=action,
            state_version=state_version,
            policy_version=self.policy_version,
        )
        record = {
            "audit_id": f"denial_{digest[:16]}",
            "transition_hash": digest,
            "state_version": state_version,
            "policy_version": self.policy_version,
            "status": "blocked",
            "result": result,
            "created_at": _utc_now(),
        }
        records = [
            item
            for item in self._load_denials()
            if item.get("transition_hash") != digest
        ]
        records.append(record)
        self._save_denials(records[-128:])
        return record

    def lookup_denial(
        self,
        *,
        action: str,
        state_version: str,
    ) -> dict[str, Any] | None:
        digest = _guard_transition_hash(
            action=action,
            state_version=state_version,
            policy_version=self.policy_version,
        )
        for record in reversed(self._load_denials()):
            if record.get("transition_hash") == digest:
                return record
        return None

    def count(self) -> int:
        return len(self._load()) + len(self._load_denials())

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def _save(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _load_denials(self) -> list[dict[str, Any]]:
        if not self.denial_path.exists():
            return []
        data = json.loads(self.denial_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def _save_denials(self, records: list[dict[str, Any]]) -> None:
        self.denial_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.denial_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.denial_path)


def _result_is_reusable(result: dict[str, Any], certificate: dict[str, Any]) -> bool:
    if result.get("allow") is not True:
        return False
    if result.get("violations") or result.get("missing_evidence") or result.get("needs_backfill"):
        return False
    uncertainties = certificate.get("uncertainties")
    if not isinstance(uncertainties, list) or any(str(item).strip() for item in uncertainties):
        return False
    audit = result.get("decision_audit")
    if isinstance(audit, dict) and any(value is False for value in audit.values() if isinstance(value, bool)):
        return False
    validation = result.get("decision_certificate_validation")
    return not isinstance(validation, dict) or validation.get("valid") is True


def _semantic_certificate(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _semantic_certificate(item)
            for key, item in value.items()
            if key not in {"evidence", "evidence_state_ids", "source_event_ids", "source_span_ids"}
        }
    if isinstance(value, list):
        return [_semantic_certificate(item) for item in value]
    return value


def _guard_transition_hash(*, action: str, state_version: str, policy_version: str) -> str:
    command = action.split("DecisionCertificate:", 1)[0].strip().lower()
    return canonical_json_hash(
        {
            "command": command,
            "certificate_hash": certificate_hash(action),
            "state_version": state_version,
            "policy_version": policy_version,
        }
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
