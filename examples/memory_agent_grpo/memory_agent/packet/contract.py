"""Strict packet validation with two explicit, non-semantic field normalizations.

Protocol validity establishes structure and reference membership, not evidence truth,
requirement coverage, or task success. No text, existing reference, or need ID is repaired.
"""

import json
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

_FIELDS = {
    "needs": {"id", "text", "source_refs"},
    "evidence": {"id", "need_ids", "text", "kind", "source_refs"},
    "unresolved": {"need_ids", "text", "source_refs"},
}
_LIMITS = {"needs": 16, "evidence": 20, "unresolved": 16}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Nonstandard JSON constant")


def _validate_packet(packet: Any, source_ids: set[str]) -> list[str]:
    """Keep the frozen strict field, array, ID, and reference rules unchanged."""
    if not isinstance(packet, dict) or set(packet) != set(_FIELDS):
        return ["TOP_LEVEL_SCHEMA"]
    errors: list[str] = []
    seen: set[str] = set()
    needs = packet["needs"]
    need_ids = {
        entry.get("id") for entry in needs
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    } if isinstance(needs, list) else set()
    for group, fields in _FIELDS.items():
        entries = packet[group]
        if (not isinstance(entries, list) or len(entries) > _LIMITS[group]
                or (group == "needs" and not entries)):
            errors.append(f"{group}:ARRAY_SHAPE")
            continue
        for index, entry in enumerate(entries):
            where = f"{group}[{index}]"
            if not isinstance(entry, dict) or set(entry) != fields:
                errors.append(f"{where}:ENTRY_SCHEMA")
                continue
            if not isinstance(entry["text"], str) or not entry["text"].strip():
                errors.append(f"{where}:EMPTY_TEXT")
            if group != "unresolved":
                identifier = entry["id"]
                if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
                    errors.append(f"{where}:INVALID_OR_DUPLICATE_ID")
                else:
                    seen.add(identifier)
            refs = entry["source_refs"]
            if (not isinstance(refs, list)
                    or any(not isinstance(ref, str) or ref not in source_ids for ref in refs)
                    or (group != "unresolved" and not refs)):
                errors.append(f"{where}:INVALID_SOURCE_REFS")
            if group != "needs":
                links = entry["need_ids"]
                if (not isinstance(links, list) or not links
                        or any(not isinstance(link, str) or link not in need_ids for link in links)):
                    errors.append(f"{where}:INVALID_NEED_IDS")
            if group == "evidence" and entry["kind"] not in ("observed", "reported", "derived"):
                errors.append(f"{where}:INVALID_EVIDENCE_KIND")
    return errors


def _normalize_packet(packet: Any) -> tuple[Any, list[dict[str, Any]]]:
    canonical = deepcopy(packet)
    events: list[dict[str, Any]] = []
    if not isinstance(canonical, dict) or not isinstance(canonical.get("unresolved"), list):
        return canonical, events
    for index, entry in enumerate(canonical["unresolved"]):
        if not isinstance(entry, dict):
            continue
        path = f"/unresolved/{index}"
        if "source_refs" not in entry:
            entry["source_refs"] = []
            events.append({"path": f"{path}/source_refs", "operation": "add_missing", "value": []})
        if "id" in entry and isinstance(entry["id"], str) and entry["id"].strip():
            events.append({"path": f"{path}/id", "operation": "remove_metadata", "value": entry.pop("id")})
    return canonical, events


def parse_packet(content: str, *, source_ids: Iterable[str]) -> dict[str, Any]:
    """Parse one complete JSON packet and report strict and compatible validity.

    ``raw_output`` is preserved. Compatibility only supplies an absent unresolved
    ``source_refs`` as an empty list, and removes its optional nonempty string ``id``
    while recording the original value. Existing null/invalid references and invalid
    metadata IDs remain errors. ``packet`` is canonical only when validation succeeds;
    all failures return it as ``None``. Strict errors always describe the original JSON.

    ``source_ids`` must be an iterable of nonempty string IDs, not a single string.
    Invalid caller configuration raises ``ValueError``; invalid model output is returned
    as a validation result. No normalization asserts the correctness of cited evidence.
    """
    if isinstance(source_ids, (str, bytes)):
        raise ValueError("source_ids must be an iterable of nonempty string IDs")
    try:
        available = list(source_ids)
    except TypeError as error:
        raise ValueError("source_ids must be an iterable of nonempty string IDs") from error
    if any(not isinstance(value, str) or not value.strip() for value in available):
        raise ValueError("source_ids must contain nonempty strings")
    available_ids = set(available)
    result: dict[str, Any] = {
        "raw_output": content,
        "json_valid": False,
        "strict_protocol_valid": False,
        "protocol_valid": False,
        "strict_errors": ["INVALID_JSON"],
        "errors": ["INVALID_JSON"],
        "normalizations": [],
        "packet": None,
    }
    try:
        parsed = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (TypeError, ValueError, RecursionError):
        return result
    strict_errors = _validate_packet(parsed, available_ids)
    canonical, events = _normalize_packet(parsed)
    errors = _validate_packet(canonical, available_ids)
    result.update({
        "json_valid": True,
        "strict_protocol_valid": not strict_errors,
        "protocol_valid": not errors,
        "strict_errors": strict_errors,
        "errors": errors,
        "normalizations": events,
        "packet": canonical if not errors else None,
    })
    return result
