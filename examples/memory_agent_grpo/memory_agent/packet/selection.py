"""Materialize an evidence-selection draft into the unchanged packet interface."""

import json
from copy import deepcopy

from .contract import _reject_constant, _unique_object, parse_packet
from .excerpts import checked_excerpt_index, source_index


def compile_selection(content, *, sources, catalog):
    """Resolve exact IDs only. Invalid drafts never produce a usable packet.

    The model writes needs, deductions and unknowns, but cannot supply or override
    the text/kind/source of a selected excerpt. This enforces text provenance, not
    correct selection, sufficient context, faithful needs, or valid deductions.
    """
    source_ids = source_index(sources)
    excerpts = checked_excerpt_index(sources, catalog)
    result = {
        "raw_output": content,
        "json_valid": False,
        "selection_valid": False,
        "strict_protocol_valid": False,
        "errors": ["INVALID_JSON"],
        "normalizations": [],
        "bindings": [],
        "packet": None,
    }
    try:
        draft = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (TypeError, ValueError, RecursionError):
        return result
    result["json_valid"] = True
    if not isinstance(draft, dict) or set(draft) != {"needs", "evidence", "unresolved"}:
        result["errors"] = ["TOP_LEVEL_SCHEMA"]
        return result
    if not isinstance(draft["evidence"], list):
        result["errors"] = ["evidence:ARRAY_SHAPE"]
        return result
    packet, errors = deepcopy(draft), []
    for position, entry in enumerate(packet["evidence"]):
        where = f"evidence[{position}]"
        if not isinstance(entry, dict):
            errors.append(f"{where}:ENTRY_SCHEMA")
        elif set(entry) == {"id", "need_ids", "excerpt_id"}:
            eid = entry["excerpt_id"]
            if not isinstance(eid, str) or eid not in excerpts:
                errors.append(f"{where}:UNKNOWN_EXCERPT_ID")
                continue
            excerpt = excerpts[eid]
            packet["evidence"][position] = {
                "id": entry["id"],
                "need_ids": entry["need_ids"],
                "text": excerpt["text"],
                "kind": excerpt["evidence_kind"],
                "source_refs": [excerpt["source_id"]],
            }
            result["bindings"].append(
                {
                    "evidence_id": entry["id"],
                    "excerpt_id": eid,
                    "source_id": excerpt["source_id"],
                    "start": excerpt["start"],
                    "end": excerpt["end"],
                }
            )
        elif set(entry) != {"id", "need_ids", "text", "kind", "source_refs"} or entry["kind"] != "derived":
            errors.append(f"{where}:SELECTION_OR_DERIVED_SCHEMA")
    if errors:
        result["errors"] = errors
        return result
    parsed = parse_packet(json.dumps(packet, ensure_ascii=False), source_ids=source_ids)
    result.update(
        {
            "selection_valid": parsed["protocol_valid"],
            "strict_protocol_valid": parsed["strict_protocol_valid"],
            "errors": parsed["errors"],
            "strict_errors": parsed["strict_errors"],
            "normalizations": parsed["normalizations"],
            "packet": parsed["packet"],
        }
    )
    return result
