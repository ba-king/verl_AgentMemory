"""Require deductions to name selected source excerpts as their direct premises."""

import json
from copy import deepcopy

from .contract import _reject_constant, _unique_object
from .excerpts import checked_excerpt_index
from .selection import compile_selection


def compile_premise_selection(content, *, sources, catalog):
    """Enforce premise membership and derive provenance, not semantic entailment.

    Only direct excerpt selections may be premises; no inference chains or cycles.
    Selected entries can appear anywhere in the evidence array. No missing premise
    is invented, selected, or repaired by this compiler.
    """
    excerpts = checked_excerpt_index(sources, catalog)
    result = {
        "raw_output": content,
        "json_valid": False,
        "selection_valid": False,
        "strict_protocol_valid": False,
        "errors": ["INVALID_JSON"],
        "normalizations": [],
        "bindings": [],
        "premise_bindings": [],
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
    selected, seen, errors = {}, set(), []
    for position, entry in enumerate(draft["evidence"]):
        where = f"evidence[{position}]"
        if not isinstance(entry, dict):
            errors.append(f"{where}:ENTRY_SCHEMA")
            continue
        eid = entry.get("id")
        if not isinstance(eid, str) or not eid.strip() or eid in seen:
            errors.append(f"{where}:INVALID_OR_DUPLICATE_ID")
            continue
        seen.add(eid)
        if set(entry) == {"id", "need_ids", "excerpt_id"}:
            excerpt_id = entry["excerpt_id"]
            if not isinstance(excerpt_id, str) or excerpt_id not in excerpts:
                errors.append(f"{where}:UNKNOWN_EXCERPT_ID")
            else:
                selected[eid] = excerpts[excerpt_id]
        elif set(entry) != {"id", "need_ids", "text", "kind", "premise_ids"} or entry["kind"] != "derived":
            errors.append(f"{where}:SELECTION_OR_PREMISE_SCHEMA")
    if errors:
        result["errors"] = errors
        return result
    translated = deepcopy(draft)
    for position, entry in enumerate(translated["evidence"]):
        if "excerpt_id" in entry:
            continue
        premises = entry["premise_ids"]
        if (
            not isinstance(premises, list)
            or not premises
            or any(not isinstance(pid, str) or pid not in selected for pid in premises)
            or len(set(premises)) != len(premises)
        ):
            errors.append(f"evidence[{position}]:INVALID_SELECTED_PREMISES")
            continue
        source_refs = list(dict.fromkeys(selected[pid]["source_id"] for pid in premises))
        result["premise_bindings"].append(
            {
                "evidence_id": entry["id"],
                "premise_ids": list(premises),
                "excerpt_ids": [selected[pid]["excerpt_id"] for pid in premises],
                "source_refs": source_refs,
            }
        )
        entry.pop("premise_ids")
        entry["source_refs"] = source_refs
    if errors:
        result["errors"] = errors
        return result
    compiled = compile_selection(json.dumps(translated, ensure_ascii=False), sources=sources, catalog=catalog)
    result.update({k: v for k, v in compiled.items() if k != "raw_output"})
    return result
