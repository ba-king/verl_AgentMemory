"""Integrity checks for model-selected evidence materialization."""

import json
from copy import deepcopy

import pytest

from examples.memory_agent_grpo.memory_agent.packet.excerpts import build_excerpt_catalog
from examples.memory_agent_grpo.memory_agent.packet.selection import compile_selection


def sources():
    return [
        {"source_id": "T", "kind": "task", "text": "Compare the two objects using verified values."},
        {"source_id": "A", "kind": "reference_observation", "text": "Object A has value 7."},
        {"source_id": "B", "kind": "memory", "text": "Object B reportedly has value 9."},
    ]


def draft(catalog):
    by_source = {e["source_id"]: e["excerpt_id"] for e in catalog}
    return {
        "needs": [{"id": "N1", "text": "Compare the verified values", "source_refs": ["T"]}],
        "evidence": [
            {"id": "E1", "need_ids": ["N1"], "excerpt_id": by_source["B"]},
            {"id": "E2", "need_ids": ["N1"], "excerpt_id": by_source["A"]},
            {
                "id": "E3",
                "need_ids": ["N1"],
                "text": "If B's report is verified, 9 > 7.",
                "kind": "derived",
                "source_refs": ["A", "B"],
            },
        ],
        "unresolved": [{"need_ids": ["N1"], "text": "B's reported value needs verification.", "source_refs": ["B"]}],
    }


def test_lossless_unicode_escaping_and_cross_boundary_context():
    text = '对象甲：7。\\n保留字面转义。\n\nObject B: "9". ' * 20
    catalog = build_excerpt_catalog([{"source_id": "S", "kind": "history", "text": text}], max_chars=43)
    assert "".join(e["text"] for e in catalog) == text
    assert catalog[0]["start"] == 0 and catalog[-1]["end"] == len(text)
    assert all(a["end"] == b["start"] for a, b in zip(catalog, catalog[1:], strict=False))
    assert all(e["text"] == text[e["start"] : e["end"]] and len(e["text"]) <= 43 for e in catalog)


def test_materialization_keeps_object_source_and_reported_status():
    src = sources()
    catalog = build_excerpt_catalog(src)
    planned = draft(catalog)
    result = compile_selection(json.dumps(planned), sources=src, catalog=catalog)
    assert result["selection_valid"] and result["strict_protocol_valid"]
    entries = result["packet"]["evidence"]
    assert entries[0] == {
        "id": "E1",
        "need_ids": ["N1"],
        "text": src[2]["text"],
        "kind": "reported",
        "source_refs": ["B"],
    }
    assert entries[1]["text"] == src[1]["text"] and entries[1]["kind"] == "observed"
    assert entries[2] == planned["evidence"][2]


@pytest.mark.parametrize("mutation", ["unknown_id", "override_text", "freeform_observed", "dangling_need"])
def test_invalid_selection_never_produces_a_usable_packet(mutation):
    src = sources()
    catalog = build_excerpt_catalog(src)
    planned = draft(catalog)
    entry = planned["evidence"][0]
    if mutation == "unknown_id":
        entry["excerpt_id"] = "F9999"
    elif mutation == "override_text":
        entry["text"] = "Object B is proven best."
    elif mutation == "freeform_observed":
        entry.pop("excerpt_id")
        entry.update({"text": "Object B is proven best.", "kind": "observed", "source_refs": ["B"]})
    else:
        entry["need_ids"] = ["MISSING"]
    raw = json.dumps(planned)
    result = compile_selection(raw, sources=src, catalog=catalog)
    assert not result["selection_valid"] and result["packet"] is None
    assert result["raw_output"] == raw and result["errors"]


def test_duplicate_sources_and_stale_catalog_rejected():
    src = sources()
    with pytest.raises(ValueError, match="Duplicate source"):
        build_excerpt_catalog([*src, src[1]])
    catalog = build_excerpt_catalog(src)
    raw = json.dumps(draft(catalog))
    altered = deepcopy(catalog)
    altered[2]["text"] = "Object B is proven best."
    with pytest.raises(ValueError, match="differs"):
        compile_selection(raw, sources=src, catalog=altered)
    altered = deepcopy(catalog)
    altered[2]["evidence_kind"] = "observed"
    with pytest.raises(ValueError, match="differs"):
        compile_selection(raw, sources=src, catalog=altered)


def test_invalid_json_is_not_repaired():
    src = sources()
    catalog = build_excerpt_catalog(src)
    for raw in ('{"needs": [], "needs": []}', '{"needs": [', "NaN"):
        result = compile_selection(raw, sources=src, catalog=catalog)
        assert not result["selection_valid"] and result["packet"] is None


def test_compiler_does_not_claim_or_fix_semantic_validity():
    src = sources()
    catalog = build_excerpt_catalog(src)
    planned = draft(catalog)
    planned["evidence"][2]["text"] = "Unsupported claim: every object has value 100."
    result = compile_selection(json.dumps(planned), sources=src, catalog=catalog)
    assert result["selection_valid"]  # Structure/provenance does not prove this deduction.
    assert result["packet"]["evidence"][2]["text"] == planned["evidence"][2]["text"]
