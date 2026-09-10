"""Premise links must resolve to selected evidence, without certifying deductions."""

import json

import pytest

from examples.memory_agent_grpo.memory_agent.packet.excerpts import build_excerpt_catalog
from examples.memory_agent_grpo.memory_agent.packet.premises import compile_premise_selection


def fixture():
    sources = [
        {"source_id": "T", "kind": "task", "text": "Compare A and B."},
        {"source_id": "A", "kind": "observation", "text": "A has value 7."},
        {"source_id": "B", "kind": "memory", "text": "B reportedly has value 9."},
        {"source_id": "UNSELECTED", "kind": "observation", "text": "A different result is 100."},
    ]
    catalog = build_excerpt_catalog(sources)
    plan = {
        "needs": [{"id": "N1", "text": "Compare A and B", "source_refs": ["T"]}],
        "evidence": [
            {"id": "E1", "need_ids": ["N1"], "excerpt_id": "F0002"},
            {"id": "E2", "need_ids": ["N1"], "excerpt_id": "F0003"},
            {
                "id": "E3",
                "need_ids": ["N1"],
                "text": "If B is verified, 9 > 7.",
                "kind": "derived",
                "premise_ids": ["E2", "E1"],
            },
        ],
        "unresolved": [],
    }
    return sources, catalog, plan


def test_resolves_selected_premises_and_preserves_text_and_raw_draft():
    sources, catalog, plan = fixture()
    raw = json.dumps(plan)
    result = compile_premise_selection(raw, sources=sources, catalog=catalog)
    assert result["selection_valid"] and result["strict_protocol_valid"]
    assert result["raw_output"] == raw
    assert result["packet"]["evidence"][2] == {
        "id": "E3",
        "need_ids": ["N1"],
        "text": plan["evidence"][2]["text"],
        "kind": "derived",
        "source_refs": ["B", "A"],
    }
    assert result["packet"]["evidence"][1]["kind"] == "reported"
    assert result["premise_bindings"] == [
        {"evidence_id": "E3", "premise_ids": ["E2", "E1"], "excerpt_ids": ["F0003", "F0002"], "source_refs": ["B", "A"]}
    ]


@pytest.mark.parametrize(
    "premises", [[], ["E999"], ["UNSELECTED"], ["F0004"], ["N1"], ["E3"], ["E1", "E1"], "E1", [None], [["E1"]]]
)
def test_rejects_missing_or_nonselected_premises(premises):
    sources, catalog, plan = fixture()
    plan["evidence"][2]["premise_ids"] = premises
    result = compile_premise_selection(json.dumps(plan), sources=sources, catalog=catalog)
    assert not result["selection_valid"] and result["packet"] is None


def test_cannot_override_provenance_or_use_legacy_free_source_deduction():
    sources, catalog, plan = fixture()
    plan["evidence"][2]["source_refs"] = ["UNSELECTED"]
    result = compile_premise_selection(json.dumps(plan), sources=sources, catalog=catalog)
    assert not result["selection_valid"] and result["packet"] is None
    plan["evidence"][2].pop("premise_ids")
    result = compile_premise_selection(json.dumps(plan), sources=sources, catalog=catalog)
    assert not result["selection_valid"] and result["packet"] is None


def test_derived_chains_and_cycles_are_rejected():
    sources, catalog, plan = fixture()
    plan["evidence"].append(
        {"id": "E4", "need_ids": ["N1"], "text": "A chained deduction.", "kind": "derived", "premise_ids": ["E3"]}
    )
    result = compile_premise_selection(json.dumps(plan), sources=sources, catalog=catalog)
    assert not result["selection_valid"] and result["packet"] is None
    plan["evidence"][2]["premise_ids"] = ["E4"]
    result = compile_premise_selection(json.dumps(plan), sources=sources, catalog=catalog)
    assert not result["selection_valid"] and result["packet"] is None


def test_selected_entries_can_follow_deduction_and_same_source_is_deduplicated():
    sources, catalog, plan = fixture()
    plan["evidence"][1]["excerpt_id"] = "F0002"
    plan["evidence"] = [plan["evidence"][2], *plan["evidence"][:2]]
    result = compile_premise_selection(json.dumps(plan), sources=sources, catalog=catalog)
    assert result["selection_valid"]
    assert result["packet"]["evidence"][0]["source_refs"] == ["A"]


def test_duplicate_evidence_or_missing_excerpt_fails_without_packet():
    sources, catalog, plan = fixture()
    plan["evidence"][1]["id"] = "E1"
    result = compile_premise_selection(json.dumps(plan), sources=sources, catalog=catalog)
    assert not result["selection_valid"] and result["packet"] is None
    plan["evidence"][1]["id"] = "E2"
    plan["evidence"][1]["excerpt_id"] = "F9999"
    result = compile_premise_selection(json.dumps(plan), sources=sources, catalog=catalog)
    assert not result["selection_valid"] and result["packet"] is None


def test_unsupported_deduction_is_preserved_not_semantically_certified():
    sources, catalog, plan = fixture()
    plan["evidence"][2]["text"] = "Unsupported: A and B are both 100."
    result = compile_premise_selection(json.dumps(plan), sources=sources, catalog=catalog)
    assert result["selection_valid"]
    assert result["packet"]["evidence"][2]["text"] == plan["evidence"][2]["text"]
