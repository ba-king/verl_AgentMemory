from examples.p2r_cmc.memory_system.retrieval.aspect_extractor import extract_query_aspects
from examples.p2r_cmc.memory_system.retrieval.coverage_selector import select_evidence_with_coverage
from examples.p2r_cmc.memory_system.retrieval.merger import build_evidence_packet
from examples.p2r_cmc.memory_system.schemas import EvidenceItem


def evidence(memory_id: str, content: str, score: float) -> EvidenceItem:
    return EvidenceItem(
        memory_id=memory_id,
        content=content,
        summary=content,
        score=score,
        facets=["preference"],
        status="active",
        rationale="rrf:bm25_sparse,dense_embedding",
    )


def test_coverage_selection_keeps_lower_score_item_for_explicit_aspect():
    query = "Please consider daily pace, budget, and raw fish or beef restrictions."
    items = [
        evidence("pace", "The user prefers a relaxed daily pace.", 0.99),
        evidence("budget", "The user prefers mid-range hotels and simple meals for budget control.", 0.98),
        evidence("pace_duplicate", "The user likes late starts and one major attraction each day.", 0.97),
        evidence("food", "The travel partner does not eat raw fish and avoids beef.", 0.20),
    ]

    packet = build_evidence_packet(query, items, max_items=3, token_budget=1200)
    selected_ids = [item.memory_id for item in packet.evidence_items]

    assert "food" in selected_ids
    assert "pace_duplicate" in packet.omitted_memory_ids
    assert any("raw fish" in aspect.text.lower() or "beef" in aspect.text.lower() for aspect in packet.required_aspects)
    assert not any("raw fish" in text.lower() for text in packet.missing_aspects)


def test_coverage_selection_falls_back_to_score_without_aspects():
    items = [
        evidence("top", "Top scored item.", 0.99),
        evidence("middle", "Middle scored item.", 0.50),
        evidence("low", "Low scored item.", 0.10),
    ]

    selection = select_evidence_with_coverage(items, extract_query_aspects("What should I do?"), max_items=2, token_budget=1200)

    assert [item.memory_id for item in selection.selected] == ["top", "middle"]
