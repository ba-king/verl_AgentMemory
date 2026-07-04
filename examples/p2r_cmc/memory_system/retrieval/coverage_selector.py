from __future__ import annotations

from dataclasses import dataclass

from ..schemas import EvidenceItem, QueryAspect
from .aspect_extractor import aspect_match_score, coverage_for_items, evidence_covers_aspect, missing_aspect_texts
from .reranker import rerank_evidence


@dataclass
class CoverageSelection:
    selected: list[EvidenceItem]
    omitted_memory_ids: list[str]
    token_estimate: int
    coverage: dict[str, list[str]]
    missing_aspects: list[str]


def select_evidence_with_coverage(
    items: list[EvidenceItem],
    aspects: list[QueryAspect],
    *,
    max_items: int,
    token_budget: int,
) -> CoverageSelection:
    ranked = rerank_evidence(items, top_k=len(items))
    if not ranked:
        return CoverageSelection([], [], 0, {}, missing_aspect_texts([], aspects))

    required_aspects = [aspect for aspect in aspects if aspect.required and aspect.requires_evidence]
    selected: list[EvidenceItem] = []
    selected_ids: set[str] = set()
    covered_aspect_ids: set[str] = set()
    token_count = 0

    while len(selected) < max_items:
        best_item: EvidenceItem | None = None
        best_key: tuple[int, float, float] | None = None
        for item in ranked:
            if item.memory_id in selected_ids:
                continue
            item_tokens = _estimate_item_tokens(item)
            if token_count + item_tokens > token_budget:
                continue
            aspect_scores = []
            for aspect in required_aspects:
                if aspect.aspect_id in covered_aspect_ids:
                    continue
                score = aspect_match_score(item, aspect)
                if evidence_covers_aspect(item, aspect):
                    aspect_scores.append(score)
            new_coverage = len(aspect_scores)
            if new_coverage == 0:
                continue
            coverage_strength = sum(aspect_scores)
            key = (new_coverage, coverage_strength, item.score)
            if best_key is None or key > best_key:
                best_key = key
                best_item = item
        if best_item is None:
            break
        _add_selected(best_item, selected, selected_ids)
        token_count += _estimate_item_tokens(best_item)
        for aspect in required_aspects:
            if evidence_covers_aspect(best_item, aspect):
                covered_aspect_ids.add(aspect.aspect_id)
        if len(covered_aspect_ids) == len(required_aspects):
            break

    for item in ranked:
        if len(selected) >= max_items:
            break
        if item.memory_id in selected_ids:
            continue
        item_tokens = _estimate_item_tokens(item)
        if token_count + item_tokens > token_budget:
            continue
        _add_selected(item, selected, selected_ids)
        token_count += item_tokens

    omitted = [item.memory_id for item in ranked if item.memory_id not in selected_ids]
    selected = rerank_evidence(selected, top_k=len(selected))
    return CoverageSelection(
        selected=selected,
        omitted_memory_ids=omitted,
        token_estimate=sum(_estimate_item_tokens(item) for item in selected),
        coverage=coverage_for_items(selected, aspects),
        missing_aspects=missing_aspect_texts(selected, aspects, candidate_items=ranked),
    )


def _add_selected(item: EvidenceItem, selected: list[EvidenceItem], selected_ids: set[str]) -> None:
    selected.append(item)
    selected_ids.add(item.memory_id)


def _estimate_item_tokens(item: EvidenceItem) -> int:
    return max(1, len(item.content) // 4)
