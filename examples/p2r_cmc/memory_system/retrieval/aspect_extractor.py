from __future__ import annotations

import re
from collections.abc import Iterable

from ..clustering.metrics import tokenize
from ..schemas import EvidenceItem, QueryAspect

MAX_ASPECTS = 24
MIN_MATCH_SCORE = 0.34

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "the",
    "this",
    "to",
    "use",
    "using",
    "what",
    "when",
    "why",
    "with",
    "you",
    "your",
    "请",
    "我",
    "你",
    "和",
    "或",
    "以及",
}

LIST_MARKERS = (
    "请同时考虑",
    "同时考虑",
    "请考虑",
    "考虑",
    "包括",
    "覆盖",
    "分别说明",
    "比较",
    "consider",
    "include",
    "cover",
    "address",
    "compare",
    "evaluate",
    "retrieval hints:",
)

SPLIT_PATTERN = re.compile(r"\s*(?:、|，|,|；|;|\n|\band\b|\balso\b|以及|并且|同时|还有)\s*", re.IGNORECASE)
SENTENCE_END_PATTERN = re.compile(r"[。！？!?]")
CARDINALITY_PATTERN = re.compile(
    r"(?:exactly|at least|at most|no more than|不少于|不超过|至少|最多)?\s*"
    r"(?:\d+|[一二三四五六七八九十]+)\s*"
    r"(?:days?|items?|options?|steps?|examples?|actions?|sections?|天|个|项|步|次|条|点|部分)",
    re.IGNORECASE,
)
CONSTRAINT_PATTERN = re.compile(
    r"(?:不要|避免|不能|必须|需要|至少|最多|不超过|不少于|avoid|must|without|at least|no more than|exactly)"
    r"[^，。；;,.!?\n]{1,80}",
    re.IGNORECASE,
)
EXCLUSIVE_PATTERN = re.compile(
    r"[^，。；;,.!?\n]{1,60}(?:\b(?:or|either|choose one)\b|二选一|或者|或)[^，。；;,.!?\n]{1,60}",
    re.IGNORECASE,
)


def extract_query_aspects(query: str, *, max_aspects: int = MAX_ASPECTS) -> list[QueryAspect]:
    """Extract explicit query requirements without calling a model.

    This intentionally keeps raw spans instead of mapping them into a fixed
    ontology. The output can work across travel, engineering, research, shopping,
    and other domains.
    """

    candidates: list[tuple[str, str, bool]] = []
    candidates.extend((span, "topic", True) for span in _extract_list_spans(query))
    candidates.extend((span, "constraint", True) for span in _regex_spans(CONSTRAINT_PATTERN, query))
    candidates.extend((span, "exclusive_alternative", True) for span in _regex_spans(EXCLUSIVE_PATTERN, query))
    candidates.extend((span, "cardinality", False) for span in _regex_spans(CARDINALITY_PATTERN, query))

    aspects: list[QueryAspect] = []
    seen: set[str] = set()
    for raw_text, kind, requires_evidence in candidates:
        text = _clean_aspect_text(raw_text)
        if not _is_useful_aspect(text):
            continue
        key = _canonical(text)
        if key in seen:
            continue
        seen.add(key)
        aspects.append(
            QueryAspect(
                text=text,
                kind=kind,  # type: ignore[arg-type]
                requires_evidence=requires_evidence,
                keywords=_keywords_for_text(text),
            )
        )
        if len(aspects) >= max_aspects:
            break
    return aspects


def evidence_covers_aspect(item: EvidenceItem, aspect: QueryAspect) -> bool:
    return aspect_match_score(item, aspect) >= MIN_MATCH_SCORE


def aspect_match_score(item: EvidenceItem, aspect: QueryAspect) -> float:
    item_text = _canonical(
        " ".join(
            [
                item.content,
                item.summary,
                " ".join(item.facets),
                " ".join(item.source_refs),
            ]
        )
    )
    if not item_text:
        return 0.0

    aspect_text = _canonical(aspect.text)
    if aspect_text and aspect_text in item_text:
        return 1.0

    keywords = [keyword for keyword in aspect.keywords if keyword not in STOPWORDS]
    if not keywords:
        return 0.0
    hits = 0
    for keyword in keywords:
        if _canonical(keyword) in item_text:
            hits += 1
    return hits / max(1, len(keywords))


def coverage_for_items(items: list[EvidenceItem], aspects: list[QueryAspect]) -> dict[str, list[str]]:
    required = [aspect for aspect in aspects if aspect.required and aspect.requires_evidence]
    coverage: dict[str, list[str]] = {aspect.aspect_id: [] for aspect in required}
    for aspect in required:
        for item in items:
            if evidence_covers_aspect(item, aspect):
                coverage[aspect.aspect_id].append(item.memory_id)
    return {aspect_id: memory_ids for aspect_id, memory_ids in coverage.items() if memory_ids}


def missing_aspect_texts(
    items: list[EvidenceItem],
    aspects: list[QueryAspect],
    *,
    candidate_items: list[EvidenceItem] | None = None,
) -> list[str]:
    covered = coverage_for_items(items, aspects)
    candidate_covered = coverage_for_items(candidate_items, aspects) if candidate_items is not None else None
    return [
        aspect.text
        for aspect in aspects
        if aspect.required
        and aspect.requires_evidence
        and aspect.aspect_id not in covered
        and (candidate_covered is None or aspect.aspect_id in candidate_covered)
    ]


def _extract_list_spans(query: str) -> list[str]:
    spans: list[str] = []
    for marker in LIST_MARKERS:
        for start, end in _marker_spans(query, marker):
            phrase = query[end:]
            phrase = SENTENCE_END_PATTERN.split(phrase, maxsplit=1)[0]
            for part in SPLIT_PATTERN.split(phrase):
                cleaned = _clean_aspect_text(part)
                if cleaned:
                    spans.append(cleaned)
    return spans


def _marker_spans(query: str, marker: str) -> list[tuple[int, int]]:
    if re.fullmatch(r"[a-z: ]+", marker, flags=re.IGNORECASE):
        escaped = re.escape(marker).replace("\\ ", r"\s+")
        pattern = re.compile(rf"(?<![a-z0-9_-]){escaped}(?![a-z0-9_-])\s*:?", re.IGNORECASE)
        return [(match.start(), match.end()) for match in pattern.finditer(query)]
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = query.find(marker, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(marker)))
        start = idx + len(marker)
    return spans


def _regex_spans(pattern: re.Pattern[str], query: str) -> list[str]:
    return [match.group(0) for match in pattern.finditer(query)]


def _clean_aspect_text(text: str) -> str:
    cleaned = text.strip(" \t\r\n:：,，;；.。")
    cleaned = re.sub(r"^(and|also|with|about|on|for|请|以及|和|并且)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _is_useful_aspect(text: str) -> bool:
    if len(text) < 2 or len(text) > 120:
        return False
    lowered = text.lower()
    if lowered in STOPWORDS:
        return False
    if lowered.startswith(("回答用", "answer in", "user query")):
        return False
    return bool(_keywords_for_text(text))


def _keywords_for_text(text: str) -> list[str]:
    tokens = [token for token in tokenize(text) if token not in STOPWORDS]
    phrase_tokens = _phrase_keywords(text)
    return _dedupe([*tokens, *phrase_tokens])


def _phrase_keywords(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9][a-z0-9_-]*", lowered)
    phrases: list[str] = []
    for size in (2, 3):
        for idx in range(0, max(0, len(words) - size + 1)):
            phrase = " ".join(words[idx : idx + size])
            if any(word in STOPWORDS for word in phrase.split()):
                continue
            phrases.append(phrase)
    return phrases


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _canonical(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()
