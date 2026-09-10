"""Lossless, domain-independent source slicing with explicit provenance."""


def source_index(sources):
    """Validate caller-owned source records without inferring their factual truth."""
    index = {}
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"source_id", "kind", "text"}:
            raise ValueError("Expected source_id/kind/text records")
        if any(not isinstance(value, str) or not value.strip() for value in source.values()):
            raise ValueError("Source fields must be nonempty strings")
        if source["source_id"] in index:
            raise ValueError("Duplicate source ID")
        index[source["source_id"]] = source
    if not index:
        raise ValueError("No sources")
    return index


def evidence_kind(source_kind):
    """Mixed history remains reported unless the caller identifies an observation."""
    return "observed" if source_kind in {"task", "observation", "reference_observation", "context"} else "reported"


def build_excerpt_catalog(sources, *, max_chars=1200):
    """Cover every source character once; prefer a nearby newline or space boundary.

    Offsets count decoded Python Unicode characters, not bytes or model tokens.
    No cleaning, unescaping, fact extraction, ranking, or task-specific splitting.
    A boundary can split a semantic unit; the model can select adjacent excerpts.
    """
    if type(max_chars) is not int or max_chars < 2:
        raise ValueError("max_chars must be an integer >= 2")
    catalog = []
    for source in source_index(sources).values():
        start, text = 0, source["text"]
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                lower = start + max_chars // 2
                boundary = text.rfind("\n", lower, end)
                if boundary < 0:
                    boundary = text.rfind(" ", lower, end)
                if boundary >= lower:
                    end = boundary + 1
            catalog.append(
                {
                    "excerpt_id": f"F{len(catalog) + 1:04}",
                    "source_id": source["source_id"],
                    "source_kind": source["kind"],
                    "evidence_kind": evidence_kind(source["kind"]),
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                }
            )
            start = end
    return catalog


def checked_excerpt_index(sources, catalog):
    """Reject stale or altered text/metadata before materializing model selections."""
    sources = source_index(sources)
    index = {}
    fields = {"excerpt_id", "source_id", "source_kind", "evidence_kind", "start", "end", "text"}
    for excerpt in catalog:
        if not isinstance(excerpt, dict) or set(excerpt) != fields:
            raise ValueError("Malformed excerpt record")
        eid, sid = excerpt["excerpt_id"], excerpt["source_id"]
        if not isinstance(eid, str) or not eid.strip() or eid in index:
            raise ValueError("Invalid or duplicate excerpt ID")
        if not isinstance(sid, str) or sid not in sources:
            raise ValueError("Unknown excerpt source")
        source = sources[sid]
        start, end = excerpt["start"], excerpt["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source["text"]):
            raise ValueError("Invalid excerpt offsets")
        if (
            excerpt["text"] != source["text"][start:end]
            or excerpt["source_kind"] != source["kind"]
            or excerpt["evidence_kind"] != evidence_kind(source["kind"])
        ):
            raise ValueError("Excerpt differs from its source")
        index[eid] = excerpt
    return index
