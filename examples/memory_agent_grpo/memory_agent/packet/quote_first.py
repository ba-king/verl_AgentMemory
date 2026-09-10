"""Experimental verbatim evidence discipline within the existing packet schema."""

from .support_review import build_support_review_messages

QUOTE_FIRST = """Use this concrete evidence discipline for this packet:
For an observed or reported evidence entry, text must be ONE exact, contiguous,
nonempty excerpt from ONE cited source record, copied without paraphrase or
added commentary. JSON-escape the excerpt normally; do not add quote markers,
labels, or ellipses that are not in the source. Choose enough context to identify
the object or situation the excerpt concerns. Do not copy a whole source when
a shorter useful excerpt suffices. You may use multiple entries if necessary.
For reported entries, copy what the memory actually says; quotation does not
establish truth or applicability. Put useful comparisons and other deductions
in separate derived entries, with the sources of their premises and their
conditions. A derived label is not permission to fill an unknown attribute.
When an eligibility or comparison premise is absent from the excerpts, record
that specific unknown in unresolved and keep the supported partial comparison.
Do not replace the task's candidate scope with only the currently visible set.
Do not repeat the needs as evidence unless doing so adds a necessary premise.
Keep the existing needs/evidence/unresolved interface and limits. No new fields.
"""


def build_quote_first_messages(sources):
    """Add extractive evidence requirements to the opt-in source-support review."""
    messages = build_support_review_messages(sources)
    messages[0]["content"] += "\n" + QUOTE_FIRST
    return messages
