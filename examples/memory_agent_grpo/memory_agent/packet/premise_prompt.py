"""Minimal prompt delta for deductions grounded in selected evidence IDs."""

from .selection_prompt import build_selection_messages

_REPLACEMENTS = (
    ('"kind": "derived", "source_refs": ["S001", "S002"]}', '"kind": "derived", "premise_ids": ["E1"]}'),
    (
        "with kind=derived, cite original source IDs for all material premises, and state\nconditions.",
        "with kind=derived, list all supporting selected evidence IDs in premise_ids,\n"
        "and state conditions. Each premise must be an evidence entry with excerpt_id\n"
        "in this same draft. Select the supporting excerpts as evidence first; listing\n"
        "a source somewhere in the input is insufficient. Do not supply source_refs\n"
        "on deductions: code derives them from the selected premises. Premises must\n"
        "be a nonempty list of unique evidence IDs, not source or excerpt IDs; they\n"
        "cannot refer to needs, the deduction itself, or other derived entries.",
    ),
    (
        "earlier memory. Derived evidence needs supporting sources. unresolved may have",
        "earlier memory. Deductions use premise_ids instead of source_refs. unresolved may have",
    ),
)


def build_premise_messages(catalog):
    """Change only the deduction contract; preserve the complete catalog message."""
    messages = build_selection_messages(catalog)
    for before, after in _REPLACEMENTS:
        if messages[0]["content"].count(before) != 1:
            raise ValueError("Base prompt no longer matches the documented premise delta")
        messages[0]["content"] = messages[0]["content"].replace(before, after, 1)
    return messages
