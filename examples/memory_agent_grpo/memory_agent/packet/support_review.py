"""Opt-in source-support review for the existing domain-independent packet prompt."""

from .prompts import build_messages

SUPPORT_REVIEW = """Before emitting the packet, perform this review within the same response:
1. Identify the current requirements and preferences, including their scope and
dependencies. Keep the exact objects, alternatives, times, units, and conditions
that could change a decision. Earlier context remains conditional on its source
and applicability; an earlier result is not a current requirement.
2. For each selected factual claim, locate the supporting content in its cited
records. Check the particular object and attribute, not just a matching name or
family. A requirement for an attribute is not evidence that an object has it.
Missing support means unknown, not true or false. Keep directly supplied facts
as observed, memory claims as reported, and your inferences as derived, citing
all material premises. Do not label a mixed fact-and-inference entry observed.
3. Before claiming eligibility, superiority, or feasibility, check the premises
that make that comparison valid. State the supported conditional comparison;
do not include an unresolved candidate as confirmed eligible or call a partial
check a complete solution. In unresolved, name the specific missing premise
and affected object or scope, rather than asking to confirm a choice whose
premises have not been established.
4. Check that useful supported facts and current requirements survived this
review. Uncertainty about one condition should not erase other known facts.
Merge repeated claims and keep the packet concise. This is an internal review,
not an extra section, a final task answer, or permission to invent evidence.
"""


def build_support_review_messages(sources):
    """Keep source text and the existing schema; add only the review instruction."""
    messages = build_messages(sources)
    messages[0]["content"] += "\n" + SUPPORT_REVIEW
    return messages
