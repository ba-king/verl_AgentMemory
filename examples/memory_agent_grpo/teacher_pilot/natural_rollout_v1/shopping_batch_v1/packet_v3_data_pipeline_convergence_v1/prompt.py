"""Append the authorized finite language discipline to the original PACKET prompt."""
import difflib
from .common import HERE
from ..packet_interface_v2_validation.interface import PROMPT as ORIGINAL, messages as original_messages

APPENDIX = '''

PACKET_V3_CANONICAL_LANGUAGE_V1 (Shopping text discipline; unchanged tool schema):
New need.text must be exactly one canonical form below. Copy the actual subject,
field and literal label from the CURRENT public task; examples are grammar, not
facts for this task. Do not put a guessed dynamic number or a winner in a need.
CURRENT_FULL|field=material|dimensions=material,price
CURRENT_FULL|field=color|dimensions=color,price
CURRENT_FRAGMENT|subjects=C1,C2|field=material|dimensions=material,price
PREVIOUS_ATTRIBUTE|subject=P2|field=color
PUBLIC_MAPPING|subject=P2|from=color|to=material
INITIAL_ANCHOR|field=material|allowed=glass
REMAINING_BUDGET|basis=prior_purchases
REMAINING_BUDGET|basis=public_cap
LOCAL_INSPECT|subject=C3|purpose=identity
LOCAL_INSPECT|subject=C3|purpose=attribute
LOCAL_RETRY|subject=C3|purpose=search
For CURRENT_FRAGMENT, use actual listed subjects and dimensions field, price, or
field,price. CURRENT_FULL covers every current candidate and both dimensions.
scope_refs must name all subjects of CURRENT_FULL, or exactly the declared
fragment/local subject(s). PREVIOUS_ATTRIBUTE scope is its P subject. Mapping
scope may be its P subject or T0. Initial anchor scope is T0. Budget scope is T0,
all previous P subjects, or the required previous P subject. Use public_cap only
when there are no prior purchases. Candidate-slot scopes cannot be T0.

evidence.fact must be ONE atomic claim. Supported forms (values below are only
examples): material=iron; color=bronze; price=49.99;
remaining_budget=933.05; allowed_material=faux leather; allowed_color=purple;
comparison_ready=true. Each semicolon here separates EXAMPLES; never put multiple
claims in one fact. Price/budget numbers have no currency symbol; units remain
the task currency. Preserve full literal labels including percentages. Do not
infer Material/Color from a name. comparison_ready=true requires all relevant
public premises, correct candidate comparisons and budget to be established.
Keep atomic candidate facts linked to the appropriate candidate inspection need;
previous attributes to the actual historical P need; allowed labels to mapping
or anchor; remaining_budget to budget. Task-level derived facts use T0 and cite
task plus the real source refs establishing all required premises. Cite the
actual retrieved version(s) for historical premises. Resolve a need only when
all claims in its stated scope are supported; local inspection cannot certify
the entire candidate comparison. Retain cited premises for derived claims.

The subject_ref and source_refs namespaces are different:
subject_ref / scope_refs: C1, C2, ..., P1, P2, ..., T0.
source_refs: task, e000, e001, ... actually visible events, exact supplied
versioned Memory IDs, or another explicitly supplied trusted source ID.
Correct: subject_ref=T0, source_refs=["task"], fact="allowed_material=glass"
when the current public anchor really says glass.
Incorrect: source_refs=["T0"], ["C1"], or ["P2"] just because those are subjects.
Never invent a source ID. A subject ID is not a source unless independently
exposed as such. A retrieved P1 record is not evidence that P2 has the same value.

unresolved entries use only MISSING|subject=C3|predicate=material,
MISSING|subject=C4|predicate=price, UNKNOWN|subject=C2|predicate=identity,
or CONFLICT|subject=C3|predicate=material, with actual subject and predicate.
Allowed predicates are material, color, price, identity, remaining_budget,
allowed_material, allowed_color. Use T0 for task-level budget/allowed-label gaps.
Remove a gap once established; keep unresolved premises that remain missing.

Failed-attempt wording supersedes the earlier shorthand "failures/repetitions":
use FAILED_SEARCH|subject=C3|query=... or FAILED_CLICK|subject=C2|target=...
ONLY when that executed action has an explicit publicly failed result (e.g. no
results, not clickable, rejected). Copy the executed query/target and cite that
event. Repeated queries with normal results and reasonable revisits are NOT
failed attempts. A failed search never proves the candidate nonexistent. Do not
invent a next query. If a new observation supplies no required state change,
an unchanged delta is allowed. Delete superseded/wrong facts by their previous
indices, preserve valid prior facts, and maintain the pending/resolved state.
'''

PROMPT = ORIGINAL + APPENDIX


def messages(inp):
    result=original_messages(inp)
    result[0]['content']=PROMPT
    return result


def save_artifacts():
    (HERE/'teacher_packet_prompt_v3_canonical_v1.txt').write_text(PROMPT)
    diff=''.join(difflib.unified_diff(ORIGINAL.splitlines(True),PROMPT.splitlines(True),fromfile='frozen_packet_prompt',tofile='packet_prompt_canonical_v1'))
    (HERE/'prompt_diff.md').write_text('# PACKET prompt only\n\nOriginal bytes retained; finite language/namespace/failure appendix added. No Answer, READ, Bank, tool schema or delta change.\n\n```diff\n'+diff+'\n```\n')
