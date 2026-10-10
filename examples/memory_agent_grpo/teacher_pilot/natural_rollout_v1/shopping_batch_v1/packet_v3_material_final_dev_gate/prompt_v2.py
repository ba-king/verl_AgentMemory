"""Only the authorized short Teacher policy appendix changes model input."""
import difflib
from .common import HERE
from ..packet_v3_data_pipeline_convergence_v1.prompt import PROMPT as V1, messages as prior_messages

RULES = '''

SCOPE_AND_INIT_RULES_V2
1. CURRENT_FULL scope_refs must equal ALL current candidate refs.
2. CURRENT_FRAGMENT subjects in need.text must exactly equal scope_refs.
3. PREVIOUS_ATTRIBUTE field must match the public task dependency; price is
   budget evidence, unless previous price is explicitly the dependency.
4. Scope is responsibility, NOT provenance. For REMAINING_BUDGET use ["T0"].
   Cite historical proof in source_refs using real source IDs, not P subjects.
5. Pending and resolved are disjoint: never add_pending and resolve the same
   need in one delta. Resolve only with sufficient evidence; clear its existing
   pending flag via remove_pending. Otherwise leave it pending.
6. subject_ref names the entity asserted about; source_refs name public sources.
   Never substitute namespaces.
Compact examples (grammar only, not task facts):
- CURRENT_FRAGMENT|subjects=C1|field=material|dimensions=material,price with
  scope_refs=["C1"]: valid. Text subjects=C1,C2 but scope_refs=["C1"]: invalid.
- REMAINING_BUDGET|basis=prior_purchases with scope_refs=["T0"]: valid;
  scope_refs=["T0","P1"]: invalid. P1's real Memory ID belongs in source_refs.
- New N2 fully supported: resolve=["N2"], add_pending=[]; unsupported: reverse.
  resolve=["N2"] AND add_pending=["N2"]: invalid.
'''
PROMPT = V1 + RULES

def messages(inp):
    result = prior_messages(inp)
    result[0]['content'] = PROMPT
    return result

def save_artifacts():
    dest = HERE/'prompt'; dest.mkdir(exist_ok=True)
    (dest/'teacher_packet_prompt_v3_canonical_v2.txt').write_text(PROMPT)
    diff = ''.join(difflib.unified_diff(V1.splitlines(True), PROMPT.splitlines(True),
        fromfile='canonical_v1', tofile='canonical_v2'))
    (dest/'prompt_diff.md').write_text('# PACKET prompt only\n\nOriginal prompt retained verbatim. Six short structural rules and three examples appended.\n'
        f'Characters: {len(V1)} → {len(PROMPT)} (+{len(RULES)}, {len(RULES)/len(V1):.1%}).\n'
        'Examples are grammar only. Budget T0 is already allowed by the frozen contract.\n\n```diff\n'+diff+'\n```\n')
