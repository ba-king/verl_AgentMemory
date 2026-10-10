"""Scaffold ownership contract plus the single requested budget-source rule."""
from .common import HERE
from ..packet_v3_material_final_dev_gate.prompt_v2 import PROMPT as PREVIOUS, messages as previous_messages

SCAFFOLD_RULE='''

TRUSTED_SCAFFOLD_RUNTIME
The supplied Previous Packet already contains every trusted global gating need.
The first call is FIRST_PACKET_UPDATE, executed with UPDATE_PACKET semantics.
Maintain evidence, progress, repairs and readiness. Do not add global gating
needs again; add_needs is reserved for dynamic LOCAL_INSPECT or LOCAL_RETRY.
'''
BUDGET_RULE='''For REMAINING_BUDGET evidence, cite the public task source for the cap plus the
actual prior-purchase/certified cumulative sources required by the proof route.
'''
PROMPT=PREVIOUS+SCAFFOLD_RULE+BUDGET_RULE

def messages(inp):
    result=previous_messages(inp);result[0]['content']=PROMPT;return result

def save_artifacts():
    import difflib
    (HERE/'teacher_packet_scaffold_prompt.txt').write_text(PROMPT)
    diff=''.join(difflib.unified_diff(PREVIOUS.splitlines(True),PROMPT.splitlines(True),fromfile='canonical_v2',tofile='scaffold_runtime'))
    (HERE/'prompt_diff.md').write_text('# Runtime ownership and budget provenance only\n\n```diff\n'+diff+'\n```\n')
