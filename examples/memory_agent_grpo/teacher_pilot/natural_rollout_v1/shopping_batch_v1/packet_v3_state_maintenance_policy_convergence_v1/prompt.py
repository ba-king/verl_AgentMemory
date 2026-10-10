"""One appended algorithm and four examples; prior prompt remains byte-exact."""
import difflib

from .common import HERE
from ..trusted_packet_scaffold_convergence_v1 import prompt as frozen

STATE_MAINTENANCE_ALGORITHM = '''

STATE_MAINTENANCE_ALGORITHM
On each call: OBSERVE -> MATERIALIZE -> DERIVE -> BIND -> PROGRESS -> CLEANUP -> READINESS.
1. Read pending scaffold needs. Use task, retrieved memories, previous Packet and latest public observation; do not recreate global needs (only runtime LOCAL_INSPECT/LOCAL_RETRY may be added).
2. Materialize directly known atomic predicate=value facts, including first-update task anchor and cap. Do not emit empty delta when such required facts are absent.
3. With a public mapping and verified historical premise, record BOTH: the premise under its actual Pk, and allowed_material/allowed_color under T0, even if the mapping need scopes Pk. All remaining_budget facts also belong to T0.
4. Cite actual source_refs: direct fact -> its source; mapping -> task + premise source; budget -> task cap + all necessary verified purchase sources or a certified cumulative source. Subject IDs are not sources. Uncertain premises cannot justify conclusions.
5. Materialize sufficient proof, then resolve the satisfied need. Resolve removes pending by frozen apply semantics; explicit remove_pending uses existing need IDs, NOT indices. Do not resolve unsupported needs. Keep unseen candidate attributes/prices pending; names do not prove literal attributes.
6. In this same delta, replace set_unresolved to remove markers now satisfied by evidence; retain genuinely missing/conflicting facts. Remove wrong/stale evidence only with supported replacement.
7. Set ready only when every gating requirement has sufficient proof. Otherwise keep false. Emit empty delta only if no required change remains.
Examples (abbreviated operations, use the frozen tool schema):
A. Pending anchor; task says Material=glass: add T0 allowed_material=glass [task]; resolve anchor. Materialize task cap as remaining_budget too.
B. P2 color=bronze [M@version]; task maps bronze -> faux leather: add P2 color=bronze [M@version] and T0 allowed_material=faux leather [task,M@version]; resolve previous/mapping needs with proof.
C. CURRENT_FULL; only candidate names visible: keep pending, no invented material/price, no resolve.
D. Old MISSING|subject=P4|predicate=color; now verified clear: add P4 color=clear [actual source]; remove that marker via set_unresolved; resolve its need if satisfied.
'''

PROMPT = frozen.PROMPT + STATE_MAINTENANCE_ALGORITHM


def messages(inp):
    result = frozen.messages(inp)
    result[0]['content'] = PROMPT
    return result


def save_artifacts():
    assert len(STATE_MAINTENANCE_ALGORITHM) <= 2500
    (HERE / 'teacher_packet_prompt_v3_state_maintenance_v1.txt').write_text(PROMPT)
    diff = ''.join(difflib.unified_diff(
        frozen.PROMPT.splitlines(True), PROMPT.splitlines(True),
        fromfile='frozen_scaffold_prompt', tofile='state_maintenance_v1'))
    (HERE / 'prompt_diff.md').write_text(
        '# One appended state-maintenance algorithm\n\n'
        f'Added characters: {len(STATE_MAINTENANCE_ALGORITHM)}; prior prompt unchanged.\n\n'
        '```diff\n' + diff + '\n```\n')
