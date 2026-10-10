"""Only append a domain-neutral maintenance rule to the frozen Packet prompt."""
from final_unified_travel_rp_packet_v3_demo_v1 import packet as original
from travel_lifecycle_fix_and_gated_rerun_v1.implementation.packet_runtime import Session as BatchedSession
from teacher_pilot.travel_v0.single_action.session import bindings

STATE_MAINTENANCE_RULE = '''

Current working-state maintenance:
The Packet represents the best current working state, not a chronological or
append-only log. For every newly supported fact, examine existing active evidence
about the same need, subject, and state dimension. Determine whether the old and
new claims can simultaneously remain true in the current state.
If trustworthy new evidence makes an old active claim stale, obsolete,
superseded, or contradictory, retract that old evidence in the SAME delta using
remove_evidence (indices in the previous Packet), then add the supported current
claim. Both having been true at different times does not justify keeping both
active. A corrected value replaces the old value for that same state dimension.
Statements such as "no X has been retrieved yet", "X is still missing", or
"we currently lack Y" describe temporary absence/progress, not durable facts.
Once actual new observations supply that information, retract the obsolete
absence claim and clear its corresponding unresolved entry with set_unresolved.
Retain a missing requirement only if it concerns an explicitly distinct,
still-unestablished premise. Do not equate receipt of some evidence with fully
resolving every requirement or with readiness to commit.
Preserve compatible facts: a new attribute does not invalidate a different
attribute. Preserve distinct subjects and distinct need/state dimensions.
Never delete all evidence for a subject merely because new evidence arrived.
Use need_id, subject_ref, proposition meaning and cited public evidence together;
shared words alone do not establish a contradiction. Do not invent a corrected
value, future result, or final decision to make the Packet appear consistent.
Before emitting the delta, check the resulting active evidence and unresolved
state for incompatible old/new claims and duplicate superseded statements.
'''

PROMPT = original.PROMPT + STATE_MAINTENANCE_RULE


class Client(original.Client):
    def request(self, inp):
        request=super().request(inp)
        request['messages'][0]['content']=PROMPT
        return request


class Session(BatchedSession):
    def __init__(self,*args,**kwargs):
        # Synchronous construction only. Event delivery, Travel adapter, V3
        # parser/apply, source handling and request counts stay unchanged.
        with bindings(original,Client=Client):
            super().__init__(*args,**kwargs)
