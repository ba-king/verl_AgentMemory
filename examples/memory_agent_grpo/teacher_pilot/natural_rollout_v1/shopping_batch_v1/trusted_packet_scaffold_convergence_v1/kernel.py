"""Same model delta and frozen audit for archived/live maintenance transitions."""
import copy
from . import prompt
from .scaffold.builder import guard_delta
from ..packet_interface_v2_validation.transport import PacketClient as BaseClient
from ..packet_interface_v3_minimal_validation import interface_v3 as V3
from ..packet_v3_data_pipeline_convergence_v1 import audit
from ..packet_v3_live_data_pilot_v1.packet_runtime import public_input

class Client(BaseClient):
    def request(self,inp):
        return {**{k:self.config[k] for k in ('model','temperature','max_tokens','thinking','stream')},
            'messages':prompt.messages(inp),'tools':V3.TOOLS,
            'tool_choice':{'type':'function','function':{'name':'emit_packet_delta'}},'parallel_tool_calls':False}

def model_input(public,previous,trusted):
    inp=public_input(public,previous,trusted)
    inp['operation']='UPDATE_PACKET'
    return inp

def apply_response(row,response,trusted):
    row['raw_response']=response
    try:
        d=V3.parse(response);row.update(schema_valid=True,parsed_packet_delta=d)
        applied=V3.apply(copy.deepcopy(row['previous_packet']),d,trusted['source_ownership_registry'],init=False)
        row.update(apply_valid=True,applied_packet=applied)
    except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)
    return row

def evaluate(row,public,trusted,contract,histories,scaffold):
    # Preserve the frozen auditor's INITIAL PUBLIC CUTOFF obligation checks.
    # This alias exists only in the audit copy, never in Teacher input or V3
    # apply. Renaming the model call must not silently admit empty omissions.
    audit_row=copy.deepcopy(row)
    audit_row['operation']='INIT_PACKET' if row['first_update'] else 'UPDATE_PACKET'
    frozen=audit.evaluate(audit_row,public,trusted,contract,histories)
    guard=guard_delta(row.get('parsed_packet_delta'),scaffold,contract)
    active=(row['applied_packet']!=row['previous_packet']) if row['apply_valid'] else bool(row.get('parsed_packet_delta'))
    return {'frozen_audit':frozen,'scaffold_guard':guard,
        'auditor_phase_alias':audit_row['operation'],
        'alias_reason':'Preserve existing first-cutoff public obligation checks; model/apply use UPDATE.',
        'nonempty':active,'local_pass':bool(frozen['gold_candidate'] and guard['passed']),
        'nonempty_local_gold':bool(active and frozen['gold_candidate'] and guard['passed'])}
