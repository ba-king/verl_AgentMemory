"""Public-only mode inputs and prompts. No scorer, hidden task or Bank-body access."""
import copy
import re
from ..artifacts import text
from ..single_action import protocol,session
from ..single_action.session import bindings

COMMON = """You are one Memory Teacher with a persistent Memory Bank and an explicit
lifecycle decision_mode. You are separate from the Answer Agent. Use only the
current public input, actual public observations/submitted text, legitimate
retrieved memories and actual action receipts. Bank metadata is not access to
the Bank's contents. At session reset, all short-term context and visible IDs
are cleared; only the Bank persists. No hidden answer or evaluator feedback is
available. A submitted proposal establishes what was submitted, not its correctness.

Choose exactly ONE next memory action. Return exactly one JSON object with keys
"action" and "arguments", conforming to the CURRENT decision_schema. No arrays,
multiple objects, wrappers, native calls, commentary or reasoning. If several
operations are needed, emit only the first, observe its REAL result on the next
turn, then decide again. Use only currently available actions and visible IDs.
YIELD ends this opportunity. Respect the remaining decision budget.
"""

READ_PROMPT = COMMON + """
You are now in READ mode. Your only job is to decide whether the current task
needs information from persistent memory. If all historical information required
by the CURRENT task is already available in the visible state, YIELD. If required
past people, selections, decisions, results or constraints are missing, retrieve
them if that action is available. After a real retrieval result, check whether
the CURRENT task's historical needs are fully covered. If not, you may issue a
more targeted retrieval. Do not yield merely because a retrieval technically
succeeded. Avoid repeating an unproductive query on an unchanged Bank.
If needed history does not exist or cannot be recovered, do not invent it.
Do not write or modify persistent memory in READ mode. Do not plan or solve the
current task for the Answer Agent. Successful retrieval results are delivered
verbatim to the Answer Agent after you YIELD; there is no packet.
"""

CONSOLIDATE_PROMPT = COMMON + """
You are now in CONSOLIDATE mode. The current session is ending. Its short-term
conversation and observations will not be available after reset. submitted_result
is the actual public submission, not a gold answer or evaluator-corrected result.

Your job is to maintain persistent memory so newly established durable state from
THIS session is not lost. Compare what was already known or persisted with what
the submitted result newly establishes. Preserve useful new durable state before
yielding, with its subject and scope. Focus on newly selected entities, person-
specific decisions, exact database object names, dates, prices, IDs, times, routes,
room types, explicit slot values (including '-' or empty values), and relationships
such as one person's state inheriting another's except stated differences.

A compact representation is preferred, but it must allow sufficient reconstruction
of the newly established state. Preserve exact structured values when they matter.
Do not copy the entire trajectory or chain-of-thought. Do not store failed
alternatives, intermediate reasoning, or a merely listed option as a final choice.
Record a submitted plan as submitted; you cannot certify an external booking or
hidden correctness. Do not speculate about future tasks or what will be asked.

If you need to inspect existing related memory to decide whether to create,
revise or remove a record, retrieve it first and use its real result. Prefer
revision over duplication when appropriate. Do not retrieve to solve the now-ended
Answer task. YIELD only when no useful new durable information from this session
remains to be persisted. The runtime does not force a write or fill in missing
fields: you must choose each operation yourself.
"""

SLOT_LABELS={'Current City':'current_city','Transportation':'transportation','Breakfast':'breakfast',
    'Attraction':'attraction','Lunch':'lunch','Dinner':'dinner','Accommodation':'accommodation'}


def public_submission(history):
    """Extract literal fields from the actual public final event; never fill gaps.

    No native canonicalizer: absent fields stay absent, no defaults or correction.
    The raw public text is retained even if it is not a valid plan.
    """
    event=next((e for e in reversed(history) if 'final_itinerary' in e.get('action',{})),None)
    if event is None: raise ValueError('CONSOLIDATE_REQUIRES_ACTUAL_PUBLIC_SUBMISSION')
    content=event['action']['final_itinerary']
    header=re.search(r"(?m)^===\s*(.+?)'s Plan\s*===\s*$",content)
    result={'source_event_id':event['event_id'],'source':'actual_public_submission','public_text':content}
    if header:
        result['traveler']=header.group(1)
        days=list(re.finditer(r'(?m)^Day (\d+):\s*$',content))
        plans=[]
        for i,d in enumerate(days):
            piece=content[d.end():days[i+1].start() if i+1<len(days) else len(content)]
            day={'days':int(d.group(1))}
            for label,key in SLOT_LABELS.items():
                matched=re.findall(r'(?m)^'+re.escape(label)+r':([^\n]*)$',piece)
                if len(matched)==1: day[key]=matched[0].strip()
            plans.append(day)
        result['daily_plans']=plans
    return result


def current_state(bank,current,history,remaining,opportunity):
    state=protocol.current_state(bank,current,history,remaining,opportunity)
    stage=opportunity['stage']
    if stage not in {'session_start','after_travel_tool','after_final_itinerary'}:
        raise ValueError('UNKNOWN_DECISION_STAGE')
    mode='CONSOLIDATE' if stage=='after_final_itinerary' else 'READ'
    allowed=state['available_actions'] if mode=='CONSOLIDATE' else [a for a in state['available_actions'] if a in {'RETRIEVE','YIELD'}]
    state.update(decision_mode=mode,available_actions=allowed,
        decision_schema=protocol.action_schema(allowed,state['visible_memory_ids']))
    if mode=='CONSOLIDATE':
        state['session_public_history']=state.pop('answer_public_history')
        state['submitted_result']=public_submission(state['session_public_history'])
        state['successful_memory_action_receipts']=state.pop('successful_write_receipts')
    return state


def messages_for(state):
    prompt={'READ':READ_PROMPT,'CONSOLIDATE':CONSOLIDATE_PROMPT}[state['decision_mode']]
    descriptions='\n'.join(f'- {a}: {protocol.DESCRIPTIONS[a]}' for a in state['available_actions'])
    return [{'role':'system','content':prompt+'\nCurrent available actions:\n'+descriptions},
            {'role':'user','content':text(state)}]


class ModeSession(session.SingleActionSession):
    def opportunity(self,stage,defer_delivery=False):
        with bindings(session,current_state=current_state,messages_for=messages_for):
            return super().opportunity(stage,defer_delivery=defer_delivery)

    def persist(self,row):
        row['decision_mode']=row['student_input']['decision_mode']
        super().persist(row)
