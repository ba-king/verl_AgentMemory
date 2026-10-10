"""Synthetic safety calibration, never labeled as live evidence."""
import copy
import json
import pytest
from .implementation.renderer import *
from .offline import check_one


def fixture(task='Keep exact Material, Color and $30.\n第二行 "quoted"'):
    obs='WebShop [SEP] Instruction: [SEP] '+task+' [SEP] Features [SEP] Material: glass [SEP] Color: Gray [SEP] Price: $12'
    block=json.dumps({'subtask_input':obs,'retrieved_memories':[]},ensure_ascii=True,indent=2)
    return {'model':'frozen','temperature':0,'tools':[{'name':'click','parameters':{}}],
        'messages':[{'role':'system','content':'frozen'},
          {'role':'user','content':'<answer_agent_subtask_json>\n'+block+'\n</answer_agent_subtask_json>'},
          {'role':'user','content':'[Retrieved Memory]\nM1 price=15'},
          {'role':'user','content':'[Working Packet]\n{"evidence":[]}'},
          {'role':'assistant','content':None,'tool_calls':[{'function':{'name':'click','arguments':'{"item":"Features"}'}}]},
          {'role':'tool','tool_call_id':'id1','content':obs},
          {'role':'tool','tool_call_id':'id2','content':obs+' [SEP] error: keep this too'}]}


def test_exact_render_restores_even_embedded_json_whitespace():
    body=fixture();snapshot=copy.deepcopy(body);out=render_dedup(body,role='ANSWER')
    assert body==snapshot and restore_task_refs(out)==body
    assert out.receipt['task_reference_count']==2
    assert all(v is not False for v in check_one(body,out).values())
    assert out.body['messages'][1]==body['messages'][1]
    assert 'Material: glass [SEP] Color: Gray [SEP] Price: $12' in out.body['messages'][-1]['content']


@pytest.mark.parametrize('change',[lambda t:t.upper(),lambda t:t+' ',lambda t:t.replace('$30','$31')])
def test_near_duplicate_never_removed(change):
    body=fixture('Exact Task $30');body['messages']=body['messages'][:5]
    body['messages'].append({'role':'tool','content':'Instruction: [SEP] '+change('Exact Task $30')+' [SEP] price $12'})
    out=render_dedup(body,role='ANSWER');assert out.body==body


def test_json_observation_header_and_instruction_only():
    task='Exact task';p={'current_task':{'instruction':task,'initial_observation':'Instruction: [SEP] '+task+' [SEP] Search'},
        'previous_packet':{'fact':'other'},'retrieved_memories':[{'fact':'other'}],
        'latest_observation':{'observation':'Instruction: [SEP] '+task+' [SEP] purchase price: $10'}}
    b={'messages':[{'role':'user','content':json.dumps(p,indent=3)}]}
    o=render_dedup(b,role='PACKET');assert o.receipt['complete_task_occurrences_after']==1
    assert restore_task_refs(o)==b
    new=json.loads(o.body['messages'][0]['content'])
    assert new['previous_packet']==p['previous_packet'] and new['retrieved_memories']==p['retrieved_memories']


def test_second_raw_task_cannot_be_deleted_to_force_once():
    body=fixture('Task');body['messages'][2]['content']='[Retrieved Memory]\nTask'
    with pytest.raises(ValueError,match='NOT_EXACTLY_ONCE'):render_dedup(body,role='ANSWER')


def test_no_authority_no_answer_request():
    with pytest.raises(ValueError,match='NOT_FOUND'):render_dedup({'messages':[{'role':'tool','content':'some page'}]},role='ANSWER')


def test_reserved_marker_and_modified_render_fail_closed():
    body=fixture();body['messages'][-1]['content']+=MARKER
    with pytest.raises(ValueError,match='RESERVED'):render_dedup(body,role='ANSWER')
    out=render_dedup(fixture(),role='ANSWER');out.body['temperature']=1
    with pytest.raises(ValueError,match='CHANGED'):restore_task_refs(out)


def test_fresh_task_retained_on_each_independent_answer_request():
    for count in (0,2):
        body=fixture()
        if count==0:body['messages']=body['messages'][:4]
        out=render_dedup(body,role='ANSWER')
        assert out.receipt['complete_task_occurrences_after']==1


def test_health_is_explicit_identity_not_fake_task_pass():
    body={'messages':[{'role':'user','content':'Reply OK'}]}
    out=render_dedup(body,role='HEALTH')
    assert out.body==body and out.receipt['authoritative_task'] is None
