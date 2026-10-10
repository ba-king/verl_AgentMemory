"""Model-independent teacher-format reader, with explicit supervision fields."""
import copy
import json


def validate(row):
    assert row['audit']['accepted'] is True and row['audit']['replay_pass'] is True
    assert row['audit']['semantic_pass'] is True
    messages=row['messages']; assert len(messages) in (4,6)
    assert [m['role'] for m in messages[:2]]==['system','user']
    indices=list(range(2,len(messages),2)); seen=set()
    assert [x['assistant_message_index'] for x in row['tools_by_turn']]==indices
    assert [x['assistant_message_index'] for x in row['supervision']]==indices
    for i,s in zip(indices,row['supervision']):
        m,t=messages[i:i+2]; assert m['role']=='assistant' and t['role']=='tool'
        assert len(m['tool_calls'])==1; call=m['tool_calls'][0]
        assert call['id'] not in seen and t['tool_call_id']==call['id'];seen.add(call['id'])
        assert json.loads(t['content'])['status']=='SUCCEEDED'
        json.loads(call['function']['arguments'])
        assert 'tool_calls' in s['fields']
        assert ('reasoning_content' in s['fields'])==bool(m.get('reasoning_content','').strip())
        if call['function']['name']!='YIELD' or len(indices)==1:
            assert m.get('reasoning_content','').strip(), 'MISSING_DECISION_REASONING'
    assert messages[-2]['tool_calls'][0]['function']['name']=='YIELD'


class SessionDataset:
    def __init__(self,path):
        with open(path) as f:self.rows=[json.loads(line) for line in f if line.strip()]
        assert len({r['sample_id'] for r in self.rows})==len(self.rows)
        for r in self.rows:validate(r)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return copy.deepcopy(self.rows[i])


def requests_for_session(row):
    validate(row)
    tools={t['assistant_message_index']:t['tools'] for t in row['tools_by_turn']}
    for s in row['supervision']:
        i=s['assistant_message_index']
        yield {'sample_id':row['sample_id'],'group_id':row['group_id'],
            'assistant_message_index':i,
            'input':{'messages':copy.deepcopy(row['messages'][:i]),'tools':copy.deepcopy(tools[i])},
            'target':copy.deepcopy(row['messages'][i]),'supervision':copy.deepcopy(s)}


def labels_from_verified_spans(input_ids,spans):
    labels=[-100]*len(input_ids);weights=[0.0]*len(input_ids)
    for start,end,weight in spans:
        if not (0<=start<end<=len(input_ids) and weight>0):raise ValueError('BAD_TARGET_SPAN')
        if any(weights[start:end]):raise ValueError('OVERLAPPING_TARGET_SPANS')
        labels[start:end]=input_ids[start:end];weights[start:end]=[float(weight)]*(end-start)
    return {'labels':labels,'loss_weights':weights}
