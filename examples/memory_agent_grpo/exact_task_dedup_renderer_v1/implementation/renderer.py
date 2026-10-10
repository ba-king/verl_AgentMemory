"""Exact task-reference rendering; no runtime, model, tokenizer or environment imports.

The original trusted task field is the authoritative anchor on EVERY request.
Messages, tools, raw retrieval and Packet are not rearranged. Only recognized
task fields / native observation instruction headers can be elided. Receipts
stay local; only ``RenderedRequest.body`` is sent to the provider.
"""
import copy
import hashlib
import json
import re
from dataclasses import dataclass

MARKER = '<REF:AUTHORITATIVE_TASK>'
WRAPPER = re.compile(r'<answer_agent_subtask_json>\s*(.*?)\s*</answer_agent_subtask_json>', re.S)
HEADER = re.compile(r'^(?:WebShop \[SEP\] )?Instruction: \[SEP\] (.*?) \[SEP\]', re.S)
PROTECTED = {'retrieved_memories','new_retrieved_memories','visible_memories','visible_retrieved_memories',
             'previous_packet','packet','working_packet','tool_calls','arguments','tool_arguments',
             'decision_schema','schema','memory_action_schema'}
OBSERVATION_KEYS = {'observation','environment_observation','initial_observation','latest_observation',
                    'current_observation','tool_result','result','subtask_input','public_text'}
TASK_KEYS = {'instruction','task','current_task'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Leaf:
    message: int
    path: tuple
    start: int
    end: int
    value: str
    json_encoded: bool


@dataclass
class RenderedRequest:
    body: dict
    receipt: dict


def leaves_for_content(content, index):
    """Track exact JSON string offsets without reserializing unrelated values."""
    if not isinstance(content, str):return []
    wrapper = WRAPPER.search(content)
    text = wrapper.group(1) if wrapper else content
    offset = wrapper.start(1) if wrapper else 0
    try:json.loads(text)
    except (ValueError, TypeError):return [Leaf(index, (), 0, len(content), content, False)]
    decoder = json.JSONDecoder();leaves = []
    def ws(i):
        while i < len(text) and text[i].isspace():i += 1
        return i
    def walk(i, path):
        i = ws(i);start = i
        if text[i] == '{':
            i = ws(i + 1)
            while text[i] != '}':
                key, i = decoder.raw_decode(text, i);i = ws(i)
                assert text[i] == ':'
                i = ws(walk(i + 1, path + (key,)))
                if text[i] == ',':i = ws(i + 1)
                else:break
            assert text[i] == '}'
            return i + 1
        if text[i] == '[':
            i = ws(i + 1);j = 0
            while text[i] != ']':
                i = ws(walk(i, path + (j,)));j += 1
                if text[i] == ',':i = ws(i + 1)
                else:break
            assert text[i] == ']'
            return i + 1
        value, end = decoder.raw_decode(text, i)
        if isinstance(value, str):leaves.append(Leaf(index, path, offset + start, offset + end, value, True))
        return end
    walk(0, ())
    return leaves


def request_leaves(body):
    return [leaf for i,m in enumerate(body.get('messages', [])) for leaf in leaves_for_content(m.get('content'), i)]


def authoritative_task(body, leaves=None):
    leaves = request_leaves(body) if leaves is None else leaves
    for leaf in leaves:
        if body['messages'][leaf.message]['role'] != 'user':continue
        if leaf.path in (('instruction',), ('current_task',), ('current_task','instruction'), ('task',), ('task','instruction')):
            return leaf.value,leaf
        if leaf.path == ('subtask_input',):
            match = HEADER.match(leaf.value)
            if match:return match.group(1),leaf
    return None,None


def allowed_replacement(leaf, message, task):
    if any(k in PROTECTED for k in leaf.path):return None
    content = message.get('content') or ''
    if message['role'] in ('system','assistant'):return None
    if content.startswith(('[Retrieved Memory]', '[Working Packet]')):return None
    # An exact task field is distinct from a coincidentally similar fact.
    if leaf.path and leaf.path[-1] in TASK_KEYS and leaf.value == task:
        return MARKER,'EXACT_TASK_FIELD'
    is_observation = message['role'] == 'tool' or any(k in OBSERVATION_KEYS for k in leaf.path)
    if not is_observation:return None
    match = HEADER.match(leaf.value)
    if match and match.group(1) == task:
        return leaf.value[:match.start(1)] + MARKER + leaf.value[match.end(1):],'EXACT_NATIVE_INSTRUCTION_HEADER'
    return None


def render_dedup(original_input, *, role):
    """Create a provider body and a local reversible receipt; fail closed."""
    body = copy.deepcopy(original_input)
    leaves = request_leaves(body);task,anchor = authoritative_task(body, leaves)
    if role == 'ANSWER' and not task:raise ValueError('ANSWER_AUTHORITATIVE_TASK_NOT_FOUND')
    if any(MARKER in leaf.value for leaf in leaves):raise ValueError('RESERVED_MARKER_ALREADY_PRESENT')
    edits=[]
    if task:
        for leaf in leaves:
            if leaf == anchor:continue
            replacement=allowed_replacement(leaf, body['messages'][leaf.message], task)
            if replacement is None:continue
            value,reason=replacement
            content=body['messages'][leaf.message]['content']
            encoded=json.dumps(value,ensure_ascii=False) if leaf.json_encoded else value
            edits.append({'message':leaf.message,'path':list(leaf.path),'old_start':leaf.start,'old_end':leaf.end,
                          'before':content[leaf.start:leaf.end],'after':encoded,'reason':reason,
                          'original_value':leaf.value,'rendered_value':value})
        for i,message in enumerate(body['messages']):
            relevant=sorted([e for e in edits if e['message']==i],key=lambda e:e['old_start'])
            if not relevant:continue
            source=message['content'];parts=[];cursor=0;outsize=0
            for edit in relevant:
                unchanged=source[cursor:edit['old_start']];parts.append(unchanged);outsize+=len(unchanged)
                edit['new_start']=outsize;parts.append(edit['after']);outsize+=len(edit['after'])
                cursor=edit['old_end']
            parts.append(source[cursor:]);message['content']=''.join(parts)
    after_leaves=request_leaves(body)
    def exact_occurrences(value):
        # A similar instruction with an added space/constraint is NOT a duplicate.
        header=HEADER.match(value)
        if header:return int(header.group(1)==task)
        return value.count(task)
    full_occurrences=sum(exact_occurrences(l.value) for l in after_leaves) if task else None
    anchor_count=next((l.value.count(task) for l in after_leaves if anchor and (l.message,l.path)==(anchor.message,anchor.path)),None)
    if role=='ANSWER' and (anchor_count!=1 or full_occurrences!=1):
        raise ValueError(f'ANSWER_TASK_NOT_EXACTLY_ONCE: {full_occurrences}')
    receipt={'version':'EXACT_TASK_DEDUP_V1','role':role,'authoritative_task':task,
             'authoritative_location':{'message':anchor.message,'path':list(anchor.path)} if anchor else None,
             'authoritative_anchor_occurrences':anchor_count,'complete_task_occurrences_after':full_occurrences,
             'task_reference_count':len(edits),'edits':edits,'original_sha256':digest(original_input),
             'rendered_sha256':digest(body),'metadata_sent_to_provider':False}
    rendered=RenderedRequest(body,receipt)
    if restore_task_refs(rendered)!=original_input:raise ValueError('RESTORE_NOT_EXACT')
    return rendered


def restore_task_refs(rendered):
    """Undo only verified elisions; preserve even embedded-JSON whitespace."""
    body=copy.deepcopy(rendered.body);receipt=rendered.receipt
    if digest(body)!=receipt['rendered_sha256']:raise ValueError('RENDERED_BODY_CHANGED')
    for edit in sorted(receipt['edits'],key=lambda e:(e['message'],e['new_start']),reverse=True):
        i=edit['message'];start=edit['new_start'];text=body['messages'][i]['content'];end=start+len(edit['after'])
        if text[start:end]!=edit['after']:raise ValueError('REFERENCE_RECEIPT_MISMATCH')
        body['messages'][i]['content']=text[:start]+edit['before']+text[end:]
    if digest(body)!=receipt['original_sha256']:raise ValueError('RESTORED_BODY_CHANGED')
    return body
