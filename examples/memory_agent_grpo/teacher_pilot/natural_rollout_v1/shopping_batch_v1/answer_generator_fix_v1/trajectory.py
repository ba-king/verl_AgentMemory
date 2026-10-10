"""Read-only public trajectory reconstruction and descriptive loop metrics."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'policy_labeler_v2_regression'
IDS = [f'shopping_joint_v1_{kind}_cumulative_{i:03d}' for kind in ('material','mixed') for i in (0,1)]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def norm(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def body(observation):
    """Ignore repeated task prefix/terminal transcript, retain actual page text."""
    observation = observation.split('WEBSHOPPLUS SOFT EXIT')[0].rstrip('=\n ')
    marker = '[SEP] Back to Search [SEP]'
    if marker in observation:
        observation = observation.split(marker, 1)[1]
    elif observation.endswith('[SEP] Search'):
        observation = '[PUBLIC_SEARCH_HOME]'
    return observation


def page_key(observation):
    page = body(observation)
    # Page numbers advancing through an empty result list are not new evidence.
    if 'Page ' in page and not re.search(r'\bB[A-Z0-9]{9}\b', page):
        page = re.sub(r'Page \d+', 'Page [EMPTY]', page)
    return norm(page)


def repetition(text):
    words = re.findall(r"[\w$%.]+", norm(text))
    grams = [tuple(words[i:i+8]) for i in range(max(0, len(words)-7))]
    counts = Counter(grams)
    return {'word_count': len(words), 'repeated_8gram_ratio':
            (sum(n-1 for n in counts.values()) / len(grams)) if grams else 0,
            'repeated_phrases': [{'text': ' '.join(g), 'count': n}
                for g,n in counts.most_common(12) if n >= 3],
            'metric_note': 'Lexical repeated 8-gram occurrence fraction, not hidden reasoning tokens or semantic truth.'}


def actions(response):
    out = []
    for c in response['choices'][0]['message'].get('tool_calls', []):
        f = c.get('function', {}); arguments = f.get('arguments')
        try: arguments = json.loads(arguments)
        except (TypeError, ValueError): pass
        out.append({'tool': f.get('name'), 'arguments': arguments})
    return out


def session(root, eid, sid, prefix_valid):
    dest = Path(root)/'episodes'/eid/sid
    summary = read(dest/'session_summary.json')
    events = [json.loads(s) for s in (dest/'answer_events.jsonl').read_text().splitlines()]
    event_map = defaultdict(list)
    for e in events: event_map[Path(e['request_ref']).parent.name].append(e)
    calls = []
    for p in sorted((dest/'api').glob('answer_*/request.json')):
        request = read(p); response = read(p.with_name('response.json'))
        msg = response['choices'][0]['message']; i = int(p.parent.name.rsplit('_',1)[1])
        calls.append({'answer_index': i, 'request_ref': str(p), 'response_ref': str(p.with_name('response.json')),
            'request_sha256': sha(p), 'response_sha256': sha(p.with_name('response.json')),
            'actual_request': request, 'actual_response': response, 'content': msg.get('content') or '',
            'finish_reason': response['choices'][0]['finish_reason'], 'usage': response.get('usage', {}),
            'actions': actions(response), 'events': event_map[p.parent.name],
            'content_repetition': repetition(msg.get('content') or '')})
    failed = not summary['strict_validation']['passed']
    return {'episode_id': eid, 'session_id': sid, 'public_input': read(dest/'public_input.json'),
        'prefix_state_on_entry': 'PREFIX_VALID' if prefix_valid else 'PREFIX_ALREADY_BROKEN',
        'failure_prefix_class': ('SAME_SESSION_ROOT_FAILURE' if prefix_valid else 'PREFIX_ALREADY_BROKEN') if failed else None,
        'strict_outcome_offline_only': summary['strict_validation'], 'summary_ref': str(dest/'session_summary.json'),
        'strict_pass': not failed, 'native_purchase_completed': summary.get('purchase_completed', False),
        'termination': summary['termination_reason'], 'runner_error': summary.get('answer_error_code'),
        'error': summary.get('error'), 'infrastructure_invalid': summary.get('infrastructure_invalid', False),
        'calls': calls, 'source_trajectory_is_public': True,
        'outcome_scope': 'Scorer result used only offline for prefix partition, never copied into replay messages.'}


def loops(s):
    findings = []; queries = defaultdict(list); asins = defaultdict(list); pages = defaultdict(list)
    feature_pages = defaultdict(list); active = None; flat = []
    for c in s['calls']:
        for e in c['events']:
            i=c['answer_index']; a=e['action']; arg=a['arguments']; flat.append((i,a,e))
            if a['tool']=='search': queries[norm(arg['keywords'])].append(i); active=None
            if a['tool']=='click' and re.fullmatch(r'B[A-Z0-9]{9}',arg.get('item','')):
                asins[arg['item']].append(i)
                if e['execution_status']=='success': active=arg['item']
            if e['execution_status']=='success':
                if a['tool']=='click' and norm(arg.get('item',''))=='back to search': active=None
                key=page_key(e['observation']); pages[key].append(i)
                if active and a['tool']=='click' and norm(arg.get('item','')) in ('features','description'):
                    feature_pages[(active,key)].append(i)
    def add(kind, indices, key, extra=None):
        chosen=[c for c in s['calls'] if c['answer_index'] in indices]
        findings.append({'episode_id':s['episode_id'],'session_id':s['session_id'],'kind':kind,
            'answer_indices':indices,'loop_length':len(indices),'span_calls':max(indices)-min(indices)+1,
            'normalized_repeated_content':key,'tool_sequence':[a for c in chosen for a in c['actions']],
            'prefix_state':s['prefix_state_on_entry'],'failure_prefix_class':s['failure_prefix_class'],
            'session_eventually_failed':not s['strict_pass'],
            'causal_failure_claim':False, **(extra or {})})
    for name,group in [('REPEATED_SEARCH_QUERY',queries),('REPEATED_ASIN_CLICK',asins),('REVISITED_PAGE',pages)]:
        for key, indices in group.items():
            if len(indices)>1: add(name,indices,key)
    for (asin,key), indices in feature_pages.items():
        if len(indices)>1:add('CANDIDATE_RECHECK_LOOP',indices,key,{'asin':asin})
    # Consecutive identical navigation and exact consecutive same-page actions.
    for key_fn,kind in [(lambda v:digest(v[1]),'REPEATED_CONSECUTIVE_ACTION'),
                        (lambda v:page_key(v[2]['observation']),'SAME_STATE_STAGNATION')]:
        start=0
        while start<len(flat):
            end=start+1
            while end<len(flat) and key_fn(flat[end])==key_fn(flat[start]):end+=1
            if end-start>=2:add(kind,[v[0] for v in flat[start:end]],key_fn(flat[start]))
            start=end
    for a,b in zip(s['calls'],s['calls'][1:]):
        wa=set(re.findall(r'\w+',norm(a['content']))); wb=set(re.findall(r'\w+',norm(b['content'])))
        if len(wa)>=12 and len(wb)>=12 and len(wa&wb)/len(wa|wb)>=.85:
            add('CONSECUTIVE_SIMILAR_CONTENT',[a['answer_index'],b['answer_index']],norm(a['content']),
                {'word_set_jaccard':len(wa&wb)/len(wa|wb)})
    return findings
