"""Predeclared read-only paired measurements; no scorer/prompt/actor mutation."""
import json
import statistics
from collections import Counter
from pathlib import Path
from .labeler import actual_input, public_facts, ownership, norm, role
from ....travel_v0.artifacts import read, rows


def token_counter():
    from transformers import AutoTokenizer
    from ....travel_v0.artifacts import WORKSPACE
    tokenizer = AutoTokenizer.from_pretrained(str(WORKSPACE / 'models/Qwen3-Embedding-0.6B'), local_files_only=True)
    return lambda records: sum(len(tokenizer.encode(r['content'], add_special_tokens=False)) for r in records)


def api_accounting(root):
    roles = {}; status = Counter(); returned_costs = []
    for p in sorted(Path(root).glob('episodes/*/p*/api/*/request.json')):
        role_name = 'Memory' if p.parent.name.startswith('memory_teacher') else 'Answer'
        v = roles.setdefault(role_name, {'HTTP_attempts': 0, 'successful_responses': 0, 'prompt_tokens': 0,
            'completion_tokens': 0, 'total_tokens': 0, 'cache_hit_tokens': 0, 'cache_miss_tokens': 0,
            'prompt_tokens_samples': [], 'input_chars_samples': [], 'length_responses': 0})
        v['HTTP_attempts'] += 1
        request = read(p)
        v['input_chars_samples'].append(len(json.dumps(request['messages'], ensure_ascii=False)))
        transport = p.with_name('transport.json')
        status[str(read(transport)['http_status']) if transport.exists() else 'NO_RESPONSE_STATUS_RECORDED'] += 1
        response = p.with_name('response.json')
        if not response.exists(): continue
        body = read(response); usage = body.get('usage', {})
        v['successful_responses'] += 1
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            v[key] += usage.get(key, 0)
        v['cache_hit_tokens'] += usage.get('prompt_cache_hit_tokens', usage.get('prompt_tokens_details', {}).get('cached_tokens', 0))
        v['cache_miss_tokens'] += usage.get('prompt_cache_miss_tokens', 0)
        v['prompt_tokens_samples'].append(usage.get('prompt_tokens', 0))
        v['length_responses'] += int(body.get('choices', [{}])[0].get('finish_reason') == 'length')
        for key in ('cost', 'total_cost', 'monetary_cost'):
            if key in usage or key in body: returned_costs.append({'response_ref': str(response), 'key': key, 'value': usage.get(key, body.get(key))})
    for v in roles.values():
        for key in ('prompt_tokens_samples', 'input_chars_samples'):
            a = v[key]
            v[key + '_mean'] = statistics.mean(a) if a else None
            v[key + '_max'] = max(a) if a else None
    return {'by_role': roles, 'total_HTTP_attempts': sum(v['HTTP_attempts'] for v in roles.values()),
            'HTTP_status_counts': dict(status), 'provider_returned_cost_fields': returned_costs,
            'monetary_cost': None, 'cost_note': 'No monetary amount inferred from tokens. Missing provider billing fields remain unknown.',
            'scope': 'Logical trajectory API directories; V1 historical billing/failed attempts reported separately, replay copies not charged again.'}


def retention_session(dest, eid, session_summary):
    # Include a completed actual Memory request even if no parsed decision follows.
    sources = []
    for req in sorted((dest / 'api').glob('memory_teacher_*/request.json')):
        request = read(req); state = json.loads(request['messages'][1]['content'])
        if state.get('decision_mode') != 'CONSOLIDATE': continue
        transport = req.with_name('transport.json')
        if transport.exists() and read(transport).get('http_status') != 200: continue
        facts = public_facts(state, str(req))
        if facts['completed_purchase']: sources.append((req, state, facts))
    answer_events = rows(dest / 'answer_events.jsonl')
    if not sources:
        return {'session': dest.name, 'eligible': 0, 'retained': 0, 'visible_but_use_unestablished': 0,
                'reason': 'NO_ACTUAL_CONSOLIDATE_PUBLIC_PURCHASE_INPUT', 'facts': [],
                'completed_purchase': bool(session_summary.get('purchase_completed'))}
    req, state, facts = sources[-1]
    field = facts['current_attribute']; fact = facts['facts'].get(field)
    if fact is None or field in facts['conflicts']:
        return {'session': dest.name, 'eligible': 0, 'retained': 0, 'visible_but_use_unestablished': 0,
                'reason': 'CURRENT_TASK_ATTRIBUTE_NOT_UNAMBIGUOUSLY_VISIBLE', 'facts': [],
                'public_facts': facts}
    usages = []
    for e in answer_events:
        if not e.get('response_ref') or e.get('execution_status') != 'success': continue
        content = read(e['response_ref'])['choices'][0]['message'].get('content') or ''
        if field in norm(content) and norm(fact['value']) in norm(content):
            usages.append({'event_id': e['event_id'], 'action': e['action'], 'response_ref': e['response_ref'],
                           'answer_content': content})
    # Conservative observable-use gate, applied identically to V1/V2. Visible
    # attributes without verbal use are exposed separately, never silently lost.
    eligible = bool(usages) and bool(session_summary.get('purchase_completed'))
    bank = read(dest / 'bank_after.json')['records']
    matches = []
    name_only = []
    for entry in bank:
        own = ownership(entry['record'], facts, eid)
        if not own['passed']: continue
        for f in entry['record']['fields']:
            if role(f['path']) == field and norm(f['value']) == norm(fact['value']):
                matches.append({'memory_id': entry['memory_id'], 'field': f, 'record': entry['record']})
            elif role(f['path']) == 'name' and norm(fact['value']) in norm(f['value']):
                name_only.append({'memory_id': entry['memory_id'], 'field': f})
    return {'session': dest.name, 'eligible': int(eligible), 'retained': int(eligible and bool(matches)),
            'visible_but_use_unestablished': int(not eligible),
            'eligibility_rule': 'Actual completed purchase; chosen-product explicit page label in actual Memory input; current public task field; actual Answer content uses that role/value during executed action. This is observable validation evidence, not causal attribution.',
            'source_request_ref': str(req), 'asin': facts['asin'], 'field': field, 'value': fact['value'],
            'fact': fact, 'public_declarations': facts['literal_declarations'],
            'actual_Answer_usage_evidence': usages, 'correctly_bound_retained_fields': matches,
            'name_tokens_only_do_not_count': name_only,
            'classification': 'RETAINED' if matches else 'OMITTED', 'public_facts': facts}


def collect(root, label_result, count_tokens):
    from .config import IDS
    root = Path(root); table = []
    retrieval = rows(root / 'analysis/retrieval_attribution.jsonl')
    chains = rows(root / 'analysis/cross_session_success_chains.jsonl')
    all_labels = label_result['results']
    for eid in IDS:
        ep_root = root / 'episodes' / eid
        summary_path = ep_root / 'episode_summary.json'
        if not summary_path.exists():
            table.append({'episode_id': eid, 'status': 'NOT_RUN', 'complete': False}); continue
        ep = read(summary_path); sessions = []
        for s in ep['sessions']:
            dest = ep_root / s['session_id']; bank = read(dest / 'bank_after.json')['records']
            decisions = rows(dest / 'memory_decisions.jsonl')
            events = rows(dest / 'answer_events.jsonl')
            ret = retention_session(dest, eid, s)
            field_counts = Counter(role(f['path']) or 'OTHER_ROLE' for r in bank for f in r['record']['fields'])
            current_account = api_accounting_session(dest)
            sessions.append({'session': s['session_id'], 'strict_success': s['strict_validation']['passed'],
                'native_purchase_completed': s.get('purchase_completed', False),
                'termination': s.get('termination_reason'), 'error': s.get('error'),
                'infrastructure_invalid': s.get('infrastructure_invalid', False),
                'retention': ret, 'Bank_records': len(bank), 'Bank_fields': sum(len(r['record']['fields']) for r in bank),
                'Bank_chars': sum(len(r['content']) for r in bank), 'Bank_tokens_Qwen_proxy': count_tokens(bank),
                'Bank_field_roles': dict(field_counts), 'Memory_decisions': len(decisions),
                'ADD_count': sum((d.get('parsed_memory_action') or {}).get('action') == 'ADD' for d in decisions),
                'rejected_native_actions': [e for e in events if e['execution_status'] != 'success'],
                'navigation_failure': not s.get('purchase_completed', False) and not s.get('infrastructure_invalid', False) and not str(s.get('error', '')).endswith('INCOMPLETE_MODEL_OUTPUT_length'),
                'api': current_account, 'session_summary_ref': str(dest / 'session_summary.json')})
        labels = [r for r in all_labels if r['episode_id'] == eid]
        pos = Counter(r['action'] for r in labels if r['label'] == 'POSITIVE')
        rc = [r for r in retrieval if r['episode_id'] == eid]; ec = [c for c in chains if c['episode_id'] == eid]
        rank = [c['top5_rank'] for c in ec]
        eligible = sum(s['retention']['eligible'] for s in sessions)
        retained = sum(s['retention']['retained'] for s in sessions)
        answer_prompts = [n for s in sessions for n in s['api']['Answer']['prompt_tokens_samples']]
        memory_prompts = [n for s in sessions for n in s['api']['Memory']['prompt_tokens_samples']]
        table.append({'episode_id': eid, 'status': ep['status'], 'complete': len(sessions) == 6 and ep['status'] == 'COMPLETED_NATURAL_ATTEMPT',
            'purchases': sum(s['strict_success'] for s in sessions), 'native_purchases': sum(s['native_purchase_completed'] for s in sessions),
            'raw_Memory_decisions': len(labels), 'ADD_count': sum(s['ADD_count'] for s in sessions),
            'positive_by_action': dict(pos), 'positive_WRITE': pos['ADD'] + pos['UPDATE'],
            'positive_RETRIEVE': pos['RETRIEVE'], 'total_positive_SFT': sum(pos.values()),
            'label_counts': dict(Counter(r['label'] for r in labels)), 'success_chains': len(ec),
            'success_chain_details': ec, 'useful_rank_samples': rank,
            'useful_rank_median': statistics.median(rank) if rank else None,
            'attribute_retention': {'eligible': eligible, 'retained': retained, 'rate': retained / eligible if eligible else None},
            'true_Top5_miss': sum(r['category'] == 'TRUE_TOP5_MISS' for r in rc),
            'present_but_unused': sum(r['category'] == 'TOP5_PRESENT_BUT_UNUSED' for r in rc),
            'usage_unestablished': sum('USAGE_NOT_ESTABLISHED' in r['reason'] for r in rc),
            'Answer_truncations': sum(s['api']['Answer']['length_responses'] for s in sessions),
            'navigation_failures': sum(s['navigation_failure'] for s in sessions),
            'final_Bank_records': sessions[-1]['Bank_records'] if sessions else 0,
            'final_Bank_tokens_Qwen_proxy': sessions[-1]['Bank_tokens_Qwen_proxy'] if sessions else 0,
            'final_Bank_fields': sessions[-1]['Bank_fields'] if sessions else 0,
            'Answer_mean_prompt_tokens': statistics.mean(answer_prompts) if answer_prompts else None,
            'Memory_mean_prompt_tokens': statistics.mean(memory_prompts) if memory_prompts else None,
            'sessions': sessions})
    return table


def api_accounting_session(dest):
    result = {r: {'requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'cache_hit_tokens': 0,
                   'cache_miss_tokens': 0, 'prompt_tokens_samples': [], 'length_responses': 0} for r in ('Answer', 'Memory')}
    for request in (dest / 'api').glob('*/request.json'):
        r = result['Memory' if request.parent.name.startswith('memory_teacher') else 'Answer']
        r['requests'] += 1
        path = request.with_name('response.json')
        if not path.exists(): continue
        response = read(path); u = response.get('usage', {})
        r['prompt_tokens'] += u.get('prompt_tokens', 0); r['completion_tokens'] += u.get('completion_tokens', 0)
        r['cache_hit_tokens'] += u.get('prompt_cache_hit_tokens', 0); r['cache_miss_tokens'] += u.get('prompt_cache_miss_tokens', 0)
        r['prompt_tokens_samples'].append(u.get('prompt_tokens', 0))
        r['length_responses'] += int(response['choices'][0].get('finish_reason') == 'length')
    return result


def judge(v1, v2, labeler_validated):
    from .config import IDS, JUDGMENT as j
    old = {r['episode_id']: r for r in v1}; new = {r['episode_id']: r for r in v2}
    m1 = old[IDS[1]]; m2 = new[IDS[1]]; x1 = old[IDS[2]]; x2 = new[IDS[2]]
    complete = all(r.get('complete') for r in v2)
    if not complete:
        return {'POLICY_RESULT': None, 'policy_verdict_evidence_complete': False,
                'verdict_note': 'Incomplete regression cannot establish improvement; infrastructure is not Teacher failure.',
                'LABELER_RESULT': 'LABELER_FIX_VALIDATED' if labeler_validated else 'LABELER_UNSAFE_OR_FAILED',
                'SCALE_DECISION': 'DO_NOT_SCALE_YET'}
    mixed0 = x1['attribute_retention']; mixed = x2['attribute_retention']
    bank_ratio = m2['final_Bank_tokens_Qwen_proxy'] / max(1, m1['final_Bank_tokens_Qwen_proxy'])
    context_ratio = (m2['Answer_mean_prompt_tokens'] or 0) / max(1, m1['Answer_mean_prompt_tokens'] or 1)
    rank_delta = (m2['useful_rank_median'] or 0) - (m1['useful_rank_median'] or 0)
    retention_improved = mixed['rate'] is not None and mixed['rate'] > (mixed0['rate'] or 0)
    bloat = bank_ratio > j['material_bank_bloat_ratio']
    context_worse = context_ratio > j['material_answer_mean_context_growth_ratio']
    retrieval_worse = m2['true_Top5_miss'] > m1['true_Top5_miss'] or rank_delta > j['material_rank_median_degradation']
    material_regression = m1['purchases'] - m2['purchases'] >= j['material_clear_regression_success_loss']
    clear = retention_improved and mixed['rate'] >= j['clear_mixed_retention_min_rate'] and mixed['eligible'] >= j['minimum_mixed_eligible_for_scale'] and m2['purchases'] >= j['clear_material_min_success'] and m2['success_chains'] >= j['clear_material_min_chains'] and not (bloat or context_worse or retrieval_worse)
    if material_regression:
        policy_result = 'POLICY_REGRESSION'
    elif clear:
        policy_result = 'POLICY_CLEAR_IMPROVEMENT'
    elif retention_improved:
        policy_result = 'POLICY_PARTIAL_IMPROVEMENT'
    else:
        policy_result = 'POLICY_NO_IMPROVEMENT'
    write_count = sum(r['positive_WRITE'] for r in v2)
    chain_count = sum(r['success_chains'] for r in v2)
    scale = (policy_result in {'POLICY_CLEAR_IMPROVEMENT', 'POLICY_PARTIAL_IMPROVEMENT'} and labeler_validated
        and mixed['eligible'] >= j['minimum_mixed_eligible_for_scale'] and (mixed['rate'] or 0) >= j['clear_mixed_retention_min_rate']
        and m2['purchases'] >= 5 and not (bloat or context_worse or retrieval_worse) and write_count > 0 and chain_count > 0)
    return {'POLICY_RESULT': policy_result, 'policy_verdict_evidence_complete': True,
        'LABELER_RESULT': 'LABELER_FIX_VALIDATED' if labeler_validated else 'LABELER_UNSAFE_OR_FAILED',
        'SCALE_DECISION': 'RUN_REMAINING_8' if scale else 'DO_NOT_SCALE_YET',
        'side_effect_flags': {'material_purchase_regression': material_regression, 'material_Bank_bloat': bloat,
            'material_Answer_context_growth': context_worse, 'material_retrieval_degradation': retrieval_worse},
        'material_Bank_token_ratio': bank_ratio, 'material_Answer_mean_context_ratio': context_ratio,
        'material_useful_rank_median_delta': rank_delta,
        'judgment_limit': 'Single paired development run; changed navigation/content length is not proven caused by memory size. Raw counts and natural trajectory evidence remain primary.'}
