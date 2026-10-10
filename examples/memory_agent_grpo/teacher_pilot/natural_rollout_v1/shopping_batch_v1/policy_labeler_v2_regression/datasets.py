"""Offline corpus adapter: preserve V1 labels; add field-verified WRITE credit."""
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from .labeler import ProvenanceLedger, actual_input, audit_write
from ....travel_v0.artifacts import read, rows, save, append
from ....travel_v0.structured_record import policy as frozen
from ....travel_v0.structured_record.record_renderer import render


def downstream_check(chain):
    """Recheck actual delivery/use/action/success, never success alone."""
    try:
        r = read(chain['retrieval_decision_ref'])
        event_path = Path(chain['retrieval_decision_ref']).parents[1] / 'answer_events.jsonl'
        events = {e['event_id']: e for e in rows(event_path)}
        summary = read(event_path.parent / 'session_summary.json')
        assert summary['strict_validation']['passed'] and chain['successful_outcome'] == summary['strict_validation']
        assert chain['source_session'] != chain['retrieval_session']
        assert any(h['memory_id'] == chain['memory_id'] and h['content'] == chain['persisted_content']
                   for h in r['retrieval_result']['hits'])
        assert chain['usage_evidence']
        confirmed = []
        for u in chain['usage_evidence']:
            ev = events[u['event_id']]
            assert ev['execution_status'] == 'success' and ev['action'] == u['action']
            req = read(u['request_ref']); resp = read(u['response_ref'])
            assert any(m.get('content') == chain['answer_visible_packet'] for m in req['messages'])
            if u.get('answer_raw_content') is not None:
                assert resp['choices'][0]['message']['content'] == u['answer_raw_content']
            else:
                assert ev['action']['tool'] == 'search'
            confirmed.append(u['event_id'])
        return {'passed': True, 'confirmed_usage_events': confirmed,
                'strict_outcome_used_only_for_downstream_gate': True}
    except (AssertionError, KeyError, OSError, TypeError) as error:
        return {'passed': False, 'reason': type(error).__name__ + ': ' + str(error)}


def authenticate_write(d):
    state = actual_input(d)
    raw = read(d['response_ref'])
    assert raw == d['raw_teacher_response']
    raw_action, normalization = frozen.parse_decision(raw, d['memory_action_schema'])
    # Exactly the existing parser and its documented formatting normalization.
    # No semantic repair, synthetic field patch or action replacement is added.
    assert raw_action == d['parsed_memory_action']
    assert d['memory_action_result']['status'] == 'SUCCEEDED'
    entry = d['memory_action_result']['record']
    assert entry['record'] == raw_action['arguments']['record']
    assert entry['content'] == render(entry['record'])
    after = read(d['memory_bank_after_ref'])['records']
    before = read(d['memory_bank_before_ref'])['records']
    assert entry in after and entry not in before
    if raw_action['action'] == 'UPDATE':
        assert entry['memory_id'] == raw_action['arguments']['memory_id']
    return state, entry


def evaluate_corpus(root, base_analysis=None):
    root = Path(root); base = Path(base_analysis or root / 'analysis')
    raw = rows(base / 'raw_memory_decisions.jsonl')
    chains = rows(base / 'cross_session_success_chains.jsonl')
    legacy_positive = rows(base / 'positive_sft_candidates.jsonl')
    chain_map = defaultdict(list)
    chain_checks = []
    for i, ch in enumerate(chains, 1):
        check = {'chain_id': f'chain_{i:03d}', **downstream_check(ch)}
        chain_checks.append(check)
        if check['passed']:
            chain_map[(ch['episode_id'], ch['source_decision'])].append((ch, check))
    ledger = {}; write_results = []; results = []; decisions = {}
    legacy_retrieve = {(r['episode_id'], r['decision_id']) for r in legacy_positive
                       if r['action']['action'] == 'RETRIEVE'}
    for row in raw:
        key = (row['episode_id'], row['decision_id']); action = row.get('action') or {}
        d = read(row['source_ref']); decisions[key] = d
        label = 'UNCERTAIN'; detail = None
        if action.get('action') in {'ADD', 'UPDATE'} and action.get('arguments', {}).get('record'):
            history = ledger.setdefault(row['episode_id'], ProvenanceLedger())
            try:
                state, entry = authenticate_write(d)
                linked = [c for c, check in chain_map[key] if c['memory_id'] == entry['memory_id']
                          and c['persisted_content'] == entry['content']]
                detail = audit_write(d, state, history, True, True, bool(linked), row['source_ref'])
                detail['downstream_chain_ids'] = [check['chain_id'] for c, check in chain_map[key] if c in linked]
                history.register(detail, entry)
            except (AssertionError, ValueError, KeyError, OSError, TypeError) as error:
                detail = {'episode_id': row['episode_id'], 'decision_id': row['decision_id'],
                    'action': action['action'], 'label': 'WRITE_UNCERTAIN', 'fields': [],
                    'source_decision_ref': row['source_ref'],
                    'authentication_error': type(error).__name__ + ': ' + str(error)}
            write_results.append(detail)
            label = 'POSITIVE' if detail['label'] == 'WRITE_POSITIVE_STRICT' else 'NEGATIVE' if detail['label'] == 'WRITE_NEGATIVE' else 'UNCERTAIN'
        elif key in legacy_retrieve:
            # Unchanged frozen RETRIEVE eligibility; all corresponding chain gates
            # must retain their actual delivery/use/outcome evidence.
            corresponding = [(c, chk) for c, chk in zip(chains, chain_checks)
                             if (c['episode_id'], c['retrieval_decision']) == key]
            label = 'POSITIVE' if corresponding and all(chk['passed'] for c, chk in corresponding) else 'UNCERTAIN'
        results.append({'episode_id': row['episode_id'], 'decision_id': row['decision_id'],
                        'action': action.get('action'), 'label': label,
                        'source_ref': row['source_ref'], 'write_audit': detail})
    positive = [r for r in results if r['label'] == 'POSITIVE']
    return {'results': results, 'write_results': write_results, 'chain_checks': chain_checks,
            'positive': positive, 'counts': dict(Counter(r['label'] for r in results)),
            'positive_by_action': dict(Counter(r['action'] for r in positive)),
            'positive_by_episode': dict(Counter(r['episode_id'] for r in positive)),
            'legacy_positive_count': len(legacy_positive), 'raw_count': len(raw)}


def export_corpus(root, destination, base_analysis=None, counterfactual=False):
    destination = Path(destination)
    assert not destination.exists(), 'REFUSE_LABELER_OUTPUT_OVERWRITE'
    result = evaluate_corpus(root, base_analysis)
    destination.mkdir(parents=True)
    for name, key in [('decision_labels', 'results'), ('write_field_audits', 'write_results'),
                      ('positive_sft_candidates', 'positive')]:
        path = destination / (name + '.jsonl'); path.write_text('')
        for row in result[key]:
            append(path, row)
    messages = destination / 'positive_sft_messages.jsonl'; messages.write_text('')
    for row in result['positive']:
        d = read(row['source_ref']); state = actual_input(d)
        response = read(d['response_ref'])['choices'][0]['message']['content']
        append(messages, {'episode_id': row['episode_id'], 'decision_id': row['decision_id'],
            'messages': d['memory_teacher_messages'] + [{'role': 'assistant', 'content': response}],
            'label': 'SFT_CANDIDATE_NOT_GOLD', 'counterfactual_only': counterfactual,
            'training_started': False, 'source_ref': row['source_ref']})
    save(destination / 'summary.json', {k: v for k, v in result.items() if k not in {'results', 'write_results', 'positive'}})
    save(destination / 'manifest.json', {'source_root': str(root), 'counterfactual_only': counterfactual,
        'original_labels_modified': False, 'LLM_judge_calls': 0,
        'safety': 'All WRITE fields grounded plus authentic successful Bank write, ownership and frozen downstream chain gate.'})
    return result
