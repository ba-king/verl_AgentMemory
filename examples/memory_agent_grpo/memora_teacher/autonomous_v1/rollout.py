"""Reference-free autonomous loop. Wrong class is observed, not label-blocked."""
import copy
from pathlib import Path

from ...memora_sft_dev.teacher_rollout_v1.interface import execute_message, restore, snapshot, tools_for
from ..artifacts import canonical, digest, read, save
from .prepare import verify


def input_messages(prompt, bank, item):
    if set(item) != {'current_session'}:
        raise ValueError('ONLY_PUBLIC_SESSION_ALLOWED_NO_HINT')
    session = item['current_session']
    if set(session) != {'date', 'session_id', 'conversation'} or any(
            set(turn) != {'turn', 'speaker', 'message'} for turn in session['conversation']):
        raise ValueError('PRIVATE_SESSION_FIELDS')
    payload = {'mode': 'CONSOLIDATE', 'owner': 'user',
        'bank': [{'memory_id': r['memory_id'], 'record': copy.deepcopy(r['record'])}
                 for r in snapshot(bank)['records']], 'current_session': copy.deepcopy(session)}
    return [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': canonical(payload)}]


def apply_response(body, bank, wrote, seen_call_ids):
    choices = body.get('choices', [])
    if len(choices) != 1 or choices[0].get('finish_reason') != 'tool_calls':
        raise ValueError('ONE_COMPLETE_TOOL_TURN_REQUIRED')
    msg = choices[0]['message']
    calls = msg.get('tool_calls', [])
    if msg.get('role') != 'assistant' or len(calls) != 1:
        raise ValueError('ONE_ASSISTANT_TOOL_CALL_REQUIRED')
    call = calls[0]
    if not isinstance(call.get('id'), str) or not call['id'] or call['id'] in seen_call_ids:
        raise ValueError('INVALID_OR_REUSED_CALL_ID')
    # Same session write budget as the paired run; no expected label is available.
    if wrote and call['function']['name'] != 'YIELD':
        raise ValueError('ONE_WRITE_PER_SESSION_BUDGET')
    result = execute_message(body, bank)
    seen_call_ids.add(call['id'])
    return result


def run_window(prepared, out, client):
    prepared, out = Path(prepared), Path(out)
    if out.exists():
        raise FileExistsError('REFUSE_SECOND_ATTEMPT_OR_OVERWRITE')
    manifest = verify(prepared)
    items = read(prepared/'public/sessions.json')
    ids = [item['current_session']['session_id'] for item in items]
    if ids != manifest['session_ids'] or ids != list(range(ids[0], ids[0]+len(ids))):
        raise ValueError('CONTIGUOUS_SESSION_IDENTITY_FAILED')
    config = read(prepared/'config.json')
    prompt = (prepared/'public/prompt.txt').read_text()
    bank = restore(read(prepared/'public/initial_bank.json'))
    out.mkdir(parents=True)
    save(out/'ATTEMPT_STARTED.json', {'origin': client.evidence_origin,
        'prepared_manifest_digest': digest(manifest), 'label_conditioned': False})
    results, candidates, status = [], [], 'WINDOW_COMPLETED'
    for item in items:
        verify(prepared)
        sid = item['current_session']['session_id']
        dest = out/'sessions'/f'{sid:04d}'
        bank.begin_session()
        bank.visible_ids = set(bank.records)
        bank.visible = snapshot(bank)['records']
        before = snapshot(bank)
        messages = input_messages(prompt, bank, item)
        save(dest/'bank_before.json', before)
        save(dest/'initial_messages.json', messages)
        steps, wrote, seen = [], False, set()
        session_status, error = 'CALL_LIMIT_STOP', None
        for i in range(1, config['max_calls_per_session']+1):
            try:
                verify(prepared)
                tools, _ = tools_for(bank)
                save(dest/f'requests/{i:02d}.json', {'messages': messages, 'tools': tools})
                body, meta = client.call(copy.deepcopy(messages), tools)
                save(dest/f'responses/{i:02d}.json', {'body': body, 'meta': meta})
                step = apply_response(body, bank, wrote, seen)
                step['api'] = meta
                save(dest/f'steps/{i:02d}.json', step)
                steps.append(step)
                messages.extend([step['assistant_message'], step['tool_message']])
                if step['receipt']['status'] == 'REJECTED':
                    session_status = 'EXECUTOR_REJECTION_STOP'
                    break
                if step['action']['action'] == 'YIELD':
                    # No comparison with reference labels, even for early YIELD.
                    session_status = 'YIELDED'
                    break
                wrote = True
            except Exception as exc:
                session_status = 'FAILED_NO_RETRY'
                error = {'type': type(exc).__name__, 'detail': 'See saved raw response/transport; no retry.'}
                break
        after = snapshot(bank)
        save(dest/'bank_after.json', after)
        save(dest/'trajectory_messages.json', messages)
        row = {'session_id': sid, 'status': session_status, 'actions': [s['action']['action'] for s in steps],
            'before_sha256': digest(before), 'after_sha256': digest(after), 'error': error,
            'reasoning_turns': sum(bool(s['assistant_message'].get('reasoning_content')) for s in steps),
            'label_conditioned': False, 'manual_repairs': 0, 'reference_resets': 0}
        save(dest/'result.json', row)
        results.append(row)
        if session_status != 'YIELDED':
            status = 'STOPPED_AT_FIRST_EXECUTION_FAILURE'
            break
        candidates.append({'chain': manifest['chain'], 'session_id': sid,
            'evidence_origin': client.evidence_origin, 'label_conditioned': False,
            'messages': messages, 'tool_schemas_per_turn': [read(dest/f'requests/{i:02d}.json')['tools']
                                                          for i in range(1, len(steps)+1)],
            'state_before_sha256': digest(before), 'state_after_sha256': digest(after),
            'review_status': 'PENDING', 'training_ready': False, 'loss_mask_status': 'NOT_BUILT'})
    save(out/'final_bank.json', snapshot(bank))
    (out/'trajectory_candidates.jsonl').write_text(''.join(canonical(c)+'\n' for c in candidates))
    result = {'status': status, 'sessions': results, 'evidence_origin': client.evidence_origin,
        'planned_sessions': len(items), 'completed_sessions': len(candidates),
        'remaining_sessions_not_run': len(items)-len(results), 'label_conditioned': False,
        'operation_selection_evaluated': True, 'max_writes_per_session': 1,
        'training_ready': False, 'training_runs': 0}
    save(out/'RESULT.json', result)
    return result
