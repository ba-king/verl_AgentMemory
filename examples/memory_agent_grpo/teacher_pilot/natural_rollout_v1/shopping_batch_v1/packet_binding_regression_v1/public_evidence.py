"""Bounded public-page extraction for regression, not a new runtime adapter.

Only exact listed titles and actual public navigation establish page ownership.
Names NEVER establish a literal attribute. Future events are sliced before any
identity map, attribute ledger or decision calculation is built.
"""
import re
from decimal import Decimal


def norm(value):
    return re.sub(r'\s+', ' ', str(value)).strip().casefold()


def canonical(value):
    value = norm(value).strip(' .\"\'')
    return {'grey': 'gray', 'aluminium': 'aluminum'}.get(value, value)


def page(observation):
    return observation.split('[SEP] Back to Search [SEP]', 1)[-1]


def atom(subject, predicate, value, kind='literal'):
    if predicate == 'price':
        value = format(Decimal(str(value)).normalize(), 'f')
    return {'subject_ref': subject, 'predicate': predicate,
            'value': canonical(value), 'kind': kind}


def ledger(export, cutoff):
    task = export['current_task']
    names = re.findall(r'^- (.+)$', task.split('Available Options:', 1)[1], re.M)
    entities = {f'C{i}': {'name': name, 'visible_asins': []} for i, name in enumerate(names, 1)}
    by_name = {norm(e['name']): sid for sid, e in entities.items()}
    events = export['public_prefix_events_only'][:cutoff]
    sources = {'task': {'subject_refs': list(entities), 'kind': 'task', 'public_text': task},
               'e000': {'subject_refs': [], 'kind': 'initial', 'public_text': export['initial_observation']['observation']}}
    claims = []; literal_latest = []; asin_map = {}; active = None; view = 'home'
    for memory in export['actual_retrieved_memories']:
        sources[memory['source_ref']] = {'subject_refs': ['history:' + memory['memory_id']],
            'kind': 'retrieved_memory', 'public_text': memory['content'], 'version_sha256': memory['version_sha256']}
    for event in events:
        ref = event['event_id']; action = event['action']; arg = action['arguments']
        ob = page(event['observation']); pieces = [s.strip() for s in ob.split('[SEP]')]
        rejected = 'not clickable' in norm(ob) or 'rejected' in norm(ob)
        owned = []
        if action['tool'] == 'search' and not rejected:
            active = None; view = 'search'
            for i, piece in enumerate(pieces[:-2]):
                if not re.fullmatch(r'B[A-Z0-9]{9}', piece):
                    continue
                title, price = pieces[i + 1], pieces[i + 2]
                if norm(title) not in by_name:
                    continue
                sid = by_name[norm(title)]; asin_map[piece] = sid
                if piece not in entities[sid]['visible_asins']:
                    entities[sid]['visible_asins'].append(piece)
                owned.append(sid)
                if re.fullmatch(r'\$[\d.]+', price):
                    claims.append({**atom(sid, 'price', str(Decimal(price[1:]))),
                        'source_ref': ref, 'literal_span': price})
        elif action['tool'] == 'click' and not rejected:
            item = arg['item']
            if re.fullmatch(r'B[A-Z0-9]{9}', item):
                active = asin_map.get(item); view = 'detail'
            elif norm(item) in {'features', 'description'}:
                view = norm(item)
            elif norm(item) == 'back to search':
                active = None; view = 'home'
            elif norm(item) == '< prev':
                view = 'detail'
            if active:
                owned = [active]
                prices = re.findall(r'Price:\s*\$([\d.]+)', ob)
                for value in prices:
                    claims.append({**atom(active, 'price', str(Decimal(value))),
                        'source_ref': ref, 'literal_span': 'Price: $' + value})
                if view in {'features', 'description'}:
                    for piece in pieces:
                        for match in re.finditer(r'(?i)\b(Material|Color):\s*([^;\n]+)', piece):
                            # Leg Material is a different predicate, not Material.
                            prefix = piece[:match.start()].strip().casefold()
                            if prefix.endswith(('leg', 'frame')):
                                continue
                            predicate = match[1].casefold(); value = match[2].strip().rstrip('.')
                            claims.append({**atom(active, predicate, value), 'source_ref': ref,
                                           'literal_span': match[0]})
        elif active:
            owned = [active]
        sources[ref] = {'subject_refs': list(dict.fromkeys(owned)), 'kind': view,
            'public_text': ob, 'action': action, 'rejected': rejected,
            'query_subject_refs': [sid for sid, e in entities.items() if action['tool'] == 'search'
                                  and norm(arg['keywords']) == norm(e['name'])]}
    literal_latest = [c for c in claims if c['source_ref'] == f'e{cutoff:03d}']
    return {'entities': entities, 'source_subject_map': sources,
        'allowed_claims_up_to_event': claims, 'allowed_claims_from_latest_event': literal_latest,
        'public_events': events}


def derivations(export, ledger_data):
    """Public arithmetic and current-task rule only. Never reads a scorer."""
    task = export['current_task']; memories = export['actual_retrieved_memories']
    previous = re.search(r'Use the Material label of the item actually purchased for Product (\d+)', task)
    if not previous:
        return []
    idx = previous[1]
    memory = next(m for m in memories if f'Subject: "Product {idx} purchase"' in m['content'])
    prev_material = re.search(r'- material = "([^"]+)"', memory['content'])[1]
    remaining = Decimal(re.search(r'- remaining_budget_after_purchase = "\$([\d.]+)"', memory['content'])[1])
    pairs = re.findall(r'Previous Material "([^"]+)" allows current (Material|Color) "([^"]+)"', task)
    _, field, allowed = next(p for p in pairs if canonical(p[0]) == canonical(prev_material))
    field = field.casefold(); known = {}
    for c in ledger_data['allowed_claims_up_to_event']:
        known[c['subject_ref'], c['predicate']] = c
    result = []; eligible = []; complete = True
    for sid, entity in ledger_data['entities'].items():
        label, price = known.get((sid, field)), known.get((sid, 'price'))
        if not label or not price:
            complete = False
            continue
        passed = canonical(label['value']) == canonical(allowed) and Decimal(price['value']) <= remaining
        result.append({**atom(sid, 'eligible', str(passed), 'derived'),
            'required_source_refs': [label['source_ref'], price['source_ref'], memory['source_ref']],
            'required_memory_ref': memory['source_ref'],
            'premises': [atom(sid, field, label['value']), atom(sid, 'price', price['value'])],
            'basis': {'field': field, 'allowed_value': canonical(allowed), 'remaining_budget': str(remaining),
                      'label': label['value'], 'price': price['value'], 'task_ref_implicitly_available': True}})
        if passed:
            eligible.append((Decimal(price['value']), entity['name'], sid))
    if complete and eligible:
        winner = sorted(eligible, key=lambda x: (-x[0], x[1]))[0][2]
        refs = [c['source_ref'] for c in known.values() if c['predicate'] in {field, 'price'}]
        result.append({**atom('decision', 'winner', winner, 'derived'),
            'required_source_refs': sorted(set(refs + [memory['source_ref']])),
            'required_memory_ref': memory['source_ref'],
            'premises': [atom(c['subject_ref'], c['predicate'], c['value']) for c in known.values() if c['predicate'] in {field, 'price'}],
            'basis': {'public_all_candidates_complete': True, 'eligible': [x[2] for x in eligible],
                      'ordering': 'descending price then ascending exact task name', 'task_ref_implicitly_available': True}})
    return result
