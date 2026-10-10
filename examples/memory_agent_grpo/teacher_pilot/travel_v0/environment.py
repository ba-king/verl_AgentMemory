"""Private evaluation boundary and original native Travel tools/submission.

Only current public input and projected final acknowledgement leave this boundary.
The Oracle method is privileged and must never be called by the Split controller.
"""
import copy
import sys
from .artifacts import ACCEPTANCE, ARENA, TRAVEL, IDS, rows, read, public_session

sys.path.insert(0, str(ACCEPTANCE))
sys.path.insert(0, str(ARENA))
import replay_travel as accepted
from env.env_systems.travel_planner_env.tool_schemas import TOOLS


class TravelBoundary:
    def __init__(self, public):
        if public['episode_id'] not in IDS:
            raise ValueError('unapproved_episode')
        self.public = public
        self.hidden = next(r for r in rows(TRAVEL / 'hidden_episodes.jsonl')
                           if r['episode_id'] == public['episode_id'])
        self.executor = accepted.ToolExecutor(str(accepted.old.DB))
        self.env = accepted.native_environment(public, self.hidden)
        city = self.hidden['destination_city']
        self.pools = {
            'meal': self.executor.restaurants.data[self.executor.restaurants.data.City == city].to_dict(orient='records'),
            'hotel': self.executor.accommodations.data[self.executor.accommodations.data.city == city].to_dict(orient='records'),
            'attraction': self.executor.attractions.data[self.executor.attractions.data.City == city].to_dict(orient='records'),
            'flight_records': self.executor.flights.data[self.executor.flights.data['Flight Number'].isin(
                [self.hidden['base_source_provenance'][k]['Flight Number'] for k in ['outbound_flight', 'return_flight']])].to_dict(orient='records')}
        self.actual_history = []
        self.stopped = False

    def reset(self, index):
        if self.stopped or len(self.actual_history) != index:
            raise RuntimeError('invalid_or_incomplete_actual_prefix')
        private_reset = self.env.reset()
        if self.env.history or self.env.step_count:
            raise RuntimeError('native_reset_did_not_clear_history')
        return public_session(self.public, index), private_reset

    def tool(self, name, arguments):
        defs = {t['function']['name']: t['function']['parameters'] for t in TOOLS}
        if name not in defs or not isinstance(arguments, dict):
            raise ValueError('invalid_travel_tool')
        definition = defs[name]
        if not set(definition['required']) <= set(arguments) or set(arguments) - set(definition['properties']):
            raise ValueError('invalid_travel_arguments')
        if any(not isinstance(v, str) for v in arguments.values()):
            raise ValueError('nonstring_travel_arguments')
        return self.executor.execute(name, arguments)

    def submit(self, content, index):
        name = accepted.old.NAMES[index]
        output = self.env.step(content, ground_truth={'name': name,
            'daily_plans': self.hidden['sessions'][index]['valid_answers_or_targets'][0]}, need_judge=True)
        parsed = accepted.canonical_plan(accepted.native._parse_person_plan_from_result(content, name))
        try:
            strict = accepted.strict_plan(parsed, index, self.public, self.hidden, self.pools, self.actual_history)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            strict = {'passed': False, 'reason': str(error)}
        self.actual_history.append(parsed)  # Actual submission, never a gold replacement.
        self.stopped = not strict['passed']
        # The only native observation fields exposed: step and actual final text.
        visible = {k: output[0][k] for k in ['step', 'final']}
        return visible, {'native_score': bool(output[1]), 'native_output': output,
                         'strict_validation': strict, 'parsed_plan': parsed,
                         'termination_reason': 'submitted' if strict['passed'] else 'strict_prefix_failure'}

    def required_facts(self, index):
        """Private diagnostic facts, including the initial-context dependencies."""
        if index == 0:
            return []
        h = self.hidden['sessions'][index]
        base = self.public['initial_observation']['base_person']
        facts = [{'id': 'bootstrap:query', 'kind': 'bootstrap', 'traveler': 'Morgan',
                  'content': base['query'], 'value': base['query'], 'source': 'public:s1:bootstrap:query'}]
        modified = set()
        rules = h['structured_goal']['selection_rules']
        if 'hotel' in rules:
            modified.update([(1, 'accommodation'), (2, 'accommodation')])
        if 'meal' in rules:
            modified.add(tuple(accepted.old.MEAL_SLOTS[index]))
        for d in base['daily_plans']:
            for slot in accepted.old.SLOTS:
                if (d['days'], slot) in modified:
                    continue
                value = d[slot]
                facts.append({'id': f"bootstrap:d{d['days']}:{slot}", 'kind': 'bootstrap_slot',
                    'traveler': 'Morgan', 'day': d['days'], 'slot': slot, 'value': value,
                    'content': f"Morgan confirmed Day {d['days']} {slot}: {value}.",
                    'source': f"public:s1:bootstrap:d{d['days']}:{slot}"})
        for dep in h['depends_on']:
            j = int(dep['session_id'][1:]) - 1
            day, slot = dep['required_fact']['day_slot']
            # Only prior independently validated actual plans; no current target.
            previous = self.actual_history[j] if j < len(self.actual_history) else None
            if previous is None:
                raise RuntimeError('oracle_requires_actual_validated_prefix')
            value = previous[day - 1][slot]
            who = accepted.old.NAMES[j]
            facts.append({'id': f'{dep["session_id"]}:{slot}', 'kind': 'previous_choice',
                'traveler': who, 'day': day, 'slot': slot, 'value': value,
                'content': f'{who} submitted Day {day} {slot}: {value}.',
                'source': f'actual_submission:{dep["session_id"]}:d{day}:{slot}'})
            if 'price' in dep['dependency_type']:
                kind = dep['required_fact']['kind']
                record = next(r for r in self.pools[kind] if accepted.old.named(r, kind) == value)
                attrs = {'listed_price': accepted.old.price(record, kind)}
                if kind == 'hotel':
                    attrs['room_type'] = record['room type']
                for key, val in attrs.items():
                    facts.append({'id': f'{dep["session_id"]}:{key}', 'kind': 'previous_attribute',
                        'traveler': who, 'entity': value, 'field': key, 'value': val,
                        'content': f'{who}\'s submitted {value} has public {key}: {val}.',
                        'source': f'public_database:{kind}:{value}:{key}'})
        return facts

    def oracle(self, index):
        return [{'content': f['content']} for f in self.required_facts(index)]

    def close(self):
        self.env.close()
