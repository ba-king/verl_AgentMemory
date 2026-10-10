"""Deterministic real-CSV Travel pilot. No network, model, reward-pipeline edits."""
from __future__ import annotations

import ast
import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MA = next(p for p in HERE.parents if (p / 'pyproject.toml').is_file()
          and (p / 'examples/memory_agent_grpo').is_dir()).parent / 'MemoryArena'
DB = MA / 'env/env_systems/travel_planner_env/database'
SEED = 2026091307
NAMES = ['Avery','Blair','Casey','Drew','Ellis']
MEAL_SLOTS = [(2,'dinner'), (3,'lunch'), (1,'dinner'), (2,'breakfast'), (3,'dinner')]
SLOTS = ['current_city','transportation','breakfast','attraction','lunch','dinner','accommodation']
FILES = {'meal':'restaurants/clean_restaurant_2022.csv','hotel':'accommodations/clean_accommodations_2022.csv',
         'attraction':'attractions/attractions.csv','flight':'flights/clean_Flights_2022.csv'}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',',':'), ensure_ascii=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def normalized(text):
    return re.sub(r'[^a-z0-9]+',' ',text.lower()).strip()


def object_normalized(text):
    return re.sub(r'\([^()]*\)\s*$', '', text.strip()).strip().lower()


def named(row,kind):
    return row['Name' if kind != 'hotel' else 'NAME'] + ', ' + row['city' if kind=='hotel' else 'City']


def price(row,kind):
    return float(row['Average Cost' if kind=='meal' else 'price'])


def selection(rows,kind,rule,previous=None):
    """Verifier: enumerate original runtime-visible rows, apply predicates, then rank."""
    eligible=[]
    for row in rows:
        if kind=='hotel' and (float(row['minimum nights'])>2 or float(row['maximum occupancy'])<1):
            continue
        if kind=='hotel' and 'room_type' in rule and row['room type'] != rule['room_type']:
            continue
        if rule['op']=='same':
            if named(row,kind)!=previous:
                continue
        elif rule['op']=='min_ge':
            if price(row,kind)<rule['minimum_price']:
                continue
        elif rule['op']=='next_price':
            refs=[x for x in rows if named(x,kind)==previous]
            if len(refs)!=1:
                raise ValueError('ambiguous_or_missing_reference')
            if price(row,kind)<=price(refs[0],kind):
                continue
            if kind=='hotel' and row['room type']!=refs[0]['room type']:
                continue
        else:
            raise ValueError('unknown_operation')
        eligible.append(row)
    if not eligible:
        raise ValueError('unsatisfiable')
    eligible.sort(key=lambda x:(price(x,kind),x['Name' if kind=='meal' else 'NAME']))
    if rule['op']=='same':
        best=eligible
    else:
        best=[x for x in eligible if (price(x,kind),x['Name' if kind=='meal' else 'NAME']) ==
              (price(eligible[0],kind),eligible[0]['Name' if kind=='meal' else 'NAME'])]
    if len(best)!=1:
        raise ValueError('ambiguous_identity')
    return best[0],len(eligible)


def plan_text(name,plan):
    lines=[f"=== {name}'s Plan ==="]
    for day in plan:
        lines.append(f"Day {day['days']}:")
        for slot in SLOTS:
            lines.append(f"{slot.replace('_',' ').title()}: {day[slot]}")
    return '\n'.join(lines)


def instruction(index,city,rules):
    who=NAMES[index]
    previous=NAMES[index-1] if index else None
    parts=[f'I am {who}. I am joining Morgan for this three-day trip to {city}.',
           'Keep every unmentioned entry from Morgan\'s confirmed itinerary. Make a separate booking for this single adult.']
    # JSON canonical serialization sorts keys; wording must not depend on dict insertion order.
    for kind in ['meal','hotel']:
        if kind not in rules:continue
        rule=rules[kind]
        if kind=='meal':
            day,slot=MEAL_SLOTS[index]
            target=f'{slot} on day {day}'
            if index:
                pd,ps=MEAL_SLOTS[index-1]
                ref=f"{previous}'s {ps} on day {pd}"
            domain=f'a restaurant in {city}'
        else:
            target='the accommodation on both nights'
            ref=f"{previous}'s first-night accommodation" if index else ''
            room_constraint=f"room type {rule['room_type']}, " if 'room_type' in rule else ''
            domain=f"an accommodation in {city} with {room_constraint}minimum nights at most 2 and maximum occupancy at least 1"
        if rule['op']=='same':
            parts.append(f'For {target}, use the same choice as {ref}.')
        elif rule['op']=='min_ge':
            parts.append(f"For {target}, choose {domain} with the lowest {'average cost' if kind=='meal' else 'nightly price'} that is at least ${rule['minimum_price']:g}.")
        else:
            parts.append(f"For {target}, choose {domain} with the lowest {'average cost' if kind=='meal' else 'nightly price'} strictly above that of {ref}." +
                         (' It must also have the same room type as that earlier booking.' if kind=='hotel' else ''))
        if rule['op']!='same':
            parts.append('Break an equal-price tie by ascending exact database name (case-sensitive).')
    parts.append(f"Output only the complete three-day plan in the standard === {who}'s Plan === format.")
    return '\n'.join(parts)


def fill_plan(base,index,chosen):
    plan=copy.deepcopy(base)
    for kind,obj in chosen.items():
        if kind=='meal':
            day,slot=MEAL_SLOTS[index]
            plan[day-1][slot]=named(obj,kind)
        else:
            for day in (0,1):
                plan[day]['accommodation']=named(obj,kind)
    return plan


def native_pure():
    import difflib
    ns={'re':re,'difflib':difflib,'SIM_TH':.7,'SLOTS':SLOTS}
    src=ast.parse((MA/'env/env_systems/travel_env.py').read_text())
    nodes=[x for x in src.body if isinstance(x,ast.FunctionDef) and x.name in {'_similarity','_get_day','_parse_person_plan_from_result'}]
    cl=next(x for x in src.body if isinstance(x,ast.ClassDef) and x.name=='TravelPlannerEnvironment')
    evaluator=copy.deepcopy(next(x for x in cl.body if isinstance(x,ast.FunctionDef) and x.name=='_evaluate_slots'))
    nodes.append(evaluator)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'native_travel_pure','exec'),ns)
    return ns


def official_inventory():
    official=[json.loads(s) for s in (MA/'benchmark_data/fixed100_source/group_travel_planner.jsonl').read_text().splitlines()]
    question_hashes={digest(q) for row in official for q in row['questions']}
    norm_hashes={digest(normalized(q)) for row in official for q in row['questions']}
    plan_hashes={digest(plan) for row in official for plan in row['answers']}
    windows={digest(row['answers'][i:i+5]) for row in official for i in range(len(row['answers'])-4)}
    objects=set()
    for row in official:
        for plan in [row['base_person']['daily_plans'],*row['answers']]:
            objects.update(plan_objects(plan))
    return official,question_hashes,norm_hashes,plan_hashes,windows,objects


def plan_objects(plan):
    values=[]
    for d in plan:
        for k in ['breakfast','lunch','dinner','accommodation']:
            if d[k]!='-':values.append(object_normalized(d[k]))
        values.extend(object_normalized(x) for x in d['attraction'].split(';') if x and x!='-')
        values.extend(re.findall(r'F\d+',d['transportation']))
    return set(values)


def validate_physical_plan(plan,hidden,pools,budget):
    flights=hidden['base_source_provenance']
    f=flights['outbound_flight'];ret=flights['return_flight']
    date,end=hidden['flight_dates']
    if (pd.Timestamp(end)-pd.Timestamp(date)).days!=2 or f['FlightDate']!=date or ret['FlightDate']!=end:
        raise ValueError('invalid_dates')
    if f['OriginCityName']!=ret['DestCityName'] or f['DestCityName']!=ret['OriginCityName']:
        raise ValueError('disconnected_route')
    if not (f['DepTime']<f['ArrTime']<='16:00' and '20:00'<=ret['DepTime']<ret['ArrTime']):
        raise ValueError('unsupported_flight_schedule')
    if 'flight_records' in pools:
        for record in [f,ret]:
            if record not in pools['flight_records']:
                raise ValueError('flight_not_in_database')
    trans=lambda x:f"Flight Number: {x['Flight Number']}, from {x['OriginCityName']} to {x['DestCityName']}, Departure Time: {x['DepTime']}, Arrival Time: {x['ArrTime']}"
    expected_cities=[f"from {f['OriginCityName']} to {f['DestCityName']}",f['DestCityName'],f"from {ret['OriginCityName']} to {ret['DestCityName']}"]
    if len(plan)!=3 or [d['days'] for d in plan]!=[1,2,3] or [d['current_city'] for d in plan]!=expected_cities:
        raise ValueError('invalid_day_city_sequence')
    if [d['transportation'] for d in plan]!=[trans(f),'-',trans(ret)]:
        raise ValueError('invalid_transportation')
    total=float(f['Price'])+float(ret['Price'])
    for d in plan:
        for slot in ['breakfast','lunch','dinner']:
            if d[slot]=='-':continue
            matches=[x for x in pools['meal'] if named(x,'meal')==d[slot]]
            if len(matches)!=1:raise ValueError('unknown_or_ambiguous_restaurant')
            total+=price(matches[0],'meal')
        if d['accommodation']!='-':
            matches=[x for x in pools['hotel'] if named(x,'hotel')==d['accommodation']]
            if len(matches)!=1:raise ValueError('unknown_or_ambiguous_accommodation')
            if matches[0]['minimum nights']>2 or matches[0]['maximum occupancy']<1:
                raise ValueError('incompatible_stay')
            total+=price(matches[0],'hotel')
        if d['attraction']!='-' and not any(named(x,'attraction')==d['attraction'] for x in pools['attraction']):
            raise ValueError('unknown_attraction')
    if plan[0]['accommodation']!=plan[1]['accommodation'] or plan[2]['accommodation']!='-':
        raise ValueError('invalid_contiguous_hotel_stay')
    if total>budget:raise ValueError('budget_exceeded')
    return total


def validate_episode(public,hidden,pools,native):
    if set(public)!={'schema_version','episode_id','task_family','initial_observation','context_policy','sessions'}:
        raise ValueError('public_top_level_fields_violation')
    if public['schema_version']!='memoryarena.synthetic.v0' or public['task_family']!='travel':
        raise ValueError('invalid_public_schema')
    if public['context_policy']!={'context_reset_between_sessions':True,'replay_initial_observation':False,
                                   'initial_observation_delivery':'once_before_first_session'}:
        raise ValueError('invalid_context_policy')
    if len(public['sessions'])!=5 or len(hidden['sessions'])!=5:
        raise ValueError('invalid_session_count')
    if set(public['initial_observation'])!={'base_person'} or set(public['initial_observation']['base_person'])!={'name','query','daily_plans'}:
        raise ValueError('invalid_initial_observation_fields')
    if public['episode_id']!=hidden['episode_id']:
        raise ValueError('id_mismatch')
    base=public['initial_observation']['base_person']['daily_plans']
    validate_physical_plan(base,hidden,pools,10000)
    prior={}; count=0
    for i,(ps,hs) in enumerate(zip(public['sessions'],hidden['sessions'])):
        if set(ps)!={'session_id','instruction','environment_config'}:
            raise ValueError('public_session_fields_violation')
        expected_config={'environment':'travel_planner','database_root':str(DB),
                            'tool_schema_path':str(MA/'env/env_systems/travel_planner_env/tool_schemas.py'),
                            'action_style':'native_function_call_then_formatted_plan','max_steps':30,
                            'context_reset_between_sessions':True,'replay_initial_observation':False}
        if ps['environment_config']!=expected_config:
            raise ValueError('public_environment_config_violation')
        if ps['session_id']!=f's{i+1}':raise ValueError('invalid_session_order')
        if ps['session_id']!=hs['session_id']:
            raise ValueError('session_id_mismatch')
        rules=hs['structured_goal']['selection_rules']
        expected_instruction=instruction(i,hidden['destination_city'],rules)
        if ps['instruction']!=expected_instruction:
            raise ValueError('instruction_goal_mismatch')
        chosen={}
        for kind,rule in rules.items():
            prev=prior.get(kind)
            target,eligible=selection(pools[kind],kind,rule,prev)
            chosen[kind]=target
            if target is None or eligible<1:raise ValueError('no_answer')
            if i and named(target,kind) in ps['instruction']:
                raise ValueError('active_answer_leaked')
            if kind=='meal':
                current_day,current_slot=MEAL_SLOTS[i]
                base_value=base[current_day-1][current_slot]
            else:
                base_value=base[0]['accommodation']
            if named(target,kind)==base_value:
                raise ValueError('base_copy_shortcut')
            if named(target,kind) in canonical(public['initial_observation']):
                raise ValueError('active_target_in_bootstrap')
        expected=fill_plan(base,i,chosen)
        if hs['valid_answers_or_targets']!=[expected]:
            raise ValueError('incorrect_or_ambiguous_gold')
        if hs['structured_goal']['plan']!=expected:
            raise ValueError('structured_plan_mismatch')
        validate_physical_plan(expected,hidden,pools,hs['structured_goal']['constraints']['budget'])
        for kind,source in hs['source_provenance'].items():
            if source['record'] not in pools[kind] or named(source['record'],kind)!=hs['structured_goal']['active_targets'][kind]:
                raise ValueError('invalid_source_provenance')
        parsed=native['_parse_person_plan_from_result'](plan_text(NAMES[i],expected),NAMES[i])
        if not native['_evaluate_slots'](None,parsed,expected):
            raise ValueError('native_roundtrip_failure')
        if i:
            if len(hs['depends_on'])!=len(rules) or len(hs['dependency_witness'])!=len(rules):
                raise ValueError('dependency_count_mismatch')
            for dep,w in zip(hs['depends_on'],hs['dependency_witness']):
                kind=w['kind']
                if dep['session_id'] != public['sessions'][i-1]['session_id']:
                    raise ValueError('bad_dependency_reference')
                if dep['required_fact']['value']!=prior[kind]:
                    raise ValueError('incorrect_required_fact')
                expected_type=('same_restaurant' if kind=='meal' else 'same_accommodation') if rules[kind]['op']=='same' else ('relative_meal_price' if kind=='meal' else 'relative_hotel_price_same_room_type')
                expected_slot=list(MEAL_SLOTS[i-1]) if kind=='meal' else [1,'accommodation']
                if dep['dependency_type']!=expected_type or dep['required_fact']['kind']!=kind or list(dep['required_fact']['day_slot'])!=expected_slot:
                    raise ValueError('incorrect_dependency_semantics')
                if w['current_instruction']!=ps['instruction']:
                    raise ValueError('counterfactual_changed_current_input')
                # Independently replay both legal prefixes from their own first request.
                replayed=[]
                for label in ['actual_history','alternative_history']:
                    prefix=w[label]
                    if len(prefix)!=i:raise ValueError('incomplete_counterfactual_history')
                    prior_cf=None
                    for j,item in enumerate(prefix):
                        if item['instruction']!=instruction(j,hidden['destination_city'],item['all_selection_rules']):
                            raise ValueError('counterfactual_upstream_instruction_mismatch')
                        obj,_=selection(pools[kind],kind,item['all_selection_rules'][kind],prior_cf)
                        if named(obj,kind)!=item['selected_value']:
                            raise ValueError('counterfactual_illegal_prefix')
                        prior_cf=named(obj,kind)
                    result,_=selection(pools[kind],kind,rules[kind],prior_cf)
                    replayed.append(named(result,kind))
                if replayed[0]==replayed[1] or replayed!=w['current_targets']:
                    raise ValueError('dependency_not_effective')
                if hs['structured_goal']['active_targets'][kind]!=replayed[0]:
                    raise ValueError('counterfactual_actual_target_mismatch')
                count+=1
        prior={k:named(v,k) for k,v in chosen.items()}
    return count


def main():
    rng=random.Random(SEED)
    frames={k:pd.read_csv(DB/v).dropna() for k,v in FILES.items()}
    # Runtime returns no stable IDs; disallow ambiguous same-name city rows in selected cities.
    restaurants=frames['meal']; hotels=frames['hotel']; attractions=frames['attraction']; flights=frames['flight']
    pools_by_city={}
    for city,g in restaurants.groupby('City'):
        if g.Name.duplicated().any():continue
        h=hotels[hotels.city==city]
        t=attractions[attractions.City==city]
        if len(t)<2 or t.Name.duplicated().any() or h.NAME.duplicated().any():continue
        h=h[(h['minimum nights']<=2)&(h['maximum occupancy']>=1)]
        room_types=[typ for typ,hg in h.groupby('room type') if hg.price.nunique()>=5]
        if g['Average Cost'].nunique()<5 or not room_types:continue
        pools_by_city[city]={'meal':g.to_dict(orient='records'),'hotel':h.to_dict(orient='records'),
                             'attraction':t.to_dict(orient='records'),'room_types':sorted(room_types)}
    # Flight raw rows are genuine database records. Conservative visible-clock filters avoid overnight ambiguity.
    outbound=flights[(flights.FlightDate>='2022-03-01')&(flights.FlightDate<='2022-03-27') &
                    (flights.DepTime<flights.ArrTime)&(flights.ArrTime<='16:00') &
                    flights.DestCityName.isin(pools_by_city)]
    returns=flights[(flights.FlightDate>='2022-03-03')&(flights.FlightDate<='2022-03-29')&
                   (flights.DepTime>='20:00')&(flights.DepTime<flights.ArrTime)]
    ret_groups={key:g for key,g in returns.groupby(['OriginCityName','DestCityName','FlightDate'])}
    outbound_records=outbound.to_dict(orient='records')
    rng.shuffle(outbound_records)
    official,qhash,nhash,phash,windows,official_objects=official_inventory()
    native=native_pure(); public_rows=[];hidden_rows=[];seen_routes=set();rejections=Counter();edge_counts=Counter()
    kinds_by_template={'meal_copy_price_chain':['meal'],'hotel_copy_price_chain':['hotel'],
                       'mixed_meal_hotel_chain':['meal','hotel']}
    route_index=0
    for template,kinds in kinds_by_template.items():
        for ordinal in range(8):
            while True:
                f=outbound_records[route_index];route_index+=1
                dest=f['DestCityName'];origin=f['OriginCityName'];date=f['FlightDate']
                end=(pd.Timestamp(date)+pd.Timedelta(days=2)).strftime('%Y-%m-%d')
                key=(dest,origin,end)
                if key not in ret_groups:
                    rejections['no_late_return_flight']+=1;continue
                route=(origin,dest,date)
                if route in seen_routes:
                    rejections['duplicate_route_date']+=1;continue
                seen_routes.add(route)
                ret=ret_groups[key].iloc[rng.randrange(len(ret_groups[key]))].to_dict()
                pool=pools_by_city[dest];room_type=rng.choice(pool['room_types'])
                generated_candidates={}
                for kind in kinds:
                    group=[x for x in pool[kind] if kind!='hotel' or x['room type']==room_type]
                    byprice={}
                    for x in sorted(group,key=lambda x:(price(x,kind),x['Name' if kind=='meal' else 'NAME'])):
                        byprice.setdefault(price(x,kind),x)
                    values=list(byprice.values())
                    start=rng.randrange(len(values)-3)
                    generated_candidates[kind]=values[start:start+4]
                excluded_hotels={named(x,'hotel') for x in generated_candidates.get('hotel',[])}
                excluded_meals={named(x,'meal') for x in generated_candidates.get('meal',[])}
                selected_base_hotel=rng.choice([x for x in pool['hotel'] if named(x,'hotel') not in excluded_hotels])
                base_meals=rng.sample([x for x in pool['meal'] if named(x,'meal') not in excluded_meals],7)
                base_attr=rng.sample(pool['attraction'],2)
                trans=lambda x:f"Flight Number: {x['Flight Number']}, from {x['OriginCityName']} to {x['DestCityName']}, Departure Time: {x['DepTime']}, Arrival Time: {x['ArrTime']}"
                base=[{'days':1,'current_city':f'from {origin} to {dest}','transportation':trans(f),'breakfast':'-',
                       'attraction':'-','lunch':'-','dinner':named(base_meals[0],'meal'),'accommodation':named(selected_base_hotel,'hotel')},
                      {'days':2,'current_city':dest,'transportation':'-','breakfast':named(base_meals[1],'meal'),
                       'attraction':named(base_attr[0],'attraction'),'lunch':named(base_meals[2],'meal'),
                       'dinner':named(base_meals[3],'meal'),'accommodation':named(selected_base_hotel,'hotel')},
                      {'days':3,'current_city':f'from {dest} to {origin}','transportation':trans(ret),
                       'breakfast':named(base_meals[4],'meal'),'attraction':named(base_attr[1],'attraction'),
                       'lunch':named(base_meals[5],'meal'),'dinner':named(base_meals[6],'meal'),'accommodation':'-'}]
                eid=f'travel_syn_{SEED}_{len(public_rows):03d}'
                public={'schema_version':'memoryarena.synthetic.v0','episode_id':eid,'task_family':'travel',
                        'initial_observation':{'base_person':{'name':'Morgan',
                            'query':f'I am Morgan. My single-adult three-day trip from {origin} to {dest} is confirmed for {date} through {end}. My budget is $10000 for listed flights, meals and lodging; attraction fees and unlisted local transit are outside this accounting.',
                            'daily_plans':base}},
                        'context_policy':{'context_reset_between_sessions':True,'replay_initial_observation':False,
                                          'initial_observation_delivery':'once_before_first_session'},'sessions':[]}
                hidden={'episode_id':eid,'seed':SEED,'template_id':template,'destination_city':dest,
                        'flight_dates':[date,end],'base_source_provenance':{'outbound_flight':f,'return_flight':ret},
                        'sessions':[]}
                rules_history=[];chosen_history=[];plans=[]
                for i in range(5):
                    rule_index=[0,0,1,1,2][i]
                    chosen={kind:generated_candidates[kind][rule_index] for kind in kinds}
                    rules={kind:dict(op='min_ge' if i==0 else ('same' if i in [1,3] else 'next_price')) for kind in kinds}
                    for kind in kinds:
                        if i==0:rules[kind]['minimum_price']=price(chosen[kind],kind)
                        if kind=='hotel' and i==0:rules[kind]['room_type']=room_type
                    rules_history.append(rules);chosen_history.append(chosen)
                    instruction_text=instruction(i,dest,rules);plan=fill_plan(base,i,chosen);plans.append(plan)
                    sid=f's{i+1}'
                    public['sessions'].append({'session_id':sid,'instruction':instruction_text,
                        'environment_config':{'environment':'travel_planner','database_root':str(DB),
                            'tool_schema_path':str(MA/'env/env_systems/travel_planner_env/tool_schemas.py'),
                            'action_style':'native_function_call_then_formatted_plan','max_steps':30,
                            'context_reset_between_sessions':True,'replay_initial_observation':False}})
                    deps=[];witnesses=[]
                    if i:
                        for kind in kinds:
                            dependency_type=('same_restaurant' if kind=='meal' else 'same_accommodation') if i in [1,3] else ('relative_meal_price' if kind=='meal' else 'relative_hotel_price_same_room_type')
                            edge_counts[dependency_type]+=1
                            actual_value=named(chosen_history[i-1][kind],kind)
                            deps.append({'session_id':f's{i}','dependency_type':dependency_type,
                                         'required_fact':{'kind':kind,'value':actual_value,
                                             'day_slot':MEAL_SLOTS[i-1] if kind=='meal' else [1,'accommodation']}})
                            prefixes=[]
                            for shift in [0,1]:
                                prefix=[]
                                for j in range(i):
                                    jrules=copy.deepcopy(rules_history[j])
                                    if j==0:jrules[kind]['minimum_price']=price(generated_candidates[kind][shift],kind)
                                    cf=generated_candidates[kind][[0,0,1,1,2][j]+shift]
                                    prefix.append({'session_id':f's{j+1}','instruction':instruction(j,dest,jrules),
                                                   'all_selection_rules':jrules,'selected_value':named(cf,kind)})
                                prefixes.append(prefix)
                            witnesses.append({'kind':kind,'current_instruction':instruction_text,
                                'actual_history':prefixes[0],'alternative_history':prefixes[1],
                                'current_targets':[named(chosen[kind],kind),named(generated_candidates[kind][rule_index+1],kind)],
                                'ablation_possible_target_count_lower_bound':2,
                                'witness_scope':'Two valid upstream requests/histories under the same environment and identical current input; upstream first threshold changes, its query is also included.'})
                    hidden['sessions'].append({'session_id':sid,
                        'structured_goal':{'selection_rules':rules,'plan':plan,'active_targets':{k:named(v,k) for k,v in chosen.items()},
                            'constraints':{'trip_days':3,'nights':2,'travelers_per_booking':1,'budget':10000,
                                           'copy_unmentioned_slots_from_initial_context':True}},
                        'depends_on':deps,'valid_answers_or_targets':[plan],
                        'source_provenance':{kind:{'file':str(DB/FILES[kind]),'csv_row_index_zero_based':next(int(ix) for ix,x in frames[kind].iterrows() if named(x.to_dict(),kind)==named(value,kind)),
                                                   'record':value} for kind,value in chosen.items()},
                        'initial_context_dependencies':{'context':'base_person','fields':['dates','route','unmentioned_plan_slots']},
                        'dependency_witness':witnesses})
                overlap={'official_exact_instruction_matches':sum(digest(x['instruction']) in qhash for x in public['sessions']),
                    'official_normalized_instruction_matches':sum(digest(normalized(x['instruction'])) in nhash for x in public['sessions']),
                    'official_full_subtask_plan_matches':sum(digest(x) in phash for x in plans),
                    'official_five_plan_chain_matches':int(digest(plans) in windows),
                    'source_objects_shared_with_official':len(set.union(*(plan_objects(p) for p in plans)) & official_objects),
                    'source_object_count':len(set.union(*(plan_objects(p) for p in plans))),
                    'dependency_template_overlap':'intentional: same-choice and relative-price relations occur in official natural-language tasks',
                    'official_structured_dependency_signature_comparison':'not fully available: official graph has no explicit fields; no claim of complete semantic graph disjointness'}
                if any(overlap[k] for k in ['official_exact_instruction_matches','official_normalized_instruction_matches','official_full_subtask_plan_matches','official_five_plan_chain_matches']):
                    rejections['official_overlap']+=1;continue
                hidden['overlap']=overlap
                edges=validate_episode(public,hidden,pool,native)
                # Identity/city/stay/schedule/budget checker is separate from target equality.
                for plan in [base,*plans]:
                    total=float(f['Price'])+float(ret['Price'])
                    for d in plan:
                        for slot in ['breakfast','lunch','dinner']:
                            if d[slot]=='-':continue
                            found=[x for x in pool['meal'] if named(x,'meal')==d[slot]]
                            assert len(found)==1
                            total+=price(found[0],'meal')
                        if d['accommodation']!='-':
                            found=[x for x in pool['hotel'] if named(x,'hotel')==d['accommodation']]
                            assert len(found)==1 and found[0]['minimum nights']<=2 and found[0]['maximum occupancy']>=1
                            total+=price(found[0],'hotel')
                        if d['attraction']!='-':assert any(named(x,'attraction')==d['attraction'] for x in pool['attraction'])
                    assert plan[0]['accommodation']==plan[1]['accommodation'] and total<=10000
                hidden['validation']={'status':'pass','session_count':5,'unique_legal_plan_per_session':True,
                    'counterfactual_edges_verified':edges,'native_parser_and_seven_slot_scorer_roundtrip':True,
                    'identity_city_stay_schedule_budget_pass':True,'model_calls':0,
                    'limits':['No hotel date inventory, opening hours, meal service times, attraction admission costs, intra-city transit or flight seat availability are present in this benchmark DB.']}
                public_rows.append(public);hidden_rows.append(hidden);break
    # Mutation checks prove that validation is capable of rejecting bad records.
    negative=[]
    p=public_rows[-1];h=hidden_rows[-1];pool=pools_by_city[h['destination_city']]
    def mutate_case(name,mutator):
        pp=copy.deepcopy(p);hh=copy.deepcopy(h);mutator(pp,hh)
        try:validate_episode(pp,hh,pool,native)
        except (ValueError,KeyError) as exc:negative.append({'case':name,'rejected':True,'reason':str(exc)})
        else:negative.append({'case':name,'rejected':False})
    mutate_case('wrong_gold',lambda p,h:h['sessions'][2]['valid_answers_or_targets'][0][0].update(dinner='Fabricated Restaurant, nowhere'))
    mutate_case('broken_dependency_reference',lambda p,h:h['sessions'][1]['depends_on'][0].update(session_id='future'))
    mutate_case('counterfactual_target_unchanged',lambda p,h:h['sessions'][1]['dependency_witness'][0]['current_targets'].__setitem__(1,h['sessions'][1]['dependency_witness'][0]['current_targets'][0]))
    mutate_case('counterfactual_illegal_history',lambda p,h:h['sessions'][1]['dependency_witness'][0]['alternative_history'][0].update(selected_value='Fabricated'))
    mutate_case('public_gold_injection',lambda p,h:p['sessions'][1].update(instruction=p['sessions'][1]['instruction']+' '+h['sessions'][1]['structured_goal']['active_targets']['meal']))
    mutate_case('empty_target_set',lambda p,h:h['sessions'][2].update(valid_answers_or_targets=[]))
    mutate_case('infeasible_budget',lambda p,h:h['sessions'][0]['structured_goal']['constraints'].update(budget=0))
    mutate_case('invalid_trip_dates',lambda p,h:h.update(flight_dates=[h['flight_dates'][0],h['flight_dates'][0]]))
    mutate_case('fabricated_source_record',lambda p,h:h['sessions'][0]['source_provenance']['meal']['record'].update(Name='Fabricated'))
    mutate_case('extra_public_session',lambda p,h:p['sessions'].append(copy.deepcopy(p['sessions'][-1])))
    mutate_case('missing_hidden_session',lambda p,h:h['sessions'].pop())
    mutate_case('bootstrap_replay_shortcut',lambda p,h:p['context_policy'].update(replay_initial_observation=True))
    mutate_case('hidden_config_injection',lambda p,h:p['sessions'][1]['environment_config'].update(gold=h['sessions'][1]['valid_answers_or_targets']))
    mutate_case('wrong_dependency_type',lambda p,h:h['sessions'][1]['depends_on'][0].update(dependency_type='unsupported'))
    mutate_case('wrong_dependency_slot',lambda p,h:h['sessions'][1]['depends_on'][0]['required_fact'].update(day_slot=[1,'breakfast']))
    assert all(x['rejected'] for x in negative)
    for file,rows in [('public_episodes.jsonl',public_rows),('hidden_episodes.jsonl',hidden_rows)]:
        (HERE/file).write_text(''.join(canonical(row)+'\n' for row in rows))
    (HERE/'negative_validator_checks.json').write_text(json.dumps(negative,indent=2)+'\n')
    stats={'task_family':'travel','seed':SEED,'official_evaluation_episodes':len(official),'synthetic_episodes':len(public_rows),
        'scored_sessions':sum(len(x['sessions']) for x in public_rows),'average_scored_sessions':5,
        'bootstrap_contexts':len(public_rows),'template_distribution':dict(Counter(x['template_id'] for x in hidden_rows)),
        'dependency_type_distribution':dict(edge_counts),'dependency_edges':sum(edge_counts.values()),
        'validity_pass_rate':1.0,'invalid_final_episodes':0,'generation_rejections':dict(rejections),
        'negative_checks_passed':sum(x['rejected'] for x in negative),'negative_checks_total':len(negative),
        'official_exact_instruction_matches':sum(x['overlap']['official_exact_instruction_matches'] for x in hidden_rows),
        'official_normalized_instruction_matches':sum(x['overlap']['official_normalized_instruction_matches'] for x in hidden_rows),
        'official_full_subtask_plan_matches':sum(x['overlap']['official_full_subtask_plan_matches'] for x in hidden_rows),
        'official_five_plan_chain_matches':sum(x['overlap']['official_five_plan_chain_matches'] for x in hidden_rows),
        'episodes_sharing_source_objects_with_official':sum(x['overlap']['source_objects_shared_with_official']>0 for x in hidden_rows),
        'per_episode_shared_source_object_sum':sum(x['overlap']['source_objects_shared_with_official'] for x in hidden_rows),
        'different_route_date_keys':len({tuple(x['flight_dates'])+(x['destination_city'],) for x in hidden_rows}),
        'eligible_cities_before_route_filter':len(pools_by_city),'outbound_candidate_records':len(outbound_records),
        'public_sha256':hashlib.sha256((HERE/'public_episodes.jsonl').read_bytes()).hexdigest(),
        'hidden_sha256':hashlib.sha256((HERE/'hidden_episodes.jsonl').read_bytes()).hexdigest(),
        'no_model_calls':True,'no_sft_or_memory_action_labels':True,'native_pipeline_unchanged':True}
    (HERE/'pilot_statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2))


def validate_saved():
    public=[json.loads(line) for line in (HERE/'public_episodes.jsonl').read_text().splitlines()]
    hidden=[json.loads(line) for line in (HERE/'hidden_episodes.jsonl').read_text().splitlines()]
    if len(public)!=len(hidden):raise ValueError('public_hidden_episode_count_mismatch')
    if len({x['episode_id'] for x in public})!=len(public):raise ValueError('duplicate_episode_id')
    frames={k:pd.read_csv(DB/v).dropna() for k,v in FILES.items()}
    wanted={x['base_source_provenance'][k]['Flight Number'] for x in hidden for k in ['outbound_flight','return_flight']}
    flight_records=frames['flight'][frames['flight']['Flight Number'].isin(wanted)].to_dict(orient='records')
    native=native_pure();edges=0
    official,qhash,nhash,phash,windows,objects=official_inventory()
    for p,h in zip(public,hidden):
        city=h['destination_city']
        pool={'meal':frames['meal'][frames['meal'].City==city].to_dict(orient='records'),
              'hotel':frames['hotel'][frames['hotel'].city==city].to_dict(orient='records'),
              'attraction':frames['attraction'][frames['attraction'].City==city].to_dict(orient='records'),
              'flight_records':flight_records}
        edges+=validate_episode(p,h,pool,native)
        if any(digest(x['instruction']) in qhash or digest(normalized(x['instruction'])) in nhash for x in p['sessions']):
            raise ValueError('official_instruction_overlap')
        plans=[x['valid_answers_or_targets'][0] for x in h['sessions']]
        if any(digest(plan) in phash for plan in plans) or digest(plans) in windows:
            raise ValueError('official_plan_overlap')
    print(json.dumps({'saved_episodes_validated':len(public),'counterfactual_edges_recomputed':edges,
                      'original_csv_flight_records_verified':len(flight_records),'official_overlap':'pass'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    validate_saved() if args.validate_only else main()
