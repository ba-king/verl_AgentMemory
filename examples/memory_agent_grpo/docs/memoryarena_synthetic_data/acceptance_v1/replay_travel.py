"""Real native local tools + original submission/parser/scorer, without a model.

Only the dataset loader is bound to our saved synthetic group. No tool response,
environment transition or score is mocked; evaluator fields stay private.
"""
import copy
import importlib
import re
from decimal import Decimal
from common import *

sys.path.insert(0,str(ARENA))
from env.env_systems.travel_planner_env.tool_executor import ToolExecutor
from env.env_systems.travel_planner_env.tool_schemas import TOOLS
native=importlib.import_module('env.env_systems.travel_env')
old=module('saved_travel_generator',TRAVEL/'generate_pilot.py')


def canonical_plan(plan):
    return [{'days':d.get('days',d.get('day')),**{k:d.get(k,'-') for k in old.SLOTS}} for d in plan]

def native_environment(public,hidden):
    group={'id':public['episode_id'],'base_person':public['initial_observation']['base_person'],
           'questions':[{'name':old.NAMES[i],'query':s['instruction'],'round_idx':i+1} for i,s in enumerate(public['sessions'])],
           'answers':[{'round_idx':i+1,'daily_plans':s['valid_answers_or_targets'][0]} for i,s in enumerate(hidden['sessions'])]}
    loader=native.load_travel_data
    try:
        native.load_travel_data=lambda:[group]  # New task data only; avoids HF/network.
        env=native.TravelPlannerEnvironment({'judgement_mode':'none'})
    finally:native.load_travel_data=loader
    return env


def strict_plan(plan,index,public,hidden,pools,actual_history):
    if len(actual_history)!=index:raise ValueError('incomplete_actual_prefix')
    rule=hidden['sessions'][index]['structured_goal']['selection_rules'];chosen={}
    for kind,condition in rule.items():
        previous=None
        if index:
            prev=actual_history[-1]
            if kind=='meal':day,slot=old.MEAL_SLOTS[index-1];previous=prev[day-1][slot]
            else:previous=prev[0]['accommodation']
        chosen[kind]=old.selection(pools[kind],kind,condition,previous)[0]
    expected=old.fill_plan(public['initial_observation']['base_person']['daily_plans'],index,chosen)
    if canonical_plan(plan)!=canonical_plan(expected):raise ValueError('exact_plan_or_history_constraint_mismatch')
    total=old.validate_physical_plan(canonical_plan(plan),hidden,pools,10000)
    return {'passed':True,'total_listed_cost':total,'expected_from_actual_history':expected}


def field_visible(value,text):
    if isinstance(value,(int,float)):
        try:
            target=Decimal(str(value))
            return any(Decimal(x)==target for x in re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])',text))
        except Exception:return False
    return ' '.join(str(value).split()) in ' '.join(text.split())


def main():
    require_preflight();selected=read(HERE/'selection.json')['selected']['travel']
    publics={r['episode_id']:r for r in rows(TRAVEL/'public_episodes.jsonl')};hiddens={r['episode_id']:r for r in rows(TRAVEL/'hidden_episodes.jsonl')}
    executor=ToolExecutor(str(old.DB));summaries=[]
    modes=[(r,False) for r in selected]+[(selected[0],True)]
    for chosen,injected in modes:
        eid=chosen['episode_id'];pub=publics[eid];hidden=hiddens[eid];gate=PrefixGate();env=native_environment(pub,hidden)
        dest=REPLAY/'travel'/('wrong_prefix_'+eid if injected else eid);dest.mkdir(parents=True,exist_ok=True)
        save(dest/'private_reference.json',hidden)
        ep={'episode_id':eid,'template_id':chosen['template_id'],'reference_aware':True,'injection':injected,'sessions':[],'complete':False}
        city=hidden['destination_city'];f=hidden['base_source_provenance']['outbound_flight'];ret=hidden['base_source_provenance']['return_flight']
        pools={'meal':executor.restaurants.data[executor.restaurants.data.City==city].to_dict(orient='records'),
               'hotel':executor.accommodations.data[executor.accommodations.data.city==city].to_dict(orient='records'),
               'attraction':executor.attractions.data[executor.attractions.data.City==city].to_dict(orient='records'),
               'flight_records':executor.flights.data[executor.flights.data['Flight Number'].isin([f['Flight Number'],ret['Flight Number']])].to_dict(orient='records')}
        for i,(ps,hs) in enumerate(zip(pub['sessions'],hidden['sessions'])):
            if not gate.next_allowed():break
            prefix=dest/ps['session_id'];prefix.mkdir(exist_ok=True);events=[]
            raw_reset=env.reset();save(prefix/'private_native_reset.json',raw_reset)
            visible=public_input(pub,i);save(prefix/'public_input.json',visible);save(prefix/'model_input_projection.json',{**visible,'tools':TOOLS,'current_session_events':[],'retrieved_memories':[]})
            session={'session_id':ps['session_id'],'passed':False,'checks':{'reset_history_empty':env.history==[] and env.step_count==0,
                     'bootstrap_once':('initial_observation' in visible)==(i==0),'current_input_allowlist':set(visible)==({'session_id','instruction','initial_observation'} if i==0 else {'session_id','instruction'})}}
            try:
                queries=[('FlightSearch',{'origin':r['OriginCityName'],'destination':r['DestCityName'],'date':r['FlightDate']}) for r in (f,ret)]
                queries += [('RestaurantSearch',{'city':city}),('AccommodationSearch',{'city':city}),('AttractionSearch',{'city':city})]
                states={c:next((s for s,cs in executor.cities.data.items() if c in cs),None) for c in (f['OriginCityName'],city)}
                for c,state in states.items():
                    if state is not None:queries.append(('CitySearch',{'state':state}))
                queries.append(('DistanceMatrix',{'origin':f['OriginCityName'],'destination':city,'mode':'driving'}))
                observations={}
                for name,args in queries:
                    text=executor.execute(name,args)  # Original formatter, including any truncation.
                    event={'action':{'tool':name,'arguments':args},'observation':text}
                    events.append(event);observations.setdefault(name,[]).append(text)
                save_rows(prefix/'actions_observations.jsonl',events)
                submitted=copy.deepcopy(hs['valid_answers_or_targets'][0])
                # Explicit malformed-result injection, not a substituted tool response.
                if injected:
                    submitted[0]['accommodation']=submitted[0]['accommodation'][:1]
                witness_checks=[]
                original=hs['valid_answers_or_targets'][0]
                for kind,tool,fields in [('meal','RestaurantSearch',['Name','Average Cost','Cuisines','Aggregate Rating','City']),
                                         ('hotel','AccommodationSearch',['NAME','room type','price','minimum nights','maximum occupancy','city']),
                                         ('attraction','AttractionSearch',['Name','City'])]:
                    used=set()
                    for day in original:
                        slots=['breakfast','lunch','dinner'] if kind=='meal' else (['accommodation'] if kind=='hotel' else ['attraction'])
                        used.update(day[s] for s in slots if day[s]!='-')
                    table='\n'.join(observations[tool])
                    for value in sorted(used):
                        matches=[r for r in pools[kind] if old.named(r,kind)==value]
                        namekey='NAME' if kind=='hotel' else 'Name'
                        matching_lines=[line for line in table.splitlines() if matches and matches[0][namekey] in line]
                        evidence='\n'.join(matching_lines)
                        fields_ok={k:len(matches)==1 and field_visible(matches[0][k],evidence) for k in fields}
                        witness_checks.append({'kind':kind,'object':value,'unique_db_match':len(matches)==1,'fields_visible_in_native_projection':fields_ok,'visible_line':evidence})
                for r,text in zip((f,ret),observations['FlightSearch']):
                    line='\n'.join(v for v in text.splitlines() if r['Flight Number'] in v)
                    fields={k:field_visible(r[k],line) for k in ('Flight Number','Price','FlightDate','OriginCityName','DestCityName','DepTime','ArrTime')}
                    witness_checks.append({'kind':'flight','object':r['Flight Number'],'fields_visible_in_native_projection':fields,'visible_line':line})
                session['evidence_checks']=witness_checks
                session['checks']['all_required_fields_visible']=all(all(w['fields_visible_in_native_projection'].values()) for w in witness_checks)
                session['checks']['city_lookup']=all(state is not None and c in '\n'.join(observations.get('CitySearch',[])) for c,state in states.items())
                session['distance_observation']=observations['DistanceMatrix'][0]
                session['distance_required_for_gold']=False  # Flights used; local transit explicitly outside listed-cost cap.
                text=old.plan_text(old.NAMES[i],submitted)
                events.append({'action':{'submit_plan':text},'observation':None})
                # Original step + parser + native seven-slot score, no LLM judge.
                native_out=env.step(text,ground_truth={'name':old.NAMES[i],'daily_plans':hs['valid_answers_or_targets'][0]},need_judge=True)
                _,score,info=native_out
                parsed=canonical_plan(native._parse_person_plan_from_result(text,old.NAMES[i]))
                session['native_score']=bool(score);session['native_info']=info
                session['checks']['normal_parser_roundtrip']=parsed==canonical_plan(submitted)
                try:strict=strict_plan(parsed,i,pub,hidden,pools,gate.actual)
                except ValueError as e:strict={'passed':False,'reason':str(e)}
                session['strict_validation']=strict;session['checks']['strict_plan']=strict['passed']
                session['native_strict_disagreement']=bool(score)!=strict['passed']
                session['passed']=all(session['checks'].values())
                session['termination']='submitted' if session['passed'] else 'strict_or_visibility_failure'
                gate.record(parsed,session['passed'])
                save(prefix/'private_submission.json',{'submitted_text':text,'parsed_plan':parsed,'native_output':native_out,'strict':strict})
            except Exception as e:
                session['error']=type(e).__name__+': '+str(e);session['termination']='execution_error';gate.record({'error':session['error']},False)
            save_rows(prefix/'actions_observations.jsonl',events);save(prefix/'private_score.json',session)
            ep['sessions'].append(session);save(dest/'episode_summary.json',ep)
        ep['complete']=len(ep['sessions'])==5 and all(s['passed'] for s in ep['sessions']);ep['stopped_after_failure']=gate.stopped
        ep['actual_history']=gate.actual;ep['gold_replacement_performed']=False
        save(dest/'episode_summary.json',ep);summaries.append(ep);save(REPLAY/'travel_summary.json',summaries);env.close()
    print(json.dumps({'attempted':len(selected),'complete':sum(e['complete'] for e in summaries if not e['injection']),
                      'errors':[(e['episode_id'],s.get('error'),s['checks']) for e in summaries for s in e['sessions'] if not s['passed']]}))

if __name__=='__main__':main()
