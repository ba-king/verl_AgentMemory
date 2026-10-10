"""Joint attribute + full-prefix-budget semantics for new tasks only."""
from decimal import Decimal
from common import *
from attribute_parsers import price_cents,strict_visible_label,tokens

SCENE_PREFIXES=('Home & Kitchen › Furniture','Home & Kitchen › Home Décor Products',
                'Tools & Home Improvement › Lighting & Ceiling Fans')
ENV={'environment':'MemoryArena.WebShop','action_format':'react','include_history':False,
     'enable_feedback':False,'show_attrs':False,'max_observation_chars':16000}

def cents(row):
    value=price_cents(row.get('pricing'))
    if value is None:raise ValueError('nonfixed_price')
    return value
def label(row,kind):return strict_visible_label(row,kind)[0]
def money(value):return f'${value//100}.{value%100:02d}'

def select(goal,products,history,cap,*,without_budget=False,without_attribute=False):
    if len(history)!=goal['step']-1:raise ValueError('incomplete_actual_prefix')
    spent=sum(cents(products[a]) for a in history)
    required=goal.get('anchor_label') if not history else goal['mapping'].get(label(products[history[-1]],goal['previous_field']))
    if required is None:raise ValueError('unmapped_history_label')
    scored=[]
    for a in goal['candidate_asins']:
        row=products[a];p=cents(row)
        if not without_budget and p>cap-spent:continue
        if not without_attribute and label(row,goal['current_field'])!=required:continue
        order=p if goal['ranking']=='lowest_price' else -p
        scored.append(((order,row['name']),a))
    if not scored:raise ValueError('no_feasible_candidate')
    best=min(t for t,a in scored);winners=[a for t,a in scored if t==best]
    if len(winners)!=1:raise ValueError('nonunique_winner')
    return winners[0],{'spent_cents':spent,'remaining_cents':cap-spent,'required_label':required,
                      'eligible_asins':[a for _,a in sorted(scored)]}

def instruction(goal,products,cap):
    step=goal['step'];field=goal['current_field'].title()
    prefix=(f'Buy a six-item indoor furnishing and decoration bundle, one item per Product in order. '
            f'The fixed total budget for all six purchases is {money(cap)}. Before each purchase, subtract the prices actually paid for ALL earlier Products from this cap. '
            'There are no shipping fees or price-changing options. These label mappings are my bundle preferences, not physical compatibility claims. '
            'Use only literal Material: or Color: declarations on Features or Description pages; ignore case, treating gray/grey and aluminium/aluminum as equal. '
            'Keep all other label words, including percentages. Search listed candidate names individually.\n\n')
    text=f'Product {step}:\nThe field for the current item is {field}.\n'
    if step==1:text+=f'Only candidates with {field} label "{goal["anchor_label"]}" are eligible.\n'
    else:
        text+=f'Use the {goal["previous_field"].title()} label of the item actually purchased for Product {step-1}. This complete rule applies only to Product {step}:\n'
        for left,right in sorted(goal['mapping'].items()):text+=f'- Previous {goal["previous_field"].title()} "{left}" allows current {field} "{right}"; all other current labels are forbidden.\n'
    text+=('Among listed candidates that satisfy BOTH the label rule and the remaining total budget, buy the '+
           ('lowest' if goal['ranking']=='lowest_price' else 'highest')+'-priced item. Break a price tie by ascending exact product name (case-sensitive).\nAvailable Options:\n')
    return prefix+text+'\n'.join('- '+products[a]['name'] for a in goal['candidate_asins'])

def solve(goals,products,cap,anchor_override=None):
    history=[];details=[]
    for i,goal in enumerate(goals):
        active={**goal,**anchor_override} if i==0 and anchor_override else goal
        a,d=select(active,products,history,cap);history.append(a);details.append(d)
    return history,details

def strict_purchase(goal,products,history,cap,observed):
    """Check actual transaction facts without substituting a saved target chain."""
    if any(r.get('asin') not in products or r.get('price_cents')!=cents(products[r['asin']]) for r in history):
        return {'passed':False,'reason':'invalid_actual_prefix'}
    try:expected,details=select(goal,products,[r['asin'] for r in history],cap)
    except ValueError as e:return {'passed':False,'reason':str(e)}
    a=observed.get('asin');p=observed.get('price_cents')
    valid=a==expected and a in products and type(p) is int and p==cents(products[a])
    return {'passed':valid,'reason':None if valid else 'incorrect_choice_or_transaction_price',
            'actual_prefix_spent_cents':sum(r['price_cents'] for r in history),'accounting':details}

def validate(pub,hidden,products,official):
    if set(pub)!={'schema_version','episode_id','task_family','context_reset_between_sessions','replay_prior_instructions','sessions'}:raise ValueError('public_fields')
    if pub['schema_version']!='memoryarena.synthetic.shopping_joint.v1' or pub['task_family']!='shopping':raise ValueError('public_schema')
    if pub['context_reset_between_sessions'] is not True or pub['replay_prior_instructions'] is not False:raise ValueError('history_channel')
    if len(pub['sessions'])!=6 or len(hidden['sessions'])!=6 or pub['episode_id']!=hidden['episode_id']:raise ValueError('episode_shape')
    cap=hidden['total_cap_cents']
    if type(cap) is not int or cap<=0:raise ValueError('cap')
    banned={a['target_asin'] for r in official for a in r['answers']};questions={tokens(q) for r in official for q in r['questions']}
    goals=[s['structured_goal'] for s in hidden['sessions']];used=set()
    for i,(p,h,g) in enumerate(zip(pub['sessions'],hidden['sessions'],goals)):
        if set(p)!={'session_id','instruction','environment_config'} or p['environment_config']!=ENV:raise ValueError('public_config')
        if p['session_id']!=f'p{i+1}' or h['session_id']!=p['session_id'] or g['step']!=i+1:raise ValueError('step_scope')
        if p['instruction']!=instruction(g,products,cap):raise ValueError('instruction_goal')
        if tokens(p['instruction']) in questions:raise ValueError('official_instruction_overlap')
        if i and g['previous_field']!=goals[i-1]['current_field']:raise ValueError('field_scope')
        ids=g['candidate_asins']
        if len(ids)!=len(set(ids)) or used.intersection(ids):raise ValueError('duplicate_product_in_episode')
        used.update(ids)
        provenance=[]
        for a in ids:
            r=products[a]
            if a in banned or not any(r['product_category'].startswith(s) for s in SCENE_PREFIXES):raise ValueError('source_exclusion_or_scene')
            if r.get('customization_options'):raise ValueError('options')
            value,evidence=strict_visible_label(r,g['current_field'])
            provenance.append({'asin':a,'cents':cents(r),'field':g['current_field'],'value':value,'evidence':evidence,'raw_sha256':digest(r)})
        if h['source_provenance']!=provenance:raise ValueError('provenance')
    actual,details=solve(goals,products,cap)
    if actual!=[s['valid_answers_or_targets'][0] for s in hidden['sessions']]:raise ValueError('wrong_gold')
    if any(len(s['valid_answers_or_targets'])!=1 for s in hidden['sessions']):raise ValueError('unexpected_answer_set')
    for i,s in enumerate(hidden['sessions']):
        if s['history_accounting']!=details[i]:raise ValueError('stale_accounting')
        no_b,_=select(goals[i],products,actual[:i],cap,without_budget=True)
        no_a,_=select(goals[i],products,actual[:i],cap,without_attribute=True)
        expected={'without_budget':no_b,'budget_changes_selection':no_b!=actual[i],
                  'without_attribute':no_a,'attribute_changes_selection':no_a!=actual[i]}
        if s['constraint_ablations']!=expected:raise ValueError('stale_ablation')
    for kind,index in [('attribute',1),('cumulative_budget',5)]:
        w=hidden['witnesses'][kind];override=w['alternate_anchor_override']
        if set(override)-{'anchor_label','ranking'}:raise ValueError('illegal_history_intervention')
        alternative,_=solve(goals,products,cap,override)
        if w['alternate_anchor_instruction']!=instruction({**goals[0],**override},products,cap):raise ValueError('anchor_request')
        if w['actual_history']!=actual[:index] or w['alternate_history']!=alternative[:index]:raise ValueError('illegal_prefix')
        if w['current_input_sha256']!=digest(pub['sessions'][index]) or w['actual_target']!=actual[index] or w['alternate_target']!=alternative[index] or actual[index]==alternative[index]:raise ValueError('ineffective_witness')
        pa=sum(cents(products[a]) for a in actual[:index]);pb=sum(cents(products[a]) for a in alternative[:index])
        if kind=='attribute':
            if pa!=pb or label(products[actual[0]],goals[0]['current_field'])==label(products[alternative[0]],goals[0]['current_field']):raise ValueError('attribute_witness_confounded_budget')
        else:
            if actual[index-1]!=alternative[index-1] or pa==pb or actual[1:index]!=alternative[1:index]:raise ValueError('not_earlier_cumulative_spend')
    if not hidden['sessions'][5]['constraint_ablations']['budget_changes_selection']:raise ValueError('budget_not_binding_on_actual_path')
    if not any(s['constraint_ablations']['attribute_changes_selection'] for s in hidden['sessions'][1:]):raise ValueError('attribute_never_binds')
    return {'valid':True,'actual_chain':actual,'total_cents':sum(cents(products[a]) for a in actual),'cap_cents':cap,
            'attribute_witness_product':2,'budget_witness_product':6,'same_previous_product_in_budget_witness':True}
