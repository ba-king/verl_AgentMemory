"""One native environment smoke; public candidate navigation, zero model calls."""
import re
import sys
from .config import HERE,ARENA,SHOP,read,rows,save,append,now


def native_task(public,private):
    target=private['valid_answers_or_targets'][0]
    return dict(task_id=public['session_id'],task_description='Shopping natural batch native preflight',
        agent_instruction=public['instruction'],steps=[dict(step=1,target_asin=target,
        requirements=dict(attributes=[],options={},price_constraints={}))],
        target_products=[target],global_constraints=dict(purchase_order='sequence',max_steps=1))


def main():
    sys.path.insert(0,str(ARENA))
    from env.env_systems.web_shopping_env.webshop_plus_client import WebshopPlusEnvClient
    from examples.memory_agent_grpo.adapters.memoryarena.shopping.observation_policy import project_public_observation
    public=read(HERE/'preflight/public_episode.json');private=read(HERE/'preflight/private_episode.json')
    catalog={p['name']:p for p in rows(SHOP/'source_products.jsonl')}
    client=None;sessions=[]
    try:
        for ps,hs in zip(public['sessions'],private['sessions'],strict=True):
            sid=ps['session_id'];path=HERE/'preflight'/sid/'private_native_task.json'
            save(str(path.relative_to(HERE)),native_task(ps,hs))
            if client is None:
                save('preflight/worker_stage.json',dict(stage='NATIVE_CREATE_REQUESTED',time=now(),session_id=sid))
                client=WebshopPlusEnvClient('http://127.0.0.1:36036',str(path),1,timeout=300,
                    max_rounds=60,enable_feedback=False,action_format='react')
                save('preflight/worker_stage.json',dict(stage='NATIVE_CREATE_COMPLETED',time=now(),env_id=client.env_id))
            else:client._init_task_state(str(path),60,False,None,None)
            client.reset(0)
            def observe(action,response=None):
                raw=client.observe() if response is None else response.state
                visible=project_public_observation({'state':raw},max_observation_chars=16000).state
                clicks=client._get('available_actions').get('clickables',[]) if not client.episode_done else []
                append(f'preflight/{sid}/actions_observations.jsonl',dict(time=now(),action=action,observation=visible,clickables=clicks))
                return visible,clicks
            initial,clicks=observe('reset')
            assert ps['instruction'] in initial
            assert client.purchased_asins==[] and client.purchased_prices==[]
            assert all(other['instruction'] not in initial for other in public['sessions'] if other['session_id']!=sid)
            # Identity is from a listed PUBLIC name, never a hidden target selector.
            name=ps['instruction'].split('Available Options:\n',1)[1].splitlines()[0][2:]
            chosen=catalog[name]['asin']
            def step(action):
                if action.startswith('click['):
                    allowed=client._get('available_actions')['clickables']
                    assert action[6:-1].casefold() in {str(v).casefold() for v in allowed},'CLICK_NOT_CURRENTLY_AVAILABLE'
                response=client.step(action)
                return (*observe(action,response),response)
            located=False
            for query in dict.fromkeys([name,' '.join(re.findall(r'[A-Za-z0-9]+',name)[:10])]):
                _,clicks,_=step('search['+query+']')
                for _ in range(3):
                    if chosen.casefold() in {str(x).casefold() for x in clicks}:
                        _,clicks,_=step('click['+chosen+']');located=True;break
                    nxt=next((str(x) for x in clicks if str(x).casefold()=='next >'),None)
                    if not nxt:break
                    _,clicks,_=step('click['+nxt+']')
                if located:break
            assert located,'PUBLIC_FIRST_CANDIDATE_NOT_DISCOVERED_WITHIN_PREFLIGHT_LIMIT'
            for page in ['Features','Description']:
                _,clicks,_=step('click['+page+']')
                back=next(str(x) for x in clicks if str(x).casefold() in {'< prev','prev','< prev page'})
                step('click['+back+']')
            observed,_,response=step('click[Buy Now]')
            assert response.done and client.purchased_asins==[chosen] and chosen.casefold() in observed.casefold()
            sessions.append(dict(session_id=sid,reset=True,search=True,click=True,purchase=True,
                actual_purchase=dict(asin=chosen,price=client.purchased_prices[-1]),
                strict_natural_trajectory=False,controller='PUBLIC_FIRST_LISTED_ENVIRONMENT_SMOKE'))
            save('preflight/worker_progress.json',sessions)
        save('preflight/worker_result.json',dict(status='SHOPPING_ENVIRONMENT_READY',sessions=sessions,api_requests=0))
    except Exception as error:
        import traceback
        save('preflight/worker_result.json',dict(status='SHOPPING_ENVIRONMENT_BLOCKED',sessions=sessions,
            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),api_requests=0))
        raise
    finally:
        if client is not None:client.close()


if __name__=='__main__':main()
