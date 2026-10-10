"""Unchanged native client, public adapter and scorer; private state never enters actors."""
import json
import sys
from decimal import Decimal
from .freeze import HERE,PROJECT,SHOP
from ....travel_v0.artifacts import save,read,rows,append
from ..native_worker import native_task
from examples.memory_agent_grpo.adapters.memoryarena.shopping.environment_session import ShoppingEnvironmentSession
from examples.memory_agent_grpo.adapters.memoryarena.shopping.executor_adapter import ShoppingToolExecutorAdapter
from examples.memory_agent_grpo.adapters.memoryarena.shopping.task_input_adapter import ShoppingTaskInputAdapter
from env.env_systems.web_shopping_env.webshop_plus_client import WebshopPlusEnvClient

sys.path.insert(0,str(PROJECT/'docs/memoryarena_synthetic_data/acceptance_v1'))
from shopping_constraints import strict_purchase


class LoggedNativeClient(WebshopPlusEnvClient):
    log_directory=None
    def _post(self,path,data):
        result=super()._post(path,data)
        if self.log_directory:
            append(self.log_directory/'private/native_http.jsonl',dict(endpoint=path,request=data,response=result))
        return result


class Boundary:
    def __init__(self,base_url):
        self.base_url=base_url;self.client=None
        self.hidden={e['episode_id']:e for e in rows(SHOP/'hidden_episodes.jsonl')}
        self.products={p['asin']:p for p in rows(SHOP/'source_products.jsonl')}
        self.actual=[]

    def begin_episode(self,eid):
        self.eid=eid;self.actual=[]

    def reset(self,public,dest,index):
        self.dest=dest;self.index=index
        private=self.hidden[self.eid]['sessions'][index]
        path=dest/'private/native_task.json';save(path,native_task(public,private))
        if self.client is None:
            LoggedNativeClient.log_directory=dest
            self.client=LoggedNativeClient(self.base_url,str(path),1,timeout=300,max_rounds=60,
                enable_feedback=False,action_format='react')
        else:
            self.client.log_directory=dest
            self.client._init_task_state(str(path),60,False,None,None)
        self.session=ShoppingEnvironmentSession(upstream_client=self.client,max_observation_chars=16000)
        initial=self.session.reset(seed=0)
        assert public['instruction'] in initial.state
        assert not self.client.purchased_asins and not self.client.purchased_prices
        self.executor=ShoppingToolExecutorAdapter(session=self.session)
        subtask=ShoppingTaskInputAdapter(max_observation_chars=16000).build_subtask_input({'state':initial.state})
        save(dest/'public_input.json',dict(session_id=public['session_id'],instruction=public['instruction'],
            initial_observation=initial.state,available_actions=self.client._get('available_actions')))
        save(dest/'reset_check.json',dict(fresh_conversation=True,client_purchase_history_empty=True,
            replay_prior_instructions=False,only_current_product=True))
        return subtask

    def score(self):
        observed={'asin':self.client.purchased_asins[-1] if self.client.purchased_asins else None,
                  'price_cents':int(Decimal(str(self.client.purchased_prices[-1]))*100) if self.client.purchased_prices else None}
        private=self.hidden[self.eid]
        outcome=strict_purchase(private['sessions'][self.index]['structured_goal'],self.products,
            self.actual,private['total_cap_cents'],observed)
        native=[r for r in rows(self.dest/'private/native_http.jsonl') if r['endpoint']=='step']
        result=dict(strict_validation=outcome,actual_purchase=observed,
            purchase_completed=bool(self.client.purchased_asins),native_score=native[-1]['response'] if native else None,
            actual_history_before=list(self.actual),scorer_target_never_delivered=True)
        # Preserve failed/missing transactions; never substitute the hidden target chain.
        self.actual.append(observed)
        save(self.dest/'private/outcome.json',result)
        save(self.dest.parent/'private/actual_history.json',self.actual)
        return result

    def close(self):
        if self.client:self.client.close()


class RecordingExecutor:
    def __init__(self,boundary,memory,answer,dest):
        self.boundary=boundary;self.memory=memory;self.answer=answer;self.dest=dest;self.events=[]

    def execute(self,call):
        allowed=self.boundary.client._get('available_actions')
        result=self.boundary.executor.execute(call)
        # Public error messages are retained as emitted by the existing executor.
        event=dict(event_id=f"{self.memory.eid}:{self.memory.current['session_id']}:a{len(self.events)+1}",
            action={'tool':call.name,'arguments':json.loads(call.arguments_json)},
            observation=result.tool_observation.content,available_actions_before=allowed,
            available_actions_after=self.boundary.client._get('available_actions') if not result.is_terminal else {},
            is_terminal=result.is_terminal,execution_status=result.tool_observation.status.value,
            request_ref=self.answer.last_meta['request_ref'],response_ref=self.answer.last_meta['response_ref'])
        self.events.append(event);append(self.dest/'answer_events.jsonl',event)
        self.memory.public_event(event)
        return result
