"""Shared hard cap around unchanged provider clients, no retries/health calls."""
import json
import time
from .common import HERE, save


def compact_wire(data):
    """JSON whitespace only; every model-visible string/value stays identical."""
    body=json.loads(data)
    encoded=json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert json.loads(encoded)==body
    return encoded


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self, phase='phase3'):
        if phase not in {'phase1','phase2','phase3'}:
            raise ValueError(phase)
        self.call_limit, self.token_limit = {'phase1':(3,40000),'phase2':(5,50000),'phase3':(60,250000)}[phase]
        self.path = HERE/phase/'api_accounting.json'
        if self.path.exists():
            raise RuntimeError('REFUSE_SECOND_PHASE_BUDGET')
        self.data = {'requests': 0, 'tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0,
                     'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 0,
                     'roles': {}, 'max_requests': self.call_limit, 'max_total_tokens': self.token_limit,
                     'response_ids': [], 'transport_errors': 0, 'http_errors': 0}
        self.last = 0
        self.persist()

    def persist(self):
        save(self.path, self.data)

    def consume_call(self):
        if self.data['requests'] >= self.call_limit or self.data['tokens'] >= self.token_limit:
            raise BudgetStop('INCOMPLETE_BUDGET_STOP')
        time.sleep(max(0, 1 - (time.monotonic() - self.last)))

    def consume_usage(self, usage):
        # Accounting happens at the actual HTTP boundary, including valid
        # responses subsequently rejected by the original client/parser.
        pass

    def wrap(self, client, role):
        client.transport = Transport(client.transport, self, role)


class Transport:
    def __init__(self, underlying, budget, role):
        self.underlying, self.budget, self.role = underlying, budget, role

    def post(self, url, **kwargs):
        assert url in {'https://api.deepseek.com/chat/completions', 'https://api.deepseek.com/beta/chat/completions'}
        kwargs['data']=compact_wire(kwargs['data'])
        body = json.loads(kwargs['data'])
        role = self.role
        if role == 'MEMORY':
            state = json.loads(body['messages'][1]['content'])
            role = state['decision_mode']
            assert role in {'READ', 'CONSOLIDATE'}
        data = self.budget.data
        # Byte upper bound plus ample message/tool framing reserve, independent
        # of a tokenizer estimate; never reduce the frozen 4096 output cap.
        reserve = len(kwargs['data']) + 512 * len(body['messages']) + body['max_tokens'] + 8192
        if data['requests'] >= self.budget.call_limit or data['tokens'] + reserve > self.budget.token_limit:
            data['stop_reason'] = 'INCOMPLETE_BUDGET_STOP'
            data['next_request_token_reserve'] = reserve
            self.budget.persist()
            raise BudgetStop('INCOMPLETE_BUDGET_STOP: conservative request reserve')
        data['requests'] += 1
        entry = data['roles'].setdefault(role, {'requests': 0, 'tokens': 0, 'prompt_tokens': 0,
            'completion_tokens': 0, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 0})
        entry['requests'] += 1
        self.budget.last = time.monotonic()
        self.budget.persist()
        try:
            response = self.underlying.post(url, **kwargs)
        except Exception:
            data['transport_errors'] += 1
            self.budget.persist()
            raise
        if response.status_code != 200:
            data['http_errors'] += 1
        else:
            obj = response.json()
            usage = obj.get('usage')
            if not usage or 'total_tokens' not in usage:
                data['usage_unknown'] = True
                self.budget.persist()
                raise RuntimeError('INFRA_USAGE_MISSING_STOP')
            data['response_ids'].append(obj.get('id'))
            for key in ('tokens', 'prompt_tokens', 'completion_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens'):
                count = usage.get('total_tokens' if key == 'tokens' else key, 0)
                data[key] += count
                entry[key] += count
            if data['tokens'] > self.budget.token_limit:
                data['stop_reason'] = 'INCOMPLETE_BUDGET_STOP'
        self.budget.persist()
        return response

    def close(self):
        self.underlying.close()
