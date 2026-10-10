"""Fixed independent API clients, bounded requests, full raw logging, no retries."""
import copy
import time
from pathlib import Path
import requests
from .artifacts import PROVIDER, read, save, append, text


def model_config():
    source = read(PROVIDER)
    model = source['models'][0]
    provider = source['providers'][model['provider']]
    assert model['model'] == 'deepseek-v4-pro' and provider['base_url'] == 'https://api.deepseek.com'
    return {'model': model['model'], 'accepted_models': model['accepted_models'],
        'base_url': provider['base_url'], 'key_file': provider['key_file'],
        'temperature': 0, 'max_tokens': 4096, 'thinking': {'type': 'disabled'}, 'stream': False,
        'timeout_seconds': source['timeout_seconds'], 'minimum_interval_seconds': 1,
        'retries': 0, 'max_response_bytes': source['max_response_bytes']}


def credential_ready(config):
    p = Path(config['key_file'])
    if not p.is_file():
        return False
    key = p.read_text().strip()
    return bool(key) and all(0x21 <= ord(c) <= 0x7e for c in key)


def request_for(messages, tools, role):
    cfg = model_config()
    request = {k: cfg[k] for k in ['model', 'temperature', 'max_tokens', 'thinking', 'stream']}
    return {**request, 'messages': copy.deepcopy(messages), 'tools': copy.deepcopy(tools),
            'tool_choice': 'required' if role == 'memory_teacher' else 'auto',
            'parallel_tool_calls': False}


class Budget:
    def __init__(self, path, call_limit=120, token_limit=800000):
        self.path = path
        self.data = read(path) if Path(path).exists() else {'requests': 0, 'tokens': 0}
        self.call_limit, self.token_limit = call_limit, token_limit
        self.last = 0

    def consume_call(self):
        if self.data['requests'] >= self.call_limit or self.data['tokens'] >= self.token_limit:
            raise RuntimeError('PREDECLARED_API_BUDGET_EXHAUSTED')
        time.sleep(max(0, 1 - (time.monotonic() - self.last)))
        self.last = time.monotonic()
        self.data['requests'] += 1
        save(self.path, self.data)

    def consume_usage(self, usage):
        self.data['tokens'] += usage.get('total_tokens', 0)
        save(self.path, self.data)


class FixedClient:
    def __init__(self, role, directory, budget):
        self.role, self.directory, self.budget = role, Path(directory), budget
        self.config = model_config()
        if not credential_ready(self.config):
            raise RuntimeError('MISSING_CONFIGURED_DEEPSEEK_CREDENTIAL')
        self.key = Path(self.config['key_file']).read_text().strip()
        self.count = 0
        # Independent transport instance per agent; messages never stored globally.
        self.transport = requests.Session()

    def call(self, messages, tools):
        self.count += 1
        dest = self.directory / f'{self.role}_{self.count:03d}'
        if dest.exists():
            raise RuntimeError('refuse_to_overwrite_existing_request')
        request = request_for(messages, tools, self.role)
        encoded = text(request)
        if self.key in encoded:
            raise RuntimeError('credential_in_request_body')
        if len(encoded) > 180000:
            raise RuntimeError('PREDECLARED_INPUT_SIZE_EXCEEDED')
        self.budget.consume_call()
        save(dest / 'request.json', request)
        start = time.monotonic()
        try:
            response = self.transport.post(self.config['base_url'] + '/chat/completions',
                headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'},
                data=encoded.encode(), timeout=self.config['timeout_seconds'], allow_redirects=False)
            body_text = response.text.replace(self.key, '[REDACTED]')
            (dest / 'response_body.txt').write_text(body_text)
            meta = {'http_status': response.status_code, 'latency_seconds': time.monotonic() - start,
                    'response_bytes': len(response.content), 'role': self.role}
            save(dest / 'transport.json', meta)
            if response.status_code != 200:
                raise RuntimeError(f'HTTP_{response.status_code}')
            if len(response.content) > self.config['max_response_bytes']:
                raise RuntimeError('RESPONSE_TOO_LARGE')
            body = response.json()
            save(dest / 'response.json', body)
            self.budget.consume_usage(body.get('usage', {}))
            if body.get('model') not in self.config['accepted_models']:
                raise RuntimeError('UNEXPECTED_RESPONSE_MODEL')
            if len(body.get('choices', [])) != 1:
                raise RuntimeError('UNEXPECTED_CHOICE_COUNT')
            choice = body['choices'][0]
            if choice.get('finish_reason') not in {'stop', 'tool_calls'}:
                raise RuntimeError('INCOMPLETE_MODEL_OUTPUT_' + str(choice.get('finish_reason')))
            return body, {'request_ref': str(dest / 'request.json'), 'response_ref': str(dest / 'response.json'),
                'token_usage': body.get('usage', {}), 'latency': meta['latency_seconds'],
                'model_config': self.config, 'actual_model': body['model']}
        except Exception as error:
            save(dest / 'error.json', {'type': type(error).__name__,
                'message': str(error).replace(self.key, '[REDACTED]'), 'latency': time.monotonic() - start})
            raise
