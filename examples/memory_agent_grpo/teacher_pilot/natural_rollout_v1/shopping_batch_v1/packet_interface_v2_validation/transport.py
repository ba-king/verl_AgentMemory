"""Strict /beta Packet client. Does not mutate frozen provider or Answer client."""
import time
from pathlib import Path
import requests
from .common import save
from .interface import TOOLS, messages
from ....travel_v0.provider import model_config, credential_ready
from ....travel_v0.artifacts import text


class PacketClient:
    def __init__(self, budget):
        self.config = model_config()
        if not credential_ready(self.config):
            raise RuntimeError('MISSING_CONFIGURED_CREDENTIAL')
        self.key = Path(self.config['key_file']).read_text().strip()
        self.budget = budget
        self.transport = requests.Session()

    def request(self, inp):
        return {**{k: self.config[k] for k in ['model', 'temperature', 'max_tokens', 'thinking', 'stream']},
            'messages': messages(inp), 'tools': TOOLS,
            'tool_choice': {'type': 'function', 'function': {'name': 'emit_packet_delta'}},
            'parallel_tool_calls': False}

    def call(self, inp, dest):
        dest = Path(dest)
        if dest.exists():
            raise RuntimeError('REFUSE_SECOND_ATTEMPT')
        request = self.request(inp)
        encoded = text(request)
        if self.key in encoded or len(encoded) > 180000:
            raise RuntimeError('INPUT_GUARD')
        self.budget.consume_call()
        save(dest / 'request.json', request)
        start = time.monotonic()
        endpoint = self.config['base_url'] + '/beta/chat/completions'
        try:
            response = self.transport.post(endpoint,
                headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'},
                data=encoded.encode(), timeout=self.config['timeout_seconds'], allow_redirects=False)
            (dest / 'response_body.txt').write_text(response.text.replace(self.key, '[REDACTED]'))
            save(dest / 'transport.json', {'endpoint': endpoint, 'http_status': response.status_code,
                'latency_seconds': time.monotonic() - start, 'response_bytes': len(response.content),
                'role': 'packet_memory_teacher', 'strict_requested': True, 'retries': 0})
            if response.status_code != 200:
                raise RuntimeError('HTTP_' + str(response.status_code))
            if len(response.content) > self.config['max_response_bytes']:
                raise RuntimeError('RESPONSE_TOO_LARGE')
            body = response.json()
            save(dest / 'response.json', body)
            self.budget.consume_usage(body.get('usage', {}))
            if body.get('model') not in self.config['accepted_models']:
                raise RuntimeError('UNEXPECTED_RESPONSE_MODEL')
            return body
        except Exception as error:
            save(dest / 'error.json', {'type': type(error).__name__,
                'message': str(error).replace(self.key, '[REDACTED]'),
                'latency_seconds': time.monotonic() - start})
            raise
