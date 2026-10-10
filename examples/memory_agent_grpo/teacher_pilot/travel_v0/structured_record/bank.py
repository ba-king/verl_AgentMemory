"""Structured records mapped to the unchanged content-indexed Bank/executor."""
import copy
from ..memory_runtime import BankBranch, FrozenHybridAdapter, DenseEncoder, HybridRetriever
from ..artifacts import WORKSPACE, FROZEN, read
from .record_renderer import render


class FreshHybridAdapter(FrozenHybridAdapter):
    def __init__(self, cache_root):
        frozen = read(FROZEN)
        self.encoder = DenseEncoder(WORKSPACE/'models/Qwen3-Embedding-0.6B', cache_root,
            device='cpu', batch_size=4, max_length=frozen['embedding']['max_length'])
        assert self.encoder.identity == frozen['embedding']
        config = dict(frozen['retriever']); config.pop('rrf_weights')
        self.backend = HybridRetriever(self.encoder, **config)
        assert self.backend.config == frozen['retriever']
        self.last_rankings = None


class StructuredBank(BankBranch):
    def __init__(self, retriever):
        self.records = {}
        super().__init__(retriever)

    def snapshot(self):
        snapshot = super().snapshot()
        for row in snapshot['records']:
            row['record'] = copy.deepcopy(self.records[row['memory_id']])
            assert row['content'] == render(row['record'])
        return snapshot

    def execute(self, action):
        name, args = action['action'], action['arguments']
        internal = copy.deepcopy(action)
        if name in {'ADD', 'UPDATE'}:
            internal['arguments'].pop('record')
            internal['arguments']['content'] = render(args['record'])
        actions_before = len(self.actions)
        result, hits = super().execute(internal)
        if len(self.actions) > actions_before:
            self.actions[-1]['action'] = copy.deepcopy(action)
        if result['status'] != 'REJECTED' and name in {'ADD', 'UPDATE'}:
            mid = result['record']['memory_id']
            self.records[mid] = copy.deepcopy(args['record'])
            result['record']['record'] = copy.deepcopy(args['record'])
            self.receipts[-1] = copy.deepcopy(result)
        elif result['status'] != 'REJECTED' and name == 'DELETE':
            self.records.pop(args['memory_id'])
        if hits is not None:
            for hit in hits:
                hit['record'] = copy.deepcopy(self.records[hit['memory_id']])
                assert hit['content'] == render(hit['record'])
        return result, hits
