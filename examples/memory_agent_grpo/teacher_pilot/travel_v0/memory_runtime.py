"""Adapter to the unchanged Bank/executor and frozen Stage 3 hybrid retrieval."""
from dataclasses import asdict
from types import SimpleNamespace
from examples.memory_agent_grpo.memory_agent.bank.store import MemoryBankStore
from examples.memory_agent_grpo.memory_agent.action.protocol import (
    MemoryAction, MemoryActionName, AddArguments, UpdateArguments, DeleteArguments, RetrieveArguments)
from examples.memory_agent_grpo.memory_agent.executor.engine import MemoryActionExecutor
from examples.memory_agent_grpo.memory_agent.retrieval.result import RetrievalResult, RetrievalMatch
from examples.memory_agent_grpo.audits.stage3_reward_feasibility.retrieval import DenseEncoder, HybridRetriever
from .artifacts import FROZEN, WORKSPACE, PROJECT, read, digest


class FrozenHybridAdapter:
    def __init__(self):
        frozen = read(FROZEN)
        self.encoder = DenseEncoder(WORKSPACE / 'models/Qwen3-Embedding-0.6B',
            PROJECT / 'runs/stage3_reward_feasibility_20260911_r1/embedding_cache',
            device='cpu', batch_size=4, max_length=frozen['embedding']['max_length'])
        if self.encoder.identity != frozen['embedding']:
            raise ValueError('frozen_embedding_identity_mismatch')
        cfg = dict(frozen['retriever'])
        cfg.pop('rrf_weights')
        self.backend = HybridRetriever(self.encoder, **cfg)
        assert self.backend.config == frozen['retriever']
        self.last_rankings = None

    def retrieve(self, query, *, state):
        # Empty audit registry: no hidden labels/provenance influence retrieval.
        audited_state = SimpleNamespace(bank=state, roots={r.memory_id: r.memory_id for r in state.records})
        self.last_rankings = self.backend.retrieve(audited_state, query, {})
        hits = self.last_rankings['hybrid'][:self.backend.config['top_k']]
        by_id = {r.memory_id: r for r in state.records}
        return RetrievalResult(query, tuple(RetrievalMatch(by_id[h['memory_id']], h['score']) for h in hits))


class BankBranch:
    def __init__(self, retriever):
        self.retriever = retriever
        self.executor = MemoryActionExecutor(store=MemoryBankStore(), retriever=retriever)
        self.begin_session()

    def begin_session(self):
        self.visible_ids = set()
        self.visible = []
        self.receipts = []
        self.actions = []

    def snapshot(self):
        return asdict(self.executor.state)

    def execute(self, action):
        name, args = action['action'], action['arguments']
        if name == 'YIELD':
            return {'status': 'SUCCEEDED', 'code': 'YIELDED'}, None
        if name in {'UPDATE', 'DELETE'} and args['memory_id'] not in self.visible_ids:
            return {'status': 'REJECTED', 'code': 'ID_NOT_VISIBLE_IN_CURRENT_SESSION'}, None
        typed = {'ADD': AddArguments, 'UPDATE': UpdateArguments, 'DELETE': DeleteArguments,
                 'RETRIEVE': RetrieveArguments}[name](**args)
        result = self.executor.execute(MemoryAction(MemoryActionName(name), typed))
        out = {'status': result.status.value, 'code': result.code.value}
        hits = None
        if result.retrieval is not None:
            hits = [{**r.to_dict(), 'rank': i} for i, r in enumerate(result.retrieval.matches, 1)]
            out.update(query=args['query'], matches=hits)
            for h in hits:
                self.visible_ids.add(h['memory_id'])
            self.visible.extend(hits)
        if result.record is not None and name != 'RETRIEVE':
            out['record'] = asdict(result.record)
            if result.status.value != 'REJECTED' and name in {'ADD', 'UPDATE'}:
                self.visible_ids.add(result.record.memory_id)
                self.receipts.append(out.copy())
        self.actions.append({'action': action, 'result': out})
        return out, hits


def delta(before, after):
    b = {r['memory_id']: r['content'] for r in before['records']}
    a = {r['memory_id']: r['content'] for r in after['records']}
    return {'added': sorted(set(a) - set(b)), 'deleted': sorted(set(b) - set(a)),
            'updated': sorted(k for k in set(a) & set(b) if a[k] != b[k]),
            'before_sha256': digest(before), 'after_sha256': digest(after)}
