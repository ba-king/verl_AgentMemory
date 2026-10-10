"""Frozen BM25 / Qwen3-Embedding / equal RRF, with content-addressed cache."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from ...memory_agent.retrieval.bm25 import _score_document, _tokenize
from .data import digest, source_refs

EMBEDDING_REPO = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QUERY_INSTRUCTION = (
    "Given a question about a user's past conversations, retrieve the personal memories needed to answer it."
)


def bm25_rank(state, query, top_n):
    """Use EXACT production tokenizer, scoring constants and stable tie rule."""
    docs = [_tokenize(r.content) for r in state.bank.records]
    n, total = len(docs), sum(map(len, docs))
    if not n or not total:
        return []
    df = Counter()
    for d in docs:
        df.update(set(d))
    ranked = []
    for i, (r, d) in enumerate(zip(state.bank.records, docs, strict=True)):
        score = _score_document(
            query_terms=tuple(sorted(set(_tokenize(query)))),
            term_frequencies=Counter(d),
            document_frequencies=df,
            document_count=n,
            document_length=len(d),
            average_document_length=total / n,
        )
        if score > 0:
            ranked.append({"memory_id": r.memory_id, "score": score, "position": i})
    return sorted(ranked, key=lambda r: (-r["score"], r["position"]))[:top_n]


def rrf(sparse, dense, constant=60):
    scores, position = Counter(), {}
    for ranked in (sparse, dense):
        for rank, r in enumerate(ranked, 1):
            mid = r["memory_id"]
            scores[mid] += 1.0 / (constant + rank)
            position[mid] = r["position"]
    return [
        {"memory_id": mid, "score": scores[mid], "position": position[mid]}
        for mid in sorted(scores, key=lambda mid: (-scores[mid], position[mid]))
    ]


class DenseEncoder:
    def __init__(self, model_root, cache_root, *, device="cpu", batch_size=8, max_length=4096):
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        model_root, cache_root = Path(model_root), Path(cache_root)
        files = sorted([*model_root.glob("*.safetensors"), *model_root.glob("*.json")])
        if not list(model_root.glob("*.safetensors")):
            raise ValueError("Explicit local Qwen3-Embedding-0.6B weights are required")
        hashes = {}
        for p in files:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for b in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(b)
            hashes[p.name] = h.hexdigest()
        config = json.loads((model_root / "config.json").read_text())
        if config.get("hidden_size") != 1024 or config.get("num_hidden_layers") != 28:
            raise ValueError("Expected Qwen3-Embedding-0.6B configuration")
        self.identity = {
            "model": EMBEDDING_REPO,
            "download_revision": EMBEDDING_REVISION,
            "actual_file_sha256": hashes,
            "pooling": "last_non_padding_token",
            "normalize": True,
            "query_instruction": QUERY_INSTRUCTION,
            "max_length": max_length,
            "dtype": "float32",
            "device": device,
            "attention": "eager",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        self.directory = cache_root / digest(self.identity)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True, padding_side="left")
        if device == "cpu":
            torch.set_num_threads(4)
        self.model = AutoModel.from_pretrained(
            model_root, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
        )
        self.model.eval().requires_grad_(False).to(device)
        self.np, self.torch = np, torch
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.stats = {
            "encoded_documents": 0,
            "encoded_queries": 0,
            "cache_hits": 0,
            "max_observed_tokens": 0,
            "truncated_documents": 0,
        }

    def encode(self, texts, *, query=False):
        paths = [self.directory / (digest([query, text]) + ".npy") for text in texts]
        missing = {}
        for text, p in zip(texts, paths, strict=True):
            if p.exists():
                self.stats["cache_hits"] += 1
            else:
                missing[p] = text
        pending = list(missing.items())
        for i in range(0, len(pending), self.batch_size):
            batch = pending[i : i + self.batch_size]
            strings = [f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}" if query else text for _, text in batch]
            raw = self.tokenizer(strings, padding=True, truncation=False, return_tensors="pt")
            longest = int(raw["attention_mask"].sum(dim=1).max().item())
            self.stats["max_observed_tokens"] = max(self.stats["max_observed_tokens"], longest)
            if longest > self.max_length:
                raise ValueError(f"Refusing silent evidence truncation: {longest} > {self.max_length}")
            raw = {k: v.to(self.device) for k, v in raw.items()}
            with self.torch.inference_mode():
                hidden = self.model(**raw).last_hidden_state
                assert bool(raw["attention_mask"][:, -1].all())
                vectors = self.torch.nn.functional.normalize(hidden[:, -1], p=2, dim=1).cpu().numpy()
            for (p, _), vector in zip(batch, vectors, strict=True):
                if not self.np.isfinite(vector).all():
                    raise ValueError("Nonfinite embedding")
                if not self.np.isclose(self.np.linalg.norm(vector), 1.0, atol=1e-5):
                    raise ValueError("Embedding is not unit normalized")
                self.np.save(p, vector)
            self.stats["encoded_queries" if query else "encoded_documents"] += len(batch)
        if not paths:
            return self.np.empty((0, 1024), dtype=self.np.float32)
        return self.np.stack([self.np.load(p, allow_pickle=False) for p in paths])


class HybridRetriever:
    def __init__(self, encoder=None, *, sparse_top_n=20, dense_top_n=20, top_k=3, rrf_constant=60):
        if min(sparse_top_n, dense_top_n, top_k) < 1 or rrf_constant < 0:
            raise ValueError("Invalid retrieval limits")
        self.encoder = encoder
        self.config = {
            "sparse_top_n": sparse_top_n,
            "dense_top_n": dense_top_n,
            "top_k": top_k,
            "rrf_constant": rrf_constant,
            "rrf_weights": [1, 1],
        }

    def retrieve(self, state, query, registry):
        sparse = bm25_rank(state, query, self.config["sparse_top_n"])
        ranked = {"bm25": sparse}
        if self.encoder is not None:
            # Encode only current records: DELETE cannot leave a stale indexed vector.
            documents = self.encoder.encode([r.content for r in state.bank.records])
            q = self.encoder.encode([query], query=True)[0]
            scored = [
                {"memory_id": r.memory_id, "score": float(score), "position": i}
                for i, (r, score) in enumerate(zip(state.bank.records, documents @ q, strict=True))
            ]
            dense = sorted(scored, key=lambda r: (-r["score"], r["position"]))[: self.config["dense_top_n"]]
            ranked.update(dense=dense, hybrid=rrf(sparse, dense, self.config["rrf_constant"]))
        refs = source_refs(state, registry)
        records = {r.memory_id: r for r in state.bank.records}
        return {
            backend: [
                {**r, "rank": i, "content": records[r["memory_id"]].content, "source_refs": refs[r["memory_id"]]}
                for i, r in enumerate(items, 1)
            ]
            for backend, items in ranked.items()
        }


def utility(ranked, positive, stale, top_k=3):
    import math

    if not positive:
        raise ValueError("Positive-empty QA has no defined utility")
    ranks = {e: min((r["rank"] for r in ranked if e in r["source_refs"]), default=None) for e in positive}
    rank_reward = sum(1 / math.log2(rank + 1) if rank else 0 for rank in ranks.values()) / len(positive)
    recall = sum(rank is not None and rank <= top_k for rank in ranks.values()) / len(positive)
    stale_hits = sorted({e for r in ranked[:top_k] for e in r["source_refs"] if e in stale})
    return {
        "evidence_ranks": ranks,
        "R_rank": rank_reward,
        f"Recall@{top_k}": recall,
        "utility": 0.7 * rank_reward + 0.3 * recall,
        "stale_evidence_top_k": stale_hits,
        "stale_hit_count": len(stale_hits),
        "stale_record_count": sum(bool(set(r["source_refs"]) & set(stale)) for r in ranked[:top_k]),
        "rank_scope": "all returned Top-N candidates; hybrid union up to sparse_N+dense_N",
    }
