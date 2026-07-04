# P2R-CMC Stage 1 Demo

This example is the prompt-based Stage 1 demo for agent memory. Stage 2
SFT/RL/GRPO is not included here.

The demo shows one complete memory loop:

```text
session turns
-> session-end memory extraction
-> SQLite memory bank
-> maintenance trigger check
-> cluster maintenance when needed
-> Qwen dense retrieval + BM25 sparse retrieval
-> RRF fusion
-> coverage-aware evidence selection
-> verifier as a fixed post-retrieval check
-> EvidencePacket
```

## Quick Start

The local Qwen3-Embedding-0.6B checkpoint is expected at:

```bash
export P2R_LOCAL_EMBEDDING_MODEL_PATH="$HOME/.cache/p2r_cmc/models/Qwen3-Embedding-0.6B"
export P2R_LOCAL_EMBEDDING_DEVICE="cpu"
```

Run the travel planning demo:

```bash
python examples/p2r_cmc/scripts/run_p2r_cmc_travel_demo.py
```

This default run uses local Qwen3-Embedding-0.6B, the rule-based memory-agent
fallback, and no main answer API call, so it does not consume DeepSeek credits.
The scenario is a realistic Kansai travel planning conversation with pace,
budget, food, mobility, flight, weather backup, and stale-conflict memories.

The use-stage memory agent only chooses the retrieval range:

```text
EXPAND_CLUSTER / SEARCH_GLOBAL / ANSWER_NO_MEMORY
```

Conflict and safety checks are handled after retrieval by the verifier.

Cluster maintenance is the heavier long-term index-cleanup path. New memories
are assigned to clusters immediately and mark cluster cards dirty. The system
first runs a cheap trigger check, then runs maintenance only when there are
enough new/updated memories, dirty clusters, oversized or low-cohesion clusters,
redundant cluster pairs, recent verifier issues, or recent missing-aspect
retrieval signals. With `--memory-agent openai`, the memory agent proposes
constrained JSON actions in cluster-index-only mode:

```text
REFRESH_CLUSTER_CARD / SPLIT_CLUSTER / MERGE_CLUSTER / NO_OP
```

The model never mutates SQLite directly. Deterministic executors validate IDs,
apply cluster-card updates, split groups, merge redirects, and write
maintenance traces. MemoryEntry content/status changes are intentionally left
to a separate future consolidation pass.

## Run With DeepSeek

Use this when you want both the memory agent and main answer agent to call the
real DeepSeek API, while keeping Qwen3-Embedding-0.6B as the local embedding
model:

```bash
export P2R_MEMORY_BASE_URL="https://api.deepseek.com"
export P2R_MEMORY_API_KEY="your_real_deepseek_api_key"
export P2R_MEMORY_MODEL="deepseek-v4-flash"
export P2R_MEMORY_TIMEOUT="120"

export P2R_MAIN_BASE_URL="https://api.deepseek.com"
export P2R_MAIN_API_KEY="your_real_deepseek_api_key"
export P2R_MAIN_MODEL="deepseek-v4-flash"
export P2R_MAIN_TIMEOUT="180"

export P2R_LOCAL_EMBEDDING_MODEL_PATH="$HOME/.cache/p2r_cmc/models/Qwen3-Embedding-0.6B"
export P2R_LOCAL_EMBEDDING_DEVICE="cpu"

python examples/p2r_cmc/scripts/run_p2r_cmc_travel_demo.py \
  --reset \
  --embedding local-qwen \
  --memory-agent openai \
  --main-agent openai \
  --db-path examples/p2r_cmc/data/travel_demo.sqlite \
  --dump-json examples/p2r_cmc/data/travel_demo_dump.json
```

Replace `your_real_deepseek_api_key` with your actual DeepSeek key. A successful
real-API run should end with `MAIN_AGENT_ANSWER`.

## Save SQLite Output

To persist the memory bank:

```bash
python examples/p2r_cmc/scripts/run_p2r_cmc_travel_demo.py \
  --reset \
  --embedding local-qwen \
  --db-path examples/p2r_cmc/data/travel_demo.sqlite \
  --dump-json examples/p2r_cmc/data/travel_demo_dump.json
```

Inspect a saved store:

```bash
python examples/p2r_cmc/scripts/inspect_store.py \
  --db-path examples/p2r_cmc/data/travel_demo.sqlite \
  --dump-json examples/p2r_cmc/data/travel_demo_dump.json
```

Open the SQLite file in VS Code with your SQLite viewer extension:

```text
examples/p2r_cmc/data/travel_demo.sqlite
```

If the VS Code viewer is unavailable, use the lightweight `sqlite_web` fallback:

```bash
~/.local/bin/sqlite_web --read-only --host 127.0.0.1 --port 8088 --no-browser examples/p2r_cmc/data/travel_demo.sqlite
```

Then open:

```text
http://127.0.0.1:8088
```

## What To Look For

The demo output has five steps:

```text
STEP 1: session buffer exists, long-term memory is still empty
STEP 2: durable memories are written only at session end
STEP 3: one stale conflicting memory is added for verifier testing
STEP 4: trigger policy decides whether cluster maintenance should run
STEP 5: retrieval returns coverage-aware EvidencePacket after verifier filtering
```

In `STEP 5`, a good output should include evidence rationales like:

```text
rrf:bm25_sparse,dense_embedding,recent_memory
```

This means the memory was supported by BM25, Qwen dense retrieval, and recency,
then merged by RRF.

The EvidencePacket also includes:

```text
required_aspects: explicit query requirements extracted by rules
coverage: which memories cover which extracted aspects
missing_aspects: aspects that had candidate evidence but were not selected
```

This prevents a lower-scoring but explicitly requested memory, such as a dietary
restriction or safety constraint, from being dropped merely because pure top-k
ranking filled the budget first.

If the verifier finds a stale conflicting memory, it appears under:

```json
"conflicts": [
  "mem_xxx: contradicted - conflicts with newer selected memory mem_yyy"
]
```

## Key Files

```text
examples/p2r_cmc/scripts/run_p2r_cmc_travel_demo.py
examples/p2r_cmc/memory_system/controller/session_pipeline.py
examples/p2r_cmc/memory_system/controller/remember_pipeline.py
examples/p2r_cmc/memory_system/controller/use_pipeline.py
examples/p2r_cmc/memory_system/controller/verifier.py
examples/p2r_cmc/memory_system/clustering/maintenance_trigger.py
examples/p2r_cmc/memory_system/clustering/maintenance_loop.py
examples/p2r_cmc/memory_system/prompts/cluster_maintenance.md
examples/p2r_cmc/memory_system/retrieval/aspect_extractor.py
examples/p2r_cmc/memory_system/retrieval/coverage_selector.py
examples/p2r_cmc/memory_system/retrieval/bm25.py
examples/p2r_cmc/memory_system/retrieval/merger.py
examples/p2r_cmc/memory_system/prompts/
```

## Tests

`pytest` may segfault in the local Anaconda Python 3.13 environment before
running tests. A basic compile check is:

```bash
python -m py_compile $(find examples/p2r_cmc tests/examples/p2r_cmc -name '*.py')
```

The relevant tests are under:

```text
tests/examples/p2r_cmc/
```
