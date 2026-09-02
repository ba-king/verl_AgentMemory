# Memory Agent GRPO — Shopping inference pipeline

This directory contains the reviewable inference-side foundation for a future
Memory Agent GRPO training pipeline. The current runnable target is one
six-Product MemoryArena Shopping episode. It verifies that a model can emit
Memory actions and that those actions pass through validation, execution,
persistent storage, retrieval, and Answer Agent injection.

It does **not** train a model or claim that memory improves task accuracy.

## Runtime flow

For each of the six Products:

1. Open a fresh WebShop session and fresh Answer Agent conversation.
2. Ask the Memory Agent for one `PRE_TASK` decision.
3. Execute an optional `RETRIEVE` against the shared Memory Bank.
4. Inject exact retrieval matches into the fixed Answer Agent input.
5. Let the Answer Agent call one WebShop tool per inference until the
   environment terminates.
6. Ask the Memory Agent for one `POST_TASK` decision.
7. If that decision is `RETRIEVE`, make one
   `POST_TASK_AFTER_RETRIEVAL` decision.

The Answer conversation is isolated per Product. The Memory Bank persists
across all six Products.

## Directory map

```text
memory_agent/                    action protocol, context, Bank, BM25, Executor
answer_agent/                    fixed Task/Answer Agent and DeepSeek provider
episode/                         shared decision trace and runtime boundaries
adapters/memoryarena/shopping/   thin MemoryArena Shopping integration
model_io/                        bounded HTTP transport
smoke/                           local/live composition entry points
tests/                           offline unit and integration checks
```

Primary files:

- `adapters/memoryarena/shopping/memory_episode_runtime.py` — six-Product
  orchestration.
- `adapters/memoryarena/shopping/memory_episode_contract.py` — result and
  failure contracts.
- `adapters/memoryarena/shopping/memory_episode_validation.py` — cross-stage
  invariants.
- `adapters/memoryarena/shopping/memory_episode_summary.py` — allowlisted,
  content-free live summary.
- `smoke/memoryarena_shopping_memory_episode_smoke.py` — runnable live entry.

Some smaller smoke modules are retained because the primary entry currently
imports their provider and MemoryArena setup helpers.

## Offline verification

From the `verl` repository root, with the sibling `MemoryArena` runtime
available:

```bash
PYTHONDONTWRITEBYTECODE=1 ../MemoryArena/.venv-runtime/bin/python \
  -m pytest -q examples/memory_agent_grpo/tests
```

The tests use scripted clients or fake transports. They do not call DeepSeek
or consume API credit.

For a short review focused on the complete Shopping composition:

```bash
PYTHONDONTWRITEBYTECODE=1 ../MemoryArena/.venv-runtime/bin/python \
  -m pytest -q \
  examples/memory_agent_grpo/tests/adapters/memoryarena/shopping/test_memory_episode_runtime.py \
  examples/memory_agent_grpo/tests/smoke/test_memoryarena_shopping_memory_episode_smoke.py
```

## Live six-Product smoke

The command requires explicit confirmation and reads the key without echoing
it. It does not save the trajectory.

```bash
printf 'DeepSeek key: '
IFS= read -r -s deepseek_key
printf '\n'
P2R_ANSWER_API_KEY="$deepseek_key" \
P2R_MEMORY_API_KEY="$deepseek_key" \
PYTHONDONTWRITEBYTECODE=1 \
../MemoryArena/.venv-runtime/bin/python \
  -m examples.memory_agent_grpo.smoke.memoryarena_shopping_memory_episode_smoke \
  --confirm-live-api \
  --memory-output-correction-budget 1
unset deepseek_key
```

The live smoke permits at most one episode-wide correction after a rejected
Memory output. The rejected raw text is neither replayed nor executed. The
generic Shopping runner defaults this budget to `0`; training and evaluation
rollouts should explicitly keep it at `0` so protocol failures remain part of
the learning signal.

## Result interpretation

- `pipeline_completed=true` means all six structural control flows completed.
- `target_match_count` measures environment-side product correctness.
- `memory_effect_observed=true` means an earlier mutation was followed by a
  later non-empty retrieval injected into the Answer Agent. It is not evidence
  of a causal accuracy improvement.
- `saved_trajectory=false` confirms the live entry emitted only an allowlisted
  summary rather than raw tasks, observations, model outputs, or memory text.
