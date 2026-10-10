# 当前方法开发入口

继续开发 AgentMemory 方法，请先读 [TEACHER_HANDOFF_CN.md](TEACHER_HANDOFF_CN.md)。
它标出了当前 Bank / 字段级 UPDATE、Packet V3、RP 接口、harness 与训练数据读取器的实际代码位置。
下面是早期 Shopping pipeline 的历史说明，不能视为最新统一配置。

# Memory Agent GRPO — Shopping inference pipeline

This directory contains the reviewable inference-side foundation for a future
Memory Agent GRPO training pipeline. The current runnable target is one
six-Product MemoryArena Shopping episode. It verifies that a model can emit
Memory actions and that those actions pass through validation, execution,
persistent storage, retrieval, and Answer Agent injection.

It does **not** train a model or claim that memory improves task accuracy.

For the latest code review, experimental packet interfaces, development examples,
and what can be checked after cloning, see the
[GitHub review guide](docs/GITHUB_REVIEW_CN.md).

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

The Shopping Answer prompt follows the official React instructions with an
API-native tool-call conversion. Extra search, pre-purchase and repetition
guidance has been removed; official action descriptions remain. This is still
a custom runner and message format, not a verbatim official reproduction.
Both current arms share this Answer configuration. See the
[prompt alignment review](docs/reviews/ANSWER_SHOPPING_PROMPT_ALIGNMENT_CN.md)
for the two-file change, validation and comparison limits.

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

The shared Memory Agent client uses a 4,096-token output cap per decision,
temperature `0`, and disabled thinking. This cap applies to the complete JSON
response; it is not a limit on the total Memory Bank size.

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
  --memory-output-correction-budget 3
unset deepseek_key
```

The live smoke permits at most one model correction per Memory decision and
three per episode. Shopping first tries the deterministic ADD outer-brace repair
described below; this uses no model retry quota. Remaining invalid outputs may
be replayed as untrusted correction data. The Shopping library defaults its
model retry budget to `0`, which does not disable deterministic syntax repair.
The generic episode runtime remains strict by default for raw-output evaluation.

## Result interpretation

- `pipeline_completed=true` means all six structural control flows completed.
- `target_match_count` measures environment-side product correctness.
- `memory_effect_observed=true` means an earlier mutation was followed by a
  later non-empty retrieval injected into the Answer Agent. It is not evidence
  of a causal accuracy improvement.
- `saved_trajectory=false` confirms the live entry emitted only an allowlisted
  summary rather than raw tasks, observations, model outputs, or memory text.

## Recorded Shopping diagnostic

The separate diagnostic entry saves complete local trajectories, including
rejected Memory outputs, validation error codes, correction attempts, Bank
snapshots, Answer inputs, provider usage, and WebShop startup logs. It defaults
to fixed-15 Shopping task 32 and runs Answer-only followed by Memory with the
same Answer model, 4,096-token output cap, 16,000-character observation limit,
20 inference steps per Product, and reset seed 0. Memory output is capped at
4,096 tokens with thinking disabled. The model correction policy is at most one
retry per Memory decision and three per episode, after deterministic syntax repair.

The two Beauty category catalogs are shared by both arms. Results are local
pipeline diagnostics and are not directly comparable to full-catalog historical
benchmark scores. Incorrect purchases are reported separately from structural
completion. The task is not silently replaced if it fails.

First check all six real environment resets without model calls:

```bash
PYTHONDONTWRITEBYTECODE=1 ../MemoryArena/.venv-runtime/bin/python \
  -m examples.memory_agent_grpo.smoke.memoryarena_shopping_diagnostic \
  --preflight-only --output-dir examples/memory_agent_grpo/runs/shopping32_preflight
```

Run the paired diagnostic with a key supplied through the existing environment
variables or a local key file (the key itself is never a command-line argument):

```bash
PYTHONDONTWRITEBYTECODE=1 ../MemoryArena/.venv-runtime/bin/python \
  -m examples.memory_agent_grpo.smoke.memoryarena_shopping_diagnostic \
  --confirm-live-api --api-key-file /private/tmp/memory-agent-deepseek.key \
  --output-dir examples/memory_agent_grpo/runs/shopping32_paired
```

Every invocation requires a new output directory. `run_manifest.json` records
input/source hashes and settings; each arm stores `episode_trace.json`,
`summary.json`, and incremental model-call logs. The Memory arm additionally
stores `memory_attempts.jsonl`, `memory_decisions.jsonl`, and `products.jsonl`.
Provider truncation is recorded in HTTP logs even when no valid Memory trace
can be constructed. These files contain task and model content; directories
are created with mode 0700 and files with mode 0600, with credentials redacted.

## Fixed-prefix memory input probe

`smoke.memoryarena_shopping_prefix_probe` freezes the recorded task-32 Product 1
prefix and evaluates Product 2 with empty, recorded-current, and factual memory.
The recorded current memory is empty, so those two inputs are verified identical
and share executions; they are not counted as independent runs. Factual memory
contains only prior public purchase evidence and arithmetic on the public budget.
It is a diagnostic intervention injected through `retrieved_memories`, bypassing
the writer and retriever; no new Memory API calls or hidden target hints are used.

```bash
PYTHONDONTWRITEBYTECODE=1 ../MemoryArena/.venv-runtime/bin/python \
  -m examples.memory_agent_grpo.smoke.memoryarena_shopping_prefix_probe \
  --prepare-only --output-dir examples/memory_agent_grpo/runs/prefix_inputs

PYTHONDONTWRITEBYTECODE=1 ../MemoryArena/.venv-runtime/bin/python \
  -m examples.memory_agent_grpo.smoke.memoryarena_shopping_prefix_probe \
  --confirm-live-api --api-key-file /private/tmp/memory-agent-deepseek.key \
  --repeats 3 --output-dir examples/memory_agent_grpo/runs/prefix_probe
```

The default source is `runs/shopping32_paired_20260904_r1`; override it with
`--source-run`. Preparation makes zero environment/model calls. Live runs check
that the task, catalogs, and Product 2 public input match the source, then perform
three repetitions per distinct memory input with alternating order. Each run
records initial input, complete Answer trace, HTTP calls, and terminal evidence.
Repeated evaluations of this one case are not independent benchmark tasks.

## Shopping click validity feedback

Before a click, the shared Shopping executor reads the current public
`/available_actions` endpoint. It follows WebShop's input-lowercasing rule and
excludes `click[search]`; search behavior is unchanged. Unavailable clicks return
an ERROR with current clickable items. A failed lookup returns a separate
verification error. Neither case advances the environment, and the Answer
runner can respond using its existing error feedback loop.

Rejected clicks consume Answer inference steps and count as executor calls,
but do not increment upstream environment rounds. Tool success, purchase
completion, and target matching remain separate metrics. New model comparisons
must use the same feedback protocol in both arms.

See the [Chinese review guide](docs/reviews/SHOPPING_CLICK_VALIDATION_CN.md) for
the bounded code changes, tests, and a real-environment check using zero model
calls.

## Memory JSON correction evidence

The generic Memory system prompt now asks for evidence-faithful content and an
explicit distinction between supported facts and uncertain inferences. This
adds no dataset rules or new output fields. See the
[focused review](docs/reviews/MEMORY_EVIDENCE_FIDELITY_CN.md). An eighteen-call
fixed-input comparison found partial content improvements with no increase in
null outputs; errors remain and end-to-end benefits have not been established.

Memory syntax errors now retain parser positions and the rejected completion
for the same decision's correction request. Rejected text is JSON-encoded and
treated as untrusted data. Syntax correction asks the model to preserve a
recoverable allowed action and its content; all corrected outputs still pass
the existing strict validation and execution boundary.

Shopping CLI commands default to three corrections per episode, with at most
one retry per Memory decision; the underlying runner still defaults to zero.
A valid null is still legal, and structural recovery does not imply
preservation or factual correctness.
See the [review guide](docs/reviews/MEMORY_JSON_CORRECTION_CN.md) for the scoped
changes, regression tests, and the nine-call live probe.
The subsequent [budget review](docs/reviews/MEMORY_CORRECTION_BUDGET_CN.md)
describes the 0–3 episode limit and complete correction-attempt records.

## Deterministic ADD syntax repair

Shopping now enables `ADD_MISSING_OUTER_BRACE_V1` before the strict gateway:
only an ADD missing exactly its outermost `}` can be repaired. The original
string is preserved and exactly one character appended; the candidate must pass
all existing schema and phase checks. No content is inferred and no other action
is repaired. Other failures retain the model correction policy above.

Full traces separate `raw_output`, `output_repair` (including original validation
and repaired text), and the effective gateway result. Diagnostic native-valid
counts exclude repaired outputs; `deterministic_repair_count` reports them
separately. Safe summaries expose only validity, error/rule codes and a
`committed_memory_output_repair_count` covering completed products.

The generic `execute_memory_decision` defaults `repair_add_outer_brace=False`;
the Shopping adapter explicitly enables it. A repaired output is not a native
model success, and the appended token must not be attributed to model generation
in future GRPO rollouts. See the [focused review guide](docs/reviews/MEMORY_ADD_BRACE_REPAIR_CN.md)
for the rule, historical replay, boundary tests and limitations.
# 当前交接入口

老师使用最新 Memora SFT 数据，请先看 [TEACHER_HANDOFF_CN.md](TEACHER_HANDOFF_CN.md)。
其中列出数据读取、原生工具格式、schema、执行器、审核与导出的实际源码和离线检查命令。
以下保留项目原有说明。
