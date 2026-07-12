# DART-CSL Memory Agent Example

This example implements `method_name="dart_csl_vtcc"`:

```text
Decision-Aware, Action-Guarded, Repairable, Token-Bounded
Causal State Ledger with Visible Task Contract Compilation
```

The MemoryArena-facing API is still simple:

```text
memory.add_chunk(action_observation_reward_chunk)
memory.wrap_user_prompt(next_task_instruction)
memory.validate_transition(question, proposed_action)
memory.audit_selection(question, selected_option_id)
memory.guard_action(question, proposed_action)
memory.register_task_adapter(adapter)
```

The memory backend may read visible task instructions and tool descriptions.
Protocol adapters may normalize public fields and legal action syntax, but the
core and adapters must not reveal benchmark names in online prompts, inspect
hidden evaluator data, encode fixed item answers/attributes, or branch on a
task/sample id.

## What Changed

```text
Visible task prompt / tool description
-> TaskAdapter registry -> public TaskIR
-> TaskContractCard
-> TaskSchemaCard
-> Event path
-> StateDelta ledger
-> Dependency / evidence / invalidation graph
-> Materialized state lanes
-> no-progress detector + minimum evidence policy
-> recent active execution path
-> typed ActionDirective (advisory or required)
-> DecisionSignature
-> DecisionSupportCapsule
-> [DART-CSL STATE PACKET]
-> optional hard action guard
-> minimal DecisionReceipt for the next scope
```

Key files:

```text
examples/csl_memory_agent/csl_memory/
  csl_memory.py              DARTCSLMemorySystem
  csl_audit.py               certificate hashing, state-version binding, guard audit cache
  csl_backfill.py            canonical missing-field detection and backfill
  csl_contract.py            visible TaskContractCard compiler
  csl_extraction.py          benchmark-blind labeled/JSON event-to-delta extraction
  csl_task_adapter.py        generic TaskIR, adapter protocol/registry, receipts, progress detector
  csl_visible_entities.py    public WebShop protocol adapter and visible entity evidence logic
  csl_schema.py              Event, TaskContractCard, StateDelta, DecisionSignature, capsule
  csl_lanes.py               contract/decision/constraint/dependency/negative/evidence/recent/uncertainty/repair lanes
  csl_packet.py              DecisionSupportCapsule, state packet, and soft verifier
  csl_guard.py               final/irreversible action guard
  csl_transition.py          runner-neutral validation, one repair, and fail-closed result
  memory_arena_adapter.py    MemoryArena-compatible wrapper
```

There is one implementation file for each responsibility. The old
`dart_csl_*.py` forwarding modules and the separate 15-line `csl_verifier.py`
module were removed. Method naming remains `dart_csl_vtcc` at the configuration
and public class level.

## Selection Audit Reuse

The final-action path treats a validated selection as a reusable proof:

```text
visible selection + entity-bound evidence
-> memory-built DecisionCertificate
-> canonical JSON SHA-256
-> deterministic validation
-> one semantic memory-LLM audit
-> bind audit to the already-observed selection state version
-> buy: rebuild certificate + hash/state-version check
-> cache hit, skip memory LLM
```

The state version hashes semantic visible evidence and the active contract; it
does not change merely because state ids were rematerialized or the guard wrote
its own repair feedback. A changed certificate, visible environment event,
decision state, unresolved uncertainty, or guard policy causes a fresh semantic
guard call. An unchanged denial is cached as well, and the no-progress tracker
marks that blocked transition stale so another identical `Buy Now` cannot count
as progress. Metrics include
`guard_audit_cache_hits`, `guard_audit_cache_misses`, and
`guard_llm_calls_avoided`.

Each public `Product N` split is an opaque evidence scope. Search results,
current selections, and other transient states from Product 1 cannot become
candidate evidence for Product 2. Every scope transition publishes a compact
`DecisionReceipt`. Successful scopes use `status=committed`; timeout/failure
scopes use a lower-confidence `provisional` or `failed` summary produced by
re-ranking visible evidence rather than copying the last click. Only the chosen
semantic option, entity attributes, supporting state ids, confidence, and state
version cross the boundary. Entering a detail page records a `candidate`, not a
final decision. A visible `Rating: N.A.` is observed-but-unavailable, which is
different from never inspecting the candidate.

## Adapter Boundary And Anti-Overfitting

The method is split into three boundaries:

```text
generic DART-CSL core
  scope isolation, state ledger, dependency graph, token budget,
  no-progress detection, active path, transition validation,
  DecisionReceipt, certificate cache

task adapter
  compile only the public instruction/tool schema into TaskIR,
  bind visible entities, declare evidence levels, emit a typed ActionDirective

benchmark bridge
  pass the proposed action through the shared transition helper;
  keep protocol-local output grammar checks beside the runner
```

`TaskIR` contains objectives, constraints, dependencies, evidence requirements,
commit actions, and public metadata. The ledger and transition validator do not
contain ASINs, colors, brands, or Task0 values. A generic safety vocabulary
recognizes broad final-action verbs such as `submit`, `book`, and `buy`, while
the active protocol's adapter declares the exact `TaskIR.commit_actions`; the
core no longer has a special `Buy Now` branch. The current WebShop adapter
parses arbitrary option text and arbitrary public compatibility relations; a
Travel, Search, or Formal adapter can implement the same interface without
changing the ledger, progress detector, receipt, retrieval lanes, or guard, then
register it with `memory.register_task_adapter(adapter)`. Explicitly registered
adapters have priority over the bundled protocol adapter by default.

Adapters are per public environment protocol, not per benchmark item or
`Product N`. The single `WebShopTaskAdapter` handles every shopping subtask by
parsing values from the visible prompt; it contains no Task0 answers. WebShop is
the largest adapter because its protocol exposes ASINs, result/detail pages,
package variants, prices, ratings, and clickables. That UI grounding stays in
`csl_visible_entities.py`; transition enforcement stays in the generic core.

Consequently, running Product 1 through Product 6, another bundled-shopping
item, or a different catalog requires **no new adapter file and no item-specific
branch**. If "subtask" means one of MemoryArena's four tracks, a small
family-level protocol adapter can be useful because the public interfaces are
genuinely different: shopping exposes page actions, travel exposes itinerary
state, search exposes progressive query evidence, and formal reasoning exposes
proof/problem state. That is one adapter per public protocol, not one adapter
per example.

Other task families do not need to reproduce the WebShop parser. They can
inherit `GenericTaskAdapter` and initially override only `supports()` and, when
the public protocol exposes a deterministic evidence obligation,
`build_action_directive()`. `compile_task()` is optional unless the task needs
typed objectives, dependencies, or evidence levels. If an adapter branches on
an item id or accumulates per-sample rules, that is treated as benchmark
overfitting and is not allowed.

Minimal family-level adapter:

```python
class RouteAdapter(GenericTaskAdapter):
    name = "route"

    def supports(self, question: str) -> bool:
        return "route matrix" in question.lower()

    def build_action_directive(self, **kwargs):
        if "route-b is unresolved" not in kwargs["question"].lower():
            return None
        return ActionDirective(
            action="inspect[unresolved-branch]",
            reason="resolve the remaining public feasibility gap",
            mode="required_state_change",
            obligation_id="resolve-route-branch",
            required_state_delta=("route-b feasibility becomes known",),
            recovery_actions=(
                RecoveryAction(
                    action="revise[unresolved-branch]",
                    operator="revise",
                    reason="repair the unresolved branch",
                ),
            ),
            forbidden_actions=("submit[*]",),
        )
```

The example intentionally has no shopping fields. Directive matching,
one-transition lease binding, bounded repair, metrics, active-path rendering,
and environment blocking are inherited from the core and benchmark bridge.

Current support is intentionally stated narrowly:

| MemoryArena track | Generic CSL ledger/packet | Typed family adapter | Bounded transition hook |
| --- | --- | --- | --- |
| Bundled Web Shopping | yes | yes, one adapter for all items/products | yes |
| Group Travel Planning | yes, generic fallback | not yet specialized | shared helper ready; runner hook not yet registered |
| Progressive Web Search | yes, generic fallback | not yet specialized | shared helper ready; runner hook not yet registered |
| Formal Reasoning | yes, generic fallback | not yet specialized | shared helper ready; runner hook not yet registered |

Thus the architecture is cross-task, while the current end-to-end typed
integration has only been completed and tested for shopping. The other three
tracks do not require copies of WebShop parsing. Their eventual adapter should
only translate visible protocol fields into `TaskIR`/`ActionDirective`, and
their runner integration should call the same `enforce_required_transition()`
helper. Claiming equal task-specific behavior before those thin hooks and tests
exist would overstate the implementation.

Evidence follows a least-sufficient policy. A result page can prove a visible
price objective; a rating objective requires detail evidence; missing identity
or compatibility evidence triggers targeted retrieval. Compatibility uses
`compatible / incompatible / unknown`: explicit incompatibility is filtered,
unknown dependencies are soft audit inputs, and identity/objective gaps remain
deterministic blockers. The planner and guard consume the same certificate.
Entity hypotheses are retained top-k and ranked lexicographically: identity
validity and package/variant consistency precede the public objective. A
visible near-match may contribute a discriminating field to the next query, but
an extra bundle cannot win merely because its price is higher.

Progress is measured by decision-state change, not raw page novelty. Equivalent
observations reached through different actions, repeating 2-6 step cycles, and
unchanged candidate/constraint ledgers mark the latest path stale. Candidate ids
from historical pages are withheld as click targets, and the official answer
adapter rejects any `click[...]` absent from the final current environment page.
The packet also retains the latest four task-agnostic `action -> outcome` rows.
Required recovery actions have a configurable consecutive-use budget, so a
changing page cannot make one navigation operator repeat until an environment
soft-exit threshold is reached.

Most packet content remains advisory. A deterministic unresolved evidence gap
is emitted as `Required state change`: an evidence goal, temporarily prohibited
actions, and a set of admissible recoveries. The first recovery is a ranked
suggestion, not a unique path. The directive is leased to the current event and
semantic state version, expires after the next observation, and is recomputed.
`validate_transition()` checks the proposed action before environment
execution. A violation receives one immediate repair call; a second violation
is blocked without mutating the environment. Legacy external adapters may still
use an exact `Required evidence transition`. The mechanism is task-independent:
adapters declare goals/action affordances, while the core validates leases,
prohibitions, action sets, and repeat budgets.

The corresponding diagnostics are `no_progress_events`,
`stale_actions_detected`, `transition_validation_calls`,
`transition_validation_blocks`, `transition_action_set_blocks`,
`transition_forbidden_action_blocks`, `transition_repeat_budget_blocks`,
`decision_receipts_emitted`,
`provisional_receipts_emitted`, and
`cross_scope_receipts_retrieved`. These are method diagnostics, not replacements
for MemoryArena's official `SR`, `PS`, or token count.

Anti-overfitting checks live in
`tests/examples/csl_memory_agent/test_csl_task_adapters.py`. They permute option
order, use unseen model/capacity values, use arbitrary relation symbols such as
`Alloy-X -> Ceramic-Z`, preserve identical entity ids across scopes, and exercise
the same no-progress detector on a non-shopping itinerary action. No test uses a
Task0 answer as an online input.

## Model Roles

There are two model roles:

```text
answer agent:
  The model that solves the MemoryArena task.
  Configure it in the MemoryArena JSON under `agent`.

memory agent:
  DART-CSL refreshes a deterministic candidate/state packet on every action turn;
  the configured model is called only for semantic selection/hard-action audits
  after deterministic contract, entity-binding, certificate, and budget checks pass.
  Configure it with `CSL_MEMORY_MODEL`, `CSL_MEMORY_API_KEY`,
  and `CSL_MEMORY_BASE_URL`.

The official shopping runner defaults to `max_tokens=1024` for the answer role;
set `P2R_ANSWER_MAX_TOKENS=4096` to restore the original MemoryArena ceiling. It
accepts only one unambiguous, current-page-legal `search[...]` or `click[...]`
action from response content or reasoning. Missing, ambiguous, or stale-page
actions receive one bounded 128-token repair call. Each split `Product N`
refreshes the visible TaskContractCard once; ordinary page turns do not
recompile it.
```

For quick local tests, the memory agent can stay `mock`. For real experiments,
set both roles explicitly. The two keys may be the same if both models use the
same provider.

## Local Smoke Test

```bash
python examples/csl_memory_agent/scripts/run_csl_memory_smoke.py
python examples/csl_memory_agent/scripts/run_memoryarena_local_smoke.py
pytest -q -p no:debugging tests/examples/csl_memory_agent
```

On the local Anaconda Python 3.13 environment, plain `pytest` can segfault while
loading pytest's debugging plugin; `-p no:debugging` avoids that local issue.

## Bundled Web Shopping Item 0 Demo

This demo runs the first row of MemoryArena `bundled_shopping/data.jsonl`
(`baking_item_0`). In API mode, the answer agent is called through an
OpenAI-compatible API and controls a small webshop loop with
`search[...] -> select[...] -> click[Buy Now]`. The loop records every API
response, action, observation, DART-CSL packet, guard result, and final purchase.

The controlled demo now uses the optimized safety path:

```text
full current instruction -> scoped TaskContractCard
structured action/observation -> entity-bound StateDelta ledger
answer-agent DecisionCertificate -> deterministic grounding/optimality check
-> memory-agent semantic select guard -> state-bound deterministic buy recheck
-> post-step offline reference evaluation
```

All online candidate product ids use the same `DEMO-SxOy` format. The hidden
reference ASIN and expected option id are unavailable to both agents until the
step has completed. Therefore this is a leakage-resistant method test, but it is
still a controlled local environment rather than an official MemoryArena score.

Run one complete small Product 1 test with both DeepSeek roles:

```bash
export P2R_MEMORY_API_KEY="sk-your-deepseek-key"
export P2R_MEMORY_MODEL="deepseek-v4-flash"
export P2R_MEMORY_BASE_URL="https://api.deepseek.com"

cd /Users/lizizheng/verl
python examples/csl_memory_agent/scripts/run_bundled_shopping_item0_demo.py \
  --selection-mode api-agent \
  --max-steps 1 \
  --max-agent-turns 8 \
  --answer-model "$P2R_MEMORY_MODEL" \
  --memory-provider openai \
  --memory-model "$P2R_MEMORY_MODEL" \
  --state-dir /tmp/dart_csl_optimized_product1_state \
  --output-dir examples/csl_memory_agent/demo_outputs/optimized_product1
```

Run the complete six-product API demo:

```bash
ANSWER_MODEL=gpt-4.1-mini \
ANSWER_AGENT_API_KEY=sk-your-answer-agent-key \
ANSWER_BASE_URL=https://api.openai.com/v1 \
CSL_MEMORY_PROVIDER=openai \
CSL_MEMORY_MODEL=gpt-4.1-mini \
CSL_MEMORY_API_KEY=sk-your-memory-agent-key \
CSL_MEMORY_BASE_URL=https://api.openai.com/v1 \
bash -lc '
set -euo pipefail
cd /Users/lizizheng/verl
python examples/csl_memory_agent/scripts/run_bundled_shopping_item0_demo.py \
  --selection-mode api-agent \
  --max-steps 6 \
  --answer-model "$ANSWER_MODEL" \
  --answer-api-key "$ANSWER_AGENT_API_KEY" \
  --answer-base-url "$ANSWER_BASE_URL" \
  --memory-provider "$CSL_MEMORY_PROVIDER" \
  --memory-model "$CSL_MEMORY_MODEL" \
  --memory-api-key "${CSL_MEMORY_API_KEY:-}" \
  --memory-base-url "$CSL_MEMORY_BASE_URL" \
  --state-dir /tmp/dart_csl_bundled_shopping_item0_state_api \
  --output-dir examples/csl_memory_agent/demo_outputs
'
```

Outputs:

```text
examples/csl_memory_agent/demo_outputs/bundled_shopping_item0_demo_trace.json
examples/csl_memory_agent/demo_outputs/bundled_shopping_item0_demo_trace.md
```

For a cheaper plumbing run, set `CSL_MEMORY_PROVIDER=mock`; the answer agent will
still use the API. The answer-agent key and memory-agent key may be the same key
if both roles use the same provider.

DeepSeek/P2R-style environment variables can drive both roles:

```bash
export P2R_MEMORY_API_KEY="sk-your-deepseek-key"
export P2R_MEMORY_MODEL="deepseek-v4-flash"
export P2R_MEMORY_BASE_URL="https://api.deepseek.com"

cd /Users/lizizheng/verl
python examples/csl_memory_agent/scripts/run_bundled_shopping_item0_demo.py \
  --selection-mode api-agent \
  --max-steps 6 \
  --max-agent-turns 10 \
  --answer-model "$P2R_MEMORY_MODEL" \
  --memory-provider openai \
  --memory-model "$P2R_MEMORY_MODEL" \
  --state-dir /tmp/dart_csl_bundled_shopping_item0_state_deepseek \
  --output-dir examples/csl_memory_agent/demo_outputs/deepseek
```

To print the full packets while running:

```bash
ANSWER_MODEL=gpt-4.1-mini \
ANSWER_AGENT_API_KEY=sk-your-answer-agent-key \
ANSWER_BASE_URL=https://api.openai.com/v1 \
bash -lc '
cd /Users/lizizheng/verl
python examples/csl_memory_agent/scripts/run_bundled_shopping_item0_demo.py \
  --selection-mode api-agent \
  --max-steps 6 \
  --answer-model "$ANSWER_MODEL" \
  --answer-api-key "$ANSWER_AGENT_API_KEY" \
  --answer-base-url "$ANSWER_BASE_URL" \
  --print-packets
'
```

Offline modes are kept only for debugging and comparison. To replay the known
reference trajectory without API calls:

```bash
cd /Users/lizizheng/verl
python examples/csl_memory_agent/scripts/run_bundled_shopping_item0_demo.py \
  --max-steps 6 \
  --selection-mode reference-replay \
  --output-dir examples/csl_memory_agent/demo_outputs/reference_replay
```

To compare with a no-reference local heuristic baseline:

```bash
cd /Users/lizizheng/verl
python examples/csl_memory_agent/scripts/run_bundled_shopping_item0_demo.py \
  --selection-mode heuristic \
  --output-dir examples/csl_memory_agent/demo_outputs/heuristic
```

The reference answers are saved in the trace only under
`offline_reference_answers_do_not_feed_to_memory`; they are not included in the
online candidate catalog, answer-agent prompt, memory prompt, or DART-CSL state
packet. API mode reads the expected option only after each step completes.

## Official MemoryArena Task 0

This command runs all six sessions of official `baking_item_0` using the
official MemoryArena task file, WebShop product database, environment server,
runner, and reward script. Both model roles use the same DeepSeek model and key.
The key stays in environment variables and is not written to generated configs
or result files.

```bash
MEMORYARENA_DIR=/path/to/MemoryArena \
MEMORYARENA_PYTHON=/path/to/MemoryArena/.venv/bin/python \
P2R_MEMORY_API_KEY="sk-your-deepseek-key" \
P2R_MEMORY_MODEL="deepseek-v4-flash" \
P2R_MEMORY_BASE_URL="https://api.deepseek.com" \
MEMORYARENA_RUN_ID="task0_deepseek_$(date +%Y%m%d_%H%M%S)" \
bash /Users/lizizheng/verl/examples/csl_memory_agent/scripts/run_official_memoryarena_task0.sh
```

The script performs the complete sequence:

```text
register DART-CSL guard integration
-> start memory and official environment servers
-> run six official WebShop sessions
-> record exact answer-agent API usage
-> run the official reward scorer
-> combine answer/memory token usage
-> write item-level SR, PS, soft reward, and per-step results
```

### Official Single-Step Functional Pilot

Use Product 1 for a complete, independently decidable official API smoke test.
Both answer and memory roles use the explicitly supplied DeepSeek model and key:

```bash
MEMORYARENA_DIR=/path/to/MemoryArena \
MEMORYARENA_PYTHON=/path/to/MemoryArena/.venv/bin/python \
P2R_MEMORY_API_KEY="sk-your-deepseek-key" \
P2R_MEMORY_MODEL="deepseek-v4-flash" \
P2R_MEMORY_BASE_URL="https://api.deepseek.com" \
MEMORYARENA_PILOT_STEP=1 \
MEMORYARENA_MAX_ROUNDS=20 \
MEMORYARENA_RUN_ID="product1_pilot_$(date +%Y%m%d_%H%M%S)" \
bash /Users/lizizheng/verl/examples/csl_memory_agent/scripts/run_official_memoryarena_step_pilot.sh
```

`MEMORYARENA_PILOT_STEP` can construct any source step, but Product 2 onward
depends on earlier ground-truth purchases. An unseeded isolated Product 2-6 run
is useful for diagnostics only and is not a comparable standalone benchmark
score. Run the full task for valid dependency-aware evaluation.

The verified Product 1 run at
`official_runs/pilots/baking_item_0/product1/official_product1_exact_query_20260711_v1`
purchased exact target `B00TUDFEW2` and received official `PS_item=1.0` and
`SR_item=1`. It used 50,767 answer-agent tokens and 5,535 memory-agent tokens,
56,302 combined. This token total excludes the offline reward scorer. One
incorrect multipack purchase was blocked; the final buy
reused the bound selection audit (`guard_audit_cache_hits=1`,
`guard_llm_calls_avoided=1`). This is a one-step functional score, not the
paper's six-step item score or 150-item aggregate.

Every invocation uses an isolated run id, so failed or interrupted ledgers do
not contaminate later experiments. Reports are written to:

```text
examples/csl_memory_agent/official_runs/task0/<run-id>/memoryarena_task0_report.json
examples/csl_memory_agent/official_runs/task0/<run-id>/memoryarena_task0_report.md
```

For one complete item, `PS_item` is the mean of the six official soft step
rewards. `SR_item = 1` only when every step receives reward 1; otherwise it is
0. `hard_step_pass_rate` is reported separately for diagnosis and is not PS.
These are valid item-level metrics, not the paper's 150-item aggregate.

The later complete official run
`official_full_task0_guided_20260711_v1` produced `PS=1/6=0.1667`, `SR=0`, and
374,495 combined answer/memory tokens. It passed one of six sessions. The current
V7 behavior was implemented after that run, so those results must not be
presented as scores for the current code until another official run is completed.

The V4 rerun `official_task0_adapter_v4_20260711_rerun1` was intentionally
stopped during Product 4 and has no valid item-level SR or PS. Its 93 completed
API calls consumed 384,849 answer tokens and exposed deterministic planner/guard
disagreement, success-only receipt cascades, stale historical clicks, and
single-transition-only progress detection. Replaying its 90 saved environment
events through V5/V6 requires no API calls: Product 1-4 now produce semantic
provisional summaries for the intended option descriptions, Product 3 exposes
no stale candidate ids as legal clicks, the Product 4 target certificate has no
blocking uncertainty, and deterministic guard validation allows it. This is a
counterfactual regression check, not an official benchmark score.

The bounded V5 Product 1 pilot
`official_product1_v5_pilot_20260712_v1` produced `PS=0`, `SR=0`, and 25,863
answer tokens in ten calls. It made no purchase and no memory-LLM call. From
turn 5 onward the packet correctly required unresolved Simple Mills evidence,
but the answer model alternated between two already resolved searches. That run
predates the V6 typed transition validator and is the regression case for the
new required-obligation tests; by itself it predicts neither V6 nor V7
performance.

The official V6 Product 1 pilot
`official_product1_v6_transition_20260712_v1` used the same 20-round limit but
terminated after 17 environment actions under MemoryArena's
`repeat_action>=7` soft-exit rule. It made no purchase, received `PS=0` and
`SR=0`, and consumed 76,470 answer tokens across 28 recorded calls; the memory
LLM was not called. V6 successfully blocked unsafe purchases and enforced every
typed transition, but its exact-action policy repeatedly forced `click[Next >]`
and caused the failure it was meant to repair.

V7 was implemented from that failure analysis. It replaces a persistent exact
recovery action with a one-transition evidence goal, a set of currently legal
recoveries, explicit temporary prohibitions, and a consecutive-action budget.
It also retains top-k identity hypotheses, ranks package consistency before the
objective, and treats explicit soft exits as failures even when the environment
also emits `Done: True`. An offline replay of the saved V6 states confirms that
the third repeated `Next` is blocked, alternative refined searches and `Back`
remain admissible, and unresolved `Buy Now` attempts are prohibited. Because a
different allowed action would produce a new environment observation, this is
only a local regression check, not a counterfactual trajectory or an official
V7 SR/PS result. No paid V7 benchmark rerun has yet been completed.

The earlier `official_optimized_task0_v3` run produced `PS=0.0`, `SR=0`, and
2,039,088 combined tokens because all 17 proposed purchases were blocked. Its
raw official score also remains unchanged; fixes derived from its logs are
clearly marked as post-run changes in:

```text
examples/csl_memory_agent/official_runs/task0/official_optimized_task0_v3/memoryarena_task0_analysis.md
```

## One-Time MemoryArena Registration

The MemoryArena preview server does not know `dart_csl_vtcc` by default. Register
it once in your MemoryArena checkout:

```bash
VERL_DIR=/Users/lizizheng/verl \
MEMORYARENA_DIR=/path/to/MemoryArena \
bash -lc 'python "$VERL_DIR/examples/csl_memory_agent/scripts/register_memoryarena.py" --memoryarena-dir "$MEMORYARENA_DIR"'
```

The script is idempotent. It registers the cross-track `dart_csl_vtcc` memory
factory plus transition-validation, selection-audit, and hard-guard client
endpoints. For the shopping runner it additionally installs per-turn
candidate-state refresh, bounded answer context, a thin call to the shared
one-repair transition helper, a post-step selection hook, non-recursive guard
feedback, and WebShopPlus purchase-summary parsing. The
official task data, product database, environment dynamics, and reward formula
remain unchanged. The full catalog is still loaded; the startup patch only
avoids materializing millions of unused synthetic warm-up goals.

## Can I Choose One Of The Four MemoryArena Tasks?

Yes. Choose one track at the offline runner/config layer:

```text
travel          -> run_travel.py
shopping        -> run_shopping.py
search          -> run_search.py
formal_reasoning -> run_math.py
```

This interface is allowed for DART-CSL because the track choice is outside the
online memory prompt. Do not pass the selected track name into memory prompts,
and do not add item-specific memory extractors. The commands below exercise the
generic memory interface on all tracks; only shopping currently has the typed
family adapter and bounded runner hook listed in the support table above.

## Fully Specified Commands

Replace these placeholders before running:

```text
/path/to/MemoryArena        your MemoryArena checkout
ANSWER_AGENT_API_KEY        API key for the answer agent
ANSWER_BASE_URL             API base URL for the answer agent, e.g. https://api.openai.com/v1
CSL_MEMORY_API_KEY          API key for the memory agent
CSL_MEMORY_BASE_URL         API base URL for the memory agent
```

You can set `CSL_MEMORY_PROVIDER=mock` and omit the memory key/base URL for a
deterministic memory smoke run.

### Travel Planning

Default answer model: `gpt-4.1-mini`

Default memory model: `gpt-4.1-mini`

```bash
VERL_DIR=/Users/lizizheng/verl \
MEMORYARENA_DIR=/path/to/MemoryArena \
ANSWER_MODEL=gpt-4.1-mini \
ANSWER_AGENT_API_KEY=sk-your-answer-key \
ANSWER_BASE_URL=https://api.openai.com/v1 \
CSL_MEMORY_PROVIDER=openai \
CSL_MEMORY_MODEL=gpt-4.1-mini \
CSL_MEMORY_API_KEY=sk-your-memory-key \
CSL_MEMORY_BASE_URL=https://api.openai.com/v1 \
bash -lc '
set -euo pipefail
cd "$MEMORYARENA_DIR"
export PYTHONPATH="$VERL_DIR:$MEMORYARENA_DIR:${PYTHONPATH:-}"
export OPENAI_API_KEY="$ANSWER_AGENT_API_KEY"
export OPENAI_BASE_URL="$ANSWER_BASE_URL"
export OPENAI_API_BASE="$ANSWER_BASE_URL"
export CSL_STATE_DIR="$MEMORYARENA_DIR/.csl_state/travel"

python "$VERL_DIR/examples/csl_memory_agent/scripts/register_memoryarena.py" --memoryarena-dir "$MEMORYARENA_DIR"

python memory/server.py > /tmp/dart_csl_memory_travel.log 2>&1 &
MEMORY_PID=$!
python env/env_server.py > /tmp/dart_csl_env_travel.log 2>&1 &
ENV_PID=$!
trap "kill $MEMORY_PID $ENV_PID 2>/dev/null || true" EXIT
sleep 5

python - <<'"'"'PY'"'"'
import json, os
from pathlib import Path
root = Path(os.environ["MEMORYARENA_DIR"])
cfg = json.loads((root / "configs/travel_planner_configs/long_context-gpt-4.1-mini.json").read_text())
cfg["agent"]["model_name"] = os.environ["ANSWER_MODEL"]
cfg["agent"]["api_key"] = os.environ["ANSWER_AGENT_API_KEY"]
cfg["agent"]["base_url"] = os.environ["ANSWER_BASE_URL"]
cfg["memory"]["memory_system_name"] = "dart_csl_vtcc"
cfg["memory"]["server_url"] = "http://0.0.0.0:8000"
cfg["env"]["env_server_url"] = "http://0.0.0.0:8001"
cfg["output"]["output_dir"] = "results/travel/dart_csl_vtcc/" + os.environ["ANSWER_MODEL"]
out = Path("/tmp/dart_csl_travel.json")
out.write_text(json.dumps(cfg, indent=2))
print(out)
PY

python run_travel.py --config /tmp/dart_csl_travel.json
'
```

### Web Shopping

Default answer model: `gpt-4.1-mini`

Default memory model: `gpt-4.1-mini`

```bash
VERL_DIR=/Users/lizizheng/verl \
MEMORYARENA_DIR=/path/to/MemoryArena \
ANSWER_MODEL=gpt-4.1-mini \
ANSWER_AGENT_API_KEY=sk-your-answer-key \
ANSWER_BASE_URL=https://api.openai.com/v1 \
CSL_MEMORY_PROVIDER=openai \
CSL_MEMORY_MODEL=gpt-4.1-mini \
CSL_MEMORY_API_KEY=sk-your-memory-key \
CSL_MEMORY_BASE_URL=https://api.openai.com/v1 \
SHOPPING_TASK_FILE_LIMIT=10 \
bash -lc '
set -euo pipefail
cd "$MEMORYARENA_DIR"
export PYTHONPATH="$VERL_DIR:$MEMORYARENA_DIR:${PYTHONPATH:-}"
export OPENAI_API_KEY="$ANSWER_AGENT_API_KEY"
export OPENAI_BASE_URL="$ANSWER_BASE_URL"
export OPENAI_API_BASE="$ANSWER_BASE_URL"
export CSL_STATE_DIR="$MEMORYARENA_DIR/.csl_state/shopping"

python "$VERL_DIR/examples/csl_memory_agent/scripts/register_memoryarena.py" --memoryarena-dir "$MEMORYARENA_DIR"

python memory/server.py > /tmp/dart_csl_memory_shopping.log 2>&1 &
MEMORY_PID=$!
python env/env_server.py > /tmp/dart_csl_env_shopping.log 2>&1 &
ENV_PID=$!
trap "kill $MEMORY_PID $ENV_PID 2>/dev/null || true" EXIT
sleep 5

python - <<'"'"'PY'"'"'
import json, os
from pathlib import Path
root = Path(os.environ["MEMORYARENA_DIR"])
cfg = json.loads((root / "configs/web_shopping_configs/long_context-gpt-4.1-mini.json").read_text())
cfg["agent"]["model_name"] = os.environ["ANSWER_MODEL"]
cfg["agent"]["api_key"] = os.environ["ANSWER_AGENT_API_KEY"]
cfg["agent"]["base_url"] = os.environ["ANSWER_BASE_URL"]
cfg["memory"]["memory_system_name"] = "dart_csl_vtcc"
cfg["memory"]["server_url"] = "http://0.0.0.0:8000"
cfg["env"]["env_server_url"] = "http://0.0.0.0:8001"
cfg["task_specific"]["task_file_limit"] = int(os.environ.get("SHOPPING_TASK_FILE_LIMIT", "10"))
cfg["output"]["output_dir"] = "results/shopping/dart_csl_vtcc/" + os.environ["ANSWER_MODEL"]
out = Path("/tmp/dart_csl_shopping.json")
out.write_text(json.dumps(cfg, indent=2))
print(out)
PY

python run_shopping.py --config /tmp/dart_csl_shopping.json
'
```

### Web Search

Default answer model: `gpt-5-mini`

Default memory model: `gpt-4.1-mini`

```bash
VERL_DIR=/Users/lizizheng/verl \
MEMORYARENA_DIR=/path/to/MemoryArena \
ANSWER_MODEL=gpt-5-mini \
ANSWER_AGENT_API_KEY=sk-your-answer-key \
ANSWER_BASE_URL=https://api.openai.com/v1 \
CSL_MEMORY_PROVIDER=openai \
CSL_MEMORY_MODEL=gpt-4.1-mini \
CSL_MEMORY_API_KEY=sk-your-memory-key \
CSL_MEMORY_BASE_URL=https://api.openai.com/v1 \
SEARCH_QUERY_IDS=116 \
bash -lc '
set -euo pipefail
cd "$MEMORYARENA_DIR"
export PYTHONPATH="$VERL_DIR:$MEMORYARENA_DIR:${PYTHONPATH:-}"
export OPENAI_API_KEY="$ANSWER_AGENT_API_KEY"
export OPENAI_BASE_URL="$ANSWER_BASE_URL"
export OPENAI_API_BASE="$ANSWER_BASE_URL"
export CSL_STATE_DIR="$MEMORYARENA_DIR/.csl_state/search"

python "$VERL_DIR/examples/csl_memory_agent/scripts/register_memoryarena.py" --memoryarena-dir "$MEMORYARENA_DIR"

python memory/server.py > /tmp/dart_csl_memory_search.log 2>&1 &
MEMORY_PID=$!
python env/env_server.py > /tmp/dart_csl_env_search.log 2>&1 &
ENV_PID=$!
trap "kill $MEMORY_PID $ENV_PID 2>/dev/null || true" EXIT
sleep 5

python - <<'"'"'PY'"'"'
import json, os
from pathlib import Path
root = Path(os.environ["MEMORYARENA_DIR"])
cfg = json.loads((root / "configs/web_search_configs/search_task.json").read_text())
cfg["agent"]["model_name"] = os.environ["ANSWER_MODEL"]
cfg["memory"]["memory_system_name"] = "dart_csl_vtcc"
cfg["memory"]["memory_url"] = "http://0.0.0.0:8000"
cfg["memory"]["no_memory"] = False
cfg["env"]["env_server_url"] = "http://0.0.0.0:8001"
cfg["task_specific"]["query_ids"] = [x.strip() for x in os.environ.get("SEARCH_QUERY_IDS", "116").split(",") if x.strip()]
cfg["output"]["output_dir"] = "results/web_search/dart_csl_vtcc_" + os.environ["ANSWER_MODEL"]
out = Path("/tmp/dart_csl_search.json")
out.write_text(json.dumps(cfg, indent=2))
print(out)
PY

python run_search.py --config /tmp/dart_csl_search.json
'
```

### Formal Reasoning

Default answer model: `gpt-5-mini`

Default memory model: `gpt-4.1-mini`

Choose `FORMAL_HF_CONFIG=formal_reasoning_math` or `formal_reasoning_phys`.

```bash
VERL_DIR=/Users/lizizheng/verl \
MEMORYARENA_DIR=/path/to/MemoryArena \
ANSWER_MODEL=gpt-5-mini \
ANSWER_AGENT_API_KEY=sk-your-answer-key \
ANSWER_BASE_URL=https://api.openai.com/v1 \
CSL_MEMORY_PROVIDER=openai \
CSL_MEMORY_MODEL=gpt-4.1-mini \
CSL_MEMORY_API_KEY=sk-your-memory-key \
CSL_MEMORY_BASE_URL=https://api.openai.com/v1 \
FORMAL_HF_CONFIG=formal_reasoning_math \
bash -lc '
set -euo pipefail
cd "$MEMORYARENA_DIR"
export PYTHONPATH="$VERL_DIR:$MEMORYARENA_DIR:${PYTHONPATH:-}"
export OPENAI_API_KEY="$ANSWER_AGENT_API_KEY"
export OPENAI_BASE_URL="$ANSWER_BASE_URL"
export OPENAI_API_BASE="$ANSWER_BASE_URL"
export CSL_STATE_DIR="$MEMORYARENA_DIR/.csl_state/formal_reasoning"

python "$VERL_DIR/examples/csl_memory_agent/scripts/register_memoryarena.py" --memoryarena-dir "$MEMORYARENA_DIR"

python memory/server.py > /tmp/dart_csl_memory_formal.log 2>&1 &
MEMORY_PID=$!
python env/env_server.py > /tmp/dart_csl_env_formal.log 2>&1 &
ENV_PID=$!
trap "kill $MEMORY_PID $ENV_PID 2>/dev/null || true" EXIT
sleep 5

python - <<'"'"'PY'"'"'
import json, os
from pathlib import Path
root = Path(os.environ["MEMORYARENA_DIR"])
cfg = json.loads((root / "configs/formal_reasoning_configs/math_bm25.json").read_text())
cfg["agent"]["model_name"] = os.environ["ANSWER_MODEL"]
cfg["agent"]["base_url"] = os.environ["ANSWER_BASE_URL"]
cfg["memory"]["memory_system_name"] = "dart_csl_vtcc"
cfg["memory"]["base_url"] = "http://0.0.0.0:8000"
cfg["env"]["base_url"] = "http://0.0.0.0:8001"
cfg["env"]["env_name"] = "phys" if os.environ.get("FORMAL_HF_CONFIG") == "formal_reasoning_phys" else "math"
cfg["env"]["env_config"]["model_name"] = os.environ["ANSWER_MODEL"]
cfg["env"]["env_config"]["base_url"] = os.environ["ANSWER_BASE_URL"]
cfg["task_specific"]["dataset"]["hf_config"] = os.environ.get("FORMAL_HF_CONFIG", "formal_reasoning_math")
cfg["output"]["json_output_dir"] = "./results/json/dart_csl_vtcc"
out = Path("/tmp/dart_csl_formal.json")
out.write_text(json.dumps(cfg, indent=2))
print(out)
PY

python run_math.py -c /tmp/dart_csl_formal.json
'
```

## Notes

- The four commands intentionally choose the task at the runner/config layer.
- The memory prompt remains benchmark-blind and does not receive the track name.
- `CSL_MEMORY_PROVIDER=mock` is useful for plumbing tests; use
  `CSL_MEMORY_PROVIDER=openai` for the real semantic hard-action guard.
- If ports `8000` or `8001` are already in use, stop the old servers or edit the
  generated config/server commands consistently.
