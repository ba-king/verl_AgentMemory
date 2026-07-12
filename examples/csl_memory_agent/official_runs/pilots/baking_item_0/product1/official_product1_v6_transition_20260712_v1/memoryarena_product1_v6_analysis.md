# Product 1 V6 Transition Pilot Analysis

## Scope

- Official source item: `baking_item_0`, Product 1 only
- Answer model: `deepseek-v4-flash`
- Memory model setting: `deepseek-v4-flash`
- Maximum environment rounds: 20, matching the supplied MemoryArena shopping configs
- Actual environment actions: 17
- Termination: official WebShopPlus soft exit, `repeat_action>=7`
- This is a one-session functional pilot, not a complete six-step item or paper aggregate.

## Official Result

| Metric | Value |
| --- | ---: |
| Purchased ASIN | none |
| Target ASIN | `B00TUDFEW2` |
| PS | 0.0 |
| SR | 0 |
| Answer API calls recorded | 28 |
| Answer prompt tokens | 63,021 |
| Answer completion tokens | 13,449 |
| Answer total tokens | 76,470 |
| Memory LLM calls / tokens | 0 / 0 |
| Transition validations / initial blocks | 21 / 4 |
| Transition repair calls | 4 |
| Guard calls / blocks | 2 / 2 |
| No-progress events | 4 |

The 28 answer API records include 17 primary environment-turn calls, four
contract-repair calls, four transition-repair calls, and three second attempts.
Contract repairs consumed 12,133 tokens and transition repairs consumed 10,044
tokens, together 22,177 tokens (29.0% of the answer total).

## Executed Action Trace

| Turn | Action submitted to the environment | Important event |
| ---: | --- | --- |
| 1 | `search[Duncan Hines ... 16.5 oz]` | First listed option resolved |
| 2 | `search[Sweet 'N Low ...]` | Second listed option resolved |
| 3 | `search[Simple Mills ... muffin pans]` | Approximate bundle candidate found |
| 4 | repeated Simple Mills search | Initial `click[B08FRM79XT]` was transition-repaired |
| 5 | `click[B08FRM79XT]` | Opened a bundle/variety-pack candidate |
| 6 | `click[Buy Now]` | Guard blocked the uncertain certificate |
| 7 | `click[Buy Now]` | Guard blocked the unchanged retry |
| 8 | `click[Back to Search]` | Returned to search |
| 9 | refined Simple Mills search | Still did not expose target ASIN |
| 10 | `search[keywords]` | Low-information model action |
| 11 | `click[Next >]` | Initial new search was transition-repaired |
| 12 | `click[Next >]` | Page scan |
| 13 | `click[Next >]` | Initial new search was transition-repaired |
| 14 | `click[Next >]` | Page scan |
| 15 | `click[Next >]` | Initial Back action was transition-repaired |
| 16 | `click[Next >]` | Page scan |
| 17 | `click[Next >]` | Official `repeat_action>=7` soft exit |

`B00TUDFEW2` never appeared in any V6 observation. In the older successful
diagnostic run, the target first appeared only after the query was refined with
the visible size clue: `Simple Mills ... vanilla cake mix 11.5 oz`.

## Findings

1. **The enforcement mechanism works.** All four violating proposals were
   caught before environment execution, repaired once, and then revalidated.
   The two unsafe purchase attempts were also kept out of the environment.

2. **The enforced recovery policy was wrong after the first useful repair.**
   The adapter repeatedly emitted the concrete required action `click[Next >]`.
   It overrode semantically reasonable attempts to reformulate the search or
   leave the pagination path, eventually causing the benchmark soft exit.

3. **Entity identity is the upstream error.** `B08FRM79XT` was retained as the
   Simple Mills candidate despite adding another variety pack. Objective price
   was then applied to that approximate identity. The ledger retained only one
   best match and lost the useful near-exact `B015NBNZ98` title containing the
   `11.5oz` clue used by the historical successful recovery.

4. **Guard feedback prevented harm but did not produce an executable recovery
   obligation.** The first blocked Buy remained advisory, so the answer model
   immediately retried it. An unchanged denial should become a deterministic
   prohibition plus a required evidence-recovery obligation.

5. **CSL progress semantics disagree with the environment repetition budget.**
   Each Next action changed the page and therefore looked like progress to CSL,
   while WebShopPlus counts the action string and exits after seven repeats.

6. **The event parser mislabels the soft exit as success.** The final memory
   event has `Done: True` and was classified `is_success=true` even though the
   official scorer correctly reports no purchase. This can corrupt a later
   cross-scope receipt and must be fixed independently of planning.

## Historical Diagnostic Comparison

These runs differ in code and round budgets, so they are not a controlled
ablation and must not be used as a method-level performance claim.

| Run | Environment events | PS | SR | Combined tokens | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| V5 bounded pilot | 10 | 0.0 | 0 | 25,863 | unresolved-candidate search loop |
| V6 transition pilot | 17 | 0.0 | 0 | 76,470 | forced pagination soft exit |
| Older exact-query diagnostic | 16 | 1.0 | 1 | 56,302 | bought `B00TUDFEW2` |

V6 used 195.7% more total tokens than the V5 pilot. Normalized by environment
event, it used 4,498 versus 2,586 tokens (73.9% more), partly because contract
and transition repairs accounted for 29.0% of V6 answer tokens.

## Recommended Method Changes Before Another Paid Run

1. Represent a required obligation as an evidence goal plus an admissible
   action set, rather than one exact action that persists across states.
2. Give recovery directives a one-transition TTL and recompute after every
   successful state change; cap repeated use of the same operator.
3. Retain top-k entity hypotheses with explicit variant mismatch reasons and
   rank identity validity before objective value.
4. Convert an unchanged guard denial into a required prohibition/recovery
   directive until the evidence state version changes.
5. Treat explicit soft-exit/timeout/no-purchase observations as failures before
   generic `Done: True` success inference.
6. Re-run Product 1 with the same 20-round budget, then run an ablation with
   transition validation disabled to isolate directive quality from enforcement.

## Post-Run V7 Implementation And Offline Check

All five corrective changes above were implemented after the paid V6 run. The
new directive represents a required state change as an evidence goal, temporary
prohibitions, and multiple admissible recovery actions. It is bound to one
event/state lease and limits consecutive use of the same recovery action. The
WebShop adapter now retains top-k identity hypotheses, marks bundle conflicts,
and can use a public field from a visible near-match to refine a query. Explicit
soft exits override generic `Done: True` success inference.

Replaying the saved V6 observations and proposed actions through the V7 policy
requires no API calls and verifies the following local decisions:

- unresolved `Buy Now` proposals are blocked by the transition layer;
- the generic `search[keywords]` proposal is rejected as non-evidentiary;
- refined alternative searches and `Back to Search` remain legal recoveries;
- a third consecutive `click[Next >]` is blocked before MemoryArena's seven-repeat threshold;
- the final soft-exit event is parsed as failure, not success.

This replay validates the new validator at states that were actually observed.
It cannot determine the full V7 trajectory: accepting a different action would
change the next environment observation, so the remainder of the V6 log is no
longer a valid counterfactual rollout. V7 therefore has no official SR, PS, or
token result until a fresh benchmark run is completed.
