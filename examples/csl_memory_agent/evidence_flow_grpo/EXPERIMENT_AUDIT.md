# Evidence-Flow GRPO source audit

- Audit date: `2026-08-28`
- Overall verdict: **PASS for teacher handoff into G0, then a bounded G1 smoke**
- Scientific-result verdict: **not applicable; no GPU training result exists**
- Review class: same-family Type-A independent source review
- Cross-model/Type-B replication: **not performed**

## Scope

The reviewer inspected the data/reward contracts, stock-VERL launcher and config,
preflight, run metadata, per-seed G2 analyzer, two-seed aggregate, tests, canonical S0
artifacts, and teacher-facing claim boundaries. This is an engineering/source audit,
not evidence that CUDA launch, vLLM generation, backward propagation, checkpointing,
or held-out policy improvement works on the target server.

## Integrity findings

| Check | Verdict | Evidence boundary |
|---|---|---|
| Ground-truth provenance | PASS | Deterministic synthetic fixture; public rows carry opaque IDs; sealed targets are labeled `synthetic_proxy` |
| Raw scoring / no self-normalization | PASS | Fixed deterministic component scores; structural GRPO normalization is not presented as a reward metric |
| Result existence | PASS | Only the committed CPU S0 result is reported; GPU results are explicitly absent |
| Scope and evaluation type | PASS | `structural_mock + held_out_synthetic_proxy`; four base tasks and zero optimizer steps |
| Result-to-claim boundary | PASS | Claims stop at source-level readiness for G0/G1 smoke |
| Credential scan | PASS | No credential value in the handoff whitelist |

## Closed Major finding

The first review found that an earlier G2 analyzer accepted step-0 and checkpoint
validation JSONLs without requiring the corresponding VAL_ONLY launcher metadata.
That would have allowed a copied validation file to be relabeled too easily.

The final analyzer now requires three immutable metadata files: the training run,
the base-model VAL_ONLY run, and the checkpoint VAL_ONLY run. It checks seed, actor,
frozen data/scorer/config/reward hashes, training parameters, exact
`OUTPUT_DIR/validation/<step>.jsonl` placement, and the exact checkpoint resume path.
The two-seed aggregate also requires distinct training and validation run identities.
Regression tests reject an outside-output validation file and a checkpoint validation
run bound to the wrong resume path. The reviewer marked the original Major **closed**.

## Focused stock-VERL integration review

A second read-only same-family reviewer traced the launcher against this checkout's
stock config, dataset filtering, reward wrapper, FSDP implementation, vLLM rollout,
checkpoint naming, and G2 code. Its first pass found genuine source blockers: the
1024-token setting could filter training rows, the preflight followed stale
`setup.py` vLLM metadata instead of the checkout's vLLM 0.20 runtime, the advertised
fp16 field did not drive stock FSDP mixed precision, and PREFLIGHT_ONLY exited before
exact Hydra composition.

The final snapshot closes those findings with a measured 1280 prompt limit and
zero-filter gate; Dockerfile-pinned `vllm==0.20.2`; required NumPy/TorchData and
flash-attn checks; bf16-only actor/ref `mixed_precision`; and exact Hydra composition
before every successful preflight exit. The reviewer's final verdict is **PASS for
G0**, followed by a fresh 10-step smoke only when the output contains both
`PREFLIGHT PASS` and `Exact Hydra composition PASS`.

## Remaining limitations

- The local Mac did not execute Linux/CUDA preflight or import the full target
  PyTorch/Ray/vLLM stack. G0 remains mandatory on the teacher's server.
- No optimizer step or genuine model rollout has occurred. S0 is deterministic
  structural evidence only.
- The current gold-subset reward is a synthetic selector proxy, not the proposed
  Answerer-conditioned local `Q_t` or terminal environment reward.
- Pre-run metadata plus analyzer recomputation does not cryptographically prove that
  an adversarial editor never replaced a same-path artifact before audit. Preserve
  server run directories read-only; use a signed post-run manifest if that threat
  model matters.
- A local Hugging Face snapshot with only an index JSON passes the shallow
  preflight weight-presence test before the model loader checks referenced shards.
- G2 model provenance hashes config and weights but not tokenizer files, and records
  Git HEAD without requiring a clean worktree. Use a clean committed checkout and a
  fixed local snapshot; extend these bindings before treating G2 as hostile-tamper
  evidence.
- The reviewer was an independent same-family Codex agent. This is Type-A review,
  not cross-model Type-B verification.

## Verification recorded at handoff

```text
57 GRPO package tests + 8 fixture provenance tests = 65 passed
fresh structural sanity: PASS; 32 groups; zero-variance rate 0.0
canonical structural result byte comparison: PASS
bash -n: PASS
compileall: PASS
Black: PASS
credential/local-path scan: PASS
```

The audit permits G0 and, after both preflight success markers, a bounded 10-step
server smoke. It does not authorize a method-validity, learnability, PEV,
generalization, or GRPO-necessity claim.
