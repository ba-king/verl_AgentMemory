# Evidence-Flow GRPO handoff verification

- Snapshot: `2026-08-28`
- Evaluation type: `structural_mock + synthetic_proxy`
- Engineering status: `READY_FOR_SERVER_G0_G1_SMOKE`
- Scientific status: `NOT_READY_FOR_CORE_METHOD_CLAIM`
- GPU/API/optimizer usage in this package: none; `actual_optimizer_steps=0`

## Canonical CPU structural result

`runs/structural_sanity_v1/structural_sanity.json` was produced without loading a
policy model. It builds the frozen factorial fixture, constructs four deterministic
probe responses per public row, runs the same local reward function, and mirrors
outcome-GRPO scalar normalization.

| Metric | Result |
|---|---:|
| Source cases | 16 synthetic factorial cases |
| Independent base tasks | 4 |
| Augmented public rows | 32 |
| Split rows | train 16 / dev 8 / test 8 |
| Deterministic probe responses | 128 |
| Four-response groups | 32 |
| Structural checks | `10/10 PASS` |
| Zero-variance groups | `0/32` |
| Reward range | `[-8, 5]` |
| Actual optimizer steps | `0` |

The checks cover finite rewards, four-member grouping, invalid-output fail-closed
behavior, oracle dominance, semantic exactness, content-hash invariance, public
leakage rejection, scorer hash binding, base-task split isolation, and non-zero
within-group variance.

## Handoff verification

The clean handoff whitelist contains the GRPO package plus only the source chain
needed to regenerate `factorial_cases_v1.json`.

| Check | Result |
|---|---:|
| Handoff unit tests | `57 GRPO + 8 fixture = 65 passed` |
| Shell syntax (`bash -n`) | PASS |
| Python compilation | PASS |
| Black formatting | PASS |
| Secret-pattern scan | no credential value found |
| Remote/base identity | GitHub `main` and local base both `34530469...` before handoff |

These are local CPU/source checks. The local Mac environment does not contain the
CUDA PyTorch, Ray, vLLM, Transformers, PEFT, Datasets, or PyArrow stack required
for a real launch. No claim is made that the target server environment is already
compatible.

## Implemented

1. Strict `RETRIEVE/NO_RETRIEVE` three-field JSON action parsing with no repair.
2. Split-before-augmentation data construction and public/sealed label separation.
3. A local deterministic VERL `compute_score` with hash-bound scorer sidecar.
4. Qwen2.5-0.5B-Instruct + LoRA + FSDP + vLLM `n=4` stock-VERL launcher.
5. Linux/CUDA/NCCL/flash-attn/version/data/hash/geometry preflight, including the
   Dockerfile-pinned vLLM runtime, bf16 FSDP precision, exact actor tokenizer,
   zero-filter/full-batch prompt gate, and exact Hydra composition.
6. Exact Hydra composition before launch and exclusive metadata binding the Git
   commit, launcher, full override argv, resolved parameters, local model snapshot,
   and resumed actor hash.
7. Step-0/checkpoint held-out recomputation and a two-seed aggregate gate, with
   each validation JSONL bound to its own launcher metadata and exact resume path;
   the aggregate replays the seed analyzers from referenced raw artifacts.

## Not yet verified or implemented

- No Linux/NVIDIA preflight, vLLM generation, backward pass, optimizer update,
  checkpoint write, loss curve, or held-out model improvement has been observed.
- The current reward is a synthetic gold-subset proxy, not an empirical local
  `Q_t`, Answerer-conditioned PEV, or environment terminal reward.
- There is no online Memory Bank transition, write/delete/consolidation policy, or
  multi-step environment trajectory.
- Four base tasks are insufficient for natural-distribution or MemoryArena claims.
- No SFT/reranker/frozen-selector comparison establishes that GRPO is necessary.
- Run metadata is written before execution. The analyzer binds files to recorded
  locations and recomputes their scores/hashes, but without trusted execution or a
  separately signed post-run manifest it cannot prove that an adversarial editor
  never replaced a same-path validation file before audit.

## Decision gates

1. **G0:** Build JSONL+Parquet on the target Linux/NVIDIA environment, obtain
   `PREFLIGHT PASS`, and inspect a successful `DRY_RUN` before training.
2. **G1:** Complete a fresh 10-step smoke without OOM/NaN, with non-degenerate
   reward/advantage and a writable checkpoint. This is engineering evidence only.
3. **Evaluator/data repair:** freeze a credible verifier or empirical subset oracle
   on a new task-disjoint cohort before treating the reward as method evidence.
4. **G2:** Only after the previous gates, run two preregistered 100-step seeds and
   apply the held-out aggregate. Even a G2 PASS supports only bounded learnability
   on this synthetic proxy.
