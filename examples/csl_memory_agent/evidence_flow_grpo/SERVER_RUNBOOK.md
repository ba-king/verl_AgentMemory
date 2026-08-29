# Evidence-Flow GRPO server runbook

This handoff runs the retrieval/selection policy with stock VERL on a
Linux/NVIDIA server. It is deliberately separate from the DeepSeek Answerer:
the actor defaults to `Qwen/Qwen2.5-0.5B-Instruct`, reward computation is local
and deterministic, and no API key is read by any launcher component.

The code was reviewed against VERL commit
`34530469d482fc322f985434dbbc9ca28fe06928`. Keep that repository baseline for
the first smoke. The local Mac verification environment does not contain the
CUDA PyTorch/Ray/vLLM stack, so dependency compatibility remains a target-host
G0 check; this handoff does not claim a tested or pinned GPU container image.

## Frozen topology

```text
train.parquet + dev.parquet/test.parquet
          │
          ▼
Qwen2.5-0.5B actor ── vLLM, n=4 ──► strict short JSON candidates
          │                                  │
          │                                  ▼
          └──── FSDP + LoRA ◄──── reward.py:compute_score
                                             │
                                             ▼
                              scorer/targets.json (local sidecar)
```

The response contract is one JSON object only:

```json
{"action":"RETRIEVE","evidence_status":"SUFFICIENT","selected_memory_ids":["M1","M2"]}
```

or:

```json
{"action":"NO_RETRIEVE","evidence_status":"INSUFFICIENT","selected_memory_ids":[]}
```

`NO_RETRIEVE` may instead use `NOT_NEEDED` when memory is unnecessary. The
prompt and reward reject prose, repairs, unknown IDs, duplicate IDs, and
selections longer than three. DeepSeek may later be connected as a frozen
Answerer for downstream evaluation; it is never the actor, reference policy,
or reward model in this run.

## Server requirements

- Linux host with NVIDIA driver and `nvidia-smi`.
- At least one visible CUDA GPU with 16 GiB free; 24 GiB is the practical
  minimum and 48 GiB gives more rollout headroom.
- Python 3.10 or newer in an environment compatible with this VERL checkout.
- Working imports for PyTorch CUDA/NCCL, Ray, vLLM, Transformers, PEFT,
  Accelerate, Datasets, PyArrow, TensorDict, TorchData, NumPy, Hydra, OmegaConf,
  flash-attn, and VERL, under the pinned runtime contract described below.
- The repository must be synced as a whole. Do not copy only this directory,
  because the launcher imports the local `verl` package.

The handoff does not install packages. For the reviewed commit, the runtime
anchor is `docker/Dockerfile.stable.vllm`, which declares `vllm==0.20.2`; the
checkout's GPU CI likewise uses a `vllm020` image. Its `setup.py` vLLM extra is
stale (`vllm<=0.12.0`) and must **not** be used as the vLLM authority. The
preflight deliberately enforces the Dockerfile version, enforces the remaining
relevant `setup.py` bounds (including `numpy<2` and TensorDict), and records the
known metadata conflict in its report. Prefer building/using that official
Dockerfile, then install this checkout itself with `pip install --no-deps -e .`;
do not run `pip install -e .[vllm]` for this handoff. No container digest or GPU
execution is claimed as locally verified.

Every real launch writes `OUTPUT_DIR/run_metadata.json` before the VERL process
starts. The file records the exact launcher seed, full Hydra override argv,
resolved parameters, repository commit, and hashes of the launcher,
config/data/scorer/manifest/reward artifacts. A local base-model snapshot is
content-hashed; checkpoint evaluation also hashes the resumed actor weights.
The metadata is exclusive and cannot be replaced by rerunning the same output
directory.

## Required artifacts

By default the launcher expects:

```text
examples/csl_memory_agent/evidence_flow_grpo/
├── reward.py
├── data/factorial_v1/
│   ├── public/
│   │   ├── train.parquet
│   │   ├── train.jsonl
│   │   ├── dev.parquet
│   │   ├── dev.jsonl
│   │   ├── test.parquet
│   │   └── test.jsonl
│   ├── scorer/targets.json
│   └── split_manifest.json
├── configs/retrieval_grpo_v0.yaml
├── server_preflight.py
└── run_qwen25_05b_grpo_fsdp.sh
```

Each Parquet file must be non-empty and contain `data_source`, `prompt`,
`reward_model`, and `extra_info`. The sidecar must be non-empty valid JSON.
The matching JSONL files are immutable decoded copies; `test.jsonl` is required
by the sealed G2 audit after validation. The manifest hashes both representations.
`reward.py` must expose this VERL-compatible interface and accept the injected
sidecar path:

```python
def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    scorer_path=None,
): ...
```

The preflight rejects common network-client imports in `reward.py`; reward-time
HTTP/API calls are outside this experiment contract.

Build the immutable split bundle on the server (or build it locally and sync
the whole output directory). The builder refuses overwrites by default:

```bash
export DATA_DIR="$PWD/examples/csl_memory_agent/evidence_flow_grpo/data/factorial_v1"
python3 -m examples.csl_memory_agent.evidence_flow_grpo.build_dataset \
  --output "$DATA_DIR" \
  --parquet
```

Use a new output directory for a revised fixture or split. Do not use
`--overwrite` to turn a changed dataset into the same experiment identity.

## 1. Enter the prepared server environment

```bash
cd /absolute/path/to/verl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN=python3
"$PYTHON_BIN" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
```

Expected: CUDA is `True`, NCCL is present, and the assigned GPU is mostly free.

If the model is already downloaded, use its absolute directory:

```bash
export MODEL_PATH=/absolute/path/to/Qwen2.5-0.5B-Instruct
```

Otherwise the default Hugging Face identifier is accepted for G0/G1 and its
config/tokenizer availability is checked during preflight. A local snapshot is
required for G2 so all three runs bind exactly the same base weights.

## 2. Set paths and resources

```bash
export DATA_DIR=/absolute/path/to/evidence_flow_grpo_data/factorial_v1
export TRAIN_FILE="$DATA_DIR/public/train.parquet"
export DEV_FILE="$DATA_DIR/public/dev.parquet"
export VAL_FILE="$DEV_FILE"
export TEST_FILE="$DATA_DIR/public/test.parquet"
export SCORER_PATH="$DATA_DIR/scorer/targets.json"
export MANIFEST_PATH="$DATA_DIR/split_manifest.json"
export NGPUS=1
export ROLLOUT_TP=1
```

The default single-GPU probe uses LoRA rank 16, rollout group size 4, prompt
length 1280, response length 128, 16 prompts per training step, and at most 100
update steps. The frozen train split has one full batch per epoch, so the
default `TOTAL_EPOCHS=100` is what makes the 100-step cap reachable. The
preflight applies the actor's real chat template/tokenizer and rejects a run if
any frozen prompt would be filtered or if the post-tokenization rows cannot reach
the requested step bound. The 1280-token default was chosen after measuring the
committed fixture with the Qwen2.5 tokenizer (maximum 1149 tokens in the reviewed
environment); the target-host preflight is authoritative.

`TEST_FILE` is checked, hashed, and retained as a held-out split. It is not
passed to VERL as `data.val_files` and must not be used for tuning.

| Variable | Default | Purpose |
|---|---:|---|
| `MODEL_PATH` | `Qwen/Qwen2.5-0.5B-Instruct` | Actor model/snapshot |
| `PYTHON_BIN` | `python3` | Prepared Python >=3.10 with CUDA/VERL dependencies |
| `NGPUS` | `1` | GPUs per node |
| `NNODES` | `1` | Training nodes |
| `ROLLOUT_TP` | `1` | vLLM tensor parallel size |
| `ROLLOUT_N` | `4` | GRPO samples per prompt |
| `DTYPE` | `bfloat16` | Frozen reviewed FSDP/vLLM path; other values are rejected |
| `TRAIN_BATCH_SIZE` | `16` | Prompts per training iteration |
| `PPO_MINI_BATCH_SIZE` | `16` | Global PPO mini-batch |
| `MAX_PROMPT_LENGTH` | `1280` | Must retain every frozen prompt in preflight |
| `MAX_RESPONSE_LENGTH` | `128` | Short JSON token cap |
| `LORA_RANK` | `16` | Set `0` for full-parameter training |
| `TOTAL_TRAINING_STEPS` | `100` | Probe update limit |
| `TOTAL_EPOCHS` | `100` | Dataloader passes; must make the step limit reachable |
| `SEED` | `20260824` | Data sampler plus actor/reference FSDP seed |
| `VAL_ONLY` | `0` | Set `1` for step-0 or checkpoint-only validation |
| `RESUME_MODE` | `disable` | Use `resume_path` only with a frozen checkpoint path |
| `RESUME_FROM_PATH` | empty | `.../global_step_N` directory for checkpoint evaluation |
| `OUTPUT_DIR` | `runs/<timestamped-name>` | Logs/checkpoints/rollouts |

For `NNODES>1`, configure the Ray cluster first and export `RAY_ADDRESS`; the
preflight refuses an unconfigured multi-node launch.

## 3. Run the fail-closed preflight

```bash
PREFLIGHT_ONLY=1 bash \
  examples/csl_memory_agent/evidence_flow_grpo/run_qwen25_05b_grpo_fsdp.sh
```

It checks all of the following without starting training:

- Linux, CUDA, NCCL, GPU count, bf16 support, and free memory;
- `nvidia-smi` and all required imports;
- flash-attn, the Dockerfile-pinned `vllm==0.20.2` contract, and the remaining
  relevant `setup.py` bounds; the stale setup.py vLLM declaration is reported;
- actor config and chat tokenizer with `trust_remote_code=false`;
- stock-style token length for every train/dev/test prompt, with zero filtering;
- VERL repository resolution and frozen config invariants;
- train/validation Parquet schemas and row counts;
- scorer JSON readability and its actual SHA-256 against every public row;
- split-manifest self-hash and train/dev/test/scorer artifact hashes;
- local-only reward imports, function name, and signature;
- batch divisibility and rollout tensor-parallel geometry;
- actor identity, including an explicit rejection of any DeepSeek actor.

The preflight also composes and resolves the exact Hydra job before reporting
success. Do not bypass a failed check. Fix the environment or artifact and
rerun the same command.

## 4. Inspect the exact launch command

`DRY_RUN=1` repeats the full preflight and exact Hydra composition, then prints
the shell-escaped VERL command without launching it:

```bash
DRY_RUN=1 bash \
  examples/csl_memory_agent/evidence_flow_grpo/run_qwen25_05b_grpo_fsdp.sh
```

The reviewed defaults live in
`configs/retrieval_grpo_v0.yaml`; the launcher supplies absolute paths and
resource overrides explicitly to stock Hydra. If an invariant default changes,
update and review both files together before assigning GPU time.

## 5. Launch the bounded probe

Use a persistent terminal on the server. The launcher already writes a log,
rollouts, validation outputs, and checkpoints under `OUTPUT_DIR`.

```bash
export EXPERIMENT_NAME=retrieval_grpo_v0_qwen25_05b_seed20260824
export OUTPUT_DIR=/absolute/path/to/runs/$EXPERIMENT_NAME
bash examples/csl_memory_agent/evidence_flow_grpo/run_qwen25_05b_grpo_fsdp.sh
```

For a first G1 training smoke, override the bound rather than editing files:

```bash
TOTAL_TRAINING_STEPS=10 SAVE_FREQ=10 TEST_FREQ=5 \
  bash examples/csl_memory_agent/evidence_flow_grpo/run_qwen25_05b_grpo_fsdp.sh
```

Positional Hydra overrides are forbidden because a trailing duplicate key can
bypass preflight after it has checked the actor, data, reward, or VAL_ONLY mode.
Use only the reviewed environment variables in this table. A new parameter
requires a code/config review, not an ad-hoc command suffix.

## 6. Run the frozen held-out comparison

Do not tune on `test.parquet`. First record a step-0 baseline under its own run
directory. `VAL_ONLY=1` initializes the actor and performs validation but exits
before any optimizer step:

```bash
export VAL_FILE="$DATA_DIR/public/test.parquet"
export EXPERIMENT_NAME=retrieval_grpo_v0_step0_test_seed20260824
export OUTPUT_DIR=/absolute/path/to/runs/$EXPERIMENT_NAME
VAL_ONLY=1 bash \
  examples/csl_memory_agent/evidence_flow_grpo/run_qwen25_05b_grpo_fsdp.sh
```

After the frozen training run, evaluate the saved checkpoint on exactly the
same file and sampling settings. Point `RESUME_FROM_PATH` at the complete
`global_step_N` directory, not its `actor/` child:

```bash
export VAL_FILE="$DATA_DIR/public/test.parquet"
export RESUME_MODE=resume_path
export RESUME_FROM_PATH=/absolute/path/to/training-run/checkpoints/global_step_100
export EXPERIMENT_NAME=retrieval_grpo_v0_step100_test_seed20260824
export OUTPUT_DIR=/absolute/path/to/runs/$EXPERIMENT_NAME
VAL_ONLY=1 bash \
  examples/csl_memory_agent/evidence_flow_grpo/run_qwen25_05b_grpo_fsdp.sh
```

`RESUME_FROM_PATH` must be absolute, end exactly in `global_step_N`, and contain
the model, optimizer, and extra-state FSDP shard for every expected actor rank.
Keep a fresh `OUTPUT_DIR` for every training or VAL_ONLY run; the launcher
refuses every non-empty run directory so logs, checkpoints, rollouts, and
validation evidence cannot be silently replaced.

Preserve the two validation JSONL artifacts and logs. Use `semantic_exact` as
the primary metric; also compare `content_selection_exact`, schema validity,
action/status accuracy, precision,
recall, and unsafe false retrieval. Repeat the complete baseline/train/test
sequence with the second preregistered seed; do not select the better seed.

Run the sealed offline audit for each seed. It recomputes every logged output
with the same sidecar and refuses missing/duplicate IDs or a logged-score
mismatch. It also evaluates `empty`, `first_retrieve`, `all_up_to_3`, and
`oracle` on the same test rows. The analyzer requires the immutable metadata
from all three runs and binds each validation JSONL to its launcher-recorded
`OUTPUT_DIR`, model, seed, test/scorer hashes, and (for checkpoint validation)
exact resume path. It writes the diagnostic JSON but exits nonzero when that
seed is not eligible:

Before the step-0/training/checkpoint sequence, set `MODEL_PATH` to one immutable
local snapshot directory. G2 intentionally rejects an unpinned remote model ID.
Keep all resolved training variables, including save/test frequencies, identical
across the three runs except for `VAL_ONLY`, `VAL_FILE`, output identity, and the
checkpoint resume fields.

```bash
python3 -m examples.csl_memory_agent.evidence_flow_grpo.analyze_g2 \
  --seed 20260824 \
  --test-jsonl "$DATA_DIR/public/test.jsonl" \
  --scorer-path "$DATA_DIR/scorer/targets.json" \
  --step0-validation /absolute/path/to/step0-run/validation \
  --checkpoint-validation /absolute/path/to/step100-run/validation \
  --checkpoint-dir /absolute/path/to/training-run/checkpoints/global_step_100 \
  --training-run-metadata /absolute/path/to/training-run/run_metadata.json \
  --step0-run-metadata /absolute/path/to/step0-run/run_metadata.json \
  --checkpoint-val-run-metadata /absolute/path/to/step100-run/run_metadata.json \
  --expected-checkpoint-step 100 \
  --output /absolute/path/to/g2_seed_20260824.json
```

After the second preregistered seed has its own result, apply the two-seed
gate. The command exits nonzero on a failed gate:

```bash
python3 -m examples.csl_memory_agent.evidence_flow_grpo.aggregate_g2 \
  --seed-result /absolute/path/to/g2_seed_20260824.json \
  --seed-result /absolute/path/to/g2_seed_20260825.json \
  --expected-checkpoint-step 100 \
  --output /absolute/path/to/g2_two_seed_result.json
```

The aggregate requires two distinct seeds, positive improvement in both,
mean held-out `semantic_exact` improvement of at least `0.10`, no validity or
unsafe-retrieval regression, performance above every non-oracle trivial
baseline, both checkpoint artifacts reporting the frozen step 100, and two
different launcher-recorded training identities and actor-checkpoint content
hashes. Relabelling one real training seed, reusing/copying one checkpoint, a
10-step smoke, or a mixed-step pair therefore fails. A passing JSON still has
`evaluation_type=held_out_synthetic_proxy`.

## 7. Monitor and stop conditions

In a second server shell:

```bash
nvidia-smi
tail -f "$OUTPUT_DIR/train.log"
```

Stop instead of blindly restarting if any of these occur:

- CUDA OOM, NCCL/Ray worker loss, NaN/Inf loss or reward;
- reward loader or scorer-path mismatch;
- malformed-output rate does not move off its initial value;
- all four completions receive identical rewards repeatedly;
- unexpected Answerer/API traffic;
- validation safety regresses while training reward increases.

Preserve the log and generated rollouts before changing configuration. A retry
with changed parameters is a new run ID, not a continuation of the failed run.

## Acceptance boundary

Pipeline success means only that data → rollout → deterministic reward → GRPO
update → checkpoint/evaluation executes reproducibly. Method learnability
requires held-out improvement over the frozen prompt and supervised/non-RL
baselines while retaining schema validity, no-required safety, precision,
recall, and registered order invariance. This server smoke alone does not prove
that GRPO is necessary or that the method generalizes.
