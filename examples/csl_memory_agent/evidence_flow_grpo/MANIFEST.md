# Evidence-Flow GRPO Git handoff manifest

- Handoff date: `2026-08-28`
- Reviewed VERL base: `34530469d482fc322f985434dbbc9ca28fe06928`
- Scope: teacher-facing selector-only GRPO engineering probe
- Integrity unit: the final Git commit plus the dataset/scorer hashes generated at run time

## Included GRPO package files

```text
examples/csl_memory_agent/evidence_flow_grpo/
├── .gitignore
├── README.md
├── TEACHER_HANDOFF.md
├── SERVER_RUNBOOK.md
├── EXPERIMENT_RESULTS.md
├── EXPERIMENT_AUDIT.md
├── EXPERIMENT_AUDIT.json
├── MANIFEST.md
├── __init__.py
├── contracts.py
├── dataset.py
├── build_dataset.py
├── reward.py
├── grpo.py
├── sanity.py
├── server_preflight.py
├── run_metadata.py
├── analyze_g2.py
├── aggregate_g2.py
├── run_qwen25_05b_grpo_fsdp.sh
├── configs/
│   └── retrieval_grpo_v0.yaml
├── tests/
│   ├── __init__.py
│   ├── test_contracts_reward.py
│   ├── test_dataset.py
│   ├── test_g2_analysis.py
│   ├── test_grpo.py
│   ├── test_sanity.py
│   └── test_server_handoff.py
└── runs/structural_sanity_v1/
    ├── structural_sanity.json
    └── structural_checkpoint.json
```

The generated `runs/structural_sanity_v1/dataset/` directory is deliberately not
committed. Its JSON/JSONL content is deterministically reconstructed by the checked-in
builder and fixture; provenance uses a repository-relative source label rather than a
checkout-local absolute path.

## Included fixture provenance chain

Only these files are included from the earlier Stage -1 workspace:

```text
examples/csl_memory_agent/evidence_flow_stage_minus1/
├── __init__.py
├── schema.py
├── build_factorial_fixture.py
├── factorial_spec_v1.json
├── fixtures/
│   └── factorial_cases_v1.json
└── tests/
    └── test_factorial_fixture.py
```

The GRPO dataset builder reads `fixtures/factorial_cases_v1.json`. The remaining
files make that fixture auditable and deterministically reproducible from its small
human-readable specification.

## Explicit exclusions

- all DeepSeek/API request and response trajectories;
- natural selector-necessity r1-r4 runs and sealed natural targets;
- controlled/prompt-rescue historical runs and reports;
- internal refine/reviewer logs and `.aris` traces;
- `structural_sanity_v0` and duplicated generated v1 dataset files;
- generated Parquet, checkpoints, rollouts, validation dumps, and train logs;
- Python caches, locks, and unrelated MemoryArena experiments or scripts.

These exclusions do not affect the dependency closure of this restricted pipeline.
The package is not standalone: it must remain inside the full VERL checkout because
the launcher imports stock `verl.trainer.main_ppo` and its generated Hydra config.

## Handoff checks

```text
57 GRPO package tests + 8 fixture provenance tests = 65 passed
bash -n: PASS
compileall: PASS
Black: PASS
credential-value scan: PASS
```

These checks cover source/CPU structure only. They do not substitute for Linux/CUDA
G0 or a real GPU optimizer step.
