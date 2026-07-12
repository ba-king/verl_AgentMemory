#!/usr/bin/env bash
set -euo pipefail

: "${MEMORYARENA_DIR:?Set MEMORYARENA_DIR to the official MemoryArena checkout}"
: "${P2R_MEMORY_API_KEY:?Set P2R_MEMORY_API_KEY for both answer and memory agents}"

VERL_DIR="${VERL_DIR:-/Users/lizizheng/verl}"
PYTHON_BIN="${MEMORYARENA_PYTHON:-$MEMORYARENA_DIR/.venv/bin/python}"
SOURCE_TASK="${MEMORYARENA_SOURCE_TASK:-$MEMORYARENA_DIR/hf_task_cache/baking/item_0.json}"
PILOT_STEP="${MEMORYARENA_PILOT_STEP:-1}"

if ! [[ "$PILOT_STEP" =~ ^[1-9][0-9]*$ ]]; then
  printf 'MEMORYARENA_PILOT_STEP must be a positive integer, got: %s\n' "$PILOT_STEP" >&2
  exit 2
fi

SOURCE_TASK_ID="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["task_id"])' "$SOURCE_TASK")"
RUN_ID="${MEMORYARENA_RUN_ID:-official_${SOURCE_TASK_ID}_product${PILOT_STEP}_pilot_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${MEMORYARENA_RUN_ROOT:-$VERL_DIR/examples/csl_memory_agent/official_runs/pilots/$SOURCE_TASK_ID/product$PILOT_STEP/$RUN_ID}"
PILOT_TASK="$RUN_ROOT/product${PILOT_STEP}_pilot.json"
TASK_SCOPE="product${PILOT_STEP}_pilot"

mkdir -p "$RUN_ROOT"
test -x "$PYTHON_BIN"
test -f "$SOURCE_TASK"

PYTHONPATH="$MEMORYARENA_DIR:$MEMORYARENA_DIR/memory:${PYTHONPATH:-}" \
SOURCE_TASK="$SOURCE_TASK" PILOT_TASK="$PILOT_TASK" PILOT_STEP="$PILOT_STEP" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from env.env_systems.web_shopping_env.runtime.runner import (
    build_instruction_for_step,
    build_single_step_task,
    split_agent_instruction,
)

source_path = Path(os.environ["SOURCE_TASK"])
source = json.loads(source_path.read_text(encoding="utf-8"))
pilot_step = int(os.environ["PILOT_STEP"])
steps = source.get("steps") or []
if pilot_step > len(steps):
    raise SystemExit(
        f"MEMORYARENA_PILOT_STEP={pilot_step} exceeds source task's {len(steps)} steps"
    )

prefix, sections = split_agent_instruction(source["agent_instruction"])
instruction = build_instruction_for_step(
    prefix,
    sections,
    pilot_step,
    {},
    include_history=False,
)
pilot = build_single_step_task(
    source,
    pilot_step - 1,
    instruction,
    total_steps=len(steps),
)
source_task_id = source["task_id"]
pilot["task_id"] = f"{source_task_id}_product{pilot_step}_pilot"
# The pilot contains one official shopping episode, so the official scorer sees
# this source step as episode step 1. Source provenance remains in metadata.
pilot["steps"][0]["step"] = 1
pilot["metadata"] = {
    **(pilot.get("metadata") or {}),
    "pilot_source_file": str(source_path),
    "pilot_source_step": pilot_step,
    "score_scope": "single_official_session_functional_pilot",
    "history_seeded": False,
}
Path(os.environ["PILOT_TASK"]).write_text(
    json.dumps(pilot, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY

MEMORYARENA_TASK_FILE="$PILOT_TASK" \
MEMORYARENA_TASK_SCOPE="$TASK_SCOPE" \
MEMORYARENA_SPLIT_STEPS="false" \
MEMORYARENA_RUN_ID="$RUN_ID" \
MEMORYARENA_RUN_ROOT="$RUN_ROOT" \
MEMORYARENA_REPORT_NAME="memoryarena_product${PILOT_STEP}_pilot_report" \
MEMORYARENA_REPORT_TITLE="Official MemoryArena Product ${PILOT_STEP} Functional Pilot" \
MEMORYARENA_REPORT_SCOPE="single_official_memoryarena_step_functional_pilot" \
MEMORYARENA_SCOPE_NOTE="This is one official WebShop session used as a functional pilot, not a complete six-step item or the paper's 150-item aggregate." \
bash "$VERL_DIR/examples/csl_memory_agent/scripts/run_official_memoryarena_task0.sh"
