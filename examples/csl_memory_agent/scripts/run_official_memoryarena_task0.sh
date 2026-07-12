#!/usr/bin/env bash
set -euo pipefail

: "${MEMORYARENA_DIR:?Set MEMORYARENA_DIR to the official MemoryArena checkout}"
: "${P2R_MEMORY_API_KEY:?Set P2R_MEMORY_API_KEY for both answer and memory agents}"

VERL_DIR="${VERL_DIR:-/Users/lizizheng/verl}"
PYTHON_BIN="${MEMORYARENA_PYTHON:-$MEMORYARENA_DIR/.venv/bin/python}"
MODEL="${P2R_MEMORY_MODEL:-deepseek-v4-flash}"
BASE_URL="${P2R_MEMORY_BASE_URL:-https://api.deepseek.com}"
TASK_FILE="${MEMORYARENA_TASK_FILE:-$MEMORYARENA_DIR/hf_task_cache/baking/item_0.json}"
TASK_SCOPE="${MEMORYARENA_TASK_SCOPE:-$(basename "$TASK_FILE" .json)}"
SPLIT_STEPS="${MEMORYARENA_SPLIT_STEPS:-true}"
if [ "$SPLIT_STEPS" = "true" ]; then
  RUN_MODE="split"
else
  RUN_MODE="full"
fi
DATA_ROOT="${MEMORYARENA_DATA_ROOT:-$MEMORYARENA_DIR/data/shopping}"
RUN_ID="${MEMORYARENA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${MEMORYARENA_RUN_ROOT:-$VERL_DIR/examples/csl_memory_agent/official_runs/task0/$RUN_ID}"
OUTPUT_ROOT="${MEMORYARENA_OUTPUT_ROOT:-$RUN_ROOT/raw}"
STATE_ROOT="${CSL_STATE_DIR:-$RUN_ROOT/state}"
USAGE_LOG="$OUTPUT_ROOT/answer_api_usage.jsonl"
CONFIG_PATH="$OUTPUT_ROOT/task0_config.json"
RUN_DIR="$OUTPUT_ROOT/$TASK_SCOPE/${MODEL}-${RUN_MODE}-dart_csl_vtcc"
REPORT_NAME="${MEMORYARENA_REPORT_NAME:-memoryarena_task0_report}"
REPORT_TITLE="${MEMORYARENA_REPORT_TITLE:-Official MemoryArena Shopping Task 0 Report}"
REPORT_SCOPE="${MEMORYARENA_REPORT_SCOPE:-single_official_memoryarena_item}"
SCOPE_NOTE="${MEMORYARENA_SCOPE_NOTE:-This is one complete official item, not the 150-item aggregate reported in the paper.}"
MAX_ROUNDS="${MEMORYARENA_MAX_ROUNDS:-25}"
ANSWER_MAX_TOKENS="${P2R_ANSWER_MAX_TOKENS:-1024}"

mkdir -p "$OUTPUT_ROOT" "$STATE_ROOT"
test -x "$PYTHON_BIN"
test -f "$TASK_FILE"
test -f "$DATA_ROOT/items_shuffle.json"
test -d "$DATA_ROOT/search_engine/indexes-full"
test -d "$DATA_ROOT/product_catalog"

export PYTHONPATH="$VERL_DIR:$MEMORYARENA_DIR:$MEMORYARENA_DIR/memory:${PYTHONPATH:-}"
export OPENAI_API_KEY="$P2R_MEMORY_API_KEY"
export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_API_BASE="$BASE_URL"
export CSL_MEMORY_PROVIDER=openai
export CSL_MEMORY_MODEL="$MODEL"
export CSL_MEMORY_API_KEY="$P2R_MEMORY_API_KEY"
export CSL_MEMORY_BASE_URL="$BASE_URL"
export CSL_STATE_DIR="$STATE_ROOT"
export OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-3}"
# Local MemoryArena services must bypass desktop HTTP proxies. Otherwise a long
# /create request can be terminated by the proxy before WebShop is ready.
export NO_PROXY="127.0.0.1,localhost,0.0.0.0${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
# The upstream seed goal is replaced by MemoryArena's bundled task prompt. Build
# only that one seed goal instead of materializing goals for all 1.18M products.
export MEMORYARENA_FAST_LITE_GOALS=1

"$VERL_DIR/.venv/bin/python" \
  "$VERL_DIR/examples/csl_memory_agent/scripts/register_memoryarena.py" \
  --memoryarena-dir "$MEMORYARENA_DIR"

MEMORYARENA_DIR="$MEMORYARENA_DIR" TASK_FILE="$TASK_FILE" TASK_SCOPE="$TASK_SCOPE" SPLIT_STEPS="$SPLIT_STEPS" DATA_ROOT="$DATA_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT" CONFIG_PATH="$CONFIG_PATH" MODEL="$MODEL" BASE_URL="$BASE_URL" \
MAX_ROUNDS="$MAX_ROUNDS" ANSWER_MAX_TOKENS="$ANSWER_MAX_TOKENS" \
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

config = {
    "task_name": os.environ["TASK_SCOPE"],
    "agent": {
        "model_name": os.environ["MODEL"],
        "temperature": 0.0,
        "max_tokens": int(os.environ["ANSWER_MAX_TOKENS"]),
        "backend": "openai",
        "api_key": "",
        "base_url": os.environ["BASE_URL"],
    },
    "memory": {
        "use_step_memory": True,
        "memory_system_name": "dart_csl_vtcc",
        "server_url": "http://127.0.0.1:8000",
        "timeout": 600,
    },
    "env": {
        "env_name": "webshop",
        "env_server_url": "http://127.0.0.1:8001",
        "timeout": 3000,
        "env_config": {
            "upstream_env_server_base": "http://127.0.0.1:36004",
            "bootstrap_upstream_env": True,
            "restart_upstream_env": False,
            "upstream_launch_module": "env.env_systems.web_shopping_env.runtime.service.launch_lite",
            "upstream_webshop_data_root": os.environ["DATA_ROOT"],
            "product_catalog_dir": str(Path(os.environ["DATA_ROOT"]) / "product_catalog"),
            "domain_data_path": str(Path(os.environ["DATA_ROOT"]) / "domain_data.json"),
            "upstream_limit_goals": 1,
            "upstream_ready_timeout": 6000,
            "reuse_env": True,
            "env_id_cache_file": ".env_id_cache_{port}_memoryarena_shopping.json",
            "clear_env_id_cache_on_restart": True,
            "clear_python_cache_on_restart": False,
            "upstream_log_file": "logs/upstream_webshop_{port}.log",
            "action_format": "react",
            "enable_feedback": False,
        },
    },
    "task_specific": {
        "task_file": os.environ["TASK_FILE"],
        "max_steps": int(os.environ["MAX_ROUNDS"]),
        "split_steps": os.environ["SPLIT_STEPS"].lower() == "true",
        "resume": False,
        "include_history": False,
    },
    "output": {
        "output_dir": os.environ["OUTPUT_ROOT"],
        "save_trajectories": True,
        "save_metrics": True,
        "save_interactions": True,
    },
}
Path(os.environ["CONFIG_PATH"]).write_text(json.dumps(config, indent=2), encoding="utf-8")
PY

rm -f "$USAGE_LOG"

"$PYTHON_BIN" "$VERL_DIR/examples/csl_memory_agent/scripts/run_csl_memory_server.py" \
  > "$OUTPUT_ROOT/memory_server.log" 2>&1 &
MEMORY_PID=$!

cd "$MEMORYARENA_DIR"
"$PYTHON_BIN" env/env_server.py > "$OUTPUT_ROOT/env_server.log" 2>&1 &
ENV_PID=$!

cleanup() {
  kill "$MEMORY_PID" "$ENV_PID" 2>/dev/null || true
  UPSTREAM_PID=$(lsof -tiTCP:36004 -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$UPSTREAM_PID" ]; then
    kill "$UPSTREAM_PID" 2>/dev/null || true
  fi
  if [ -n "${SCORE_STAGE:-}" ] && [ -d "$SCORE_STAGE" ]; then
    rm -rf "$SCORE_STAGE"
  fi
}
trap cleanup EXIT

for url in http://127.0.0.1:8000/docs http://127.0.0.1:8001/docs; do
  for _ in $(seq 1 60); do
    if curl -fsS "$url" >/dev/null; then
      break
    fi
    sleep 1
  done
  curl -fsS "$url" >/dev/null
done

"$PYTHON_BIN" "$VERL_DIR/examples/csl_memory_agent/scripts/run_memoryarena_shopping_with_usage.py" \
  --memoryarena-dir "$MEMORYARENA_DIR" \
  --usage-log "$USAGE_LOG" \
  --config "$CONFIG_PATH"

SCORE_STAGE=$(mktemp -d "$MEMORYARENA_DIR/.csl_score_run.XXXXXX")
SCORE_STAGE=$(cd "$SCORE_STAGE" && pwd -P)
cp -R "$RUN_DIR" "$SCORE_STAGE/run"

"$PYTHON_BIN" env/env_systems/web_shopping_env/compute_reward.py \
  --run-dir "$SCORE_STAGE/run" \
  --llm-model "$MODEL" \
  --force
cp "$SCORE_STAGE/run/reward_report.json" "$RUN_DIR/reward_report.json"

TASK_RESULT=$(find "$RUN_DIR/step_results" -name '*_results_*.json' -type f -print | sort | tail -n 1)
"$VERL_DIR/.venv/bin/python" \
  "$VERL_DIR/examples/csl_memory_agent/scripts/summarize_memoryarena_shopping_run.py" \
  --task-result "$TASK_RESULT" \
  --reward-report "$RUN_DIR/reward_report.json" \
  --answer-usage-log "$USAGE_LOG" \
  --memory-state-root "$STATE_ROOT" \
  --output-dir "$RUN_ROOT" \
  --report-name "$REPORT_NAME" \
  --report-title "$REPORT_TITLE" \
  --report-scope "$REPORT_SCOPE" \
  --scope-note "$SCOPE_NOTE"

printf 'MemoryArena task-0 run completed: %s\n' "$RUN_ROOT"
