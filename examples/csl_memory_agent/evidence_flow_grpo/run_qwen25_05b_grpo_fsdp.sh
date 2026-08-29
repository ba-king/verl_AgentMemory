#!/usr/bin/env bash
# Stock VERL GRPO handoff for Linux/NVIDIA only. Invoke with `bash`, so the
# repository does not depend on executable-bit preservation during transfer.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"

if [[ ! -f "${REPO_ROOT}/verl/version/version" ]]; then
    echo "Cannot resolve VERL repository root from ${SCRIPT_DIR}" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/configs/retrieval_grpo_v0.yaml}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data/factorial_v1}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/public/train.parquet}"
DEV_FILE="${DEV_FILE:-${DATA_DIR}/public/dev.parquet}"
VAL_FILE="${VAL_FILE:-${DEV_FILE}}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/public/test.parquet}"
MANIFEST_PATH="${MANIFEST_PATH:-${DATA_DIR}/split_manifest.json}"
REWARD_PATH="${REWARD_PATH:-${SCRIPT_DIR}/reward.py}"
SCORER_PATH="${SCORER_PATH:-${DATA_DIR}/scorer/targets.json}"

# The trainable actor is deliberately small and local/open-weight. DeepSeek is
# reserved for a frozen future Answerer interface and is never used here.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
NNODES="${NNODES:-1}"
NGPUS="${NGPUS:-${NGPUS_PER_NODE:-1}}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
DTYPE="${DTYPE:-bfloat16}"
MIN_FREE_GPU_GIB="${MIN_FREE_GPU_GIB:-16}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1280}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
MAX_MODEL_LENGTH="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

ACTOR_LR="${ACTOR_LR:-5e-6}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-10}"
SEED="${SEED:-20260824}"
PROJECT_NAME="${PROJECT_NAME:-evidence_flow_grpo}"
DEFAULT_RUN_SUFFIX="$(date -u +%Y%m%dT%H%M%SZ)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-retrieval_grpo_v0_qwen25_05b_${DEFAULT_RUN_SUFFIX}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/${EXPERIMENT_NAME}}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
VAL_ONLY="${VAL_ONLY:-0}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

if (( $# != 0 )); then
    echo "Positional Hydra overrides are forbidden; use only the reviewed environment variables" >&2
    exit 2
fi

for binary_flag_name in PREFLIGHT_ONLY DRY_RUN VAL_ONLY VLLM_USE_V1; do
    binary_flag_value="${!binary_flag_name}"
    if [[ "${binary_flag_value}" != "0" && "${binary_flag_value}" != "1" ]]; then
        echo "${binary_flag_name} must be 0 or 1, got ${binary_flag_value}" >&2
        exit 2
    fi
done
if [[ "${VLLM_USE_V1}" != "1" ]]; then
    echo "VLLM_USE_V1 must remain 1 for the reviewed launcher" >&2
    exit 2
fi
if [[ "${DTYPE}" != "bfloat16" ]]; then
    echo "DTYPE must remain bfloat16 for the reviewed FSDP/vLLM path" >&2
    exit 2
fi

if [[ "${VAL_FILE}" != "${DEV_FILE}" && "${VAL_FILE}" != "${TEST_FILE}" ]]; then
    echo "VAL_FILE must equal the frozen DEV_FILE or TEST_FILE" >&2
    exit 2
fi

case "${VAL_ONLY}" in
    0) VAL_ONLY_HYDRA=false ;;
    1) VAL_ONLY_HYDRA=true ;;
esac
case "${RESUME_MODE}" in
    disable|auto) ;;
    resume_path)
        if [[ -z "${RESUME_FROM_PATH}" || ! -d "${RESUME_FROM_PATH}" ]]; then
            echo "RESUME_MODE=resume_path requires an existing RESUME_FROM_PATH directory" >&2
            exit 2
        fi
        if [[ "${RESUME_FROM_PATH}" != /* ]]; then
            echo "RESUME_FROM_PATH must be an absolute global_step_N directory" >&2
            exit 2
        fi
        RESUME_FROM_PATH="$(cd -- "${RESUME_FROM_PATH}" && pwd -P)"
        RESUME_BASENAME="$(basename -- "${RESUME_FROM_PATH}")"
        if [[ ! "${RESUME_BASENAME}" =~ ^global_step_[0-9]+$ ]]; then
            echo "RESUME_FROM_PATH must end in global_step_N" >&2
            exit 2
        fi
        if [[ ! -d "${RESUME_FROM_PATH}/actor" ]]; then
            echo "RESUME_FROM_PATH is missing the required actor/ checkpoint directory" >&2
            exit 2
        fi
        ACTOR_WORLD_SIZE="$((NNODES * NGPUS))"
        for ((rank = 0; rank < ACTOR_WORLD_SIZE; rank++)); do
            for shard_kind in model optim extra_state; do
                shard_path="${RESUME_FROM_PATH}/actor/${shard_kind}_world_size_${ACTOR_WORLD_SIZE}_rank_${rank}.pt"
                if [[ ! -f "${shard_path}" ]]; then
                    echo "RESUME_FROM_PATH is missing required FSDP shard: ${shard_path}" >&2
                    exit 2
                fi
            done
        done
        ;;
    *)
        echo "RESUME_MODE must be disable, auto, or resume_path" >&2
        exit 2
        ;;
esac

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VLLM_USE_V1

cd -- "${REPO_ROOT}"

PREFLIGHT=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/server_preflight.py"
    --repo-root "${REPO_ROOT}"
    --config "${CONFIG_PATH}"
    --train-file "${TRAIN_FILE}"
    --val-file "${DEV_FILE}"
    --test-file "${TEST_FILE}"
    --manifest-path "${MANIFEST_PATH}"
    --reward-path "${REWARD_PATH}"
    --reward-name compute_score
    --scorer-path "${SCORER_PATH}"
    --model-path "${MODEL_PATH}"
    --ngpus "${NGPUS}"
    --nnodes "${NNODES}"
    --rollout-tp "${ROLLOUT_TP}"
    --rollout-n "${ROLLOUT_N}"
    --train-batch-size "${TRAIN_BATCH_SIZE}"
    --ppo-mini-batch-size "${PPO_MINI_BATCH_SIZE}"
    --max-prompt-length "${MAX_PROMPT_LENGTH}"
    --total-training-steps "${TOTAL_TRAINING_STEPS}"
    --total-epochs "${TOTAL_EPOCHS}"
    --dtype "${DTYPE}"
    --min-free-gpu-gib "${MIN_FREE_GPU_GIB}"
)
"${PREFLIGHT[@]}"

TRAIN_CMD=(
    "${PYTHON_BIN}"
    -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=false
    algorithm.kl_ctrl.kl_coef=0.0
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${VAL_FILE}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    data.return_raw_chat=true
    data.filter_overlong_prompts=true
    data.truncation=error
    data.shuffle=true
    "data.seed=${SEED}"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=false
    actor_rollout_ref.model.use_remove_padding=true
    actor_rollout_ref.model.enable_gradient_checkpointing=true
    "actor_rollout_ref.model.lora_rank=${LORA_RANK}"
    "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}"
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.actor.strategy=fsdp
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_MODEL_LENGTH}"
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    actor_rollout_ref.actor.use_dynamic_bsz=false
    actor_rollout_ref.actor.use_kl_loss=true
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.actor.fsdp_config.param_offload=false
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
    "actor_rollout_ref.actor.fsdp_config.dtype=${DTYPE}"
    '+actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype:bf16,reduce_dtype:fp32,buffer_dtype:fp32}'
    "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    "actor_rollout_ref.rollout.dtype=${DTYPE}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
    actor_rollout_ref.rollout.top_k=-1
    "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}"
    actor_rollout_ref.rollout.do_sample=true
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LENGTH}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    "actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=true
    actor_rollout_ref.rollout.enforce_eager=true
    actor_rollout_ref.rollout.enable_chunked_prefill=false
    actor_rollout_ref.rollout.free_cache_engine=true
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.do_sample=false
    actor_rollout_ref.rollout.val_kwargs.n=1
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.ref.fsdp_config.param_offload=true
    "actor_rollout_ref.ref.fsdp_config.dtype=${DTYPE}"
    '+actor_rollout_ref.ref.fsdp_config.mixed_precision={param_dtype:bf16,reduce_dtype:fp32,buffer_dtype:fp32}'
    "actor_rollout_ref.ref.fsdp_config.seed=${SEED}"
    reward.num_workers=1
    "reward.custom_reward_function.path=${REWARD_PATH}"
    reward.custom_reward_function.name=compute_score
    "+reward.custom_reward_function.reward_kwargs.scorer_path=${SCORER_PATH}"
    reward.reward_manager.source=register
    reward.reward_manager.name=naive
    reward.reward_model.enable=false
    trainer.device=cuda
    "trainer.nnodes=${NNODES}"
    "trainer.n_gpus_per_node=${NGPUS}"
    'trainer.logger=["console"]'
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    trainer.balance_batch=true
    trainer.val_before_train=true
    trainer.critic_warmup=0
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.resume_mode=${RESUME_MODE}"
    "trainer.val_only=${VAL_ONLY_HYDRA}"
    "trainer.default_local_dir=${OUTPUT_DIR}/checkpoints"
    "trainer.rollout_data_dir=${OUTPUT_DIR}/rollouts"
    "trainer.validation_data_dir=${OUTPUT_DIR}/validation"
)
if [[ "${RESUME_MODE}" == "resume_path" ]]; then
    TRAIN_CMD+=("trainer.resume_from_path=${RESUME_FROM_PATH}")
fi

# Ask Hydra to compose and resolve the exact command that will be launched.
# This catches stale/invalid override paths without initializing Ray or a model.
CONFIG_COMPOSE_CMD=(
    "${TRAIN_CMD[@]:0:3}"
    --cfg job
    --resolve
    "${TRAIN_CMD[@]:3}"
)
"${CONFIG_COMPOSE_CMD[@]}" >/dev/null
echo "Exact Hydra composition PASS"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    echo "Preflight-only mode complete; no training process was started."
    exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY RUN: '
    printf '%q ' "${TRAIN_CMD[@]}"
    printf '\n'
    exit 0
fi

if [[ -e "${OUTPUT_DIR}" && ! -d "${OUTPUT_DIR}" ]]; then
    echo "OUTPUT_DIR exists and is not a directory: ${OUTPUT_DIR}" >&2
    exit 2
fi
if [[ -d "${OUTPUT_DIR}" ]]; then
    for existing_artifact in \
        "${OUTPUT_DIR}"/* \
        "${OUTPUT_DIR}"/.[!.]* \
        "${OUTPUT_DIR}"/..?*; do
        if [[ -e "${existing_artifact}" || -L "${existing_artifact}" ]]; then
            echo "Refusing to reuse non-empty OUTPUT_DIR; choose a fresh run directory: ${OUTPUT_DIR}" >&2
            exit 2
        fi
    done
fi

mkdir -p -- "${OUTPUT_DIR}"
if ! REPOSITORY_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --verify HEAD)"; then
    echo "Cannot record the exact VERL Git commit for this run" >&2
    exit 2
fi
RUN_METADATA_CMD=(
    "${PYTHON_BIN}"
    -m examples.csl_memory_agent.evidence_flow_grpo.run_metadata
    --output "${OUTPUT_DIR}/run_metadata.json"
    --output-dir "${OUTPUT_DIR}"
    --experiment-name "${EXPERIMENT_NAME}"
    --seed "${SEED}"
    --model-path "${MODEL_PATH}"
    --repository-commit "${REPOSITORY_COMMIT}"
    --launcher-file "${SCRIPT_DIR}/run_qwen25_05b_grpo_fsdp.sh"
    --config-file "${CONFIG_PATH}"
    --train-file "${TRAIN_FILE}"
    --dev-file "${DEV_FILE}"
    --val-file "${VAL_FILE}"
    --test-file "${TEST_FILE}"
    --scorer-file "${SCORER_PATH}"
    --manifest-file "${MANIFEST_PATH}"
    --reward-file "${REWARD_PATH}"
    --total-training-steps "${TOTAL_TRAINING_STEPS}"
    --total-epochs "${TOTAL_EPOCHS}"
    --nnodes "${NNODES}"
    --ngpus-per-node "${NGPUS}"
    --rollout-tp "${ROLLOUT_TP}"
    --rollout-n "${ROLLOUT_N}"
    --train-batch-size "${TRAIN_BATCH_SIZE}"
    --ppo-mini-batch-size "${PPO_MINI_BATCH_SIZE}"
    --lora-rank "${LORA_RANK}"
    --lora-alpha "${LORA_ALPHA}"
    --actor-lr "${ACTOR_LR}"
    --dtype "${DTYPE}"
    --kl-loss-coef "${KL_LOSS_COEF}"
    --ppo-micro-batch-size-per-gpu "${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    --log-prob-micro-batch-size-per-gpu "${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    --max-prompt-length "${MAX_PROMPT_LENGTH}"
    --max-response-length "${MAX_RESPONSE_LENGTH}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --rollout-gpu-memory-util "${ROLLOUT_GPU_MEM_UTIL}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE}"
    --rollout-top-p "${ROLLOUT_TOP_P}"
    --save-freq "${SAVE_FREQ}"
    --test-freq "${TEST_FREQ}"
    --val-only "${VAL_ONLY}"
    --resume-mode "${RESUME_MODE}"
)
if [[ "${RESUME_MODE}" == "resume_path" ]]; then
    RUN_METADATA_CMD+=(--resume-from-path "${RESUME_FROM_PATH}")
fi
for hydra_override in "${TRAIN_CMD[@]:3}"; do
    RUN_METADATA_CMD+=(--hydra-override "${hydra_override}")
done
"${RUN_METADATA_CMD[@]}"
LOG_FILE="${OUTPUT_DIR}/train.log"
echo "Launching ${EXPERIMENT_NAME}; log=${LOG_FILE}"
"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
