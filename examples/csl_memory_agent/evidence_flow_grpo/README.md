# Evidence-Flow GRPO

这是一个交给 Linux/NVIDIA 服务器运行的小规模 VERL/GRPO 可学习性 probe。当前已经完成数据、
严格动作协议、密封奖励、组内 advantage 结构验证和服务器启动脚手架；**尚未执行任何 GPU
训练或 optimizer step**。

当前交付状态是：**工程上可进入服务器 G0/G1 smoke，科学上尚不能验证核心方法**。这里训练的
是静态候选池上的 retrieval/selection policy，不是带在线 Memory Bank transition 的完整端到端
Memory Agent。当前 gold-subset reward 也是 `synthetic_proxy`，不是最终方法需要的可信局部
`Q_t`。老师建议先阅读 [`TEACHER_HANDOFF.md`](TEACHER_HANDOFF.md)；完整服务器步骤见
[`SERVER_RUNBOOK.md`](SERVER_RUNBOOK.md)，独立源码审计见
[`EXPERIMENT_AUDIT.md`](EXPERIMENT_AUDIT.md)。本包针对 VERL 基线提交
`34530469d482fc322f985434dbbc9ca28fe06928` 审核。

## 冻结动作协议

actor 必须只输出一个三字段 JSON 对象：

```json
{
  "action": "RETRIEVE",
  "evidence_status": "SUFFICIENT",
  "selected_memory_ids": ["mem_..."]
}
```

或者：

```json
{
  "action": "NO_RETRIEVE",
  "evidence_status": "INSUFFICIENT",
  "selected_memory_ids": []
}
```

`NO_RETRIEVE + NOT_NEEDED` 也是合法组合，但只适用于任务本身不需要记忆。解析器不提取
markdown、不截断、不修复：额外字段、未知/重复 ID、非法组合和 prose 均固定获得 fail-closed
负奖励。

## 当前实现

- `contracts.py`：严格 JSON/action schema。
- `dataset.py` / `build_dataset.py`：先按基础任务切分，再做候选重排和 ID 重命名增强。
- `reward.py`：VERL `compute_score`，只读取本地密封 sidecar，按内容哈希确定性打分。
- `grpo.py`：CPU-only 的 VERL outcome-GRPO 标量归一化镜像，仅供结构检查。
- `sanity.py`：构造 oracle、错误合法输出和非法输出，检查奖励排序与组内方差。
- `configs/retrieval_grpo_v0.yaml`：审核用冻结默认值。
- `server_preflight.py` / `run_qwen25_05b_grpo_fsdp.sh`：Linux/CUDA、依赖版本、真实
  tokenizer 长度与精确 Hydra composition 的 fail-closed 预检和 stock VERL 启动器。
- `run_metadata.py`：在模型启动前独占记录真实 seed、完整 Hydra argv、resolved run 参数、
  Git/launcher/data/model/checkpoint provenance。
- `analyze_g2.py` / `aggregate_g2.py`：重算密封 held-out reward、简单基线和两 seed
  预注册门槛；aggregate 会从原始 validation/metadata/checkpoint 重新执行每个 seed 的审计，
  不信任手改的中间 JSON。
- `SERVER_RUNBOOK.md`：老师在服务器上的完整操作顺序。

public 训练 row 只携带不透明 `sample_id`。Gold 动作、证据状态和内容哈希集合只存在于
`scorer/targets.json`；其完整文件 SHA-256 同时写入每条 public row。reward worker 在哈希不一致
时直接退出，不会静默使用错误标签。

## 本地复现 S0 结构验证

在 VERL 仓库根目录运行精简交付集的全部测试：

```bash
.venv/bin/python -m pytest \
  examples/csl_memory_agent/evidence_flow_grpo/tests \
  examples/csl_memory_agent/evidence_flow_stage_minus1/tests/test_factorial_fixture.py \
  -q
```

当前结果为 `57 个 GRPO package tests + 8 个 fixture provenance tests = 65 passed`。
这些测试不加载 CUDA 模型。

如需产生一个新的不可覆盖 run，请换一个全新的输出目录：

```bash
.venv/bin/python -m \
  examples.csl_memory_agent.evidence_flow_grpo.sanity \
  --output-dir /private/tmp/evidence_flow_grpo_sanity_new
```

最新冻结的 `runs/structural_sanity_v1` 结果为：32 个 public 样本、128 个确定性 probe
response、32 个四样本 GRPO group；所有 10 个结构检查通过，reward 范围 `[-8, 5]`，
零方差 group 比例
为 `0.0`。该结果中 `gpu_used=false`、`actual_optimizer_steps=0`。

## 服务器数据构建

GRPO builder 的唯一跨目录输入是
`examples/csl_memory_agent/evidence_flow_stage_minus1/fixtures/factorial_cases_v1.json`。
它可以从同目录提交的 `factorial_spec_v1.json`、`schema.py` 和
`build_factorial_fixture.py` 确定性重建。正式训练数据不要提交到 Git。

在装有 `datasets`/PyArrow 的服务器环境中生成 Parquet：

```bash
python3 -m examples.csl_memory_agent.evidence_flow_grpo.build_dataset \
  --output examples/csl_memory_agent/evidence_flow_grpo/data/factorial_v1 \
  --augmentations-per-case 2 \
  --parquet
```

输出布局是：

```text
data/factorial_v1/
├── public/
│   ├── train.parquet
│   ├── train.jsonl
│   ├── dev.parquet
│   ├── dev.jsonl
│   ├── test.parquet
│   └── test.jsonl
├── scorer/targets.json
└── split_manifest.json
```

train/dev/test 按 `base_task_id` 隔离。`test.parquet` 是 held-out probe，不作为默认训练或在线调参
输入；对应的 `test.jsonl` 只供训练后密封 G2 审计器逐行重算指标。默认 builder 拒绝覆盖已有
数据；要改变 seed、增强数或 split，必须使用新的输出目录和 run ID。

## GPU 入口

完整步骤见 `SERVER_RUNBOOK.md`。最短的服务器预检和 10-step smoke 是：

```bash
export DATA_DIR=/absolute/path/to/data/factorial_v1
PREFLIGHT_ONLY=1 bash \
  examples/csl_memory_agent/evidence_flow_grpo/run_qwen25_05b_grpo_fsdp.sh

DRY_RUN=1 bash \
  examples/csl_memory_agent/evidence_flow_grpo/run_qwen25_05b_grpo_fsdp.sh

TOTAL_TRAINING_STEPS=10 SAVE_FREQ=10 TEST_FREQ=5 bash \
  examples/csl_memory_agent/evidence_flow_grpo/run_qwen25_05b_grpo_fsdp.sh
```

默认 actor 是 `Qwen/Qwen2.5-0.5B-Instruct`、LoRA rank 16、vLLM rollout `n=4`、FSDP、单张
NVIDIA GPU 和冻结的 bf16 mixed-precision 路径。该 checkout 的执行契约以
`docker/Dockerfile.stable.vllm` 的 `vllm==0.20.2` 为准；`setup.py` 中 `<=0.12.0` 是已知过时
声明，预检会显式报告而不会照它错误降级。默认 prompt 上限是 1280 token；目标机 preflight
会使用真实 Qwen tokenizer，任何
冻结样本若将被 VERL 过滤都会拒绝启动。DeepSeek API key 对本阶段**完全不需要**：DeepSeek
既不作为 actor，也不在 reward
时被调用。以后只有进入 cached PEV/冻结 Answerer 评估时才可能需要 API，并且也不能把密钥
写入仓库、命令行或 run artifact。

launcher 拒绝任意尾随 Hydra override，并要求每次训练或 VAL_ONLY 使用全新的空
`OUTPUT_DIR`，避免在 preflight 后替换 actor/data/reward 或覆盖证据。G2 analyzer 会哈希实际
`global_step_100/actor` 模型权重，并同时核对训练、step-0 VAL_ONLY、checkpoint VAL_ONLY
三个 run 的 `run_metadata.json`。validation JSONL 必须位于对应 launcher 记录的
`OUTPUT_DIR/validation`，测试数据/评分器哈希必须一致，checkpoint VAL_ONLY 还必须明确从被审计
的 checkpoint 恢复。命令行 seed 与真实训练 seed 不同、复制 checkpoint，或把别处的 validation
JSONL 重新贴标签都不能通过。

远程 Hugging Face ID 可用于 G0/G1 smoke，但 G2 analyzer 会拒绝未固定的远程 base model。
step-0、训练和 checkpoint 验证必须共同使用同一个本地模型 snapshot；其 `config.json` 和权重
集合会写入 hash-bound run metadata。

## 结果边界

S0 通过只说明 public data → sealed reward → group-relative advantage 的结构闭环成立。老师完成
G1/G2 后，只有 held-out semantic exact 相对 step-0 在两个 seed 上均改善、schema-valid 不退化，
才可写成“该 synthetic proxy 上存在初步 RL learnability”。它仍不等价于 Answerer 质量提升、
PEV 主张成立、RL 必要性或 MemoryArena 泛化；这些必须由后续 cached PEV 和自然状态实验验证。
