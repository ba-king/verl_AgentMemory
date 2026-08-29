# Evidence-Flow GRPO：交接说明

## 一句话定位

这是一个放在 VERL 仓库内的、**面向静态候选记忆选择的完整工程闭环**：

```text
public decision state
  -> Qwen2.5-0.5B-Instruct actor
  -> vLLM 为同一 prompt 采样 n=4 个 JSON 动作
  -> 本地密封 deterministic reward
  -> VERL outcome-GRPO
  -> FSDP + LoRA 更新 actor
  -> step-0 / checkpoint held-out 审计
```

这里的“完整”只指上述受限 probe 的 data-to-checkpoint 工程链路。它不是完整的长程
Memory Agent：没有在线 Memory Bank 写入/删除/consolidation，没有环境多步交互，也没有可信的
Answerer-conditioned `Q_t` reward。

## 当前状态

| 层面 | 状态 | 证据边界 |
|---|---|---|
| 动作、数据、reward | 本地结构验证完成 | 严格三字段 JSON；public/sealed 分离；内容哈希评分 |
| GRPO 组内信号 | CPU synthetic proxy 通过 | 32 个四分支组，零方差 `0/32`；不是模型 rollout |
| 服务器入口 | 已实现，待目标机执行 | launcher、fail-closed preflight、metadata、防覆盖 |
| GPU 训练 | **未执行** | `actual_optimizer_steps=0`；没有真实 checkpoint/loss |
| 核心科学观点 | **未验证** | 当前 reward 是人工 synthetic subset，不是最终 `Q_t` |

本包审核所针对的 VERL 基线提交为：
`34530469d482fc322f985434dbbc9ca28fe06928`。
不确定本机是否已经 clone 过仓库时，按 `CLONE_REVIEW_GUIDE.md` 中对应 Case 操作，
不要直接覆盖已有工作区。
源码级独立审计结果见 `EXPERIMENT_AUDIT.md`：结论是可进入 G0/G1 smoke；该审计是
same-family Type-A review，不是跨模型复现，也不是 GPU 运行证据。

## 为什么目前只建议 smoke

在未随本次 Git 交付上传原始轨迹的一个内部五状态自然干预 pilot 中，双 reviewer 的保守复核
只在 `2/5` 状态看到稳健的 packet-quality 方差，只在 `1/5` 看到稳健正 PEV，稳健负 PEV 为
`0/5`；当前 target-aware 手工 ORACLE 的答案为 `0/5`。原冻结数学 Judge 还出现过等价公式
误判。这些结果不能证明 selector/GRPO 无效，但足以说明当前 oracle/evaluator 不能直接升级为
论文中的局部 reward。这里仅记录它对工程路线的影响；旧 API 原始输出和 sealed 自然目标因不属于
服务器训练依赖而明确排除在本次 branch 之外。

## 第一次运行只需要四步

在完整 VERL checkout 根目录、准备好的 Linux/NVIDIA 环境中：

```bash
python3 -m examples.csl_memory_agent.evidence_flow_grpo.build_dataset \
  --output examples/csl_memory_agent/evidence_flow_grpo/data/factorial_v1 \
  --augmentations-per-case 2 \
  --parquet

PREFLIGHT_ONLY=1 bash \
  examples/csl_memory_agent/evidence_flow_grpo/run_qwen25_05b_grpo_fsdp.sh

DRY_RUN=1 bash \
  examples/csl_memory_agent/evidence_flow_grpo/run_qwen25_05b_grpo_fsdp.sh

TOTAL_TRAINING_STEPS=10 SAVE_FREQ=10 TEST_FREQ=5 bash \
  examples/csl_memory_agent/evidence_flow_grpo/run_qwen25_05b_grpo_fsdp.sh
```

详细环境、路径、step-0/checkpoint 验证和双 seed gate 见 `SERVER_RUNBOOK.md`。10-step
smoke 必须在 preflight 与 DRY_RUN 均通过后执行，只用于确认
CUDA/vLLM/FSDP/LoRA/reward/checkpoint 链路，不得写成方法有效性结果。当前 checkout 以官方
Dockerfile 的 `vllm==0.20.2` 和 bf16 为冻结执行路径；`setup.py` 的 vLLM 上限是已知过时元数据。

## 最小代码审查地图

如果时间有限，建议按风险而不是按文件名顺序审查：

1. **研究对象是否正确**：`contracts.py`、`dataset.py` 的 system prompt 和动作协议。
2. **reward 是否是想训练的目标**：`reward.py` 的 component weights、invalid score 和 sealed
   sidecar。当前最大科学风险就在这里。
3. **VERL 是否接对**：`run_qwen25_05b_grpo_fsdp.sh` 与
   `configs/retrieval_grpo_v0.yaml`。
4. **数据是否泄漏或重复**：`dataset.py`、fixture、`tests/test_dataset.py`。
5. **服务器是否 fail closed**：`server_preflight.py`、`run_metadata.py`。
6. **结果是否能被冒充**：`analyze_g2.py`、`aggregate_g2.py`。
7. 最后运行 `tests/`，不要只相信文档中的 PASS。

## 本次 Git 交付包含什么

- `evidence_flow_grpo/` 中的训练、数据、reward、预检、评估、配置、测试和交接文档；
- `evidence_flow_stage_minus1/` 中仅用于重建 factorial fixture 的 spec、schema、builder、fixture
  和对应测试；
- canonical `structural_sanity_v1` 的两个小型结果文件。

本次明确不上传：旧 DeepSeek API 轨迹、natural pilot r1-r4、raw Answerer/Judge 输出、sealed
自然任务答案、refine logs、Python cache、生成的 Parquet、rollout、checkpoint 和训练日志。
训练 launcher 不读取 DeepSeek API key。

## 服务器 smoke 之后需要共同决定的问题

1. **训练谁**：先只训练 retrieval/selection policy，还是一开始就做带 bank transition 的完整
   Memory Agent？当前代码选择前者。
2. **reward 是什么**：继续使用 gold-subset synthetic reward，还是先构造程序 verifier / cached
   Answerer `Q(E)-Q(EMPTY)`？后者更贴近论文，但更慢且更难保证可信。
3. **先用什么数据**：建议先用可程序验证的 Formal Reasoning/Progressive Search，再进入
   shopping/travel 长程任务；当前 4 个 base task 只够 smoke。
4. **candidate bank 如何产生**：固定 retriever 后只训 selector，还是 joint train retriever/bank？
   当前固定 candidate pool，无法支持 joint-learning claim。
5. **必须比较哪些简单基线**：规则 top-k、BM25/dense top-k、frozen LLM selector、SFT/reranker，
   以及无 RL 的同预算训练。没有这些，GRPO 必要性无法成立。
6. **何时允许扩大算力**：建议先通过 G0、10-step G1，以及新数据上的 evaluator/oracle 校准，
   再执行 100-step × 2 seeds；不要用 smoke checkpoint 做论文结论。
7. **服务器环境是否接受**：第一次运行应固定当前 checkout、官方 Dockerfile 的
   `vllm==0.20.2` 与 bf16；如已有不同镜像，先以 G0/DRY_RUN 结果决定是否另做兼容分支，
   不要临时绕过预检。

另外，`run_metadata.json` 是启动前的 provenance 记录；G2 会核对位置并重算文件哈希/分数，
但在没有可信执行环境或独立签名的 post-run manifest 时，不能从密码学上证明同一路径的文件从未
被人工替换。应保留服务器原始目录为只读证据，不要把 analyzer PASS 当成防恶意篡改证明。

## 可允许的当前汇报

可以说：受限 selector GRPO 的工程链路和审计门槛已经搭好，可以进入目标服务器 smoke。

不能说：完整 Memory Agent 已训练、GRPO 已有效、PEV 已得到验证，或该方法已在自然长程任务上
优于端到端 agent。
