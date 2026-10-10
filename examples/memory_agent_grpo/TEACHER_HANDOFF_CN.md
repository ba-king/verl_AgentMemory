# AgentMemory：继续开发方法的代码入口

这份交接是让老师在现有方法上继续开发。训练数据包单独提供；批量获取数据、调用 Teacher 造数据、人工审核和 zip 打包流程不作为本次交接内容。源码、prompt、接口定义、必要配置和离线测试保留，运行日志、cache、模型权重和原始数据不上传。

## 先认清现在的实现位置

当前实现还没有收拢成一个干净的统一 runtime。基础层在 `memory_agent/`，较新的 structured Bank、Packet V3、RP 和 lifecycle 实现在各轮 `teacher_pilot/` 或 `implementation/` 目录里，存在正常的跨目录依赖。**不能只复制 memory_agent/ 一个目录就认为拿到了最新版。** 本次把下面这些实现所需的源码一起提交，并保留路径，不借同步机会重写架构。

| 开发内容 | 当前实际实现 |
| --- | --- |
| 基础动作、Bank、执行器 | `memory_agent/action/`、`memory_agent/bank/`、`memory_agent/executor/` |
| 当前字段级 UPDATE | `memory_agent/action/structured_update.py` |
| Structured Bank v2 / parser 接线 | `teacher_pilot/travel_v0/structured_record_v2/bank.py`、`policy.py` |
| Record schema / renderer | `teacher_pilot/travel_v0/structured_record/policy.py`、`record_renderer.py` |
| 原生工具调用与回执 | `memora_sft_dev/teacher_rollout_v1/interface.py` |
| 最新 YIELD wire interface | `memora_teacher/weekly_pro_yield_v2/interface.py` |
| BM25 | `memory_agent/retrieval/bm25.py` |
| Dense / Hybrid / RRF | `audits/stage3_reward_feasibility/retrieval.py`、`teacher_pilot/travel_v0/memory_runtime.py` |
| Answer Agent / provider / tool loop | `answer_agent/`、`model_io/` |
| Shopping 环境适配 | `adapters/memoryarena/shopping/` |
| Packet V3 schema / delta apply | `teacher_pilot/natural_rollout_v1/shopping_batch_v1/packet_interface_v3_minimal_validation/interface_v3.py` |
| Packet 主体与作用域 | 同目录 `trusted_registry.py` |
| Packet runtime / 公开输入边界 | `teacher_pilot/natural_rollout_v1/shopping_batch_v1/packet_v3_live_data_pilot_v1/packet_runtime.py`、`hardening/` |
| Packet state maintenance | `teacher_pilot/natural_rollout_v1/shopping_batch_v1/packet_v3_state_maintenance_policy_convergence_v1/` |
| Shopping Memory Policy v2 | `teacher_pilot/natural_rollout_v1/shopping_batch_v1/policy_labeler_v2_regression/policy.py` |
| Shopping RP 输入接线 | `teacher_pilot/natural_rollout_v1/shopping_batch_v1/shopping_p_vs_rp_check/interface.py` |
| Travel Packet / subject adapter | `final_unified_travel_rp_packet_v3_demo_v1/packet.py`、`adapter.py` |
| Travel event ledger / consolidation 义务 | `travel_lifecycle_fix_and_gated_rerun_v1/implementation/` |
| stale evidence 维护规则 / 只读 auditor | `packet_stale_retraction_and_travel_chain_gate_v1/implementation/` |
| 精确重复题面去重 renderer | `exact_task_dedup_renderer_v1/implementation/renderer.py` |
| 模型无关 SFT dataloader | `training_data/dataset.py` |

部分接口代码的目录名仍带 Memora、teacher 或 pilot，因为通用执行逻辑最初在这些实验中实现。目录名不代表老师必须使用那套数据采集流程；只有实际依赖的接口代码随方法一起保留。

Travel 的旧 scorer 还直接导入 `docs/memoryarena_synthetic_data/family_audits/travel/generate_pilot.py` 中的选择规则与约束函数，因此该文件也必须保留。它在本次交接中作为运行依赖，不需要老师再生成数据。

## 三处容易拿错的版本

1. 基础 `memory_agent/action/protocol.py` 中仍有早期字符串 UPDATE。当前 Memora 数据使用 record v1 + **字段级 UPDATE v2**，实际接线在上表 structured_record_v2。它不是已无条件替换所有历史 runtime 的全局定义。
2. `memory_agent/packet/` 是早期 packet 探索实现。**冻结 Packet V3** 位于上表 `interface_v3.py`，不要根据短目录名误认为前者较新。
3. Shopping 当前 RP 路径是 **同一次 READ 的 raw retrieved memories + Packet 一起给 Answer**。早期 Packet-only 的类仍被 RP wrapper 复用，不代表最终输入只剩 Packet。Travel 也已有 Packet V3/RP 路径，不应再描述为“Travel 没有 Packet”。

ADD 新建记录；UPDATE 设置/移除字段，也承担列表部分删除；DELETE 删除整条。字段值保持字符串编码。模型端 YIELD 为 `{"status":"done"}`，内部 canonical 为 `{}`。可直接核对 [Bank schema 文档](docs/bank_schema_current/README_CN.md)；其中已记录的一致性限制没有在这次同步中偷偷修掉。

## 训练数据读取

下面使用已经交付的 500 条数据包，两个包可以分别读取。此步骤只需 Python 标准库，不要求 Qwen、DeepSeek 或 GPU。

```bash
export PYTHONPATH="$PWD"
python -m examples.memory_agent_grpo.training_data.verify_package \
  --root /path/to/unpacked/data_package --expected-count 500
```

```python
from examples.memory_agent_grpo.training_data.dataset import (
    SessionDataset, requests_for_session, labels_from_verified_spans,
)
data = SessionDataset('/path/to/unpacked/data_package/data/sessions.jsonl')
for turn in requests_for_session(data[0]):
    messages = turn['input']['messages']
    tools = turn['input']['tools']
    target = turn['target']
    supervision = turn['supervision']
```

这里的 loader 面向现有独立 session 数据：一个写操作后 YIELD，或直接 YIELD。保留实际 `reasoning_content / tool_calls / tool 回执`。`sessions.jsonl` 与 `input_target.jsonl` 是同一数据的两种视图，不能重复混入训练。

学生模型选定后，老师仍需接该模型的 chat template，核对监督 token span，并把 weight 接入 trainer。写操作权重 1.0、YIELD 0.2；输入、工具回执、audit、padding 不计 loss。不把 audit/后态泄漏到当前输入，不静默截断。已有 verl 训练框架不等于本项目已经实现并验证了完整 Memory GRPO 训练闭环；本次没有开始训练。

## 离线开发检查

在仓库根目录安装 `requirements-method.txt` 所列 CPU 依赖后，运行：

```bash
PYTHONPATH="$PWD:$PWD/examples/memory_agent_grpo" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest \
  examples.memory_agent_grpo.tests.test_structured_update \
  examples.memory_agent_grpo.teacher_pilot.travel_v0.structured_record_v2.test_runtime \
  examples.memory_agent_grpo.tests.test_method_handoff \
  examples.memory_agent_grpo.training_data.test_contract
```

这些检查使用测试 fixture，不调用模型，不需要旧 API 日志。部分历史测试明确回放已封存在线轨迹，需要另行提供原始附件，不能把附件缺失认定为方法源码缺失。

旧 pilot 中同时存在完整包名和短包名 import，开发时按上例将仓库根目录与 `examples/memory_agent_grpo` 都加入 `PYTHONPATH`。

运行真实 Shopping/Travel 环境仍需单独准备 MemoryArena、catalog/数据库和本地模型/embedding；这些是外部依赖，不在本方法代码中。旧环境适配器默认使用与本仓库并列的 `MemoryArena/` 目录。Shopping RP 的在线类还会导入 MemoryArena 的原生 client；只做 Bank、Packet 与 dataloader 检查不需要它。没有以省略外部数据为由更换 scorer 或缩小 catalog。

本次移植修复只涉及：按仓库内容定位根目录，不再要求目录名必须叫 `verl`；仅使用 Bank/parser 时不提前加载 Travel 原生环境。Bank、动作语义、模型 prompt 未因此改变。旧实验的源文件哈希会与当前开发代码不同，不重写旧 seal；新实验需新 prepare，现成训练包继续按包内原始哈希检验。

## 本次同步已做的验证

- 从 Git 暂存内容导出到新目录，运行 Bank/字段 UPDATE、工具协议、Packet V2/V3 和 loader 的 37 项检查，全部通过。
- 精确题面去重 renderer 的 10 项测试通过。
- 使用独立副本导入 99 个方法模块，通过；涉及在线环境的模块使用本机另行安装的 MemoryArena，未导入原工作树中的方法代码，也未发出网络请求。RP 的原始 READ 消息插入检查通过。
- 用新 loader 检验两份已交付的 500 条数据包：共 1,000 个 session、1,895 个 assistant turn；输入/目标对齐及包内哈希通过。

这些是源码可导入、接口与数据读取验证，不是重新做语义审核、在线环境复现实验或模型训练。旧实验 launcher 仍依赖各自的 task/config/seal 附件；本次提供继续开发的方法源码，不声称旧实验命令在无附件时能直接复跑。
