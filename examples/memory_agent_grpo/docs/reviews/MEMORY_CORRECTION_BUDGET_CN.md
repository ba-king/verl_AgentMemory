# Memory 纠正预算：每决策一次、每 episode 三次

本次独立主题调整纠正机会与审计记录。此前整个 episode 只允许一次纠正；现在 Shopping 诊断和 Memory episode CLI 默认允许整个 episode 最多三次，每个 Memory 决策仍最多一次。底层 runner 默认 0 保持显式开启方式，0/1/2/3 均可配置，训练入口未接入重试。

## Review 顺序

| 文件 | 本次变化 |
|---|---|
| `adapters/memoryarena/shopping/memory_episode_runtime.py` | 用纠正记录列表的长度检查总预算；保留单次嵌套重试，不增加同一决策的循环重试 |
| `adapters/memoryarena/shopping/memory_episode_contract.py` | 将单条纠正字段换为不可变 tuple：memory_output_corrections；旧 memory_output_correction 属性提供第一条记录的只读视图 |
| `adapters/memoryarena/shopping/memory_episode_validation.py` | 验证预算、记录顺序、每决策唯一性、失败纠正必须是最后一条、调用次数一致性 |
| `adapters/memoryarena/shopping/memory_episode_summary.py` | 输出准确的纠正次数、每决策上限及完整记录列表；旧单条安全字段保留第一条，不能用于统计全部纠正 |
| `smoke/memoryarena_shopping_diagnostic.py`、`smoke/memoryarena_shopping_memory_episode_smoke.py` | CLI 默认 3，允许 0–3，说明每决策至多一次 |

每个决策由当前商品及阶段定位：PRE_TASK、POST_TASK、POST_TASK_AFTER_RETRIEVAL。后两个阶段是不同决策，若均输出非法且预算足够，可分别纠正。一个决策的纠正再次非法，会立刻停止，即使 episode 还剩其他额度。模型调用失败、执行错误、观察器异常不会借此增加重试。

每次纠正无论成功或失败都占用一次预算。记录仍是 `product_ordinal / phase / trigger_error_code / recovered`，不包含原始输出。`recovered` 仍表示结构／执行恢复，不承诺原信息保留或内容真实；实验另外通过原始轨迹核查实际写入和后续注入。

原始 episode_trace.json 的 dataclass 序列化现使用复数字段 memory_output_corrections。旧 Python 读取属性及安全汇总单条字段保留；构造新结果时应使用复数 tuple。已有历史结果文件不修改。

## 验证

新增 8 个测试实例：三个不同决策可各纠正并仅写入一次；第四个错误遇到总预算耗尽停止；同一决策无法消耗剩余额度连续纠正；同商品三个阶段可分别纠正；拒绝负值、超过 3、布尔值、非整数预算。成功记录列表还验证重复决策会被结果校验拒绝。

完整框架测试：1093 passed。相关文件 Ruff、git diff --check 通过。

## 两案重新验证的边界

使用 `runs/shopping_correction_budget3_20260904_r1/`，从空 Bank 和初始环境重新运行 task17 与 task61，各一次。任务、商品目录及其 hash 与旧实验一致；DeepSeek v4 flash、temperature=0、thinking disabled、4096 输出 tokens、16000 观察字符、每商品 20 步。

相对旧实验，正常／纠正提示和总纠正预算均已变化，因此不能把路径或目标匹配变化单独归因于预算。本轮检验纠正能否实际写入、暴露给后续 Answer、支持继续运行，原始模型输入与响应均保存，结束后统一分析。结果见该目录 REPORT_CN.md。
