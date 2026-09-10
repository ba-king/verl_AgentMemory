# ADD 外层括号固定修复：代码 review

## 本次解决的问题

两案上一轮共 20 次首次 Memory 决策，其中 8 次 ADD 缺少最外层最后一个 `}`。
模型纠正成功 6 次，但两个 episode 都在第 4 次错误时耗尽三次纠正预算。
本修改在 Shopping 的模型输出与严格 Gateway 之间加入确定性修复，不再为这类已识别的格式错误额外调用模型。

这只是格式恢复，不判断记忆是否真实、有用，也不保证 Answer 正确使用记忆。

## 规则与执行顺序

1. 原始输出先走原有严格校验。合法输出保持原样。
2. 只对 `INVALID_JSON` 扫描结构括号，正确处理字符串、引号和反斜杠转义。
3. 必须恰好只剩最外层对象未闭合，字符串和所有内部结构均完整。
4. 原字符串逐字符保留，只在末尾追加一个 `}`。
5. 候选必须通过完整的原有校验，且动作必须为当前阶段允许的 ADD。
6. 通过后经原有 Gateway 执行一次；否则将原输出送入原来的拒绝/模型纠正流程。

规则名为 `ADD_MISSING_OUTER_BRACE_V1`。每份输出只尝试一次，不循环补括号。
不补引号、不补字段、不推断正文、不修 DELETE/UPDATE/RETRIEVE/null。
PRE_TASK 不允许 ADD，不能借修复绕过。重复字段、空正文、额外字段等仍被拒绝。
模型客户端已判定为截断或失败的响应不会进入此输出修复步骤。

固定修复不产生额外模型调用，因此不消耗模型纠正预算。模型纠正仍为每个决策最多一次、每个 episode 最多三次（Shopping CLI 默认），两种机制分别记录。

## 建议 review 顺序

| 主题 | 文件 | 看什么 |
|---|---|---|
| 纯修复规则 | `memory_agent/action/repair.py` | 无 Bank 副作用；仅追加一个字符；完整校验后仅接受 ADD |
| 接入执行 | `episode/runtime/memory_decision.py`、`adapters/memoryarena/shopping/memory_episode_runtime.py` | 模型只调用一次，Gateway 只处理最终候选一次 |
| 保留并核实证据 | `episode/trace.py`、`episode/validation/memory_decision.py` | 原始输出与修复输出分开；从原文重算修复，拒绝伪造正文、规则和 Bank 变化 |
| 正确统计 | `adapters/memoryarena/shopping/memory_episode_summary.py`、`smoke/memoryarena_shopping_diagnostic.py` | 原生合法、固定修复、模型纠正分别计数 |
| 验证 | `tests/memory_agent/action/test_repair.py`、`tests/episode/runtime/test_output_repair.py` | 结构边界、8 份历史输出、实际 Bank 写入、记录防篡改 |
| 流程与日志 | Shopping runtime/diagnostic 对应测试 | 连续六次缺括号、零重试预算、后续注入、日志不把修复算作原生正确 |

严格解析器 `validate_memory_decision_output` 与 `ActionToExecutorGateway` 没有增加静默容错。
通用 `execute_memory_decision` 的 `repair_add_outer_brace` 默认仍为 `False`；Shopping adapter 显式传 `True`。
因此 Shopping 的模型纠正预算设为零只关闭模型重试，不关闭固定修复。通用 episode runtime 保持原生严格行为。

## 轨迹和指标

- `raw_output`：模型原文，始终不覆盖。
- `output_repair`：没有修复时为 null；修复时保留规则、追加后的文本、原始校验结果。
- `gateway_result.validation`：实际送执行器的候选校验结果，不能直接作为模型原生格式合法率。
- 诊断记录状态为 `repaired`，并保留 `raw_output_valid=false`、`raw_error_code=INVALID_JSON`。
- `first_attempt_valid_count` / `correction_valid_count` 只计模型原生合法输出。
- `deterministic_repair_count` 单独统计固定修复；`first_attempt_accepted_count` / `correction_accepted_count` 包含修复后接受的输出。
- `error_counts` 包含已经修复的原始错误，避免看起来错误消失。
- 安全摘要只暴露 `native_output_valid`、错误码和规则，不暴露原文或正文。
- `committed_memory_output_repair_count` 只覆盖完整提交商品；失败阶段的记录仍应从完整 `memory_decisions.jsonl` 读取。

未来 GRPO 必须保留原始采样 token：不能把程序追加的 `}` 作为模型生成 token 计算概率，也不能把修复率当成原生格式正确率。这里没有修改训练代码或奖励。

## 验证结果与范围

全量测试 **1153 项通过**，本主题相关 Python 文件通过 Ruff；无新增 API 调用。

1. 上轮 8 份非法输出全部可以补一个 `}` 恢复，正文完全保留。
2. 分别在各自历史 Bank 快照上，用固定输出替身通过真实 runtime/Gateway/Executor 回放：8/8 新增一条，旧记录不变，轨迹验证通过。
3. 模拟六件 Shopping 流程，每次 POST ADD 都缺最后括号：零模型纠正预算仍完成六件流程；共 12 次 Memory 调用、6 次固定修复、6 条写入，P2–P6 的 Answer 输入均含前件记忆。
4. 不完整字符串、多处缺括号、重复键、非法字段、其他动作和非法阶段均不自动执行；伪造修复内容或 Bank 变化被轨迹验证拒绝。
5. Observer 失败不重复执行已写入动作；严格模式继续拒绝原始非法输出。

旧的模型纠正预算测试改用缺两个括号的样例，继续验证三次总额度和每决策一次限制；原单括号错误由新增测试覆盖。

历史 fixture 为 `tests/memory_agent/action/fixtures/outer_brace_failures.json`，包含来源路径与 SHA-256，可对照 `runs/shopping_correction_budget3_20260904_r1/correction_audit.json`。

**这是离线固定输出回放和模拟环境测试，不是新的 DeepSeek 两案端到端结果。**
只能说明已观察到的 8 次格式阻塞可以恢复，以及程序流程不会因重复该错误而耗尽模型重试额度；真实任务成功率仍需重新运行验证。

验证命令（从 `verl/` 执行）：

```bash
PYTHONDONTWRITEBYTECODE=1 ../MemoryArena/.venv-runtime/bin/python -m pytest examples/memory_agent_grpo/tests -q -p no:cacheprovider
```
