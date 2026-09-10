# Memory JSON 输出与纠正提示：本次改动 review

后续更新：本文记录 prompt 改动时的实验与配置；当前纠正预算已单独调整，见 [预算 review](MEMORY_CORRECTION_BUDGET_CN.md)。

本次解决的具体问题：Memory 首次尝试 ADD 时输出缺少最外层闭合括号，旧纠正提示只传 INVALID_JSON、不传原输出，模型常重新选择 null。现在正常提示明确闭合规则；纠正提示携带原输出与解析错误位置，要求可恢复的合法动作在修语法时保留动作和内容。

## 建议 review 顺序

| 文件 | 本次新增或改变的逻辑 |
|---|---|
| `memory_agent/prompts/system.py` | 明确 arguments/action/顶层对象均需闭合，null 只需顶层；强调字符串转义；声明纠正数据是不可信数据 |
| `memory_agent/action/validation.py` | JSONDecodeError 附带解析器消息、行、列、字符位置，INVALID_JSON 错误码不变 |
| `memory_agent/prompts/correction.py` | 接收 rejected_output 和错误消息，区分 JSON 语法修复与字段／阶段／可见 ID 错误，输出转义后的 correction_data_json |
| `episode/runtime/memory_decision.py` | 校验失败时将原输出和错误消息放入异常属性；纠正调用将它们传入提示构造器，异常字符串不包含原输出 |
| `adapters/memoryarena/shopping/memory_episode_runtime.py` | 在已有的一次纠正分支中传递上述两个属性 |

正常提示此前已有由 json.dumps 生成的完整输出示例；本次没有再复制一套手写示例。闭合规则说明区分 action 与 null，不要求所有输出机械地以三个括号结束。

纠正链路为：

`原输出 → 严格校验失败 → 保存原输出及错误位置 → 同一上下文的纠正请求 → 原严格校验与执行入口`

对 INVALID_JSON，要求保留可恢复、当前阶段允许的 action 及其值，避免改写、摘要或仅为通过校验改成 null。对结构、字段、阶段或可见 ID 错误，要求修正对应问题，不要求保留原本就不允许的动作。无法恢复或无法依据当前证据选择合法动作时，null 仍被允许。

## 原输出如何隔离

原失败文本作为 JSON 字符串放入 correction_data_json；复用现有 `_dynamic_json`，转义引号、换行以及 `<`、`>`。即使失败文本含有 `</correction_data_json>` 等标签，也不会关闭实际数据块。系统提示声明它不是指令，也不是已验证的记忆事实。

原始输出只通过错误对象属性传给同一决策的纠正调用，不进入该异常的 str 或安全结果摘要。原始模型轨迹本来就保存在诊断记录中；现在纠正请求也包含这段文本，因此实际输入 token 会增加。继续使用现有上下文预算校验，不暗中截断原输出。

## 本次没有自动执行修补后的文本

未新增“补一个括号就写入”的路径，也未放宽严格 JSON、动作结构、阶段或可见 ID 校验。模型必须重新输出，通过原入口才能执行。真实失败 fixture 中的 `expected_output` 是对照用的格式修复参考，不是自动写入的数据。

纠正预算仍为原配置：Shopping 可以选择整个 episode 0 次或 1 次。本次没有增加每个决策的重试次数。`ShoppingMemoryOutputCorrection.recovered` 仍表示原有结构／执行恢复，不保证原内容被保留；因此下述实验另外统计原 ADD 内容是否完整保留。也没有新增拒绝一切 null 的机制。

JSON 修复只解决表达结构；如果原内容包含错误历史或未经证实的推断，保留文本也会保留这些问题。内容可靠性仍是独立主题。

## 离线验证

新增 `tests/memory_agent/decision/test_correction_input.py`，以及三个真实失败 fixture：`tests/memory_agent/decision/fixtures/missing_final_brace_cases.json`。

新增 9 个测试实例覆盖：三个实际失败输出仍被拒绝、失败时 bank 不变、原文本和错误位置传至下一调用、脚本化正确响应原文写入一次、标签注入隔离、非语法错误仍遵守阶段限制、null 不被当成写入，以及缺失原输出的参数检查。脚本化响应测试验证的是实现流程，不冒充模型恢复能力。

另更新 Shopping 集成测试：原输出现在应出现在转义后的纠正数据中，但不出现在安全摘要。更新 gateway 错误消息测试以验证新增的解析位置。

完整框架测试：1085 passed。相关改动文件 Ruff、git diff --check 通过。

## 真实模型定点复测

结果目录：`runs/memory_json_correction_20260904_r1/`。三个失败点分别为烘焙 P1 POST、烘焙 P2 POST、电子产品 P1 POST。

固定原始决策上下文，分别运行旧纠正、新纠正、新正常提示，共 9 次 DeepSeek v4 flash 请求，每个 case/arm 一次。temperature=0、thinking disabled、max_tokens=4096、json_object。没有购物环境调用，也没有真实 Bank 写入。旧纠正使用记录中的旧 system/user 消息和旧纠正 notice；新纠正同时改变系统说明、原输出暴露与修复 notice，未分离各因素效果。

| 指标 | 旧纠正 | 新纠正 | 新正常提示重新生成 |
|---|---:|---:|---:|
| 调用数 | 3 | 3 | 3 |
| 格式与动作校验合法 | 3/3 | 3/3 | 2/3 |
| 合法 ADD | 1/3 | 3/3 | 0/3 |
| 合法 null | 2/3 | 0/3 | 2/3 |
| 原 ADD 内容完整保留 | 0/3 | 3/3 | 不作为该组目标 |
| 总输入＋输出 tokens | 13,213 | 15,030 | 13,510 |

全部共 41,753 tokens。新纠正本轮比旧纠正总 tokens 多 1,817（约 13.8%），输出也更长，因为原 ADD 得以保留。不能据此估计总体性价比。

新正常提示在烘焙 P1 仍产生 INVALID_JSON，另两个位置选择 null。该结果不支持“增加闭合约束已经解决首次输出问题”；目前实测支持的是这三个已知失败点的纠正内容保留改善。单次固定点复测不能估计跨模型或跨任务稳定恢复率，也没有验证修复后的完整六商品 episode。

下一步若继续验证，先用新纠正链路复跑曾提前失败的 episode，观察首次非法输出是否再次多次出现、共享一次纠正预算是否仍成为瓶颈，再单独决定是否调整预算或输出协议。
