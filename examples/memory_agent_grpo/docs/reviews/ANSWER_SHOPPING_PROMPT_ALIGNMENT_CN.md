# Answer Shopping 提示词对齐：独立 review

日期：2026-09-06。本主题仅修改两个运行时代码文件中的固定提示词，目标是减少相对官方 Shopping React 指令新增的行为指导，保留现有 API-native 工具接口。

## 来源与准确命名

对齐来源为本地 `MemoryArena/env/env_systems/web_shopping_env/webshop_plus_client.py`：

- `IMPORTANT_INSTRUCTIONS`：每轮一个动作、可用动作限制、商品 ID、返回搜索、翻页、购买、选项选择。
- `WebshopAdapter.conversation_start_dict[ActionFormat.REACT]`：根据状态与指令行动、search 仅在可用时使用、click 值来自可用列表、关键词由 agent 自己仔细设计。

历史 Shopping baseline 使用过该 React 分支。本次没有混入官方其他动作格式分支特有的提示。

当前版本应称为“按官方 Shopping React 指令整理的 native-tool 适配”。它仍不是官方 runner 的逐字、逐消息等价复现：API tools、通用角色说明、消息包装和控制协议仍由当前框架提供。

## 主题一：Shopping adapter 的任务提示

文件：`adapters/memoryarena/shopping/task_prompt.py`。

| 原内容 | 本次处理 | 理由 |
|---|---|---|
| 关键词简洁、覆盖商品类型和关键属性 | 改为关键词由 agent 决定并仔细设计 | 恢复官方指令的指导强度 |
| 购买前选齐所有必需选项 | 改为说明点击 option value 可选择选项 | 保留动作知识，不加入先后策略 |
| 购买前再次核对公开要求 | 删除额外提醒 | 官方任务文本本身保留需求与约束 |
| 动作无效或页面不变后换有效动作，避免重复 | 删除 | 不添加针对诊断失败的恢复策略 |
| 每次动作都会改变页面 | 改为执行后等待观察，再选择下一动作 | 动作可能不改变页面；串行要求不依赖这个错误假设 |
| 返回搜索、翻页、购买、选项动作含义不完整 | 按官方动作说明补齐 | 删策略时不能一并丢失官方提供的操作知识 |

保留 search 可用性、click 精确匹配、点击商品 ID，以及单次 native 调用、无 prose final、环境决定停止等当前接口要求。
环境数据不覆盖运行协议的既有说明也保留，本次不重构通用消息边界。
提示词顶部增加来源说明，明确它经过 native 转换。

## 主题二：环境终止模式的通用协议

文件：`answer_agent/prompts/environment_terminated.py`。

仅删除：`Do not repeat a successful call with the same tool and arguments unless the new public state explicitly requires it.`

这是一条额外的动作选择指导，不是执行器的硬性权限规则。串行执行、收到观察后再做下一动作和环境终止后停止仍然保留。
另一个 `TOOL_OR_FINAL` 模式不在本次范围内。

## 作用范围与实验可比性

Shopping 的 Answer-only 与 Memory+Answer 均通过相同 adapter 和环境终止模式构建提示，因此都会使用本次版本。
本次没有调整模型、token 上限、步数、会话重置、检索、商品目录、任务切分、执行器、Memory 内容原则或括号修复。

同一新版 Answer 下重跑的对照可以用于分析 Memory 差异；本次改动不能消除历史实验中的目录、动作协议、观察上限等差异。旧结果保留为历史记录，不能改标签冒充新提示词结果。

## 验证与下一步

- 现有全量测试 **1153 项通过**，包括输入拼装、单工具协议、Shopping 两类 runner、轨迹和诊断记录。
- 两个修改文件通过 Ruff，`git diff --check` 通过。
- 没有修改或弱化已有测试，也没有为提示词文字新增重复断言。
- 未调用模型 API；不能据离线测试断言购买能力或记忆利用能力提高。
- 五任务批次继续按用户要求暂停。此前已同意的通用 Memory 内容原则保留。

```bash
PYTHONDONTWRITEBYTECODE=1 ../MemoryArena/.venv-runtime/bin/python -m pytest examples/memory_agent_grpo/tests -q -p no:cacheprovider
```

下一轮如果验证 Memory 内容原则，应使用固定 Memory 输入做前后对照；如果做整体购物对照，应将本次 Answer 提示词作为固定配置，两组一起重跑。两种实验回答不同问题，不应混为同一次干预。
