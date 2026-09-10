# Memory 评价性结论：依据、范围与不确定性

日期：2026-09-06。状态：**检查结果保留，额外 prompt 草案暂缓实施**。本轮读取已有输入、18 份输出及执行代码，没有修改运行时代码，也没有新增模型或环境调用。五个 Shopping 任务继续暂停。

## 用户确认的工作边界

优先修复框架执行、接口、信息传递和实验设置中已证实的问题，保留必要的通用提示，避免围绕少数失败案例增加冗余设计。记忆选择、摘要忠实性和检索策略等模型能力问题先记录证据与基线，后续由 GRPO 实验检验能否改善；不能预先保证训练可以解决。

因此，下文额外 prompt 草案仅保留为审计历史，不作为待实施任务。当前已生效的通用证据原则和固定 JSON 修复保留。讨论过的来源字段、长依据模板和在线 verifier 均不在当前实施范围；已有轨迹足够支持现阶段人工诊断。Answer 的决策或利用问题单独归因，不能假定只训练 Memory 就一定修复。下一步优先形成冻结版本下的对照基线；本条决定不解除五任务批次的暂停。

## 检查结论

关键问题是生成的结论超出了证据能支持的范围，同时新版出现了不必要地弱化有依据判断的现象。烘焙输入已经包含当前规则、商品完整名称、购买回执和检索到的前购记录；这些关键证据没有被截断。前购来自 M001，不能因此直接认定它等于标准答案中的历史。

当前框架可以可靠检查动作结构并执行写入，但还没有实现内容真实性检查。因此，“合法 ADD”“成功写入”与“记忆有证据支持”必须分别判断。这是尚未实现的语义验证能力；本次未发现这些案例是因关键证据未传入 Memory 而出错。

## 五条可复核证据

所有输入见 [fixed_inputs.json](../../runs/memory_evidence_prompt_20260906_r1/fixed_inputs.json)。下列路径以该 JSON 顶层案例名为起点；事件编号从 0 开始，事件 content 内还包含序列化的 tool observation。

| 输出与判断 | 输入依据 | 审计结论 |
|---|---|---|
| [旧烘焙 r1](../../runs/memory_evidence_prompt_20260906_r1/baking_p2_old_r1/result.json)：因 bundle 含 Vanilla，便认定兼容 Lemon | `baking_p2.task_input` 的 Lemon 避 Fudge/Chocolate；`pre_task_retrieval.matches[0]` 记录 Lemon；`interaction_events[3]` 标题含 Chocolate Fudge and Vanilla | 与可见规则及其自己引用的前提冲突。Vanilla 不能抵消标题中的 Fudge/Chocolate。 |
| [新烘焙 r1](../../runs/memory_evidence_prompt_20260906_r1/baking_p2_new_r1/result.json)：以 $100 八件装更贵，支持购买项不是最高价兼容选项 | `baking_p2.interaction_events[1]` 显示 $100 和 $12.50；任务另有候选与兼容约束 | 价格差有证据，但论据没有建立八件装的候选资格及兼容性。不能把价格抄对等同于评价正确，也不能由此断言购买项的否定结论必然错误。 |
| [新烘焙 r2](../../runs/memory_evidence_prompt_20260906_r1/baking_p2_new_r2/result.json)、[r3](../../runs/memory_evidence_prompt_20260906_r1/baking_p2_new_r3/result.json)：仅说兼容性未显式核实 | 与第一行相同的规则、M001 和商品属性均可见 | 未保留具体规则冲突，属于有用诊断遗漏。沿用原报告的保守口径，不新增硬错误计数。 |
| [旧电子 P1 r2](../../runs/memory_evidence_prompt_20260906_r1/electronics_p1_old_r2/result.json)：Norcent 在指定候选中最低价 | `electronics_p1.task_input` 列三个候选；`interaction_events[3]`、`[5]` 显示对应价格 $999.99、$839.99、$149.99 | 限定到这三个候选的价格结论有支持。不能因出现 lowest-priced 就判错，也不能扩大为所有搜索结果最低价或所有要求均满足。 |
| [新电子 P2 r2](../../runs/memory_evidence_prompt_20260906_r1/electronics_p2_new_r2/result.json)：因为没有 reward/feedback，将规则推断整体标为 uncertain | `electronics_p2.task_input` 预算 $420；`task_result.final_answer` 本次购买 $1232.15；M001 记录 LED，当前规则写 LED 避 Dolby Atmos | 超预算可由可见金额直接确定；兼容冲突应注明依赖 M001。无评分意味着没有官方判分，不能一概削弱局部证据支持的判断。 |

三类标签应分开：**矛盾/无依据评价、诊断遗漏、有依据推断**。本次没有重算或扩大此前 3/9 → 1/9 的错误计数。

## 代码为什么接收这些内容

- [system.py](../../memory_agent/prompts/system.py) 已要求忠于证据和区分推断，但尚未具体说明评价范围；“state its basis and uncertainty”也可能被理解为所有推断都应不确定。这只是待验证的提示词解释假设，不能认定它是原因。
- [action/validation.py](../../memory_agent/action/validation.py) 校验 JSON、字段、非空字符串、允许动作及可见 ID，不接收用于比较 content 语义的任务证据。
- [action_gateway.py](../../memory_agent/decision/action_gateway.py) 校验通过后交给 Executor；[executor/engine.py](../../memory_agent/executor/engine.py) 执行 Bank 操作，没有内容与证据的语义比较。
- [context.py](../../memory_agent/decision/context.py) 的 `COMPLETED` 表示正常结束。Shopping runner 在终止后构造 `TaskResult(final_answer=...)`，本轮输入的 reward/feedback 均为 null。[executor_adapter.py](../../adapters/memoryarena/shopping/executor_adapter.py) 的工具 success 表示调用返回了公开观察。这些状态不证明满足全部任务要求；仅凭轨迹也无法证明它们造成了模型误判。

缺少语义检查并不意味着应把它塞进 JSON 修复器。追加 `}` 可以有唯一、可重放的结构判据，兼容/最优等自然语言评价没有这种通用判据。关键词、价格出现或 source ID 存在，都不能证明结论成立。

## 建议的一个代码主题

**主题：保留评价时说明依据，限制结论范围，并把不确定性定位到具体前提。**

建议下一次只修改 `memory_agent/prompts/system.py` 中现有证据原则段，替换文字如下，其他段落保留。这是草案，尚未应用：

```text
When composing memory content, remain faithful to the visible evidence:
- Preserve observed identities, attributes, actions, and outcomes accurately.
  Do not make a claim that contradicts the evidence you are summarizing.
- Distinguish directly supported facts from inferences. For a useful inference,
  briefly state its supporting facts or rules and identify any uncertain
  premise. Prior agent claims and retrieved memories are not independently
  verified facts; identify when a conclusion depends on them. Missing external
  evaluation alone does not invalidate a conclusion supported by visible facts
  and rules. Missing information is not evidence of absence or falsity.
- For an assessment you retain, state its basis and limit it to the scope that
  basis supports. A claim that a choice is best among eligible alternatives
  requires support for eligibility and the relevant comparison attributes;
  merely observing an alternative does not establish its eligibility.
- Distinguish execution completion from satisfaction of task requirements.
  Do not present a goal, an attempted action, or an unverified assessment as an
  observed outcome. Preserve a relevant rule conflict with its supporting
  premises when the visible evidence establishes it.
- Before writing, check each claim against its evidence. If an assessment
  lacks support, retain useful supported facts and identify the missing premise
  when helpful, without adding an unsupported verdict or dropping those facts.
```

这些要求作用于准备保留的短摘要，不要求输出内部推理过程，也不要求 Memory 继续搜索、替 Answer 选商品或重做任务。购物属性、候选解释、原始规则和公开环境反馈仍属于 adapter 提供的任务数据；草案不加入具体商品、Lemon 规则、标准答案或新 JSON 字段。ADD/UPDATE 的 content 仍为普通字符串，null 仍合法。

## 怎样验证这一个改动

1. 先确认代码 diff 只有这一个 prompt 段，并运行现有流程测试检查兼容性。测试通过只证明接口没破坏，不证明内容变好。
2. 后续固定输入验证应比较**当前证据原则版与本草案**，不能再只与最早的无证据原则版比较。模型、输入、生成配置和格式修复保持一致。
3. 同时检查无依据评价、具体冲突保留、核心事实保留、null、JSON 原生合法率及成本。增加“错误弱化有依据结论”这一观察项，并对两组使用同一口径；不能只奖励多写 inference 或少写结论。
4. 三份旧输入用于开发回归；推广结论仍需要新的任务证据。如果再一次小范围修改仍没有稳定改善，应转向离线断言审计/训练监督的设计，避免围绕单个 case 不断堆提示词。当前五任务批次不由本方案自动启动。

## 与创新动机的关系

这个问题可作为原 idea 中 `v_faith`（内容是否由来源支持）和 `v_cov`（是否遗漏有用信息）的具体诊断样本。提示词调整属于基础实现优化，不能据此宣称完成了 Evidence-Flow GRPO 或证明创新有效。

后续若实现 verifier，适合先离线记录“断言—支持或冲突证据—判断—不确定前提”，人工校准后再考虑训练信号。来源存在性可以确定性检查，语义支持需要另外评估。在线阻断写入或增加纠错 LLM 会改变方法和成本，不能作为这次小改的隐含部分。

本轮另收紧了旧实验报告中一处“非任务候选”的表述，改为候选资格或兼容性尚未确认。没有修改原始输入、输出、标签或统计。
