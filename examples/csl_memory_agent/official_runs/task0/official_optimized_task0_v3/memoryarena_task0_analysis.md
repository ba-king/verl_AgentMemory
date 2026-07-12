# MemoryArena 正式 Task 0 运行分析

## 1. 实验身份与可比性

- Benchmark：官方 `ZexueHe/MemoryArena` checkout，commit `6cd9de14b71915e39ac742a20dc33785e14b6aab`。
- Track：Bundled Web Shopping，官方 item `baking_item_0`，完整执行 6/6 个 split sessions。
- Answer agent：`deepseek-v4-flash`，`max_tokens=4096`，每步最多 25 轮。
- Memory agent：`deepseek-v4-flash`，通过 DART-CSL hard-action guard 调用。
- 环境与评分：官方 WebShop 商品库、搜索索引、任务文件和 `compute_reward.py`；没有用 demo 环境或自定义 ground truth。
- 结论范围：这是一个正式 item 的 item-level 结果，不是论文表中 150 个 items 的总体均值。

## 2. 任务内容

该 item 要按顺序购买 6 类烘焙商品，并满足总预算、逐项搜索、比较全部候选、与前序 ground-truth 商品兼容等全局要求。

| Step | 子任务 | 决策目标 | 官方 target ASIN |
|---:|---|---|---|
| 1 | Cake Base | 候选中最高价，并兼顾评分偏好 | `B00TUDFEW2` |
| 2 | Frosting | 兼容候选中最高价 | `B08957C9ZH` |
| 3 | Coloring Base | 兼容候选中最低价 | `B08SC5FLRK` |
| 4 | Sprinkles | 兼容候选中最高价 | `B07FSHWK6B` |
| 5 | Theme Topper | 兼容候选中最高评分 | `B09CNVBP13` |
| 6 | Candles | 兼容候选中最低价 | `B06XG24BBK` |

## 3. 官方指标定义与结果

- `PS`（Progress Score）：官方六步 soft reward 的平均值。单步 reward 综合商品类型/类别相似度 `r_type`、要求属性匹配和价格约束；若 ASIN 精确匹配 target，单步 reward 强制为 1。
- `SR`（Success Rate）：完整 item 的六步是否全部得到 reward 1。单 item 中是 0/1；论文表格再对全部 items 求平均。
- Hard step pass rate：六步中 reward 恰为 1 的比例，仅作为诊断量，不等同于 PS。

本次正式结果：`PS=0.0`，`SR=0`，hard step pass rate `0/6`。六步都没有实际购买，因此官方 scorer 将每步归因为 `no_purchase`，attribute judge 被跳过，单步 reward 均为 0。

## 4. Token 与行为统计

| Step | Answer calls | Prompt tokens | Completion tokens | Total tokens | Buy 尝试 | 空 action | Reward |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25 | 230,315 | 19,044 | 249,359 | 1 | 0 | 0.0 |
| 2 | 25 | 343,766 | 38,154 | 381,920 | 0 | 0 | 0.0 |
| 3 | 25 | 294,921 | 48,891 | 343,812 | 4 | 1 | 0.0 |
| 4 | 25 | 290,883 | 72,946 | 363,829 | 4 | 6 | 0.0 |
| 5 | 25 | 276,319 | 44,113 | 320,432 | 2 | 1 | 0.0 |
| 6 | 25 | 257,128 | 34,787 | 291,915 | 6 | 0 | 0.0 |

- Answer agent：150 API calls，1,693,332 prompt tokens，257,935 completion tokens，合计 1,951,267。
- Memory agent：17 API calls，70,293 prompt tokens，17,528 completion tokens，合计 87,821。
- 总计：2,039,088 tokens。
- 行为：64 次 search、28 次 Back to Search、17 次 Buy Now；17 次购买都被 guard 拦截。
- Selection audit endpoint 被调用 28 次，但 audit record、cache hit、LLM avoided 均为 0。
- Ledger 最终有 2,685 个 deltas、1,454 个 active states；最终 packet 约 1,198 tokens。

## 5. 主要故障链

1. **跨步骤临时状态污染。** Step 2 的首个 memory packet 把 Step 1 被拦截的 `Sweet 'N Low Lemon Cake Mix` 作为 `Final states to preserve` 注入；Step 4 的 reasoning 又依赖 Step 3 的 coloring selection。这使 answer agent 把“上一页候选”误当成“已成功购买的前序事实”。
2. **状态语义错误。** 进入商品详情页时，`current_selection` 被标为 `final`；实际上它只表示 UI 当前候选，不表示 selection audit 或购买已经完成。
3. **证书门槛与可见字段语义不匹配。** Step 5 已找到正确 target `B09CNVBP13`，评分 4.7，并检查了五个候选；但两个页面显示 `Rating: N.A.`，旧逻辑将其视为“从未检查”，证书始终 deferred。
4. **证书出现两个误报。** `Set of 17` 被无条件判成未请求的 variant；正文中一句 `Purchase History` 被当作历史区块，导致 `$70` 总预算被错误计算为“已花费 70”。
5. **缓存版本不稳定。** 旧 `state_version` 包含 event count 和随机 state id；一次 guard feedback 就改变版本，因此即使页面和候选证据不变，也无法复用审计。
6. **repair feedback 递归嵌套。** 连续 Buy 时，新的 `Current environment remains unchanged` 包住旧 guard 文本，提示不断膨胀并重复同一修复要求。
7. **answer context 成本过高。** Prompt tokens 占 answer tokens 的 86.8%；完整对话持续累积。8 次空 action 来自 reasoning 消耗完 completion budget 后没有输出最终 action，Step 4 最严重。
8. **写入过宽。** 搜索结果页的许多非候选商品都进入 ledger，形成 1,454 个 active states；packet 虽然被截到约 1,198 tokens，但检索噪声和 materialization 成本已经发生。

## 6. 已在运行后落地的修复

- 按公开的 `Product N` 建立 opaque scope；当前决策只读取当前 scope 的 events/deltas/states。
- scope 切换前先 flush 旧事件，避免旧事件绑定到新 TaskContractCard。
- 将详情页 `current_selection` 从 `final` 改为 `candidate`。
- 增加 `rating_observed`，把可见 `N.A.` 表示为“已观察但不可排序”，而非缺失证据。
- 只有任务明确给出数量且观测数量冲突时才报告 variant mismatch。
- Purchase History 只接受独立标题区块；后续步骤没有可见历史时不伪造累计预算。
- `state_version` 改为语义状态哈希，忽略随机 id 与内部 guard feedback；certificate hash 忽略短生命周期 evidence pointer id。
- 缓存同一证书/状态下的 selection approval，也缓存同一状态下的 denial，重复 Buy 不再重复调用 memory LLM。
- guard feedback 只保留最内层真实环境页，阻止递归嵌套。
- 新增 selection audit deferred/build-failure/uncertainty 和 scope-switch 指标。

这些修复来自本次 v3 正式日志，**尚未包含在上面的 v3 分数中**；不能把修复后的预期改进写成已获得的 benchmark 分数。

## 7. 下一轮最值得验证的设计

1. 先对官方 Product 5 做单步诊断运行，验收 `N.A.`、证书签发、selection cache hit 和正确 target purchase，再决定是否花约 2M tokens 重跑完整 item。
2. 给 answer agent 增加 action-format repair：空输出或非单一 action 时，使用仅含当前页的短提示重试一次，并单独统计 repair tokens。
3. 将 memory packet 作为可替换的当前状态，而不是只在 session 第一轮注入后永久留在增长中的 conversation；采用最近若干轮的滑动窗口。
4. 搜索页只写入与五个公开候选描述高相关的实体；详情页事实保持完整，减少 state explosion。
5. repair policy 返回一个确定的下一步（例如“检查尚无 rating_observed 的候选 X”），而不是笼统要求重新搜索全部五项。

## 8. 可复核文件

- `memoryarena_task0_report.json`：机器可读指标与 token 账单。
- `raw/.../reward_report.json`：官方 scorer 原始输出。
- `raw/.../interaction_history/`：六步完整消息、reasoning、action 与 observation。
- `state/.../metrics.json`：DART-CSL 调用、缓存、ledger 与 latency 指标。
