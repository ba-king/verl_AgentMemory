本次代码主题：Shopping 点击有效性检查与错误反馈。

触发问题是 Answer 在 Description 页面点击 Buy Now。上游只返回不变页面，原适配器因调用没有抛异常而记录 success，Answer 缺少明确的动作错误反馈。现在执行前读取当前页面的结构化可点击项；不存在的点击返回 ERROR 和可点击列表，有效点击按原参数执行。

**建议 review 顺序**

| 顺序 | 文件／入口 | 负责什么 |
| --- | --- | --- |
| 1 | [executor_adapter.py](../../adapters/memoryarena/shopping/executor_adapter.py)，`ShoppingToolExecutorAdapter.execute()` | 点击前校验，区分不可点击与可点击项查询失败，构造公开错误反馈 |
| 2 | [environment_session.py](../../adapters/memoryarena/shopping/environment_session.py)，`get_clickable_items()` | 读取当前 `clickables`，检查生命周期、响应格式与文本编码，隔离私有字段与异常 |
| 3 | [test_executor_adapter.py](../../tests/adapters/memoryarena/shopping/test_executor_adapter.py) | 验证拒绝、反馈、大小写、搜索规则，以及 Answer 接收错误后继续执行 |
| 4 | [test_environment_session.py](../../tests/adapters/memoryarena/shopping/test_environment_session.py) | 验证结构化接口、数据投影、请求失败与会话状态 |

[test_answer_only_episode_runtime.py](../../tests/adapters/memoryarena/shopping/test_answer_only_episode_runtime.py) 另补了 4 行：为原有共享环境替身提供 `_get("available_actions")`。Memory episode 测试复用同一替身，真实运行逻辑没有因此改变。

**改动一：环境提供事实**

新方法 `get_clickable_items() -> tuple[str, ...]` 使用现有 WebShop client 的 `_get("available_actions")`，每次点击前重新读取，避免把旧页面按钮用于新页面。

上游已有该只读接口，返回 `has_search_bar` 和 `clickables`。新方法只取 `clickables`，不把整包响应或其他私有数据传入工具反馈。它检查顶层为 dict、该字段为 list、成员为可编码的 UTF-8 字符串；原样保留大小写与空白。空列表是有效响应，表示当前没有点击项。

会话尚未 reset、已结束或已关闭时拒绝读取。新增错误码 `AVAILABLE_ACTIONS_FAILED`、`AVAILABLE_ACTIONS_OUTPUT_INVALID`；缺少上游读取能力时使用已有 `CLIENT_INVALID`。上游异常内容及异常链不会进入公开反馈。

**改动二：执行层判断动作并反馈**

| 情况 | 新行为 |
| --- | --- |
| 搜索调用 | 使用原搜索路径，不查询可点击项，不依据 `has_search_bar` 增加限制 |
| 当前允许的点击 | 保持模型原始参数，交给 `session.step()` |
| 当前不允许的点击 | 返回 `ToolObservationStatus.ERROR`，包含当前可点击项，不调用 `session.step()` |
| 可点击项查询失败或格式异常 | 返回“无法确认当前点击项”的错误，允许下一轮重试，不假称按钮一定不存在 |
| 合法点击后页面不变 | 保持正常返回；页面不变不作为判错规则 |

大小写规则与上游一致：只对传入的 `item` 做 `.lower()` 后查询上游原始键，不去掉空格、不把整份按钮列表转小写，并排除 `click[search]`。例如 `BuY NoW` 可以匹配 `buy now`；带额外空格的 ` Buy Now ` 不会被擅自修成有效动作。这个差异对上游保留原始大小写的 radio 选项也按相同规则处理。

Description 页面上的错误反馈示例：

```text
Click rejected: the requested item is not clickable on the current page.
Current clickable items: ["back to search", "< prev"]
```

框架没有把 Buy Now 自动替换成 `< Prev`。模型选择了什么、被拒绝的是什么，仍然保留在轨迹中；下一步动作由 Answer 自己产生。

**计数与实验口径**

被拒绝的点击消耗一次 Answer 推理轮数，并进入现有连续工具错误计数。它不会增加上游 `round_count`，因为没有调用 `session.step()`。

现有 `executed_tool_count` 统计调用了 executor 的次数，包含 executor 返回错误的调用；不能把它直接当成真实环境步数。这与已有参数拒绝路径采用相同计数方式。本次真实复验有 24 次工具尝试、4 次前置拒绝、20 次上游执行。

新反馈会告知当前可点击项，因此它属于一次明确的反馈协议改动。Answer-only 与 Memory episode 共用这个适配器，后续模型对照应两组同时使用新版本，并记录代码哈希。历史成绩属于旧版本；本次不把新旧轨迹混成同一次比较。

每次点击增加一次本机 available-actions 查询。目标 ASIN 匹配仍由原环境会话的独立终止分类处理，工具 ERROR/SUCCESS 不代替目标评分。

**验证结果**

- 框架测试：1076 项通过（修改前 1042 项，新增 34 项）。
- 新增测试包括：Description 页拒绝 Buy Now、合法购买、大小写和空格、排除 click[search]、查询失败、异常文本隔离、合法操作页面不变、错误进入下一轮 Answer、连续两次错误停止、返回商品页后恢复购买。
- 独立代码 review 未发现阻塞问题；相关三个测试文件单独运行 106 项通过。
- 真实 WebShop、零模型调用：重放既有轨迹中 PIXI、Phyris、T.N. Dickinson's、Sea Breeze 四个商品的 Description 页面。四次 Buy Now 均得到 ERROR；前后按钮列表和上游轮数保持一致；每次 `< Prev` 能返回商品页，最后合法购买完成。

真实复验使用预设动作验证工程接口，没有测试模型会不会自主采取正确的恢复动作，也不能据此声称模型准确率提高。结果与完整动作见 [result.json](../../runs/shopping_click_validation_20260904_r1/result.json)、[actions.jsonl](../../runs/shopping_click_validation_20260904_r1/actions.jsonl)。

可独立查看本次五个实现／测试文件的补丁：[implementation.patch](../../runs/shopping_click_validation_20260904_r1/implementation.patch)。
