# 当前 Memora Bank schema：给老师核对的版本

本页依据本地最新 Memora 执行代码整理。这里只同步**当前实际定义**，没有修改 Bank、动作、prompt、训练数据或 dataloader。不是一份已经修好所有一致性问题的新方案。

当前三层版本要分开看：

- **记录内容**：沿用 `structured_record_v1` 的 `subject / scope / fields`。
- **UPDATE 动作**：`structured_record_v2`，改为按字段设置、删除，不要求模型重写整条记录。
- **最新 YIELD 模型接口**：`weekly_pro_yield_v2`，模型输出 `{"status":"done"}`，适配器转换成执行器内部的 `{}`。

## 1. 一条记忆实际长什么样

```json
{
  "memory_id": "M031",
  "content": "Subject: \"user\"\nScope: \"document: demo_research_plan\"\nFields:\n- kind = \"content\"\n- item = \"demo_research_plan\"\n- data.budget = \"1000000\"",
  "record": {
    "subject": "user",
    "scope": "document: demo_research_plan",
    "fields": [
      {"path": "kind", "value": "content"},
      {"path": "item", "value": "demo_research_plan"},
      {"path": "data.budget", "value": "1000000"}
    ]
  }
}
```

上面是讲解用的示意；可校验的完整例子见 [bank_snapshot.example.json](bank_snapshot.example.json)。

| 字段 | 谁负责 | 当前规则 |
| --- | --- | --- |
| `memory_id` | 执行器 | ADD 自动分配 `M001`、`M002`……；删除后不复用 |
| `record.subject` | Memory Agent | 事实属于谁或什么对象；非空字符串 |
| `record.scope` | Memory Agent | 哪个事件、文档或状态的范围；非空字符串 |
| `record.fields` | Memory Agent | 至少一个 `{path,value}`，同一 record 不允许重复 path |
| `fields[].path` | Memory Agent | 完整字段名；允许字母、数字、下划线及点分段，首段不能以数字开头 |
| `fields[].value` | Memory Agent | **必须是字符串**，允许空字符串；数字和数组也不作为原生 JSON 类型存入这里 |
| `content` | renderer | 从 `record` 唯一生成的检索文本，不能与 record 分别维护 |

`record` 顶层恰好是三个字段，不允许额外字段。`owner` 目前在模型输入中，不在 record 顶层。没有固定的 `summary`、`version`、`source_refs` 或 `timestamp` 顶层字段。

Memora 的 `item`、`preference`、`subcategory`、`data.budget` 等是 **fields 中的字段名约定**，不是新的 ADD 参数。例如偏好可以写：

```json
{
  "subject": "user",
  "scope": "preference: demo_food",
  "fields": [
    {"path": "item", "value": "coffee"},
    {"path": "preference", "value": "dislike"}
  ]
}
```

重要区别：这里的 `data.budget` 是整个字段槽位的名字，不是对一个嵌套 `data` 对象取属性。因此 `budget` 与 `data.budget` 是不同字段；执行器不会替模型纠正写错的路径。

列表在源数据适配时编码为 JSON 字符串。例如 `value` 可以是 `"[\"A\",\"B\"]"`。删除其中一项，需要把整个字段设成新的列表字符串；`remove_fields` 不能进入列表内部。源数据类型另存于审计 sidecar，运行时 Bank 本身不保存那份类型表。

机器可读定义：[record.schema.json](record.schema.json)。它只覆盖结构规则；重复 path 等仍由本地语义校验执行。

## 2. 整个 Bank 怎样保存

```json
{
  "records": [],
  "next_id": 1
}
```

非空时，`records` 每项都包含上面的 `memory_id / content / record`。`next_id` 是下次 ADD 的编号计数，不是 record 的版本号。完整格式见 [bank_snapshot.schema.json](bank_snapshot.schema.json)。

模型看到的全量 Bank 只投影 `memory_id / record`，不重复发送 `content`，也不发送 `next_id`。当前 CONSOLIDATE 输入外层是 `mode / owner / bank / current_session`。

Hash 使用 canonical JSON 的 UTF-8 SHA256：对象键排序、无多余空白、保留 Unicode、禁止 NaN。数组顺序保留，因此字段顺序也会影响 hash。当前摘要保存为十六进制字符串，没有自动附加 `sha256:` 前缀。

## 3. 当前动作参数

| 动作 | 参数 | 含义 |
| --- | --- | --- |
| ADD | `{"record": {"subject": "…", "scope": "…", "fields": […]}}` | 新建完整记录，ID 由执行器分配 |
| UPDATE | `{"memory_id":"M031","set_fields":[{"path":"data.budget","value":"1100000"}],"remove_fields":[]}` | 设置或新增指定字段，删除指定字段，其他字段保持原样 |
| DELETE | `{"memory_id":"M031"}` | 删除整条记录 |
| RETRIEVE | `{"query":"查询文本"}` | 检索；属于通用动作集合，当前 Memora CONSOLIDATE 不开放它 |
| YIELD（最新模型接口） | `{"status":"done"}` | 本次维护结束，不写 Bank |
| YIELD（内部 canonical） | `{}` | 由适配器转换后的执行器参数，不是另一个动作 |

**部分删除用 UPDATE，整条删除用 DELETE。** UPDATE 的两个数组都必须提供，至少一个非空。UPDATE 保留 `subject`、`scope`，不允许通过这个动作改它们。未提到的字段及顺序保持原样，新字段按提交顺序追加。

以下 UPDATE 会整体拒绝，不先写一半：删除不存在的字段、重复设置/删除同一路径、同时设置和删除同一路径、删除后没有任何字段、更新后的路径存在父子冲突。

UPDATE/DELETE 必须指向当时可见且仍存在的 ID。ADD 完全重复的内容，以及 UPDATE 设置成现有值，可能返回 `NO_CHANGE`，不等于执行错误。

当前 Memora CONSOLIDATE：非空且全量可见 Bank 提供 ADD/UPDATE/DELETE/YIELD；空 Bank 只提供 ADD/YIELD。本次采集循环另有“一次写入后结束”的预算限制，这是 harness 规则，不是 Bank schema 的表达限制。

两份工具定义的用途不同：

- [actions.local.example.schema.json](actions.local.example.schema.json)：本地动作参数约束，含通用 RETRIEVE 和内部 YIELD；可见 ID 固定为 `M031` **仅作示例**。
- [tools.memora_wire.example.json](tools.memora_wire.example.json)：当前 Memora 原生 function tools 的实际输出格式，含最新 YIELD；同样只是一组示例 Bank 的动态工具定义。

Provider wire schema 会移除 API 不支持的部分约束，如数组最小长度和唯一性；本地校验仍负责执行。**wire schema 合法不代表一定能写入 Bank。** 不能用 wire schema 代替本地约束，也不能把示例中的 ID 枚举固定用于所有训练样本。

## 4. 当前真实工具回执

UPDATE 成功的形状是：

```json
{
  "status": "SUCCEEDED",
  "code": "UPDATED",
  "record": {
    "memory_id": "M031",
    "content": "Subject: \"user\"\nScope: \"document: demo_research_plan\"\nFields:\n- kind = \"content\"\n- item = \"demo_research_plan\"\n- data.budget = \"1100000\"\n- data.objectives = \"Preserve this unrelated field exactly.\"",
    "record": {
      "subject": "user",
      "scope": "document: demo_research_plan",
      "fields": [
        {"path": "kind", "value": "content"},
        {"path": "item", "value": "demo_research_plan"},
        {"path": "data.budget", "value": "1100000"},
        {"path": "data.objectives", "value": "Preserve this unrelated field exactly."}
      ]
    }
  }
}
```

这是离线合成输入经实际执行器得到的回执。**当前没有 `version: 2`。** 如果后续希望引入版本号，需要另外修改执行器，不能直接补进旧轨迹冒充真实返回。

| 情况 | status / code | record 内容 |
| --- | --- | --- |
| ADD 成功 | `SUCCEEDED / ADDED` | `memory_id / content / record` |
| UPDATE 成功 | `SUCCEEDED / UPDATED` | `memory_id / content / record` |
| DELETE 成功 | `SUCCEEDED / DELETED` | 目前只有被删除的 `memory_id / content` |
| 重复 ADD | `NO_CHANGE / EXACT_DUPLICATE_ADD` | 已有记录 |
| 同值 UPDATE | `NO_CHANGE / UNCHANGED_UPDATE` | 未改变的记录 |
| YIELD | `SUCCEEDED / YIELDED` | 无 record |

工具消息按 `tool_call_id` 关联 assistant 调用，现有消息不额外写 `name`。完整执行器输出见 [executor_examples.json](executor_examples.json)。全部是本轮离线构造输入后的执行器输出，不是 Teacher rollout，不进入训练集。

## 5. 老师重点看这几处一致性问题

1. **确有一处 ADD/UPDATE 校验不一致。** 当前 ADD 允许同一 record 同时出现 `a` 和 `a.b`，UPDATE 却拒绝更新结果中出现这种组合。本轮用实际执行器复现：ADD 成功、UPDATE 拒绝且 Bank 不变。此次只记录，未修复。见 `executor_examples.json` 的 `known_inconsistency`。
2. **字段名没有统一词表。** schema 允许 `budget` 和 `data.budget`；如果模型写错名字，会新增另一字段，而不是自动修改原字段。需要判断老师说的一致性问题是否具体指这个，不能只凭 schema 看起来合法就认定内容正确。
3. **字符串承载所有值。** 原本的数字、字符串、列表等类型不在 Bank 内显式区分。这是现有表达选择，不能对外说原始 JSON 类型都在 Bank 里。
4. **DELETE 和 ADD/UPDATE 回执形状不同。** DELETE 目前不附带嵌套 structured record；训练导出要保留真实返回，不自行补齐。
5. **不同层和历史版本不能混用。** 旧 UPDATE 是整条替换；当前 UPDATE 是字段编辑。旧 YIELD 是 `{}`；当前模型端是 `{"status":"done"}`，但内部仍是 `{}`。它们不应在未标版本时写成一个统一协议。

以上没有证明现有样本已经发生了同类错误；它们是本次核对出的接口事实与一个可复现的校验差异。字段含义与对话是否一致，仍需查看具体样本。

## 6. 本次同步的范围与检查

这是一份**自包含 schema 审阅包**，不是整个最新实验工作树的发布。历史 GitHub runtime 不代表当前 Memora runtime；本地来源文件及 SHA256 全部列在 [source_manifest.json](source_manifest.json)。本页不放远端不存在的源码链接。

完整本地工作树可重新导出（不调用模型）：

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python \
  examples/memory_agent_grpo/docs/bank_schema_current/export_contract.py
```

老师仅下载这个目录也能检查 JSON schema、回执关联、哈希及记录/检索文本一致性。需要 Python 和 `jsonschema`，不需要模型或 API：

```bash
python examples/memory_agent_grpo/docs/bank_schema_current/verify.py
```

校验脚本只核验发布材料；实际执行器探针由 `export_contract.py` 在本地完整工作树执行。9 个探针包括 7 个常规操作/不变操作和 2 个已知不一致探针。校验通过不表示 Teacher 语义质量通过，也不表示这处不一致已经修好。

训练样本外层格式、监督哪些字段、YIELD 的 loss 权重、审核元数据如何打包，均留到下一步讨论；此次不作修改。
