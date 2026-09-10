# 可选构造接口：推断指向已选前提

在编号选择版本之上，推断条目改为：

```json
{
  "id": "E3",
  "need_ids": ["N1"],
  "text": "有明确条件的推断",
  "kind": "derived",
  "premise_ids": ["E1", "E2"]
}
```

E1/E2 必须是同一 evidence 数组中包含 excerpt_id 的原文选择。模型不再为 derived 填 source_refs，程序依据 E1/E2 的原文来源生成，并在 premise_bindings 保存声明的前提关系。最终 packet 仍使用原来的字段，不保留中间字段 premise_ids。

```python
from examples.memory_agent_grpo.memory_agent.packet.premise_prompt import build_premise_messages
from examples.memory_agent_grpo.memory_agent.packet.premises import compile_premise_selection

messages = build_premise_messages(catalog)
# 调用构造模型并确认请求成功且正常结束，取得 raw_content。
result = compile_premise_selection(raw_content, sources=sources, catalog=catalog)
packet = result["packet"] if result["selection_valid"] else None
```

前提只允许直接的原文选择，可在 evidence 中先后任意排列。不接受指向推断的链、循环、自引、空或重复前提、未选编号；也不接受把 source_id 或 excerpt_id 当 evidence ID。没有任何前提会由程序补选。

这能检查声明的前提是否在 packet 中，以及来源是否与这些前提一致。它不能确认模型选对片段、前提是否充分、推断是否成立、需要与未知是否准确。选中 Memory 原文也不表示其中的说法获得独立验证。原文与自由推断均原样保留，错误不自动纠正。

此接口仍为实验选项，未接入正式 Memory action 或 Answer 闭环。完整对照见 `runs/qwen_packet_premises_20260909_r1/`。
