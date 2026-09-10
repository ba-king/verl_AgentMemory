# 通用短期 packet

此模块提供构造探针使用的提示与解析接口。packet 包含 `needs`、`evidence`、`unresolved`，任务相关内容由模型生成。它尚未接入正式环境中的 Memory/Answer 闭环。

## 主题一：字段兼容

```python
from examples.memory_agent_grpo.memory_agent.packet import parse_packet

parsed = parse_packet(raw_content, source_ids={"S001", "S002"})
```

`raw_output` 保留原文，`strict_errors`/`strict_protocol_valid` 描述规范化之前的结构，`normalizations` 逐项记录兼容处理。`protocol_valid` 描述处理后的结构；只有通过时 `packet` 才包含规范形式，否则为 `None`。

只处理两类差异：缺失的 `unresolved.source_refs` 视为 `[]`；非空字符串 `unresolved.id` 留入事件记录后从规范形式移除。明确提供的错误引用、空/非法编号、其他字段、JSON 错误与语义内容均不修补。

结构通过只证明字段和引用成员关系合法。调用方仍需检查模型请求成功、正常结束以及证据是否支持内容；不能将截断或错误响应中碰巧可解析的 JSON 当成有效生成。

## 主题二：旧记忆相关要求检查

```python
from examples.memory_agent_grpo.memory_agent.packet.prompts import build_messages

messages = build_messages(sources, review_memory_requirements=True)
```

默认 `False` 保留原基础提示。显式 `True` 加入一个通用检查环节：回看旧记录中仍影响当前需求的要求、偏好和前提，保留来源及适用范围；范围不明或冲突进入 `unresolved`。这项选项用于验证，效果以实验结果为准。

`sources` 使用 `source_id/kind/text`。领域实例、人工评分清单、隐藏答案及来源选择说明不属于模型输入。输入准备方负责从允许的当前上下文中提供真实来源。

单元测试位于 `examples/memory_agent_grpo/tests/memory_agent/packet/`。2026-09-09 的本地前后对照保存在 `runs/qwen_packet_memory_requirements_20260909_r1/`。
