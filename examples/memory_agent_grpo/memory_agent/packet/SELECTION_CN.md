# 实验接口：选择证据编号，程序取回原文

这是可选的 packet 构造探针，尚未接入正式 Memory action 或环境闭环。

```python
from examples.memory_agent_grpo.memory_agent.packet.excerpts import build_excerpt_catalog
from examples.memory_agent_grpo.memory_agent.packet.selection import compile_selection
from examples.memory_agent_grpo.memory_agent.packet.selection_prompt import build_selection_messages

catalog = build_excerpt_catalog(sources)
messages = build_selection_messages(catalog)
# 将 messages 交给构造模型，得到 raw_content；调用方先确认请求成功且正常结束。
result = compile_selection(raw_content, sources=sources, catalog=catalog)
packet = result["packet"] if result["selection_valid"] else None
```

`sources` 是原有 `source_id/kind/text` 记录。分片为连续原文，最多 1200 个 Unicode 字符，按换行/空格优先切分，全量无损覆盖。`start/end` 为 Python 字符索引，end 不含在片段中；不能用作字节或 tokenizer 索引。

模型对来源证据只生成：

```json
{"id": "E1", "need_ids": ["N1"], "excerpt_id": "F0001"}
```

程序根据 F0001 填回 text、kind、source_refs。模型可以生成独立的 `kind=derived` 条目表达有条件的推导，仍须注明来源。两种条目形状不能混用，不能在选择编号时顺便覆盖其事实文本。

最终 packet 仍有 `needs/evidence/unresolved`。`excerpt_id` 仅用于构造中间结果，最终 provenance 以 source_refs 和编译记录 bindings 中的原文位置保存。

来源类型由输入准备方负责：task/observation/reference_observation/context 对应 observed；memory 与未细分的 history 对应 reported。这是来源地位的保守标记，不认证内容真伪；混合 history 中确实来自环境的事实也会暂记 reported。

编译器会拒绝非法编号、原文/元数据被改动、事实选择与自由文本混合及非法关联，不做模糊匹配或事实修补。沿用旧接口两项 unresolved 字段兼容处理，并保留原始输出和处理记录。

`selection_valid` 只说明结构与引用可解析。**它不证明选对了片段、上下文完整、需求正确或推断成立。** 编译器也没有模型 finish_reason，因此调用方须单独拒绝失败或截断的生成。

三个主要 review 文件分别是 `excerpts.py`（分片与原文校验）、`selection.py`（草稿组装）、`selection_prompt.py`（构造提示）。完整本地验证保存在 `runs/qwen_packet_selection_20260909_r1/`；生成草稿与组装后 packet 的长度分开统计。
