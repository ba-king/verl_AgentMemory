# Minimal Agent Memory Demo

这个 example 是一个**不涉及训练的最小 agent memory demo**。它的目标不是实现完整的 memory 系统，而是先验证一个最基本的闭环：

```text
写入 memory -> 后续 query 检索 memory -> 将 memory 暴露给 main agent -> main agent 基于 memory 回答
```

当前版本刻意避免复杂设计，例如 embedding、memory selector policy、update/merge/delete、reward/loss 和 GRPO。这样做是为了先让流程可以清楚地跑通，后续再逐步替换其中的 rule-based 模块。

## 设计目标

这个 demo 对应一个简化后的 agent memory 流程：

```text
User query
  -> MemoryBank.retrieve
  -> MemoryAgent.expose
  -> MainAgent.answer
  -> MemoryAgent.extract explicit Remember/Memory statements
  -> MemoryBank.write_new
  -> updated JSON memory bank
```

其中：

- `MemoryAgent` 当前是 rule-based，不调用模型。
- `MainAgent` 可以是 mock agent，也可以是真实 OpenAI-compatible 模型，例如 DeepSeek 或 OpenAI。
- memory 使用 JSON 文件存储。
- 每条 memory 只保留最小字段。

## Memory 数据结构

当前每条 memory 只包含 4 个字段：

```json
{
  "id": "unique-memory-id",
  "content": "memory content",
  "source": "where this memory came from",
  "created_at": 1780712345.0
}
```

字段含义：

- `id`: memory 的唯一 ID。
- `content`: 记忆的实际内容。
- `source`: 这条记忆来自哪里。当前格式类似 `user_query:1780712345`。
- `created_at`: 创建时间戳。

当前没有使用这些复杂字段：

```text
scope
type
importance
utility
status
updated_at
embedding
confidence
relations
merge/delete/update actions
```

这是有意简化。现阶段只验证 memory flow 是否成立。

## Memory 存储方式

memory bank 存储为 JSON 文件。默认路径是：

```text
examples/agent_memory_demo/memory_bank.json
```

也可以通过 `--memory-path` 指定：

```bash
python examples/agent_memory_demo/simple_memory_demo.py \
  --memory-path /tmp/memory_bank.json \
  --reset \
  --demo
```

如果加 `--reset`，程序启动时会删除旧的 memory bank，然后从空 memory 开始。

## 如何运行

本地 mock 版本不需要任何 API：

```bash
python examples/agent_memory_demo/simple_memory_demo.py --reset --demo
```

自定义多轮 query：

```bash
python examples/agent_memory_demo/simple_memory_demo.py --reset \
  --query "Remember: In this project, answer in concise Chinese." \
  --query "How should you answer project questions?"
```

使用真实 OpenAI-compatible 模型，例如 DeepSeek：

```bash
export OPENAI_API_KEY="your_deepseek_api_key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-v4-flash"

python examples/agent_memory_demo/simple_memory_demo.py --reset --backend openai \
  --query "Remember: I usually drink iced latte in the morning, but I avoid caffeine after 3 PM because it affects my sleep." \
  --query "Remember: I prefer simple home-cooked meals on weekdays, especially noodles, eggs, and vegetables." \
  --query "I have a busy afternoon today and want to buy a drink before a 4 PM meeting. What should I choose?" \
  --query "I am tired tonight and only want to cook something quick. What dinner would fit my usual preference?"
```

这里虽然参数名叫 `OPENAI_*`，但实际可以接任何 OpenAI-compatible API。DeepSeek API 兼容这个格式，所以可以直接使用。

## 多轮 Query 如何执行

多个 `--query` 是**一轮一轮顺序执行**，不是一次性全部送进模型。

代码逻辑等价于：

```python
for query in queries:
    result = run_turn(query, memory_agent, main_agent)
```

每一轮都会执行完整流程：

```text
1. 用当前 query 检索已有 memory
2. 把检索到的 memory 拼成 context
3. main agent 基于 query + memory context 回答
4. 检查当前 query 是否包含 Remember/Memory 记忆指令
5. 如果包含，则写入 JSON memory bank
6. 下一轮 query 可以使用前面写入的 memory
```

注意：当前 demo 不把完整历史对话 transcript 传给模型。模型看到的是当前 query，以及从 JSON memory bank 检索出的 memory。

## 写入 Memory 的方法

当前写入 memory 使用显式规则。只有当用户 query 中包含以下模式时，才会写入：

```text
Remember: ...
Memory: ...
记住：...
记忆：...
```

例如：

```text
Remember: I usually drink iced latte in the morning, but I avoid caffeine after 3 PM because it affects my sleep.
```

程序会抽取冒号后面的内容：

```text
I usually drink iced latte in the morning, but I avoid caffeine after 3 PM because it affects my sleep.
```

然后写入 JSON：

```json
[
  {
    "id": "8b491fe78c3d4761ac2af048fe43cda8",
    "content": "I usually drink iced latte in the morning, but I avoid caffeine after 3 PM because it affects my sleep.",
    "source": "user_query:1780735548",
    "created_at": 1780735548.5819268
  }
]
```

如果当前 query 不包含这些显式记忆指令，则不会写入新 memory，输出：

```text
Memory actions:
- NO_WRITE: No durable candidate memory was extracted.
```

## 检索 Memory 的方法

当前检索方法是最简单的 lexical overlap，也就是关键词重叠。

流程如下：

```text
1. 将当前 query 分词成 term set
2. 将每条 memory.content 分词成 term set
3. 计算 query terms 和 memory terms 的交集大小
4. 交集大于 0 的 memory 会被召回
5. 按重叠词数量排序
6. 取 top-k 条 memory 暴露给 main agent
```

例如：

```text
query:
I have a busy afternoon today and want to buy a drink before a 4 PM meeting.

memory:
I usually drink iced latte in the morning, but I avoid caffeine after 3 PM because it affects my sleep.
```

二者有 `I`、`drink`、`PM` 等词重叠，所以这条 memory 会被检索出来。

当前检索方法非常简单，因此可能召回不完全相关的 memory。这是已知限制，也是后续可以改进的方向。

## Memory 如何暴露给 Main Agent

检索到 memory 后，`MemoryAgent` 会把它们拼成 plain text context：

```text
Relevant memories:
1. I usually drink iced latte in the morning, but I avoid caffeine after 3 PM because it affects my sleep.
2. I prefer simple home-cooked meals on weekdays, especially noodles, eggs, and vegetables.
```

当使用真实模型时，发送给模型的用户内容大致是：

```text
Relevant memories:
1. ...
2. ...

User query: I have a busy afternoon today and want to buy a drink before a 4 PM meeting. What should I choose?
```

所以模型并不是自己长期记住了用户信息，而是当前 demo 从 JSON 中检索出 memory，再把 memory 放进本轮 prompt。

## Example 输出解析

下面以一个日常多轮例子解释中间结果。

运行：

```bash
python examples/agent_memory_demo/simple_memory_demo.py --reset --backend openai \
  --query "Remember: I usually drink iced latte in the morning, but I avoid caffeine after 3 PM because it affects my sleep." \
  --query "Remember: I prefer simple home-cooked meals on weekdays, especially noodles, eggs, and vegetables." \
  --query "I have a busy afternoon today and want to buy a drink before a 4 PM meeting. What should I choose?" \
  --query "I am tired tonight and only want to cook something quick. What dinner would fit my usual preference?"
```

第一轮：

```text
USER: Remember: I usually drink iced latte in the morning, but I avoid caffeine after 3 PM because it affects my sleep.
Relevant memories: <none>
Memory actions:
- WRITE_NEW: User explicitly asked to remember this.
```

解释：

- 因为使用 `--reset`，memory bank 一开始为空。
- 第一轮没有可检索 memory。
- query 包含 `Remember:`，所以写入饮品偏好 memory。

第二轮：

```text
USER: Remember: I prefer simple home-cooked meals on weekdays, especially noodles, eggs, and vegetables.
Relevant memories:
1. I usually drink iced latte in the morning, but I avoid caffeine after 3 PM because it affects my sleep.
Memory actions:
- WRITE_NEW: User explicitly asked to remember this.
```

解释：

- 第二轮开始时，第一轮的饮品 memory 已经存在 JSON 里。
- 当前检索方法是关键词重叠，所以可能召回第一条 memory。
- query 包含 `Remember:`，所以写入饮食偏好 memory。

第三轮：

```text
USER: I have a busy afternoon today and want to buy a drink before a 4 PM meeting. What should I choose?
Relevant memories:
1. I usually drink iced latte in the morning, but I avoid caffeine after 3 PM because it affects my sleep.
2. I prefer simple home-cooked meals on weekdays, especially noodles, eggs, and vegetables.
```

模型回答类似：

```text
Given your 4 PM meeting and your rule of avoiding caffeine after 3 PM,
an iced latte would not be a good choice.
```

解释：

- 第三轮不是写入，而是使用 memory。
- demo 检索到饮品偏好 memory。
- main agent 使用这条 memory，建议 4 PM 前不要喝含咖啡因饮品。
- query 不包含 `Remember:`，所以不写入新 memory。

第四轮：

```text
USER: I am tired tonight and only want to cook something quick. What dinner would fit my usual preference?
Relevant memories:
1. I usually drink iced latte in the morning, but I avoid caffeine after 3 PM because it affects my sleep.
2. I prefer simple home-cooked meals on weekdays, especially noodles, eggs, and vegetables.
```

模型回答类似：

```text
a simple noodle soup with eggs and vegetables
```

解释：

- 第四轮检索到饮食偏好 memory。
- main agent 根据 `noodles, eggs, vegetables` 推荐了简单晚餐。
- query 不包含 `Remember:`，所以不写入新 memory。

## 当前 Demo 证明了什么

这个 demo 证明了：

```text
1. memory 可以被写入 JSON memory bank
2. 多轮 query 是顺序执行的
3. 后续 query 可以检索前面写入的 memory
4. 检索到的 memory 会被拼进 main agent prompt
5. 真实模型可以基于 memory 改变回答
6. 没有显式 Remember/Memory 的普通 query 不会写入新 memory
```

## 当前限制

当前版本有以下限制：

- memory 写入依赖显式 `Remember:`，不是模型自动判断。
- memory 检索是关键词重叠，不是 embedding retrieval。
- 检索到的 top-k memory 全部暴露给 main agent，没有 learned selector。
- 不支持 update、merge、delete、deprecate。
- 不区分 short-term memory 和 long-term memory。
- 不涉及训练、reward、loss、GRPO。
- 每轮只传入检索到的 memory，不传完整历史 transcript。

这些限制是有意保留的，因为当前目标是验证最小流程，而不是一次性实现完整 memory system。

## 后续扩展方向

可以逐步替换当前 rule-based 模块：

```text
显式 Remember 抽取
  -> LLM-based memory extractor

关键词重叠检索
  -> embedding retriever / vector database

全部 top-k 暴露
  -> memory selector policy

只写新 memory
  -> update / merge / delete / deprecate actions

rule-based memory agent
  -> trainable memory agent

无训练 demo
  -> logging actions and rewards for GRPO-style training
```

在 verl 中，下一步可以考虑把：

- `MemoryAgent.prepare_context()` 接成 generation 前的 memory hook。
- `MemoryAgent.maintain_after_turn()` 接成 generation 后的 memory update hook。
- 每次选择和写入 action 记录下来，之后用于训练 memory selector 或 maintenance policy。
