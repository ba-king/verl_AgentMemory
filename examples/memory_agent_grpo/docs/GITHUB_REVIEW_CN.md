# 本次 GitHub 更新的 review 入口

最新 Memora Bank 定义请先看 [当前 Bank schema 审阅包](bank_schema_current/README_CN.md)：包含记录格式、动作参数、实际回执和已发现的一致性差异。此次仅同步 schema 材料；下文是之前的交付记录，不代表最新 Memora runtime 已全部发布。

本次交付范围是 `examples/memory_agent_grpo/` 的推理修复、packet 构造模块、测试和训练准备文档。训练数据来源的选择仍待讨论，本次没有决定使用 MemoryArena test 数据训练，也没有启动训练。

## 三个主题

1. **运行与 JSON 修复**：当前页面的点击检查、可配置 tool choice、有限纠错预算、ADD 缺失外层括号的确定性修复、原始输出及轨迹记录、通用 Memory 忠于证据提示。修改目的及局限见 [reviews/](reviews/)。正式 Shopping 主入口仍采用 native tools；旧诊断与当前模型/协议的结果不能直接混算。
2. **通用 packet 接口**：needs / evidence / unresolved，编号选择、原文回填和可选前提引用检查。见 [packet README](../memory_agent/packet/README.md)。这些是已测试的构造探针，尚未接入正式 Memory action/Answer 闭环，也未证明语义判断可靠。
3. **训练前准备与样例**：见 [准备清单](TRAINING_PREPARATION_CN.md)、[任务边界](training/PACKET_ORGANIZATION_V0_CN.md)、[Task61 P2 开发样本](training/shopping61_p2/REVIEW_CN.md)。样本是参考标注，实际训练集、评分器及服务器训练对接尚未完成。

## Clone 后可做的检查

Task61 样例可以仅用 Python 3.10+ 在仓库根目录检查，不需要 API、模型或原始实验日志：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 \
  examples/memory_agent_grpo/docs/training/shopping61_p2/verify_prepared.py
```

若当前 Python 环境已安装 pytest，运行整个推理框架的离线测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest \
  -q -o addopts= -p no:cacheprovider examples/memory_agent_grpo/tests
```

本机已执行 1,231 项离线测试并全部通过。全目录 Ruff 存在 12 个提交前即有的告警，与 HEAD 逐项比较未发现本次新增告警；packet 和样例脚本单独检查通过。此记录不等于 GPU 训练、联网环境或付费模型复测。

## 本地实验与 GitHub 交付的区别

原始 `runs/`、模型权重、API 凭据和其他研究目录没有加入这次提交。文档中指向 `runs/` 的历史证据链接只在保留这些文件的本地工作区可用；涉及既有决策快照的 probe 也需要相应来源目录。

Task61 已提交的样本保存公开前缀的投影、参考草稿、组装结果与哈希记录，可以独立检查编译一致性；完整原始请求核对仍需要本地存档。不能把独立样例检查当成原始实验复现。

本次代码与文档整理使用了 AI 辅助，按主题提交，方便逐项人工 review。
