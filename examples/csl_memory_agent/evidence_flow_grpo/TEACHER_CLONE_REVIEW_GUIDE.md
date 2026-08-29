# Evidence-Flow GRPO：老师 Clone 与 Review 指南

这份指南只解决一件事：无论本机是否已经 clone 过 VERL 仓库，都能安全取得本次 handoff，
并与 `main` 比较。不要直接下载 GitHub 网页上的 `/tree/.../examples` 子目录；pipeline 依赖完整
VERL checkout。

## 固定信息

- 仓库：`https://github.com/ba-king/verl_AgentMemory.git`
- 交接分支：`evidence-flow-grpo-handoff`
- 核心 pipeline commit：`96386c51b7f0add0b479a47470d1a512e61fbfc3`
- GitHub 分支页面：
  `https://github.com/ba-king/verl_AgentMemory/tree/evidence-flow-grpo-handoff`
- 与 `main` 的网页比较：
  `https://github.com/ba-king/verl_AgentMemory/compare/main...evidence-flow-grpo-handoff`

核心 commit 后面可能只有交接文档更新。取得分支后可用下面的命令确认核心代码确实包含在
当前 HEAD 中：

```bash
git merge-base --is-ancestor \
  96386c51b7f0add0b479a47470d1a512e61fbfc3 HEAD \
  && echo "Core pipeline commit present"
```

## Case A：老师从未 clone 过这个仓库

这是最简单、最推荐的情况：

```bash
git clone https://github.com/ba-king/verl_AgentMemory.git
cd verl_AgentMemory
git fetch origin
git switch --track origin/evidence-flow-grpo-handoff
git status --short --branch
```

最后一条应显示位于 `evidence-flow-grpo-handoff`，并且没有未提交文件。

查看这次相对 `main` 的全部变化：

```bash
git diff --stat origin/main...HEAD
git diff --name-status origin/main...HEAD
```

## Case B：已经 clone，工作区没有本地修改

先进入老师已有的 checkout：

```bash
cd /absolute/path/to/existing/verl_AgentMemory
git status --short
git remote -v
```

只有 `git status --short` 没有输出时，才按本 Case 继续。确认 `origin` 指向
`ba-king/verl_AgentMemory` 后执行：

```bash
git fetch origin
git branch --list evidence-flow-grpo-handoff
```

如果第二条没有输出，说明本地还没有该分支：

```bash
git switch --track origin/evidence-flow-grpo-handoff
```

如果第二条已经显示本地分支：

```bash
git switch evidence-flow-grpo-handoff
git pull --ff-only origin evidence-flow-grpo-handoff
```

`--ff-only` 会在历史分叉时停止，不会自动制造 merge commit。

## Case C：已经 clone，但里面有未提交修改

不要 stash、reset 或强行切分支。使用独立 worktree，原工作区不会被改动：

```bash
cd /absolute/path/to/existing/verl_AgentMemory
git status --short
git fetch origin evidence-flow-grpo-handoff
git worktree add --detach \
  ../verl_AgentMemory-evidence-review \
  origin/evidence-flow-grpo-handoff
cd ../verl_AgentMemory-evidence-review
git status --short --branch
```

如果 `../verl_AgentMemory-evidence-review` 已存在，请换一个全新的目录名。Review 完成前不要删除
老师原来的工作区。

## Case D：已经 clone，但 `origin` 不是这个 GitHub 项目

不要覆盖原来的 remote。增加一个新的只读来源名：

```bash
cd /absolute/path/to/existing/verl
git remote -v
git remote add teacher-handoff \
  https://github.com/ba-king/verl_AgentMemory.git
git fetch teacher-handoff
git worktree add --detach \
  ../verl_AgentMemory-evidence-review \
  teacher-handoff/evidence-flow-grpo-handoff
cd ../verl_AgentMemory-evidence-review
```

如果 `teacher-handoff` 已经存在，先用下面的只读命令核对 URL，不要重复添加：

```bash
git remote get-url teacher-handoff
```

## Case E：暂时不想 clone，只想在 GitHub 网页 Review

打开下面的 compare 页面：

```text
https://github.com/ba-king/verl_AgentMemory/compare/main...evidence-flow-grpo-handoff
```

网页 Review 适合先看整体 diff，但服务器运行前仍然需要完整 clone，因为 launcher 会导入仓库
里的 stock `verl` package 和 Hydra config。

## 推荐的 Review 顺序

先读交接和声明边界：

```bash
cd examples/csl_memory_agent/evidence_flow_grpo
less TEACHER_HANDOFF.md
less EXPERIMENT_RESULTS.md
less EXPERIMENT_AUDIT.md
less SERVER_RUNBOOK.md
less MANIFEST.md
```

然后回到仓库根目录，按风险审查代码：

```bash
cd ../../..

git diff origin/main...HEAD -- \
  examples/csl_memory_agent/evidence_flow_grpo/contracts.py \
  examples/csl_memory_agent/evidence_flow_grpo/dataset.py \
  examples/csl_memory_agent/evidence_flow_grpo/reward.py

git diff origin/main...HEAD -- \
  examples/csl_memory_agent/evidence_flow_grpo/run_qwen25_05b_grpo_fsdp.sh \
  examples/csl_memory_agent/evidence_flow_grpo/configs/retrieval_grpo_v0.yaml \
  examples/csl_memory_agent/evidence_flow_grpo/server_preflight.py

git diff origin/main...HEAD -- \
  examples/csl_memory_agent/evidence_flow_grpo/run_metadata.py \
  examples/csl_memory_agent/evidence_flow_grpo/analyze_g2.py \
  examples/csl_memory_agent/evidence_flow_grpo/aggregate_g2.py
```

如果使用的是 `teacher-handoff` remote，请把上述 `origin/main` 换成
`teacher-handoff/main`。

## 可选：只运行 CPU/source 测试

Review 阶段不需要安装 CUDA、vLLM 或下载模型：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install pytest pyyaml packaging black

python -m pytest -q \
  examples/csl_memory_agent/evidence_flow_grpo/tests \
  examples/csl_memory_agent/evidence_flow_stage_minus1/tests/test_factorial_fixture.py

black --check \
  examples/csl_memory_agent/evidence_flow_grpo/*.py \
  examples/csl_memory_agent/evidence_flow_grpo/tests/*.py

bash -n \
  examples/csl_memory_agent/evidence_flow_grpo/run_qwen25_05b_grpo_fsdp.sh
```

冻结结果应为 `65 passed`。这些只是 CPU/source 检查，不代表 GPU 训练通过。

## Review 后如何反馈

建议反馈时注明：

1. Review 的分支和 `git rev-parse HEAD` 输出；
2. 问题对应的 `文件:行号`；
3. 属于工程 blocker、科学设计问题，还是非阻塞建议；
4. 是否同意进入 G0 preflight；
5. 是否同意在 G0 同时出现 `PREFLIGHT PASS` 和
   `Exact Hydra composition PASS` 后运行 10-step smoke。

本分支包含 AI-assisted code。按照仓库 `AGENTS.md`，人类在合并或提交 PR 前必须理解并审查
全部改动。本指南不会自动创建 PR，也不建议在 Review 完成前合并到 `main`。
