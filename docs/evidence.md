# 验证证据

记录可复现的验收结论与关键片段。完整终端输出不入库，需要时按「复现方式」重跑。

## 测试规模

| 阶段 | 单测数 | 套件耗时 | 覆盖范围 |
|---|---|---|---|
| Day 1 结束 | 89 | — | LLM Provider、Agent Loop、CLI chat、`read_file` / `write_file` / `list_dir` |
| Day 2 结束 | 212 | 约 6s | 新增 `edit_file` 22 / `bash` 38 / `grep` 18，其余为 llm、loop、cli 的流式用例 |

套件耗时从 31.5s 降到约 6s，原因见 ADR-009：超时用例原先要等满子进程的睡眠时长。

复现：`.venv\Scripts\python -m pytest -q`，`.venv\Scripts\python -m ruff check src tests`。

## 关键验证用例

### 用例一：多轮 tool_call 与错误回填（Day 1）

- **任务**：`lite-agent chat "读取 README.md，把标题改成 '# LiteCoding Agent v0.1'，然后告诉我改后的第一行" --verbose`
- **结果**：退出码 0，共 2 轮。第 1 轮并发发出 `list_dir` + `read_file`；第 2 轮 `write_file` 被拒（`文件已存在，拒绝覆盖`），错误回填后模型如实回答「没有改成」，未伪造成功。README 全程未被改动。
- **说明**：当时还没有 `edit_file`，`write_file` 又拒绝覆盖，所以这个任务注定完不成——价值恰恰在于两点：多轮 tool_call 链路确实会连续调用工具；工具失败不会打断循环，错误文本回填后模型据此改口而不是幻觉成功。

### 用例二：edit_file 回归（Day 2）

- **任务**：同上，在 `edit_file` 落地后重跑。
- **结果**：退出码 0，2 轮完成。第 1 轮 `read_file`，第 2 轮 `edit_file` 成功并返回极简 diff。README 第一行变为 `# LiteCoding Agent v0.1`，验证后 `git checkout README.md` 还原。
- **说明**：同一个用例从「做不到」变成「做到」，是 Day 2 最直接的端到端证据。注意它只覆盖了唯一匹配的成功路径，多次匹配、mtime 变更等分支由 `tests/test_edit_file.py` 的 22 个用例覆盖。

## 已知问题

- **httpcore2 关闭流时的 traceback**（未解决，上游缺陷）：多轮流式调用后，进程退出阶段 stderr 会打印 `RuntimeError: generator didn't stop after athrow()`。根因在 `httpcore2/_utils.py` 的 `safe_async_iterate` 于 `finally` 中 `await aclose()`。退出码、stdout 与文件改动均不受影响。完整分析与候选方案见 ADR-007。

## 待补

- **约束保留对比实验（Day 4）**：开关约束保留机制，对比长会话中的约束违反次数。目前只有设计，无数据。
- **20 轮压缩接续验证（Day 3）**：验证四层压缩在长会话中可触发，且压缩后关键信息不丢。
