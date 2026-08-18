# AGENTS.md

本文件是 lite-coding-agent 的项目规则，agent 启动时按目录层级加载。**「关键约束」一节会被约束保留机制解析并独立存储，请勿随意改动其格式。**

一句话说明：从零实现的终端 Coding Agent，不依赖 agent 框架，核心目标是在长任务中不丢失关键约束。

## 代码规范

- 目标运行时：**Python 3.11+**（`requires-python = ">=3.11"`）。不使用 3.12+ 独有语法。
- 格式化与静态检查统一用 **ruff**（`line-length = 100`，`target-version = "py311"`），提交前跑 `ruff check .` 与 `ruff format .`。
- 所有公开函数必须写**类型注解**；`def` 返回值即使是 `None` 也要标注。文件首行用 `from __future__ import annotations`。
- **异步优先**：涉及网络、IO、LLM 调用的接口一律定义为 `async def`。工具执行器统一走异步接口，同步实现用 `asyncio.to_thread` 包装。
- 异常不吞：捕获后要么处理，要么带上原始上下文重新抛出，禁止裸 `except: pass`。
- 新增依赖写进 `pyproject.toml` 的 `dependencies`，并说明用途。

## 目录约定

| 目录 | 职责 | 不允许做的事 |
| --- | --- | --- |
| `src/agent/core/` | Agent Loop、LLM 抽象、上下文管理与压缩 | 不导入 `tools`，不做终端输出 |
| `src/agent/tools/` | 工具注册表、JSON Schema 校验、具体工具实现 | 不导入 `core`，不自己发起 LLM 调用 |
| `src/agent/memory/` | `AGENTS.md` 加载与注入、checkpoint 持久化 | 不直接操作消息历史以外的状态 |
| `src/agent/cli/` | 命令行入口、参数解析、终端渲染、依赖装配 | 不实现业务逻辑，只做装配与呈现 |
| `tests/` | 单元测试与集成测试 | 不发真实 LLM 请求，一律用 fake/mock |

依赖方向：`cli` 是装配层，依赖 `core`、`tools`、`memory`；`core` **不导入** `tools`，工具执行器以 Protocol 形式注入（见 `core/loop.py` 的 `ToolExecutor`）；`tools` 不导入 `core`。

## 禁止事项

- 不引入 **LangChain / LlamaIndex** 等 agent 框架，主循环必须自研（见 `docs/decisions.md` ADR-001）。
- 不提交任何 **密钥、Token、`.env` 文件**；密钥只从环境变量读取。
- 不直接 `git push` 到 **main**（个人项目阶段例外，见 C5）；多人协作时一律 feature 分支 + PR。
- **`core` 不得依赖 `tools`**，工具通过注册表在 `cli` 层注入。
- 不为了让测试通过而放宽断言或删测试。
- 不在代码里写死模型名、上下文窗口大小、压缩阈值等可配置项。

## 提交规范

使用 Conventional Commits：

- 前缀：`feat` / `fix` / `docs` / `refactor` / `test` / `chore` / `perf`
- 格式：`<type>: <简短描述>`，一句话说明**做了什么**，不写「update files」这类无信息量的描述
- 一个提交只做一件事；格式化和逻辑改动不要混在同一个提交里
- 破坏性变更加 `!`，如 `feat!: ...`

## 关键约束

以下约束由约束保留机制解析：每行匹配 `^- \*\*(C\d+)\*\*：` 提取 ID 与正文，存入独立的 `constraints.json`，压缩后校验 ID 完整性，缺失则重新注入。

- **C1**：必须兼容 Python 3.11 及以上版本，不得使用 3.12+ 独有的语法或标准库 API。
- **C2**：运行时只允许依赖标准库与 LLM 官方 SDK，禁止引入 LangChain、LlamaIndex 等 agent 框架。
- **C3**：禁止将任何 API Key、Token、密码写入仓库，密钥只能通过环境变量读取。
- **C4**：依赖方向必须单向，core 不得导入 tools，tools 不得导入 cli。
- **C5**：提交信息遵循 Conventional Commits；个人项目阶段允许直推 main，多人协作时改走 feature 分支。

约束编号**永久稳定、不复用**。需要废除某条约束时，在该行末尾追加 `（已废弃）` 并保留编号，不要删除整行——删除会让校验机制无法区分「约束被废弃」和「约束被压缩丢失」。

## 测试要求

- 测试框架：**pytest**，测试文件放 `tests/`，命名 `test_*.py`。
- 新增工具、压缩策略、约束提取逻辑**必须**同时提交单元测试，否则不算完成。
- 涉及 LLM 的测试一律注入假客户端，**测试套件不允许发起真实网络请求**。
- 修 bug 时先写一个能复现的失败测试，再改代码。
- 提交前必须跑通：`pytest -q`、`ruff check .` 与 `ruff format --check .`。
- 这三条同时跑在 CI 上（`.github/workflows/ci.yml`），本地漏掉会被 CI 拦下。
- 测试要断言行为而不是实现细节，避免重构时大面积改测试。

## 环境说明（Windows）

- 开发机为 Windows + PowerShell 5.1，写脚本时用 PowerShell 语法，不用 `&&`、`source`、`mkdir -p`。
- 使用仓库内解释器 `.\.venv\Scripts\python.exe`，不要用系统全局 `python`（当前指向 3.7.9）。
- PowerShell 5.1 的 `Set-Content -Encoding utf8` 会写入 BOM，需要无 BOM 时用
  `[System.IO.File]::WriteAllText(path, text, (New-Object System.Text.UTF8Encoding($false)))`。
- 控制台代码页为 936，重定向输出到文件时加 `PYTHONUTF8=1` 可避免中文乱码。