# 开发路线图

> **当前进度**：第 0 周 ✅ 已完成（脚手架 / CLI 入口 / 文档）；Day 1 ✅ 已完成（Agent Loop + LLM Provider + 3 个基础工具 + `chat` 命令）
> **策略**：5 天完成简历级核心，第 6–7 天缓冲 + 加分项
> **原则**：每天结束必须有可运行版本，每天至少 3 个 commit

## 总目标

**第 5 天结束**：能演示、有数据、有测试、文档齐全。
**第 6-7 天**：补债、扩大评测、做 MCP 或子 Agent。
**核心独有部分**：关键约束保留机制，必须有对比实验数据。

---

## Day 1：Agent Loop + LLM Provider + 基础工具

**目标**：跑通「用户输入 → LLM → 工具调用 → 回填 → 返回」完整链路。

| 任务 | 预计耗时 | 验收 |
|---|---|---|
| `core/llm.py` Provider 抽象 | 2h | 接一家 API，工具调用格式跑通 |
| `core/loop.py` 主循环 | 2h | 终止条件、错误回填、最大轮数 |
| `tools/read_file.py` `write_file.py` `list_dir.py` | 2h | 参数校验 + 截断 |
| `cli/main.py` 接上 chat 命令 | 1h | `lite-agent chat` 能跑 |
| 调试 + commit | 1h | 3 个 commit，能演示一次工具调用 |

**状态**：✅ 已完成 —— 4 个模块全部实现，89 个单测通过。提交：`552a6a1` LLM Provider、`b0948ff` 工具注册表与文件工具、`c12a2ed` Agent Loop、`cc8fb30` CLI chat。

**验收（全部通过）**：
- [x] 接一家 API，工具调用格式跑通（`OpenAICompatProvider`，OpenAI 兼容协议）
- [x] 终止条件、错误回填、最大轮数（`max_turns` 默认 10，超限返回 `stopped_reason="max_turns"`）
- [x] 参数校验 + 截断（Pydantic 校验；`read_file` 超 2000 行截断）
- [x] `lite-agent chat` 能跑
- [x] 能演示一次工具调用（`list_dir` 返回工作区目录列表）

**可演示**：输入“列出当前目录”，Agent 调 `list_dir` 并返回。
**卡点**：LLM API 工具调用格式不熟 → 先只接一家，跑通再抽象。
**风险应对**：若 API 调试超 2h，先用 mock LLM 跑通循环，再替换真实 API。

---

## Day 2：完整工具系统 + 流式输出

**目标**：6 个工具全上，Agent 能读改文件、跑命令、搜索。

| 任务 | 预计耗时 | 验收 |
|---|---|---|
| `tools/bash.py` | 1.5h | 超时、危险命令确认 |
| `tools/edit_file.py` | 2h | old_string 唯一性校验 |
| `tools/grep.py` | 1.5h | 正则搜索 + 结果截断 |
| `tools/registry.py` | 1h | 工具注册表 + JSON Schema |
| 流式输出 | 2h | 终端逐字输出 |
| 调试 + commit | 1h | 能演示读→改→跑测试 |

**可演示**：让 Agent 读一个文件、改一行、跑 `pytest`。
**卡点**：`edit_file` 唯一性校验容易出 bug → 参考 `claude-code-from-scratch` 的 `docs/02-tools.md`（edit_file 唯一性校验一节），实现对照 `python/mini_claude/tools.py` 的 `_find_actual_string`（第 265 行）与 `_edit_file`（第 290 行）。
**风险应对**：若 edit_file 超时，先只支持唯一匹配，冲突时报错让模型重试。

---

## Day 3：上下文压缩（4 层全实现）

**目标**：长会话不崩，压缩逻辑完整。

| 任务 | 预计耗时 | 验收 |
|---|---|---|
| `core/context.py` token 预算估算 | 1h | 字符数 / 4 近似 |
| Tier 1：预算截断 | 1.5h | 工具输出超阈值截断 |
| Tier 2：裁剪重复 | 1.5h | 同文件重复读取、旧搜索结果 |
| Tier 3：微压缩 | 1h | 空闲后缓存失效触发 |
| Tier 4：全量摘要 | 2h | 摘要 Prompt + 接续逻辑 |
| 调试 + commit | 1h | 构造长会话，观察压缩触发 |

**可演示**：跑一个 20 轮对话，观察压缩触发且对话能接续。
**卡点**：摘要 Prompt 写不好会导致信息丢失 → 先用简单 Prompt，后续调优。
**风险应对**：若 Tier 4 超时，先只做 Tier 1 + Tier 4，Tier 2/3 后补。

---

## Day 4：约束保留机制（核心独有）

**目标**：约束不丢，有对比数据。

| 任务 | 预计耗时 | 验收 |
|---|---|---|
| `core/constraints.py` 约束存储与加载 | 2h | `constraints.json` 结构 |
| 约束来源：AGENTS.md + 用户声明 | 1.5h | 按目录层级加载 |
| 压缩前注入 + 压缩后校验 | 2h | 校验 ID 完整性 |
| 自愈流程 | 1h | 丢失则重新注入 |
| 对比实验设计 | 1.5h | 10 个含约束的长任务用例 |
| 跑实验 + 统计 | 1h | 约束违反次数 X→Y |

**可演示**：同一长任务，开启/关闭约束保留，违反次数对比。
**卡点**：对比实验设计 → 哪怕 10 个用例，也是真实数据。
**风险应对**：若实验设计超时，先用 5 个用例跑通，第 6 天扩大到 20 个。

---

## Day 5：记忆 + checkpoint + 测试 + 文档

**目标**：简历级完成，GitHub 可展示。

| 任务 | 预计耗时 | 验收 |
|---|---|---|
| `memory/agents_md.py` AGENTS.md 加载 | 1.5h | 按目录层级查找 |
| `memory/session.py` checkpoint | 1.5h | `session.jsonl`，kill 后可恢复 |
| 单测（核心模块） | 2h | loop/context/constraints 覆盖 |
| README 架构图 | — | ✅ 已完成（Mermaid 模块图 + Agent Loop 数据流） |
| README 评测数据 | 1.5h | 待 Day 4 实验产出后填入真实数字 |
| 演示 GIF | 1h | 30 秒终端交互 |
| commit 历史整理 | 0.5h | 每天多个 commit |

**可演示**：GitHub README 完整，CI 绿，评测数据填入。
**卡点**：时间不够 → 测试优先级 > 文档 > GIF。
**风险应对**：若单测超时，先覆盖 `constraints.py` 和 `context.py`，其余第 6 天补。

---

## Day 6：缓冲 + 补债 + 扩大评测

**目标**：前 5 天验收标准全部打勾，评测规模扩大。

| 任务 | 预计耗时 | 验收 |
|---|---|---|
| 补前 5 天欠债 | 2h | 所有验收标准打勾 |
| 约束保留实验扩大到 20 用例 | 2h | 统计违反次数、任务成功率 |
| `docs/architecture.md` 与代码同步校验 | 1.5h | 文档中的模块名、状态标记与实际代码一致 |
| `docs/decisions.md` 补 ADR | 1.5h | 至少 5 条决策记录 |
| 补边界单测 | 1h | edit_file 冲突、bash 超时 |

**可演示**：约束保留实验数据更可信，文档完整。
**卡点**：补债时间不够 → 优先补约束实验和 decisions.md。
**风险应对**：若欠债太多，第 7 天继续补，MCP 可放弃。

---

## Day 7：MCP / 子 Agent（选做）

**目标**：加分项，二选一。

| 选项 | 预计耗时 | 验收 |
|---|---|---|
| MCP 客户端 | 4-5h | JSON-RPC over stdio，接入一个外部 server |
| 子 Agent 隔离 | 4-5h | 独立上下文 + 工具白名单 |

**可演示**：接入一个外部 MCP server，或子 Agent 完成一次隔离任务。
**卡点**：JSON-RPC 握手不熟 → 参考 `claude-code-from-scratch` 的 `docs/12-mcp.md`（MCP 集成）；子 Agent 隔离参考 `docs/11-multi-agent.md`。实现对照 `python/mini_claude/mcp_client.py` 与 `python/mini_claude/subagent.py`。
**风险应对**：若时间不够，不做，把第 6 天补扎实。**MCP 和子 Agent 是锦上添花，不是简历必需。**

---

## 风险与应对总表

| 风险 | 影响 | 应对 |
|---|---|---|
| Day 1 API 调试超时 | 阻塞后续 | 先用 mock LLM 跑通循环，再替换真实 API |
| Day 2 edit_file 出 bug | 工具不可用 | 先只支持唯一匹配，冲突时报错让模型重试 |
| Day 3 压缩调优超时 | 长会话崩 | 先只做 Tier 1 + Tier 4，Tier 2/3 后补 |
| Day 4 实验设计超时 | 核心卖点无数据 | 先用 5 个用例，第 6 天扩大到 20 个 |
| Day 5 时间不够 | 简历级不达标 | 测试 > 文档 > GIF，欠债第 6 天补 |
| Day 6 欠债太多 | 文档不完整 | 优先约束实验和 decisions.md，MCP 放弃 |
| Day 7 时间不够 | 少加分项 | 不做，核心链路已完整 |

---

## 里程碑

- [x] **M0**：工程脚手架 + CLI 入口 + 文档（第 0 周完成，6 个 commit）
- [x] **M1**：Agent Loop 跑通一次工具调用（Day 1 结束，89 个单测通过）
- [ ] **M2**：6 个工具可用，流式输出（Day 2 结束）
- [ ] **M3**：4 层压缩能触发，长会话不崩（Day 3 结束）
- [ ] **M4**：约束保留机制有对比数据（Day 4 结束）
- [ ] **M5**：简历级完成，GitHub 完整（Day 5 结束）
- [ ] **M6**：前 5 天验收全打勾，实验扩大到 20 用例（Day 6 结束）
- [ ] **M7**：MCP 或子 Agent 完成（Day 7 结束，选做）

---

## 每日提交建议

- 每天至少 3 个 commit，使用 Conventional Commits。
- 示例：
  - `feat: implement agent loop with tool call handling`
  - `feat: add bash and edit_file tools with validation`
  - `feat: implement tier1/tier4 context compression`
  - `feat: add constraint retention with validation`
  - `test: add unit tests for constraints and context`
  - `docs: update README with evaluation results`

---

## 项目周期说明

如果被问「为什么项目只做了 7 天」：

> 这是限时冲刺版，核心链路和约束保留机制已验证；完整版在 roadmap 中，后续会补 MCP、更大规模评测和子 Agent 隔离。

**关键**：能对着 GitHub 讲清 Agent Loop、压缩策略、约束保留机制的设计与数据，比项目做了多久更重要。