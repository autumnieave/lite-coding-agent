# 开发路线图

> **当前进度**：第 0 周 ✅、Day 1 ✅、Day 2 ✅、Day 3 ✅、Day 4 ✅ —— 脚手架、Agent Loop、6 个工具、流式输出、四层压缩、约束保留机制全部完成；约束保留的对比数据见 `docs/evidence.md` 用例五
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

**状态**：✅ 已完成 —— 6 个工具全部可用，流式输出已接。提交：`7d3f213` edit_file、`6ca3e21` bash、`5ed0851` grep、`805a427` 流式输出、`af81400` 流式收尾修复。

**验收（全部通过）**：
- [x] 超时、危险命令确认（默认 30s；危险命令需 `confirm_dangerous=true` 或交互确认；超时终止整棵进程树）
- [x] old_string 唯一性校验（唯一匹配可用；多次匹配报「出现了 N 次」并给出行号；未找到、read-before-edit、mtime 防护均有覆盖）
- [x] 正则搜索 + 结果截断（上限 100 条，超限提示省略条数；跳过 `.git` / `.venv` / `__pycache__`）
- [x] 工具注册表 + JSON Schema（`Tool.spec()` 输出 OpenAI 兼容 function 定义）
- [x] 终端逐字输出（模型文本流式写 stdout；工具调用与结果实时写 stderr）
- [x] 能演示读→改→跑测试（`lite-agent chat "读取 README.md，把标题改成 ..."` 两轮内完成 read_file → edit_file）

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

**状态**：✅ 已完成 —— 四层压缩全部落地，并通过 20 轮真实会话验证。提交：`7b9c9bb` Tier 1、`ac7e25f` Tier 4、`902b2c7` Tier 2+3；数据与复现方式见 `docs/evidence.md`。

**验收（全部通过）**：
- [x] `core/context.py` token 预算估算（字符数 / 4；工具调用参数一并计入，否则会低估）
- [x] Tier 1 预算截断（超预算的工具结果压成「头 + 截断标记 + 尾」；实测 14967 → 13830 token，1.1ms）
- [x] Tier 2 裁剪重复（同一目标只留最新一次；搜索/命令类结果超过 3 条只留最新几条；实测 20074 → 10722 token）
- [x] Tier 3 微压缩（距上次 API 调用超 5 分钟且利用率仍 ≥ 60% 时清理旧结果；实测 10722 → 8839 token）
- [x] Tier 4 全量摘要（摘要 Prompt 含四个保留项、第四项为「关键约束（如有）」；摘要以 system message 注入；保留最近 10 条；实测 13830 → 4980 token，6.6s）
- [x] 构造长会话，观察压缩触发（20 轮真实会话在第 16 轮触发；压缩后仍能答对压缩前 4 个代号，其中约束类 2/2 经摘要存活）

**可演示**：跑一个 20 轮对话，观察压缩触发且对话能接续。（已达成，见 `docs/evidence.md` 用例三）
**卡点**：摘要 Prompt 写不好会导致信息丢失 → 先用简单 Prompt，后续调优。
**风险应对**：若 Tier 4 超时，先只做 Tier 1 + Tier 4，Tier 2/3 后补。
**参考**：`claude-code-from-scratch/docs/07-context.md` —— 第 37 行「我们的实现」；分层源码位置：第 165 行执行期截断 `truncateResult`、第 204 行大结果持久化 `persistLargeResult`、第 237 行 Budget、第 291 行 Snip、第 317 行 Microcompact、第 349 行 Auto-compact、第 604 行前缀缓存；第 660 行起「真实 Claude Code 比这多做了什么」。实现对照 `python/mini_claude/agent.py`（Anthropic 压缩第 426 行、OpenAI 压缩第 503 行）与 `python/mini_claude/tools.py`（执行期截断与持久化）。
> 层数对齐：参考项目实际是 6 层，其中第 0 / 0.5 层在工具层做执行期截断与持久化（对应我们 `read_file` / `bash` 已有的 2000 行截断）；本表的 Tier 1–4 对应参考的第 1–4 层。Claude Code 原始设计是 5 级流水线，与参考项目的 4 层压缩不等价。

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

**状态**：✅ 已完成 —— 机制落地并跑完对比实验。提交：`786f165` 约束存储、`4be6385` 压缩前注入 + 压缩后自愈、`b562bcb` CLI 接线、`6f26c83` 实验脚本、`2590192` 压力档、`d54b4c5` 续跑支持、`faade72` 证据与 ADR-012。

**验收**：
- [x] `core/constraints.py` 约束存储与加载（结构 `{id, content, source, priority, created_at}`；接口 `load` / `add` / `get_all` / `verify` / `absorb`；空文件、损坏 JSON、重复 id 都是明确定义的行为，34 个单测）
- [x] 约束来源标记（`user` / `agents_md` / `agent` 三种来源已入库；`absorb` 只认用户消息，避免把助手的复述算成约束。**按目录层级读取 AGENTS.md 尚未接入**，落在 Day 5 的 `memory/agents_md.py`，`agents_md` 来源目前只有单测覆盖）
- [x] 压缩前注入 + 压缩后校验（`Compactor.compact()` 先吸收 `[CONSTRAINT] 代号 X：内容` 声明并落盘，再把 `- [ID] 原文` 清单附进摘要请求；摘要回来后按 id 逐条比对）
- [x] 自愈流程（漏掉的按原文补录成 `[约束补录]` 块接在摘要后，累计条数计入 `CompactionStats.replenished`；校验认 id 不认正文，这条边界有单测固定住）
- [x] 对比实验设计（`scripts/constraint_experiment.py`：标准档 18 轮 / 15 条随机代号约束 / 脚本化判定；压力档 25 轮 / 40 条；支持 `--resume` 续跑）
- [x] 跑实验 + 统计（标准档 10+10 打平；压力档 10+10 逐字保留实验组 400/400 vs 对照组 372/400，实验组 6/10 次遇到摘要整段丢掉 40 条、累计补录 280 条。**「违反次数 X→Y」在本次设计下无显著差异**——行为探针只覆盖 40 条约束中的 3 条）

**可演示**：同一长任务开启/关闭约束保留，对比约束原文的存活情况与补录次数（已达成，见 `docs/evidence.md` 用例五）。
**卡点**：对比实验设计 → 已解决。结论和最初设想不同：约束原文的丢失不是「慢慢衰减」，而是「整段消失」，机制的价值是兜底而不是抬平均分。
**风险应对**：实验途中撞到 API 余额中断（402），补了 `--resume` 从已有记录续跑；若再遇到，命令与逐次记录都在 `docs/evidence/*.jsonl`。
**参考**：**无直接对照** —— 约束保留是本项目独有方向，参考项目与 Claude Code 都没有「约束独立存储 + 压缩后校验自愈」机制。可借鉴的相邻设计只有两处：`docs/07-context.md` 第 688 行 Claude Code 的 Level 5 Autocompact 用「分析-摘要」两阶段（先 `<analysis>` 推理，再输出 9 段 `<summary>`，最后剥离推理只留摘要）；`docs/01-agent-loop.md` 第 244-245 行 `collapse_drain_retry` / `reactive_compact_retry` 处理 PTL 错误时的重试顺序。摘要 Prompt 与校验逻辑需自己设计，接口见 ADR-004 与 ADR-010（摘要 Prompt 已为关键约束预留字段）。

---

## Day 5：记忆 + checkpoint + 测试 + 文档

**目标**：简历级完成，GitHub 可展示。

| 任务 | 预计耗时 | 验收 | 状态 |
|---|---|---|---|
| Tier 4 触发过频（动态保留阶梯） | 1.5h | 压力档 25 轮 Tier 4 触发 ≤ 5 次 | ✅ |
| `compose_summary` 摘要叠加 | 1h | 连压 10 次只剩 1 条摘要消息 | ✅ |
| `memory/agents_md.py` AGENTS.md 加载 | 1.5h | 按目录层级查找，C1–C5 标 `source=agents_md` | ✅ 模块完成（**未接入 `Compactor`**） |
| `memory/session.py` checkpoint | 1.5h | `session.jsonl`，kill 后可恢复 | ✅ 模块完成（**未接入 loop / CLI**） |
| 单测（核心模块） | 2h | loop/context/constraints 覆盖 | ✅ 357 → 417 |
| README 架构图 | — | ✅ 已完成（Mermaid 模块图 + Agent Loop 数据流） | ✅ |
| README 评测数据 | 1.5h | 待 Day 4 实验产出后填入真实数字 | ✅ 见 `docs/evidence.md` 用例五/六 |
| 演示 GIF | 1h | 30 秒终端交互 | 📋 顺延 Day 6 |
| commit 历史整理 | 0.5h | 每天多个 commit | ✅ |

**验收（全部通过）**：

- [x] Tier 4 触发过频：`CompactionConfig.retain_ladder = (10, 5, 3, 1)`，压缩后按阶梯收窄保留窗口，直到降到触发线（`DEFAULT_COMPACT_THRESHOLD = 0.60`）以下。离线探针 25 轮会话在 600 / 3000 / 8000 / 15000 / 30000 字符假摘要下 Tier 4 均为 **3 次**，修复前为 3 / 5 / 10 / 11 / 11 次
- [x] 摘要不再叠加：`compose_summary` 的 `head` 过滤掉历史摘要消息（旧摘要仍作为输入喂给摘要器）；连压 3 次、10 次摘要消息数恒为 1（`tests/test_compaction_thrash.py`）
- [x] `memory/agents_md.py`：从 cwd 向上遍历目录层级收集 AGENTS.md，解析「关键约束」下的 C1–C5，落成 `source=agents_md` 的约束；单测 23 条覆盖多层目录、缺失文件、格式异常
- [x] `memory/session.py`：`session.jsonl` 逐轮追加，回放得到消息历史 / 轮数 / token 数 / 约束状态；尾部未配对的工具交换会被裁掉；单测 23 条
- [x] 全量 **417** 个单测通过（Day 4 收尾时 357），`ruff check .` 与 `ruff format --check .` 干净

**可演示**：GitHub README 完整，CI 绿，评测数据填入。
**卡点**：时间不够 → 测试优先级 > 文档 > GIF。
**风险应对**：若单测超时，先覆盖 `constraints.py` 和 `context.py`，其余第 6 天补。
**参考**：`docs/08-memory.md`（记忆系统）与 `docs/04-cli-session.md`（会话与恢复）；实现对照 `python/mini_claude/memory.py`、`python/mini_claude/session.py`。AGENTS.md 的目录层级加载对照 `docs/07-context.md` 第 670 行 CLAUDE.md 的「从 CWD 向上遍历目录树」。

---

## Day 6：接线 + 补债 + 复跑评测

**目标**：把 Day 5 只落了模块的记忆系统接进主流程，并用修复后的代码重跑对比实验。

| 任务 | 预计耗时 | 验收 | 状态 |
|---|---|---|---|
| `memory/agents_md.py` 接进主流程 | 1.5h | 任务启动时按目录层级读 `AGENTS.md`，C1–C5 进约束存储 | ✅ 装配在 `cli` 层 |
| `memory/session.py` 接进 CLI | 1.5h | `session.jsonl` 逐轮追加，kill 后新进程能续聊 | ✅ |
| 约束保留实验复跑 | 2h | 同参数同种子，实验组 / 对照组各 10 次 | ✅ Day 5 + Day 6 各 10 次 |
| 自愈机制专项单测 | 1h | 摘要丢 10 条必须补回 10 条；覆盖全丢 / 部分丢 / 不丢 | ✅ 10 条用例 |
| 补前 5 天欠债 | — | 所有验收标准打勾 | ✅ 见各日 |
| `docs/decisions.md` 补 ADR | 1.5h | 至少 5 条决策记录 | ✅ 已 16 条（ADR-001 ~ ADR-016） |
| 补边界单测 | 1h | edit_file 冲突、bash 超时 | ✅ 前几天已覆盖 |
| 约束注入 system prompt（收尾补的缺口） | 1h | 短任务不触发压缩也能看到约束 | ✅ ADR-016 |
| 约束保留实验扩大到 20 用例 | 2h | 统计违反次数、任务成功率 | 🚧 未做，见下 |
| `docs/architecture.md` 与代码同步校验 | 1.5h | 文档模块名与状态标记和代码一致 | 📋 顺延 Day 7 |

**验收（全部通过）**：

- [x] AGENTS.md 接线：`cli/main.py` 装配阶段调用 `agents_md.register_constraints()`。实跑 `lite-agent chat` 输出 `AGENTS.md 约束：新增 5 条`，`.lite-agent/constraints.json` 落盘 C1–C5 且 `source=agents_md`；改掉 C1 正文再跑变成「按文件刷新 1 条」
- [x] **接线时修掉一个缺口**：`register_constraints` 原本对已存在的同 ID 一律跳过，导致「编辑 AGENTS.md 让约束生效」做不到。改为「AGENTS.md 对它自己的 ID 是唯一真源」，被 `user` / `agent` 来源占用时仍跳过
- [x] 会话 checkpoint 接线：`session.jsonl` 逐轮追加 + `reset` 标记（压缩改写上下文时作废此前消息）。实跑验证：新进程打出 `已恢复会话：6 条消息`，人为注入半行坏 JSON 被静默跳过，模型不调工具答出上一轮读到的内容；三次运行后 `reset` 仍只有 1 条 → 无压缩时是纯追加
- [x] 复跑实验：修复后代码压力档 20+20 次，关闭机制逐字保留 552/800（69.0%）、开启机制 760/800（95.0%），**「上下文里仍在」开启组 800/800**；`summary_messages` 恒为 1
- [x] 自愈专项单测：`tests/test_constraint_self_healing.py` 10 条，覆盖全丢（补回 10 条）/ 部分丢（只补缺的）/ 不丢（摘要一字不改）
- [x] 约束注入 system prompt（收尾补的缺口）：`AgentLoop` 每轮重建 system prompt 并追加约束清单，多轮接力时按当前约束重建历史首条 system message。短任务实跑（verbose 打出 `压缩统计：未触发压缩`）`AGENTS.md` 的约束照样生效，改掉 C1 正文后新进程立刻看到新值；新增 `test_loop.py` 5 条 + `test_cli.py` 2 条
- [x] 全量 **451** 个单测通过（Day 5 收尾时 417；Day 6 收尾再补 7 条），`ruff check` 与 `ruff format --check` 干净

**可演示**：同一长任务开启 / 关闭约束保留的逐字保留率对比；`lite-agent chat` 跨进程续聊（`--verbose` 会打出恢复的轮数与 token）。

**卡点**：「实验扩大到 20 用例」没做成。实际做的是**同一档位重复 20+20 次**（提高的是样本量而不是用例多样性），行为探针仍然只有 3 项。要把用例多样性做上去，得先设计更多「可脚本化判定的约束类型」，落到 Day 7 或之后。

**收尾时又查出一个真缺口**：约束此前只在 Tier 4 压缩时进上下文，短任务看不到 `AGENTS.md` 的 C1——「存了但没生效」。已加 system prompt 通道（ADR-016）并实跑验证。这条是接线之后靠实跑发现的，单测和文档都盖不到。

**风险应对**：实验一度担心 Day 4 的结论被推翻——修复后对照组从 372/400 掉到 273/400。查清后发现是拆掉了一条隐式保留通道（旧摘要原文被反复复制），Day 4 那组数据高估了对照组，不是回归；两次复跑互相印证（详见 `docs/evidence.md` 用例五）。

**参考**：本日以接线与复跑为主，无新增参考实现。接线位置的选择依据 `docs/04-cli-session.md`（会话持久化）与 `docs/07-context.md` 第 670 行（CLAUDE.md 从 CWD 向上遍历目录树）。

---

## Day 7：MCP 客户端（已完成，子 Agent 未做）

**目标**：加分项，二选一。**最终选 MCP**——它有公开协议可对照（参考项目第 12 章），演示成本也比子 Agent 低；子 Agent 隔离顺延。

| 任务 | 预计耗时 | 验收 | 状态 |
|---|---|---|---|
| 同步 `docs/architecture.md` | 1.5h | 模块名与状态标记和代码一致 | ✅ 8 处 |
| 最小 echo MCP server | 0.5h | `initialize` / `tools/list` / `tools/call` 全通 | ✅ `examples/echo_mcp_server.py` |
| MCP 客户端 | 3h | JSON-RPC over stdio，工具发现 + 调用 | ✅ `mcp/client.py` |
| 接进工具注册表 + CLI | 1.5h | `--mcp-server` 启动 server 并注册工具 | ✅ |
| MCP 调用前过约束校验（自研） | 1h | 命中禁止类约束时请求不发出去 | ✅ ADR-018 |
| 子 Agent 隔离 | 4-5h | 独立上下文 + 工具白名单 | 📋 **评估后不做**（理由见下） |

**验收**：

- [x] `docs/architecture.md` 同步：`constraints.py` / `agents_md.py` / `session.py` 由 📋 改 ✅，模块图加 `mcp/` 子图，数据流图加约束注入分支与 MCP 工具约束校验，模块职责表补 `config.py` 与 `mcp/client.py`
- [x] 最小 MCP server：`examples/echo_mcp_server.py`，手写 stdio JSON-RPC，零依赖，只暴露一个 `echo` 工具；未知方法 / 未知工具分别返回 -32601 / -32602
- [x] MCP 客户端：`initialize` → `tools/list` → `tools/call` 三步流程，工具按 `mcp__<server>__<tool>` 注册；与参考项目的四处工程差异（请求登记时序、进程树终止、stderr 处理、超时收尾）见 ADR-017
- [x] CLI 接线：`--mcp-server "[NAME=]COMMAND"` 可重复；单个 server 连不上只警告并跳过。实跑 `lite-agent chat "用 echo 工具说 hello" --mcp-server "python examples/echo_mcp_server.py"` 打出 `MCP server「echo_mcp_server」已连接，发现 1 个工具`，工具调用成功
- [x] 执行层约束校验：`AGENTS.md` 写一条禁止调用 echo 工具的约束后，实跑中模型**仍然发起了** `mcp__echo__echo` 调用，被本地校验拦下（`请求没有发给 MCP server`），失败原因回填给模型
- [x] 全量 **488** 个单测通过（Day 6 收尾时 451），`ruff check` 与 `ruff format --check` 干净

**可演示**：`lite-agent chat "用 echo 工具说 hello" --mcp-server "python examples/echo_mcp_server.py" --verbose`；以及同一条命令在 `AGENTS.md` 有禁止性约束时的拦截效果。

**卡点**：JSON-RPC 握手 → 已解决。参考 `claude-code-from-scratch/docs/12-mcp.md`（对照段在第 575 行），实现对照 `python/mini_claude/mcp_client.py`。子 Agent 隔离参考 `docs/11-multi-agent.md`（对照段在第 634 行），实现对照 `python/mini_claude/subagent.py`；**评估后不做**。

**子 Agent 为什么不做**：**不是时间不够，是评估后判定没有增量**——它与「智能问数」项目在能力上重复，再花 4-5 小时做一遍「独立上下文 + 工具白名单」，简历上多不出来新东西。MCP 反而有公开协议可对照、演示成本低，所以二选一选 MCP。

**风险应对**：MCP 已完整可演示，「核心链路完整」这个结论不受子 Agent 影响。

---

## 风险与应对总表

| 风险 | 影响 | 应对 |
|---|---|---|
| Day 1 API 调试超时 | 阻塞后续 | 先用 mock LLM 跑通循环，再替换真实 API |
| Day 2 edit_file 出 bug | 工具不可用 | 先只支持唯一匹配，冲突时报错让模型重试 |
| Day 3 压缩调优超时 | 长会话崩 | 先只做 Tier 1 + Tier 4，Tier 2/3 后补 |
| Day 4 实验设计超时 | 核心卖点无数据 | 未超时；但中途撞上 API 余额中断，补了 `--resume` 从已有记录续跑 |
| Day 5 时间不够 | 简历级不达标 | 测试 > 文档 > GIF，欠债第 6 天补 |
| Day 6 欠债太多 | 文档不完整 | 优先约束实验和 decisions.md，MCP 放弃 |
| Day 7 时间不够 | 少加分项 | 不做，核心链路已完整 |

---

## 里程碑

- [x] **M0**：工程脚手架 + CLI 入口 + 文档（第 0 周完成，6 个 commit）
- [x] **M1**：Agent Loop 跑通一次工具调用（Day 1 结束，89 个单测通过）
- [x] **M2**：6 个工具可用，流式输出（Day 2 结束，212 个单测通过）
- [x] **M3**：4 层压缩能触发，长会话不崩（Day 3 结束，301 个单测通过，20 轮真实会话验证见 `docs/evidence.md`）
- [x] **M4**：约束保留机制有对比数据（Day 4 结束，357 个单测通过；压力档约束原文逐字保留 100% vs 93.0%，实验组 6/10 次遇到摘要整段丢 40 条、累计补录 280 条，见 `docs/evidence.md` 用例五）
- [x] **M5**：简历级完成，GitHub 完整（Day 5 结束，417 个单测通过；四层压缩两处技术债已修，`memory/agents_md.py` 与 `memory/session.py` 落地但尚未接入主流程，见 `docs/evidence.md` 用例六）
- [x] **M6**：前 5 天验收全打勾，评测规模扩大（Day 6 结束，451 个单测通过；**原计划的「20 用例」实际做成了「同档位 20+20 次复跑」**——提高的是样本量不是用例多样性，行为探针仍只有 3 项，这条欠债顺延 Day 7，见 Day 6 卡点）
- [x] **M7**：MCP 完成（Day 7 结束，488 个单测通过；MCP 客户端 + echo server + 执行层约束校验全部落地。**子 Agent 评估后不做**——与「智能问数」能力重复，理由见 Day 7「子 Agent 为什么不做」）

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