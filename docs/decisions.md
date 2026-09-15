# 设计决策记录（ADR）

记录项目中的关键技术决策，每条包含**背景 / 决策 / 理由**三部分，供面试追问时直接引用。
不记录尚未拍板的事项；未定项在架构文档中标 📋。

## ADR-001：不依赖 LangChain，自研 Agent Loop

**背景**：LangChain、LlamaIndex 等框架提供现成的 AgentExecutor，能快速跑通 demo，但把主循环、上下文管理、错误恢复都封装在内部。

**决策**：只用 LLM 官方 SDK 和标准库，自己实现 Agent Loop、工具调度、上下文压缩。

**理由**
- AgentExecutor 是黑盒，循环怎么终止、消息怎么组装、失败怎么重试都不可控，也无法解释。
- 本项目的差异化点是上下文压缩与约束保留，这恰好是框架封装最深的部分，魔改框架比自研更贵。
- 主循环本身只有几十行，复杂度不在循环，而在压缩策略和约束保留，框架帮不上忙。
- 依赖可控：不引入框架意味着版本升级不会打乱核心逻辑。

## ADR-002：压缩阈值选在上下文窗口的 60%

**背景**：四层压缩需要触发条件，阈值定得太高会在压缩后立即再次触发，定得太低会频繁压缩、损失信息。

**决策**：以模型上下文窗口的 60% 作为触发线，超过即启动压缩。

**理由**
- 预留 40% 缓冲，避免压缩后的上下文加上新一轮工具输出又立刻越线，形成抖动。
- 压缩本身要消耗 token 和一次 LLM 调用，频次过高反而更慢更贵。
- 60% 是经验起点，不是理论最优值。后续用实验校准，并允许按模型单独配置。
- 阈值可配置，不写死在代码里，方便做对比实验。

## ADR-003：项目记忆用 Markdown，不引入向量数据库

**背景**：Agent 需要项目级规则（编码规范、禁区、构建命令），常见做法是 RAG，把项目文档切块存进向量库再检索。

**决策**：直接读取 `AGENTS.md`，按目录层级加载，全文注入上下文，不做向量检索。

**理由**
- 项目规则是结构化短文本，总量在几千 token 级别，全文注入即可，检索反而增加不确定性。
- 规则有**层级语义**（子目录规则覆盖父目录），这是目录结构天然表达的，向量检索的相似度排序表达不了。
- 向量库引入 embedding 模型、存储、索引刷新等一整套依赖，收益与复杂度不成正比。
- 可解释：加载了哪些规则、来自哪个文件，一眼可见，便于调试和排查 agent 违规行为。
- 需要检索的是**代码**而非规则，代码检索用 grep 这类确定性工具更合适。

## ADR-004：关键约束独立存储，不参与压缩

**背景**：长任务中，压缩会把早期消息改写成摘要，用户早期设定的关键约束（如「禁止修改数据库迁移文件」）可能被摘要丢失，导致后续步骤违规。

**决策**：把约束抽取成独立结构，存入单独的 `constraints.json`，不进入会被压缩的消息流；压缩后校验约束 ID 完整性，缺失则重新注入。

**理由**
- 约束一旦进入被摘要的消息流，就变成「可能被改写」的数据，无法保证不丢失。
- 独立存储把约束从「尽力保留」变成「可校验」：压缩前后比对 ID 集合即可判定是否丢失。
- 校验发现问题后可以自愈（重新注入），形成闭环，而不是只依赖摘要 Prompt 的自觉。
- 这也让约束可观测：可以直接统计一次任务里约束的提取数、保留数、违反数，作为评测指标。
- 代价是多一份状态要维护，但相比长任务跑偏的返工成本可以接受。

## ADR-005：实现语言选 Python 而不是 TypeScript

**背景**：终端 Coding Agent 的主流选择里，Claude Code 是 TypeScript，多数开源复刻项目也用 Node。选语言会影响生态、分发和调试体验。

**决策**：主体用 Python 3.11，通过 `pyproject.toml` 打包，提供 `lite-agent` 命令。

**理由**
- 目标领域是 AI 应用，Python 的模型 SDK、tokenizer、评测工具链都是一等公民，TS 侧常有滞后的类型定义和功能缺失。
- 后续要做的约束提取、摘要质量评测都涉及文本处理与数据分析，Python 生态更顺手。
- 单文件分发不是本项目的核心诉求，放弃 Node 的启动速度优势可以接受。
- 协程模型上 Python 的 asyncio 足够支撑单会话工具调用，不需要 TS 的事件循环优势。
- 已知代价：跨平台路径处理、终端渲染不如 Node 生态成熟，需要额外注意 Windows 兼容。
- 若后续要做 IDE 插件或 Web 前端，可以用 TS 单独写，通过 MCP 协议与 Python 主体通信，不冲突。

## ADR-006：Agent Loop 的关键取舍（对照 claude-code-from-scratch）

**背景**：Agent Loop 是整个项目的心脏。参考项目 `claude-code-from-scratch` 的 `python/mini_claude/agent.py`（1951 行）给了成熟做法，但它把所有东西塞进一个 `Agent` 类：循环、工具分发、权限、压缩、流式混在一起。我们需要保留核心机制，同时让模块边界可单测。

**参考实现（第 1 章 `docs/01-agent-loop.md`）的关键事实**

| 维度 | 做法 | 位置 |
| --- | --- | --- |
| 终止条件 | 这一轮没有任何 `tool_use` / `tool_calls` 就 `break`；代码里没有任何按工具名分支的判断，循环转不转由模型决定 | `agent.py:1543`、`agent.py:1764` |
| 错误回填 | 工具层不抛异常，把失败变成字符串结果：`Error: ...`、`Unknown tool: ...`、`Warning: ...` | `tools.py:670-735` |
| 最大轮数 | `max_turns` 是可选参数，默认 `None`（不限轮数）；每轮 `current_turns += 1` 后调 `_check_budget()`，与 token 成本上限共用一套检查 | `agent.py:191,204,1548-1551` |
| tool_calls 保留 | assistant 消息原样进历史：Anthropic 存完整 content blocks；OpenAI 直接把响应里的 message dict 整个 append（含 `tool_calls`）；工具结果靠 id 配对回填 | `agent.py:1538-1541`、`agent.py:1761` |
| 超限时的一致性 | 预算或轮数超限时，先给每个未执行的调用补一条「未执行」结果再 `break`——缺了配对，下一轮请求的历史就非法 | `agent.py:1552-1560`、`agent.py:1773-1781` |

**决策与差异**
- 终止条件与参考实现同源：`if not response.tool_calls` 即返回，不做任何基于工具名的分支。
- 错误回填更早、更结构化：参考实现把参数 JSON 解析失败静默降级成空 `{}`（`agent.py:1792-1795`），模型只能看到二手的「缺少必填参数」。我们在 `Tool.run()` 里先拦 `JSONDecodeError`，直接告诉模型「参数不是合法 JSON」，Pydantic 校验失败再逐字段回报。
- **最大轮数默认收紧**：参考实现默认不限轮数，我们默认 `max_turns=10`，并返回 `stopped_reason="max_turns"`。理由是评测要跑固定任务集，需要可复现的调用次数上限，且非零退出码能让 CLI 明确报告任务未完成。
- **配对完整性天然成立**：我们的循环是「一轮内执行完全部 tool_call 并逐条回填」才进入下一轮或退出，不存在「assistant 消息已入历史但缺 tool 结果」的中间态，因此不需要像参考实现那样补占位结果。
- **依赖倒置（本项目最大的结构差异）**：参考实现把工具执行硬编码在 `Agent` 内部（`_execute_tool_call` 直接调 `tools.execute_tool`）。我们按约束 C4 让 `core` 不导入 `tools`：`core/loop.py` 只声明 `ToolExecutor` / `ToolOutcome` 两个 Protocol，由 `cli` 层注入 `ToolRegistry`。代价是多一层抽象；收益是 loop 的测试可以完全脱离工具实现，换一套工具实现也不需要改 core。
- 暂不实现参考实现的流式早期工具执行与多种错误恢复分支（第 5、15 章），属于工程加固，不在 Day 1 范围。

**后果**
- `core` 与 `tools` 可独立演进、独立测试；已用脚本核对三层导入关系：core 只依赖 core，tools 只依赖 tools，cli 是唯一装配层。
- Protocol 是隐式契约，`ToolRegistry` 若改了方法签名，静态检查不一定能发现，需要测试兜底。

**Claude Code 原始设计**：Loop 分两层——外层 `QueryEngine`（约 1155 行）管对话生命周期（用户输入、USD 预算、Token 统计、会话恢复），内层 `queryLoop`（约 1728 行）管一次查询（消息压缩、API 调用、工具执行、错误恢复）；内层是异步生成器（`async function*`），用背压替代回调、用普通 `continue`/`break` 表达控制流。循环「继续」的原因有 7 种（`next_turn`、`collapse_drain_retry`、`reactive_compact_retry`、`max_output_tokens_escalate`、`max_output_tokens_recovery`、`stop_hook_blocking`、`token_budget_continuation`），只有第 1 种是「模型调了工具」。可恢复错误先扣住不抛、跑完恢复再决定是否暴露；`StreamingToolExecutor` 在响应还没流完时就开跑已解析完的工具。
**参考项目复现**：只实现第 1 种继续原因，两层拆解、错误扣留、流式提前执行全部省略。`docs/01-agent-loop.md` 第 229 行起「真实 Claude Code 比这多做了什么」把这些列为「玩具循环和生产级引擎之间的距离」，并注明结构细节来自公开版本分析，官方文档只坐实 `tool_use` / `tool_result` 回路本身。
**本实现差异**：在参考项目的单层循环上再切一刀——`core/loop.py` 不持有工具，只声明 `ToolExecutor` / `ToolOutcome` Protocol，由 `cli` 注入 `ToolRegistry`（约束 C4）；另把参考实现默认不限轮数（`max_turns=None`）收紧为默认 10，并返回 `stopped_reason` 供 CLI 判定非零退出码。

## ADR-007：流式输出默认开启，并接受 httpcore2 关闭流时的已知噪音

**背景**：Day 2 给 CLI 加上流式输出：模型增量文本逐字写 stdout，工具调用与结果实时写 stderr。实现后发现，多轮流式调用会在进程退出阶段向 stderr 打印一段 traceback：`RuntimeError: generator didn't stop after athrow()`。根因在 `httpcore2/_utils.py` 的 `safe_async_iterate`——它在 `finally` 里 `await iterator.aclose()`，而该 async generator 此时正因 GeneratorExit 被关闭，await 一旦挂起，CPython 就判定「generator didn't stop」。消息由 CPython 的 async generator finalizer 通过 `PyErr_WriteUnraisable` 打出，不属于本项目的调用栈。

**决策**：保留流式输出，不为此加全局兜底。已做两件正确但不足以消除噪音的事：`OpenAICompatProvider.chat_stream` 在 `finally` 中显式关闭 SSE 流；`aclose()` 保证 HTTP 客户端在事件循环仍然存活时关闭。明确不引入 `sys.unraisablehook` 全局屏蔽。

**理由**：
- 这是依赖方缺陷，不是本项目逻辑错误：exit code、stdout、文件改动全部正确，噪音只出现在退出阶段。
- 全局 `unraisablehook` 会连真实的未处理异常一起吞掉，为一行噪音牺牲可观测性不划算。
- 触发条件已定位为「多轮流式」：单轮流式、以及多轮非流式都干净。因此换 provider 或把 `openai` 降级到使用 httpx 的版本都是候选方案，但要先验证再改依赖。
- 已用最小脚本复现（两次顺序流式调用即可稳定触发），排除偶发。

**结果**：演示多轮任务时结尾会多出一段与被演示功能无关的 traceback，需在演示说明中标注，避免被误判成程序 bug。待办：确认上游是否修复；若影响演示，再评估 `openai<3` 或自定义 transport。

## ADR-008：edit_file 用「唯一性校验 + read-before-edit + mtime 防护」三重约束

**背景**：模型改文件时最容易犯两类错：一是凭记忆写出文件里并不存在的原文，二是要替换的片段在文件中出现多次、结果改错了地方。参考实现 `claude-code-from-scratch/python/mini_claude/tools.py` 的 `_find_actual_string`（第 265 行）与 `_edit_file`（第 290 行）覆盖了这两类错误的一半：0 次报 `old_string not found`，多次报 `found N times, must be unique`，并额外做了弯引号到直引号的归一。

**决策**：保留参考实现的匹配、报错语义与引号容错；read-before-edit / mtime 防护沿用其思路但换了落点；唯一性报错额外补上它没有的位置信息。
- 唯一性校验：0 次报「未找到」并提示先 `read_file` 核对缩进与换行；多次报「出现了 N 次」并**列出每次出现的起始行号**，要求扩展上下文使其唯一。
- read-before-edit：`read_file` 成功时把文件快照写进 `FileTracker`；`edit_file` 拿不到快照直接拒绝（参考实现在 `executeTool` 分发器里用 `readFileState` Map 做同一件事）。
- mtime 防护：写入前比对 `mtime_ns` 与字节数，读后被外部改过则要求重读。
- 引号归一：沿用参考实现的思路，但只在归一确实改变了字符串时才做二次匹配，省掉无谓扫描。

**理由**：
- 参考实现的报错只说「出现了 N 次」，模型拿不到位置信息，只能盲目扩大 `old_string` 反复试；给出行号能让它一次改对。
- 参考实现把 read-before-edit 与 mtime 防护写在分发器里而非工具里（`docs/02-tools.md` 第 915 行起「Read-before-edit + mtime 防护」，源码是 `executeTool` 中的 `readFileState: Map<绝对路径, mtimeMs>`）。落点不同带来的是耦合方向不同：分发器统一检查要求所有写操作都经过它，而检查下沉进工具后 `edit_file` 单独构造也是安全的，代价是必须显式共享 `FileTracker`。
- 差异刻意控制在「更严」而不是「更聪明」：不擅自改写 `new_string`，引号只用于定位、写入保持原样，避免静默篡改用户内容。

**结果**：`tests/test_edit_file.py` 22 个用例覆盖唯一匹配、多次匹配、未找到、未读先编辑、mtime 变更、引号归一、路径越界等。代价是多了一个 `FileTracker`，必须在 `build_default_registry` 里显式共享，否则读写工具各持一份、read-before-edit 会永远失败。

**Claude Code 原始设计**：编辑验证是一条 14 步流水线，配合 `readFileTimestamps` 机制保证「编辑必须基于已知状态，不能盲写」；工具结果是三级大结果限制。
**参考项目复现**：把 14 步压成五项——引号容错 + 唯一性 + diff + read-before-edit + mtime（`docs/02-tools.md` 第 1181 行「我们的简化决策」表列出这条压缩，第 915 行起给出 read-before-edit + mtime 的实现）。匹配层面 `_find_actual_string`（tools.py:265）只做「精确匹配 → 引号归一后再匹配」，命中后返回文件中的原始字符串；`_edit_file`（tools.py:290）把 0 次与多次都转成错误字符串，宁可失败也不猜。
**本实现差异**：规则子集与参考项目一致，三处不同——① 多次匹配时列出每次出现的行号，参考只给次数，模型只能盲目扩大上下文重试；② 状态检查从分发器下沉进工具（`FileTracker` 注入 `ReadFileTool` / `EditFileTool`），且比对 `(mtime_ns, size)` 而不只是 mtime；③ `write_file` 改为「已存在即拒绝覆盖」，参考只要求已存在文件先读后即可覆盖，我们把「改已有文件」完全推给 `edit_file`。

## ADR-009：bash 超时改为终止整棵进程树

**背景**：第一版直接用 `subprocess.run(command, shell=True, timeout=T)`，与参考实现 `tools.py:424` 的 `_run_shell` 一致。实测发现超时形同虚设：Windows 上 `subprocess.run` 超时后只终止 `cmd.exe`，真正的子进程（例如 `python -c "time.sleep(60)"`）继续存活并占着 stdout/stderr 管道，`communicate()` 必须等到管道关闭才返回。表现为设了 `timeout_seconds=2` 的命令实际耗时 30 秒——正好是测试里子进程的睡眠时长，整个测试套件被拖到 31.5 秒。

**决策**：不再直接用 `subprocess.run`，改为 `Popen` + 手动等待 + 整棵进程树终止。
- 启动时让子进程进入独立进程组/会话：Windows 用 `CREATE_NEW_PROCESS_GROUP`，POSIX 用 `start_new_session=True`。
- 超时后按平台终止整棵树：Windows 走 `taskkill /F /T /PID`，POSIX 走 `os.killpg(os.getpgid(pid), SIGKILL)`，失败再退回 `process.kill()`。
- 终止后再给 `communicate(timeout=5)` 一次机会回收管道，然后才抛错，由上层转成 `ToolError` 回填给模型。

**理由**：
- 只杀 shell 等于没杀：`shell=True` 下 shell 只是外壳，真正执行命令的是它的子进程，`Popen.kill()` 只作用于前者。
- 参考实现有同样缺陷，但它的场景是交互式 REPL、单次调用，超时被拖长不易察觉；本项目要在测试里断言超时行为，问题立刻暴露。
- 顺带修了配套的编码问题：Windows 中文环境下 `cmd` 内建命令输出 GBK、`git` 等工具输出 UTF-8，只认一种必然有一半乱码，改为 UTF-8 优先、失败退回 `locale.getpreferredencoding()`。

**结果**：超时真正生效，测试套件从 31.5 秒降到约 6 秒；`test_timeout_actually_stops_the_command` 用计时断言（<20s）把这条行为钉住，防止将来回退。

**Claude Code 原始设计**：Shell 安全靠 AST 解析 + 沙箱，而不是正则匹配；工具结果按「选择性裁剪 + 磁盘持久化」处理，而不是单层截断。
**参考项目复现**：降级为「正则匹配 + 确认」——`DANGEROUS_PATTERNS` 命中后交由权限层裁决；`_run_shell`（tools.py:424）用 `subprocess.run(shell=True, capture_output=True, text=True, timeout=...)`，超时只捕获 `TimeoutExpired` 并返回 `Command timed out after {ms}ms`，不处理子进程树（见 `docs/02-tools.md` 第 766 行 run_shell 与第 1194 行起的简化对比表）。
**本实现差异**：保留正则匹配的路子，但把超时做实——`Popen` 启动时让子进程进入独立进程组/会话，超时后按平台终止整棵树（Windows `taskkill /F /T /PID`，POSIX `killpg(SIGKILL)`），再回收管道后才报错；另补了输出解码回退（UTF-8 优先、失败退回本地编码），因为 Windows 中文环境下 cmd 内建命令输出 GBK 而 git 输出 UTF-8。

## ADR-010：上下文压缩的阈值、估算方式与可观测指标

**背景**：ADR-002 定了「窗口 60% 触发压缩」，但没有说明压缩分几层、各层怎么触发、怎么判断压缩是否有效。Day 3 要落地四层压缩，先把这些量定下来——否则做完又是一份只有实现、没有数据的模块。

**决策**：
- **阈值**：Tier 1~3 的入口线是上下文窗口的 **60%**（沿用 ADR-002）；Tier 4 全量摘要是 **85%**，只在前面几层压不动时才动用。
- **Token 估算**：字符数 / 4。不引 tokenizer——C2 只允许标准库与 LLM 官方 SDK，而这里只需要判断「该不该压缩」，不需要精确计数。
- **摘要保留内容**：关键约束（为 Day 4 预留的字段，现在可以为空）、关键决策、未完成任务、涉及的文件路径。摘要以 system message 注入。
- **保留窗口**：任何一层压缩后都保留最近 **10 条**消息，保证当前这一步的上下文不被抹掉。
- **可观测指标**：每层的触发次数、压缩前后的 token 数、压缩耗时（毫秒）。

**理由**：
- 60% 与 85% 拉开两档，是让「便宜的局部截断」先跑、贵的全量摘要当兜底；两个阈值挨太近会导致压缩完立刻又触发，形成抖动。
- 估算用字符数 / 4 而非精确 tokenizer：中文与代码的字符/token 比不同，误差可能到 ±30%，但判断「有没有越过 60%」不需要那么准，而引 tokenizer 会带来新依赖与冷启动成本。
- 「关键约束」字段现在就写进摘要 Prompt，哪怕值为空：Day 4 的约束保留机制要依赖这个字段，先建好通道，Day 4 直接接校验即可，不必再改 Prompt 结构。
- 记录触发次数与 token 数，是为了 Day 4 的对比实验能复用同一套指标，避免重复搭埋点。

**结果**：压缩行为可配置、可观测。代价是估算不精确，可能提前或滞后触发压缩；这一点在 Day 3 的 20 轮验证里量化。

**Claude Code 原始设计**：5 级流水线——Tool Result budget（含磁盘持久化）→ History Snip → Microcompact（冷热缓存双路径）→ Context Collapse 投影 → Autocompact（约 85.5% 触发，两阶段「分析-摘要」）。
**参考项目复现**：4 层压缩（Budget 双阈值 50%/70%、Snip >60%、Microcompact 空闲 5 分钟、Auto-compact >85%），另有第 0 / 0.5 层在工具层做执行期截断与落盘，合计 6 层（`claude-code-from-scratch/docs/07-context.md` 第 165 / 204 / 237 / 291 / 317 / 349 行，对照段在第 660 行）。
**本实现差异**：层数同为 4 层、分层职责一一对应（Tier 1~4 ↔ Budget / Snip / Microcompact / Auto-compact），但四处不同——① 入口线统一为 60%，参考的 Budget 用的是 50%/70% 双阈值；② Tier 3 比参考多一条「利用率 ≥ 60%」的门，避免上下文宽裕时做无谓清理（实测会被 Tier 2 抢先压到线下而跳过）；③ 没有落盘那一层，大结果靠 `read_file` / `bash` 的 2000 行截断兜底；④ Tier 4 加了两条参考没有的护栏——切点不落在 tool 结果上（否则 assistant 与 tool 消息会失去配对，API 直接报错），以及只剩上一次摘要时不再压（否则保留窗口本身超线时会逐轮抖动）。
## ADR-011：grep 不调用系统 grep，改用纯 Python 遍历

**背景**：Claude Code 的搜索工具走 ripgrep（`rg`）——快、默认遵守 `.gitignore`、支持 `.ignore` 文件，这是它能在大型仓库里同时做到「搜得快」和「搜得干净」的基础。参考项目退了一档：优先调系统 `grep -r`，只在系统没有 grep 时才退回一个很薄的 Python walker（`claude-code-from-scratch/docs/02-tools.md` 第 704 行有独立的 `grep_search` 设计段，第 764 行明说「Claude Code 用 ripgrep，我们用系统 grep——功能够用，少一个依赖」）。它的忽略集合是写死的 `node_modules` 与 `.git`，而且只在 fallback 路径生效——走系统 grep 时不忽略任何东西。Day 2 实现 grep 时面临同样的选择。

**决策**：用纯 Python 遍历（`os.walk` + `re`）自己实现，不调系统 grep；忽略集合集中在 `agent/tools/ignore.py` 的 `IGNORED_DIRS`，由 grep 与 list_dir 共用。

**理由**：
- **少一个外部依赖**：不必假设目标机器装了什么版本的 grep，也不必维护「系统 grep + Python fallback」两条代码路径——参考项目正是这么分裂的，结果是忽略逻辑只在其中一条路径生效，同一个工具的行为随环境改变。
- **忽略集合可自定义**：`IGNORED_DIRS` 就是一个 `frozenset`，要加 `.ruff_cache`、`node_modules` 或某个业务目录改一行即可；改成调系统 grep 的话，这些规则得靠 `--exclude-dir` 逐个拼，还要赌对方支持这些参数。
- **跨平台一致**：Windows 上没有 grep（Git Bash 之外），而 `os.walk` 处处相同。本项目在 Windows 上开发、在 ubuntu 上跑 CI，只有统一走 Python 才能保证两边行为一致。
- **代价可控**：搜索范围限定在工作区，结果本来就要截断到 100 条，Python 遍历的性能劣势在这个量级上可以接受。

**结果**：grep 的行为完全由本项目决定，不随环境漂移，测试也能直接断言「跳过了哪些目录」。代价是大型仓库上比 ripgrep 慢；若将来需要提速，可以换成 `rg` 并复用同一套 `IGNORED_DIRS`，不必改动上层接口。

## ADR-012：约束保留机制（压缩前注入 + 压缩后校验 + 自愈）

**背景**：ADR-004 定了「约束独立存储、可校验」，但没有说注入与校验具体怎么做。Day 4 落地时撞到两个具体问题：① 摘要 Prompt 第 4 项只写「关键约束（如有）……逐条原样保留」，没告诉模型具体是哪几条，模型完全可以写一句「已保留相关约束」交差，校验也无从下手；② 模型会把整段约束丢掉——实测标准档 10 次里 1 次，压力档 10 次里 6 次（40 条约束、25 轮会话），只靠 Prompt 是概率性保证（见 `docs/evidence.md` 用例五）。

**决策**：
- **压缩前吸收**：`Compactor.compact()` 在任何一层压缩之前，先扫描消息里的 `[CONSTRAINT] 代号 X：内容` 声明收进存储并落盘。只认 `role == "user"` 的消息。
- **摘要前注入**：把清单（`- [ID] 原文`）逐条附在摘要指令末尾，替代原来那句泛指的「关键约束（如有）」。
- **摘要后校验**：拿摘要文本逐条比对约束 id，缺的按原文补录成 `[约束补录]` 块接在摘要正文后面，补录条数计入 `CompactionStats.replenished`。
- **校验认 id，不认正文**；`constraints=None` 时三条路径全部关闭，行为与从前完全一致。

**理由**：
- 只要求「保留约束」不够。模型不知道具体是哪几条时，摘要里容易只剩一句「已保留相关约束」——这句话在后续轮次里没有任何可执行信息。把 id 与原文一起给它，校验才有依据。
- 校验认 id 是刻意的：正文会被摘要改写，逐字比对必然误报；id 是我们自己生成的稳定锚点。代价是「模型写了 id、却把正文写歪」抓不到——这条已知边界有单测固定住（`test_verification_is_id_based_so_garbled_content_is_not_detected`），不假装它不存在。
- 自愈用追加而不是替换：不覆盖模型写的摘要，只把漏掉的原文补在后面。追加块永远在尾部，模型读得到；替换会让摘要失去连贯性。
- 吸收只认用户消息：助手消息里的同款文本多半是复述或举例，算成约束会误伤。同一 id 重复出现时「先到先得」而不是报错——扫的是自由文本，重申同一条约束是常态，为此抛异常会打断整轮对话；`add()` 仍保持严格，那是显式写入。
- 不接存储时行为不变，对照组才能干净地只改一个变量。

**三层对照**
**Claude Code 原始设计**：Autocompact 是 5 级流水线的最后手段，fork 子 Agent 调 API 生成摘要，提示词走「分析-摘要」两阶段，产出 9 段标准化 `<summary>`，最后剥离推理只留摘要（`claude-code-from-scratch/docs/07-context.md` 第 688 行）。约束没有独立的存储与校验环节，靠「分节模板不丢内容」保证。
**参考项目复现**：摘要提示词只有一句 `Summarize the conversation so far...`，system 侧加一句 `Be concise but preserve important details.`（`docs/07-context.md` 第 480 / 515 行）。没有约束概念，也没有压缩前后的校验。
**本实现差异**：在四层压缩之外单独开了一条「约束通道」——独立存储（ADR-004）+ 压缩前吸收 + 摘要时逐条注入 + 摘要后按 id 校验并补录。相对参考项目，是把「摘要要保留重要细节」这句祈使句换成可枚举、可校验、可自愈的机制；相对 Claude Code，是把「靠分节模板不丢」换成「不依赖模型自觉」。

**结果**：见 `docs/evidence.md` 用例五。标准档（15 条 / 10+10）两组打平；压力档（40 条 / 10+10）出现差异——逐字保留实验组 400/400、对照组 372/400（93.0%），对照组 10 次里有 5 次丢掉 1~7 条约束**原文**而代号仍在，实验组 10 次里有 6 次遇到摘要**整段丢掉全部 40 条**、累计补录 280 条。行为探针（JSON / snake_case / 无代码围栏）两组仍打平，因为 3 项检查只覆盖 40 条约束中的 3 条。**机制买到的是「约束原文不会丢」这条保证，不是行为指标的提升。**

代价是每次摘要请求多一段清单（15~40 行），以及多一份要维护的 `constraints.json`。另外要注意：逐字保留率的两组差距（100% vs 93.0%）来自「补录」，而补录只在摘要真丢东西时才发生——也就是说这个数字衡量的是「模型有多不稳」，不是「机制有多好」；机制的确定性在于，它把这份不稳从结果里消除了。