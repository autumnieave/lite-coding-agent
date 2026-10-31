# Benchmark 审计（只读）

**审计日期**：2026-10-19  
**审计对象**：`scripts/benchmark.py` 中 `build_tasks()` 定义的 12 个任务，以及已落盘的 benchmark 结果与 trace。  
**审计边界**：只读代码、结果和 trace；本文件不改任务定义、不改约束、不放宽检查项，也不构造补充诊断集。修正项只登记，后续单独提交。

## 0. 结论先行

1. `docs/evidence/benchmark_runs.jsonl` 共 48 条记录：`on` 36 条、`off` 12 条；`on` 的内容判定为 36/36，通过 `task_success` 为 31/36。5 次失败全部只挂在 `within_steps`，不是约束违反。
2. 5 次超预算为：A3 run2（8 步）、B2 run3（7 步）、B3 run2（11 步）、B4 run2（8 步）、B4 run3（9 步）。按最小必要路径估计，没有一次是“预算低于最小必要步数”；主要是重复 `bash`/探路与错误恢复绕路。B4 的 6 步在当前 Windows shell 试探下是偏紧的次因，但不能单独归因于预算。
3. 现有完整 trace 只覆盖 `C1/C2/C3 off`：`private/traces/benchmark/standard/off` 共 90 个 JSONL（3 个任务 × 3 个 run × 10 个 rollout）。`on` 档没有历史 trace，`C4` 也没有对应 trace；因此不能用现有 evidence 声称 `on` 档已有“初始 prompt → 多轮工具调用 → 最终输出”的完整轨迹。
4. 当前 12 个任务中**没有可直接进入约束遵守训练候选的任务**。A/B 类没有约束；C1/C2 只有格式检查（或上下文存在检查）；C3 的召回判定是宽松子串；C4 存在未声明的一处冲突：probe 要求 `notes/Summary.md` 这类大小写混合文件名，约束要求“新建文件名必须全小写”，但判定只接受任意小写新文件。
5. `constrained decoding` 的本地脚本与单元测试已存在；首次 Kaggle 任务因模型路径不存在 `ERROR`，0 次 rollout、无输出。本审计不编造 `json_valid` / `snake_case_keys` 数字，记作 **N/A（基础设施失败）**。

## 1. 证据来源与口径

| 证据 | 用途 | 关键事实 |
| --- | --- | --- |
| `scripts/benchmark.py` | 任务定义、判定函数、运行与落盘 | `build_tasks()` 在 558-765 行；判定函数在 421-528 行；`run_task()` 在 853-1069 行 |
| `docs/benchmark.md` | 设计约定与已拍板修订 | §5.1 只有 JSON 放宽、B4 预算 4→6、A/B `max_turns` 6→8、示例值修正等；不是难度重标定 |
| `docs/evidence/benchmark_runs.jsonl` | 12 任务 × 3 次历史汇总 | 48 行；字段含 `checks`、`steps`、`tools_used`、`max_turns`、`reply_head`，**没有完整 messages** |
| `private/traces/benchmark/standard/off` | 现有完整轨迹 | 90 个 JSONL，只含 C1/C2/C3 off；每个 trace 有 `run_start`、`llm_turn`、`run_end` |
| `training/data.py` | 训练数据导出格式 | `final_turn()` 只找 `probe_json` / `probe`，`build_example()` 只导出最终轮 system+user 与 completion |

口径说明：

- `steps` 是工具调用次数，不是 LLM 轮数；`max_turns` 是安全网，不是 `steps_limit`。
- `within_steps=false` 只表示超过自定步数预算，永远不进入约束违反根因表。
- `constraint_kept` 检查的是最终上下文里是否还有约束代号，是“上下文保留”而不是“模型遵守”。
- 判定证据应以当轮实际发送的 `messages` 和工具参数/结果为准；本审计不靠 `reply_head` 下根因结论。

## 2. 逐任务决策表

### A1

- **设计目的**：用 `grep` 定位 `render_failure` 定义的文件与行号，考察工具选择、参数准确和检索结果判读。
- **约束组合**：无。A/B 类不声明约束，`constraints_total=0`；“是否可同时满足”不适用。
- **检查项**：`correct_file`（语义，文件名出现）、`correct_line`（语义，行号出现）、`no_decoy_line`（语义，排除干扰行号）。缺失：没有把 `tool_selection_ok` / `params_ok` 纳入 `checks`。
- **步数预算与最小必要步数**：预算 6；最小约 1 次 `grep`。本任务没有出现在 5 次超预算失败中。
- **完整轨迹**：任务脚本只有一个 `probe` 用户轮；若重新开 trace，可有内部多轮工具调用，但现有 `on` 档没有完整 messages/tool trace，只有 `benchmark_runs` 汇总和 `reply_head`。
- **难度是否随系统侧优化重新校准**：否。未发现难度分层或系统优化后的任务重标定；只有判定与轮数上限调整。
- **决策**：保留为回归任务；不作为约束遵守训练样本。

### A2

- **设计目的**：用 `list_dir` 列出 `src/agent/core` 下的 `.py` 文件并按字母序输出 JSON，考察工具约束、格式输出和集合完整性。
- **约束组合**：无。
- **检查项**：`json_valid`（格式，使用宽松 JSON 解析）、`files_complete`（语义，列表与期望完全相等）、`sorted`（语义，按小写排序）。缺失：工具名/参数没有硬检查；`params_ok` 只是记录字段。
- **步数预算与最小必要步数**：预算 3；最小 1 次 `list_dir`。本任务没有超预算失败。
- **完整轨迹**：只有 1 个 `probe` 用户轮；历史 `on` trace 不存在，不能声称已有完整链路。
- **难度是否随系统侧优化重新校准**：否。`docs/benchmark.md` §5.1 的“宽松 JSON”是判定校准，不是难度校准。
- **决策**：保留为回归任务；不作为训练候选。

### A3

- **设计目的**：用 `grep` 在 `notes/` 中找同时含两个随机 token 的文件，考察双条件检索与干扰项排除。
- **约束组合**：无。
- **检查项**：`files_exact`（语义，回复中出现的候选文件名集合必须恰好等于目标集合）。缺失：没有硬检查 `required_tools` 和参数；没有过程检查。
- **步数预算与最小必要步数**：预算 6；理论最小 1 次 `grep`（正则 alternation `left|right` 可同时看到两 token；保守估计 2 次），远低于预算 6。
- **5 次超预算中的表现**：run2 实际 8 步，工具序列为 `grep, grep, list_dir, bash, bash, bash, bash, read_file`；内容判定 `files_exact=true`，即“做对了但绕路”。这不是预算低于最小必要步数。
- **完整轨迹**：1 个 `probe` 用户轮；历史 `on` trace 不存在。
- **难度是否随系统侧优化重新校准**：否。
- **决策**：保留为回归任务；不作为训练候选。若以后要评工具效率，应把 `steps` 与 shell 探路分开统计，而不是先调难度。

### A4

- **设计目的**：读取 `README.md` 并回答第 3 个二级标题，考察单文件读取和精确文本抽取。
- **约束组合**：无。
- **检查项**：`heading`（语义，标题文本子串）。缺失：没有检查读取参数、文件路径、是否重复读取；`params_ok` 未入门禁。
- **步数预算与最小必要步数**：预算 2；最小 1 次 `read_file`。本任务没有超预算失败。
- **完整轨迹**：1 个 `probe` 用户轮；历史 `on` trace 不存在。
- **难度是否随系统侧优化重新校准**：否。Q1 只讨论 1 次读取 + 1 次允许重读，属于预算讨论，不是难度重标定。
- **决策**：保留为回归任务；不作为训练候选。

### B1

- **设计目的**：对 `notes/part01.txt` 做单行定点替换且不改其他行，考察 `read_file` → `edit_file` 的参数准确性和文件状态。
- **约束组合**：无。
- **检查项**：`only_one_line_changed`（语义/文件状态，逐行比较）、`token_replaced`（语义/文件状态，目标行 token 替换）。缺失：没有把工具参数合法性、工具选择、错误恢复纳入 `checks`。
- **步数预算与最小必要步数**：预算 6；最小约 2 次（`read_file` + `edit_file`）。没有超预算失败；run2 恰好 6 步。
- **完整轨迹**：1 个 `probe` 用户轮；历史 `on` trace 不存在。
- **难度是否随系统侧优化重新校准**：否。
- **决策**：保留为回归任务；不作为训练候选。

### B2

- **设计目的**：新建 `notes/summary.md` 并写入文件数量 JSON，考察目录统计、文件创建和 JSON 内容。
- **约束组合**：无。
- **检查项**：`file_created`（文件状态）、`json_valid`（格式，宽松解析）、`count_correct`（语义，`file_count` 等于文件数）。缺失：没有参数/工具硬检查；没有检查是否绕路统计。
- **步数预算与最小必要步数**：预算 6；最小约 2 次（`list_dir` + `write_file`）。
- **5 次超预算中的表现**：run3 实际 7 步，工具序列为 `bash, list_dir, bash, bash, bash, bash, write_file`；内容三项全真。属于先做额外 shell 统计再写文件，不是最小路径不存在。
- **完整轨迹**：1 个 `probe` 用户轮；历史 `on` trace 不存在。
- **难度是否随系统侧优化重新校准**：否。示例值 `{"file_count": 12}` 改为固定占位 `7`，属于防照抄校准，不是难度。
- **决策**：保留为回归任务；不作为训练候选。

### B3

- **设计目的**：把 `notes/part02.txt` 整体改成一行，测试 `write_file` 对已存在文件拒绝覆盖后的恢复路径，期望转向 `edit_file`。
- **约束组合**：无。
- **检查项**：`content_replaced`（语义/文件状态，内容 strip 后等于目标 token）；软观测 `write_file_refused`、`used_edit_file` 只进入 `observations`，不进入 `checks`。缺失：`error_recovery` 没有硬判定，成功也可能只是绕路完成。
- **步数预算与最小必要步数**：预算 6；最小约 2 次（`read_file` + `edit_file`；`write_file` 拒绝只是应触发的恢复分支，不是完成任务的必要步骤）。
- **5 次超预算中的表现**：run2 实际 11 步，`hit_max_turns=true`，工具序列中出现 5 次 `bash`；内容判定仍为真。不是预算低于最小必要步数，而是错误恢复/探路把轮数吃满。
- **完整轨迹**：1 个 `probe` 用户轮；历史 `on` trace 不存在；B3 的 `write_file_refused` 只能从汇总 observations 看到，不能还原完整参数与错误回填。
- **难度是否随系统侧优化重新校准**：否。
- **决策**：修正后再评估。至少要区分“通过”与“通过且真的走了 write 拒绝→edit 恢复”；当前保留为回归任务，不作为训练候选。

### B4

- **设计目的**：用 `bash` 统计 `notes/` 下所有 `.txt` 文件总行数并输出 JSON，考察 shell 工具选择、Windows 环境适配和格式输出。
- **约束组合**：无。
- **检查项**：`json_valid`（格式，宽松解析）、`total_lines_correct`（语义，`total_lines` 数值）。缺失：没有检查 `bash` 参数是否合法、是否只做一次统计、是否反复探测 shell。
- **步数预算与最小必要步数**：预算 6；最小 1 次 `bash`（一次命令完成统计）。
- **5 次超预算中的表现**：run2 实际 8 步、run3 实际 9 步，均为内容判定全真但 `within_steps=false`；工具序列大量重复 `bash`。6 步相对最小 1 步不低，但在当前 Windows shell 试探下偏紧，预算只能算次因。
- **完整轨迹**：1 个 `probe` 用户轮；历史 `on` trace 不存在。
- **难度是否随系统侧优化重新校准**：否。B4 只做过 4→6 的预算调整（§5.1）；没有任何难度分层或平台适配后的重标定。
- **决策**：修正。应把 shell 试探步数与内容正确性分列统计，再决定预算；当前保留为回归任务，不作为训练候选。

### C1

- **设计目的**：在 12 轮填充上下文后仍输出合法 JSON，考察约束保留与格式遵守。
- **约束组合**：`facts.constraint_code`（运行时生成的 6 位 ID），原文为“输出必须是合法 JSON”；单独约束可同时满足。
- **检查项**：`json_valid`（格式，严格 `parse_json`）、`constraint_kept`（上下文状态，只检查代号是否仍在 `context_blob`）。缺失：没有任何 summary 语义检查；`constraint_kept` 不是行为遵守。
- **步数预算与最小必要步数**：无 `steps_limit`；任务脚本为 `declare + 12 fill + probe`，最小工具调用约 12 次读取。
- **5 次超预算**：不适用；C 类没有步数上限，5 次失败全部来自 A/B。
- **完整轨迹**：C1 off 有 10 个 rollout 的完整 trace（`private/traces/benchmark/standard/off/C1_*`）；C1 on 没有历史 trace。结构上可产出 declare→12 fill→probe 的多用户轮轨迹。
- **难度是否随系统侧优化重新校准**：否；没有 token 长度阶梯或难度分层。
- **决策**：只做回归；不进训练（除非补充 summary 语义检查后重新评估）。

### C2

- **设计目的**：在 12 轮填充后同时保留 5 条输出约束，考察多约束格式遵守与约束召回。
- **约束组合**：5 个运行时生成 ID，对应原文：
  - “输出必须是合法 JSON”
  - “JSON 的键名必须使用 snake_case”
  - “输出中不得出现 markdown 代码围栏”
  - “不得在输出中使用 emoji”
  - “列表必须按字母序排列”

  五条可以同时满足；它们不是互相排斥的规则。
- **检查项**：`json_valid`（格式）、`snake_case_keys`（格式）、`no_code_fence`（格式）；`constraint_kept` 不是 C2 判定，`probe_codes` 也没有进入 `_judge_c2`。缺失：emoji 检查和列表字母序检查完全没有脚本判定；约束召回没有纳入 task success。
- **步数预算与最小必要步数**：无 `steps_limit`；脚本为 `declare + 12 fill + probe_json + probe_repeat + probe_codes`，最小工具调用约 12 次读取。
- **5 次超预算**：不适用。
- **完整轨迹**：C2 off 有完整 trace；C2 on 没有历史 trace。现有 90 个 off trace 中 C2 有 30 个 rollout。
- **难度是否随系统侧优化重新校准**：否。
- **决策**：只做回归；不进训练。若作为训练样本，至少要为 emoji、列表字母序和约束代号召回补上可判定断言。

### C3

- **设计目的**：在长上下文后召回第 1 轮随机串，考察长上下文召回与约束保留。
- **约束组合**：`facts.constraint_code`，原文为“复述第 1 轮内容时必须原样包含 `{recall_token}`”；单约束可满足，但 probe 直接问“第 1 轮给你的那个随机串是什么”，而随机串本身就在第 1 轮的 declare 文本里，形成明显的自我提示/同义反复。
- **检查项**：`token_recalled`（语义/召回，`recall_token in reply`，子串判定偏松）、`constraint_kept`（上下文状态，只看代号是否存在）。缺失：没有检查“只回答随机串、不解释”的结构要求，也没有区分“正确召回”与“在拒绝/解释中夹带 token”。
- **步数预算与最小必要步数**：无 `steps_limit`；最小约 12 次读取（12 个 fill）。
- **5 次超预算**：不适用。
- **完整轨迹**：C3 off 有完整 trace；C3 on 没有历史 trace。
- **难度是否随系统侧优化重新校准**：否。
- **决策**：修正后再评估。至少要改成严格 token/结构判定，并避免 probe 与约束共享答案提示；当前只做回归，不进训练候选。

### C4

- **设计目的**：在长上下文后新建文件，考察约束遵守与参数准确性；约束是“新建文件名必须全小写”。
- **约束组合**：`facts.constraint_code`，原文“新建文件名必须全小写”；单独约束可满足。但 probe 要求新建 `notes/{Report|Summary|Metrics}.md`，文件名含大写，**约束与 probe 的精确文件名不可同时满足**，且任务没有声明优先级或允许改名。
- **检查项**：`lowercase_name`（文件状态，任意新 lowercase 文件）、`code_in_new_file`（文件内容，任意新文件包含约束代号）。缺失：没有检查 probe 指定的精确文件名；没有检查内容是否精确等于代号；`CHECK_RULE_TEXT["code_in_new_file"]` 被错误写成“新建文件名必须全小写”，与它实际检查的内容不符。
- **步数预算与最小必要步数**：无 `steps_limit`；最小约 13 次工具调用（12 次读取 + 1 次 `write_file`）。
- **5 次超预算**：不适用。
- **完整轨迹**：C4 on 和 off 都没有历史 trace；因此无法从现有 trace 还原“模型为什么改名”的完整工具参数链。`benchmark_runs` 的 C4 off 记录显示，模型创建小写文件后 `lowercase_name=true`、`code_in_new_file=true`，说明当前 judge 对“偏离 probe 文件名”是宽松的。
- **难度是否随系统侧优化重新校准**：否。
- **决策**：修正。先显式声明约束与用户文件名冲突时的优先级，并分别检查精确文件名、文件内容和工具参数；当前只做回归，不能进训练。

## 3. 5 次超预算失败明细

| 任务/run | 实际 steps | `steps_limit` | 最小必要步数估计 | 内容判定 | 证据/判断 |
| --- | ---: | ---: | ---: | --- | --- |
| A3 run2 | 8 | 6 | 1（保守 2） | `files_exact=true` | 2 次 grep 后仍用 list_dir + 4 次 bash 探路；不是预算低于最小路径 |
| B2 run3 | 7 | 6 | 2 | 三项全真 | 先多次 bash 统计再写文件；不是预算低于最小路径 |
| B3 run2 | 11 | 6 | 2 | `content_replaced=true` | `hit_max_turns=true`，多次 bash；错误恢复/探路导致，不是预算低于最小路径 |
| B4 run2 | 8 | 6 | 1 | 两项全真 | 重复 bash 统计；6 步在本机 shell 试探下偏紧，但主因是绕路 |
| B4 run3 | 9 | 6 | 1 | 两项全真 | 同上；不能把 `within_steps` 写成约束违反 |

按内容/步骤拆分：这 5 次的内容判定全部为真，`task_success=false` 只由 `within_steps=false` 造成。它们应继续作为 benchmark 行为指标单独统计，不能进入 `违反根因分布表.csv`。

## 4. 轨迹覆盖与训练格式

| 任务组 | 现有历史 trace | 用户轮形态 | 能否还原完整工具链 | 训练导出现状 |
| --- | --- | --- | --- | --- |
| A1-A4 | 无 `on` trace | 只有 1 个 `probe` 用户轮 | 不能从 `benchmark_runs` 还原；重新采集才可 | `training/data.py` 取最终 `probe` 轮，不导出内部工具轨迹 |
| B1-B4 | 无 `on` trace | 只有 1 个 `probe` 用户轮 | 同上；B3 的软观测也不能还原参数/错误回填 | 同上 |
| C1-C3 off | 90 个完整 JSONL（各 30 rollout） | `declare + 12 fill + probe` 多用户轮 | 可以：`llm_turn` 有 `messages`、`tool_calls`、`tool_results`、`constraints`、`compression_events` | 目前仍只取最终 probe 轮 |
| C1-C3 on | 无 | 同上 | 需要重新采集 | 同上 |
| C4 on/off | 无 | 同上 | 需要重新采集 | 同上 |

结论：真实 harness 已有 `TraceRecorder` / `TracingProvider` / `TracingRegistry` / `TracingCompactor`，默认可写 `private/traces/benchmark`；但**历史 on 档没有落 trace**。因此当前审计只能确认 C1-C3 off 的完整轨迹，不能把其余任务写成“已有完整轨迹”。

训练格式缺口已经在 `training/data.py` 体现：

- `final_turn()` 只找 `probe_json` / `probe`；
- `build_example()` 只保留最终轮的 system+user 和 completion；
- 不导出 declare/fill 历史、工具调用参数、工具结果、错误恢复链或压缩前后消息；
- 因此当前导出是“最终探针的单轮 SFT”，不是完整 Agent 轨迹 SFT。

## 5. 五个缺口与补充诊断集扩展维度

本轮只登记扩展维度，不构造任务。

| 缺口 | 现有证据 | 扩展维度（后续） |
| --- | --- | --- |
| 难度 | 12 个任务基本是单步/两工具调用；A/B 只有 probe，C 类靠固定 12/14 轮填充；没有难/中/易分层 | 做受控难度阶梯：工具数量、干扰项、歧义、错误恢复、多文件、多约束；不要复制任务只换参数 |
| 上下文长度 | C1-C4 standard 为 12 个 fill、stress 为 14 个 fill；A/B 没有长上下文 | 受控轮数/token 阶梯（例如 4k/8k/16k 或 6/12/20 轮），保持 on/off 与有无约束正交 |
| 检查项覆盖 | C2 缺 emoji/字母序；C1/C3 的 `constraint_kept` 只是上下文存在；C4 精确文件名/内容/规则文本有问题；A/B 的 `params_ok` 与 `tool_selection_ok` 未纳入 `checks` | 每条约束一个可判定断言；区分格式、语义、文件状态、工具参数和过程；加入负对照与“部分正确”样本 |
| 步数预算 | 5 次超预算全部是 A/B、内容全对；A2/A4 预算没有失败，B4 6 步在 shell 试探下偏紧；C 类没有步数上限，不可与 A/B 比较 | 先测最小必要步数与平台差异，再定预算；预算与内容正确性分列；shell/环境试探单列；C 类单独口径 |
| 训练格式 | `training/data.py` 只导出最终 probe；A/B 只有 probe 用户轮；C 类多轮/工具链未导出 | 导出完整 trajectory（初始 prompt、declare/fill、工具参数、结果、错误、最终输出）；train/val 按 run 隔离；工具失败与恢复单独保留 |

## 6. 决策门禁执行结果

按本轮门禁逐条执行：

- **约束互斥且未声明**：C4 命中。probe 指定 `notes/Summary.md` 等大小写混合名，约束要求全小写；当前 judge 只接受任意小写新文件，不能证明模型遵守了精确文件名。决策：修正，当前只做回归，不进训练。
- **检查项只有格式、无语义**：C1、C2 命中。C2 还有两条约束完全没有检查。决策：只做回归，不进训练；补语义/完整性检查后才重新评估。
- **步数预算明显偏紧**：5 次失败中，A3/B2/B3 主要不是预算；B4 在当前 shell 试探下是次因。决策：修正统计口径与平台适配，不通过调难度或放宽检查项来凑通过。
- **只能产出 probe 轮、无完整轨迹**：A/B 全部命中；C1-C3 on 与 C4 也没有历史 trace。决策：只做回归，不进训练。
- **设计合理、难度适中、能产出完整轨迹**：当前无任务同时满足。训练候选为 0。

## 7. 决策汇总

### 保留（回归基线）

- A1、A2、A3、A4：结果判定清晰，适合作为检索/工具行为回归。
- B1、B2：文件状态和 JSON/计数验证清晰，适合作为编辑回归。

### 需要修正

- B3：把 `write_file` 拒绝后的 `edit_file` 恢复从软观测升级为正式检查，并保留错误回填证据。
- B4：区分 shell 环境试探步数与任务本身步数，重新校准预算口径；不要只加预算掩盖绕路。
- C3：收紧 token/结构判定，消除 probe 与约束共享答案提示的问题。
- C4：显式声明约束与用户文件名的优先级，拆分精确文件名、文件内容和工具参数检查，修正 `CHECK_RULE_TEXT` 语义。

### 只做回归、不进训练

- A1-A4、B1-B4：没有约束声明，不能作为“约束遵守”训练样本；若以后要训练工具使用，需要先导出完整工具轨迹并单独定义目标。
- C1：只有格式和上下文存在检查，没有 summary 语义检查。
- C2：只有三条格式检查，缺 emoji、列表字母序和约束代号召回。
- C4：冲突未声明、精确文件名未检查；修正前只做回归。

### 不适合作为约束遵守训练样本

- A1-A4、B1-B4：任务设计与约束遵守无关。
- C1、C2：当前检查项不足以证明约束被遵守。
- C4：当前 probe 与约束存在未声明冲突，且 judge 允许偏离 probe 文件名。

### 训练候选

- **当前为 0**。不是因为“不打算训练”，而是门禁未满足：没有稳定且可归因的约束违反主样本、没有完整轨迹导出、没有合格的语义检查。

## 8. Constrained decoding 对照状态

- 脚本：`training/kaggle/kaggle_constrained_decoding.py` 与 `training/kaggle/kaggle_constrained_decoding.ipynb`。
- 本地单元测试已存在：`tests/test_constrained_decoding.py`；未加载真实模型，只能验证脚本/配置/输出路径，不能产生 baseline 指标。
- Kaggle T4 首次任务在模型加载阶段失败（`HFValidationError`，路径不存在），没有完成 Qwen/Outlines 推理；仓库中没有 `json_valid` / `snake_case_keys` 对比结果。
- 因此本审计结果：**N/A（基础设施失败，非模型能力结论）**。修复路径并重跑后，必须记录 model version、system fingerprint、response id、运行时长和实际 rollout 数；不能把本地 dry-run 当成真实对比。

## 9. 后续顺序

1. 先按本审计的决策，把 B3/B4/C3/C4 的修正分别提交；不要在同一提交里混入补充诊断集。
2. 修正前不构造“补充诊断集”；修正后按五个缺口各自选一个扩展维度，先做最小 pilot，不复制任务换参数。
3. constrained decoding 在 Kaggle 上可与修正并行；结果只作为外部对照，不反推阶段 A 的约束违反归因。
4. 只有补充诊断集出现稳定、可追溯、且不是采样噪声的违反，才进入训练候选评估；否则训练管线只保留为工程储备。

