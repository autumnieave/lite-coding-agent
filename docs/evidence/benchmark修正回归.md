# Benchmark 修正后回归（真实运行）

**日期**：2026-10-19  
**定位**：原 12 任务在 B3/B4/C3/C4 分别独立修正后的回归基线；**不是**训练集，也不用于证明历史 5 次失败可以被精确复现。  
**运行**：`scripts/benchmark.py --group both --runs 3 --profile standard --sampling-temperature 0 --sampling-seed 20261019`，每个 task × run 执行 1 次 rollout。

对应提交：`2b83892`（C3 精确 token）、`7eef4df`（C4 文件名冲突）、`3303db3`（B3 恢复硬检查）、`eadccf3`（B4 步数预算）；trace/采样元数据由 `c0e1f77` 提供。

## 1. 采样与证据

- 模型：`deepseek-v4.1-flash`（DashScope OpenAI 兼容端点；历史 `deepseek-flash` 当前不可用）。
- 采样：`temperature=0.0`、`top_p=None`、`seed=20261019`；每个 trace 的 `sampling_config` 记录该配置。
- 记录：`private/benchmark_fixed_runs.jsonl`；trace：`private/traces/benchmark_fixed/standard/{on,off}/`。
- 规模：48 次任务运行（on 36、off 12）；48 个 trace 文件，848 个 `llm_turn`、466 次工具调用、466 条工具结果。
- trace 覆盖：`run_start` / `llm_turn` / `run_end` 均为 48/48；所有工具参数 JSON 合法，路径越界拦截 0 次。
- metadata：848/848 次 LLM turn 有 `model_version`、`request_id`、`response_id`；API 未返回 `system_fingerprint`，报告不伪造该字段。
- 脱敏：扫描 trace 中本机绝对路径（`<repo>` / `<home>` 等）与常见 API key 模式，命中 0；workspace 统一写成 `<workspace>`。

## 2. 总览

“内容正确”指除 `within_steps` 外的所有检查项通过；“任务成功”还要求步数在预算内。两者分列，避免把绕路当成内容错误。

| 范围 | 次数 | 内容正确 | 任务成功 | 步数失败 | 非步数检查失败 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部 | 48 | 41/48（85.4%） | 37/48（77.1%） | 6 | 7 |
| on | 36 | 32/36（88.9%） | 28/36（77.8%） | 6 | 4 |
| off | 12 | 9/12（75.0%） | 9/12（75.0%） | 0 | 3 |

结论：修正后的判定确实能区分“内容错误”和“步数绕路”。本轮没有复现历史 5 个失败检查项；本轮出现 6 次步数失败和 7 次非步数检查失败，二者均按预注册口径分开统计。

## 3. 步数失败（单独统计，不进入约束违反根因表）

| task | run | 步数 | 预算 | 内容检查 | 说明 |
| --- | ---: | ---: | ---: | --- | --- |
| A3 | 3 | 9 | 6 | `files_exact=false` | 多轮 `bash` 探路后达到 `max_turns` |
| B1 | 2 | 8 | 6 | 内容通过 | 达到 `max_turns`，未给出最终回复 |
| B2 | 2 | 7 | 6 | 内容通过 | 重复目录/命令探路 |
| B3 | 2 | 9 | 6 | 恢复检查通过 | 内容替换成功，但超出预算 |
| B3 | 3 | 9 | 6 | `content_replaced=false` | `edit_file` 只删了一行，没有替换为目标 token；同时达到 `max_turns` |
| B4 | 1 | 11 | 9 | 内容通过 | 多次 PowerShell/命令环境试探；B4 新预算仍不足以覆盖这次绕路 |

B4 修正后的 9 步预算在 run2（6 步）和 run3（8 步）生效，但 run1 仍需 11 步；因此预算是“减少误判”的校准，不是保证所有运行通过。B3 的新硬检查在 run2/run3 都观察到 `write_file` 拒绝和随后 `edit_file` 成功，run3 仍因最终文件内容不对而失败，说明“恢复动作存在”与“任务完成”被正确分开。

## 4. 修正项回归

### B3（恢复路径硬检查）

- run1：6 步，`write_file_refused=true`、`recovered_with_edit=true`、`content_replaced=true`，任务成功。
- run2：9 步，三项检查均为真，但步数超预算，任务失败归因于步数而非恢复逻辑。
- run3：`write_file_refused=true`、`recovered_with_edit=true`，但 `content_replaced=false`；trace 显示 `edit_file` 的 `old_string` 只删除一行，没有把文件替换成目标 token。硬检查按预期暴露了“看似恢复、实际内容错误”。

### B4（步数预算 9）

- run1：11 步，内容正确（`json_valid=true`、`total_lines_correct=true`），仍超预算；
- run2：6 步，通过；
- run3：8 步，通过。

这说明 6→9 的调整降低了 B4 的历史误判，但没有把“shell 环境试探”从任务效率中抹掉；报告中继续把步数失败单列。

### C3（精确 token）

- on：3/3 给出精确 token；
- off：2/3 给出精确 token；run3 的回复引用了错误 token 并解释“工作区中不存在目标串”，`token_recalled=false`。

旧实现会被“回复中出现某个代号”误判通过；本轮 off run3 被正确判失败。C3 仍只做回归，不进入训练候选。

### C4（冲突显式化）

- on/off 共 6 次全部通过 `lowercase_name` 与 `code_in_new_file`；
- 模型均在回复中说明用户给出的混合大小写文件名与“新建文件名必须全小写”冲突，并实际创建小写目标文件。

C4 的修正把原本未声明的冲突变成可判定行为；它证明了“冲突声明 + 文件状态”可以被脚本化检查，但仍不构成“稳定训练信号”。

## 5. 非步数检查失败明细

| task | group | run | 检查 | trace 证据摘要 |
| --- | --- | ---: | --- | --- |
| A3 | on | 3 | `files_exact` | 多轮 `bash` 后达到 `max_turns`，没有完成目标列表 |
| B3 | on | 3 | `content_replaced` | `edit_file` 成功但只删除一行，最终文件不是目标 token |
| C1 | on | 2 | `json_valid` | probe 输出使用 Markdown 代码围栏，整条文本不可直接 `json.loads` |
| C1 | on | 3 | `json_valid` | 同上，代码围栏导致格式失败 |
| C1 | off | 2 | `json_valid` | 同上，代码围栏导致格式失败 |
| C2 | off | 2 | `snake_case_keys` | 最终 JSON 键名不符合 snake_case |
| C3 | off | 3 | `token_recalled` | 回复给出错误 token，且解释与约束不一致 |

## 6. 对回归基线的判断

- 修正后的 benchmark 可以作为**历史回归基线**：C3 不再接受错误 token，C4 能检查声明后的实际小写文件，B3 能区分恢复动作与内容完成，B4 的步数口径可解释。
- 它仍然不能作为训练数据：A/B 类没有约束，C 类检查项有限，且 `tool_selection_ok` 未统一纳入 `checks`；16 次运行出现 `bash` 等非首选工具，但任务仍可能通过。
- 本轮结果不支持“历史 5 次已复现”的说法；新样本只说明修正后的判定在当前模型上如何分布。原 benchmark 回归继续保留，统一数据集四组对比另行设计。

## 7. 复现命令

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\benchmark.py `
  --group both --runs 3 --profile standard `
  --sampling-temperature 0 --sampling-seed 20261019
```

输出与 trace 已相对化到 `private/`，不提交大文件；原始 API key、系统绝对路径不出现在 trace 中。
