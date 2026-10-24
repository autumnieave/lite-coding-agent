# 阶段 1.5b Pilot 评测

- Kaggle kernel：`autumnsq/lite-agent-stage-1-5-stats-pilot`
- train-v5 adapter SHA-256：`f5b685d4da795a783fbcb21b15880daebfa08c64b1661792f7e2c6caab29fea7`
- 模型：`Qwen/Qwen2.5-7B-Instruct`，NF4，transformers 离线 provider（不起 vLLM server）
- 采样：`temperature=0.0`、`top_p=None`、`seed=20261019`
- 规模：5 个任务 × 10 rollout × baseline/SFT = 100 条
- 方差口径：每个任务 10 个二元结果的样本方差；另列 Bernoulli `p(1-p)`。
- staging：5 任务 × 1 rollout × 2 组，0 error=True，合计 success=1；通过后跑 full。

## 逐任务结果

| 任务 | baseline 通过 | baseline 率 | baseline 方差 | SFT 通过 | SFT 率 | SFT 方差 | 变化 | stable(≥6/10) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `U-S15-01-EXTRACT-A` | 0/10 | 0% | 0.000000 | 0/10 | 0% | 0.000000 | 0% | baseline=False, SFT=False |
| `U-S15-04-SELECT-A` | 0/10 | 0% | 0.000000 | 0/10 | 0% | 0.000000 | 0% | baseline=False, SFT=False |
| `U-S15-07-SEARCH-A` | 0/10 | 0% | 0.000000 | 0/10 | 0% | 0.000000 | 0% | baseline=False, SFT=False |
| `U-S15-10-EDIT-A` | 0/10 | 0% | 0.000000 | 10/10 | 100% | 0.000000 | 100% | baseline=False, SFT=True |
| `U-S15-13-AUDIT-A` | 0/10 | 0% | 0.000000 | 0/10 | 0% | 0.000000 | 0% | baseline=False, SFT=False |

## 条件汇总

| 条件 | 状态 | 有效 rollout | error | task_success | 通过率 |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | completed | 50 | 0 | 0 | 0% |
| sft | completed | 50 | 0 | 10 | 20% |

## 失败检查项计数

| 任务 | baseline 失败检查（10 次中） | SFT 失败检查（10 次中） |
| --- | --- | --- |
| `U-S15-01-EXTRACT-A` | value_correct=10, policy_applied=10 | value_correct=10, policy_applied=10 |
| `U-S15-04-SELECT-A` | json_valid=10, files_exact=10, total_correct=10, selected_count_correct=10, sorted=10, tools_allowed=10 | files_exact=10, total_correct=10, selected_count_correct=10, no_excluded_read=10 |
| `U-S15-07-SEARCH-A` | ids_exact=10, files_exact=10, count_correct=10, required_tools_used=10 | ids_exact=10, files_exact=10, count_correct=10, required_tools_used=10 |
| `U-S15-10-EDIT-A` | edit_file_used=10, token_replaced=10, verified_after_edit=10 | 无 |
| `U-S15-13-AUDIT-A` | resolved_exact=10, conflicts_exact=10, sources_exact=10 | resolved_exact=10, conflicts_exact=10, sources_exact=10, conflict_count_correct=10 |

## 判定（2026-10-24 修正）

**baseline 地板效应，1.5b 验收失败；退回 1.5a 调难度，1.5c 不启动。**

- baseline 总计 `0/50`，SFT 总计 `10/50`，0 harness error；所有 rollout 均 `stopped=completed`，不是超时或评测崩溃。
- 4/5 个任务 baseline 与 SFT 双方均为 `0/10`：`U-S15-01-EXTRACT-A`、`U-S15-04-SELECT-A`、`U-S15-07-SEARCH-A`、`U-S15-13-AUDIT-A`。这不是“未饱和、有提升空间”，而是四个任务对双方都不可解。
- 唯一正收益来自 `U-S15-10-EDIT-A`（baseline `0/10` → SFT `10/10`）。效应全部来自 1 个任务，不能外推到其他族。
- 五个任务的二元结果方差均为 0，baseline 通过率全部为 0；Pilot 没有可用于功效分析的基线方差或任务间差异。
- 若按 rollout 做 McNemar，会得到单向不一致对显著；但该显著性完全由 `U-S15-10-EDIT-A` 的 10 条不一致对贡献，不能解释为总体方法差异。
- Bootstrap CI 在零方差、全 0/全 1 的边界上退化，无法给出可用区间；功效分析所需的基线通过率、方差和 discordant 比例不存在。
- “SFT 通过率 <90% 的任务数 = 4/5”不能作为进入 1.5c 的门禁；本轮唯一有效门禁应是“至少 3 个任务 baseline 通过率落在 20–80%”。本轮为 0/5。

## 根因诊断（A/B/C）

**结论：A（任务太难 + 任务定义歧义），不是 B（门禁过严），也不是 C（policy/检查实现 bug）。**

- `U-S15-01-EXTRACT-A`：baseline 10/10 输出 `key="settings.ini"`、`value` 为三个 TOKEN 的拼接。prompt 只写“键必须恰好为 key、value”，没有定义 `key` 指配置键名；属于任务定义歧义，不是模型不会 JSON。
- `U-S15-04-SELECT-A`：baseline 聚合值 42 正确，但输出带代码围栏、files 用 basename 而非 prompt 要求的工作区相对路径，并调用了不允许的 `grep`。属于格式/工具约束未在难度上分层，不是门禁误判。
- `U-S15-07-SEARCH-A`：baseline 只调用 `grep` 就把 grep 输出当最终答案，没有读取命中文件；同时把 archive 的 `NEEDLE-999` 计入。prompt 已要求读取命中文件并排除 archive，属于多步任务难度过高。
- `U-S15-10-EDIT-A`：baseline 实际调用 `edit_file`，但传 `old_string="token"`，工具因匹配 59 处拒绝执行；模型随后仍声称 `changed=true, verified=true`。SFT 10/10 通过，说明工具实现正常，失败是 baseline 能力/任务难度问题。
- `U-S15-13-AUDIT-A`：baseline 的 `resolved` 保留了 `;REV=n` 后缀，冲突键判断错误，sources 未排序。prompt 没有明确“resolved value 去掉 REV 后缀”，属于任务定义歧义叠加多约束难度。
- `policy_compliance()` 只是对已定义的 policy checks 取合取；self-test 15/15 通过。未发现门禁逻辑或 policy 实现 bug。5 个任务全部 `within_steps=True`、0 error，也排除了步数上限和 harness 崩溃。

## 统计口径修正

- 零方差 + 地板效应下，McNemar 的“显著”只反映单向不一致对，不反映总体效应；Bootstrap CI 退化；功效分析参数不存在。
- 不把“4/5 任务 SFT <90%”写成“未饱和、有提升空间”；正确表述是“baseline 地板效应，4 个任务双方均不可解”。
- 1.5c 预注册若启动，必须写清三层任务和每层 baseline 通过率目标；本轮不满足启动条件。

## 边界

- 本报告只写实际 Kaggle rollout；没有跑成功的条件不写通过。
- stable 只在每个任务 10 次 rollout 内计算，不替代 1.5c 的样本量/功效分析。
- 5 个 pilot 任务覆盖 5 个工作流族；结果不外推到全部 15 个任务。
