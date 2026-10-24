# 阶段 1.5b Pilot v2 评测（Round 2，最终轮）

- Kaggle kernel：`autumnsq/lite-agent-stage-1-5-stats-pilot-v2-round-2`
- train-v5 adapter SHA-256：`f5b685d4da795a783fbcb21b15880daebfa08c64b1661792f7e2c6caab29fea7`
- 模型：`Qwen/Qwen2.5-7B-Instruct`，NF4，transformers 离线 provider（不起 vLLM server）
- 采样：`temperature=0.7`、`top_p=0.9`、`seed=20261019`（每 rollout 使用 `base_seed + rollout_index`）
- 规模：6 个任务 × 10 rollout × baseline/SFT = 120 条
- 任务分层：easy / medium / hard 各 2 个任务；每层 baseline 目标分别为 60–80% / 40–60% / 20–40%。
- 方差口径：每个任务 10 个二元结果的样本方差；另列 Bernoulli `p(1-p)`。
- staging：6 任务 × 1 rollout × 2 组，0 error=True，合计 success=3；通过后跑 full。

## 逐任务结果

| 任务 | 层 | baseline 通过 | baseline 率 | baseline 方差 | SFT 通过 | SFT 率 | SFT 方差 | 变化 | stable(≥6/10) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `U-S15V2-01-EXTRACT-E` | easy | 10/10 | 100% | 0.000000 | 10/10 | 100% | 0.000000 | 0% | baseline=True, SFT=True |
| `U-S15V2-05-SELECT-M` | medium | 0/10 | 0% | 0.000000 | 0/10 | 0% | 0.000000 | 0% | baseline=False, SFT=False |
| `U-S15V2-08-SEARCH-M` | medium | 0/10 | 0% | 0.000000 | 0/10 | 0% | 0.000000 | 0% | baseline=False, SFT=False |
| `U-S15V2-09-SEARCH-H` | hard | 0/10 | 0% | 0.000000 | 0/10 | 0% | 0.000000 | 0% | baseline=False, SFT=False |
| `U-S15V2-10-EDIT-E` | easy | 10/10 | 100% | 0.000000 | 8/10 | 80% | 0.177778 | -20% | baseline=True, SFT=True |
| `U-S15V2-15-AUDIT-H` | hard | 0/10 | 0% | 0.000000 | 0/10 | 0% | 0.000000 | 0% | baseline=False, SFT=False |

## 按层汇总

| 层 | 任务数 | baseline 平均通过率 | baseline 范围 | SFT 平均通过率 | 层内 20–80% 任务数 |
| --- | ---: | ---: | --- | ---: | ---: |
| easy | 2 | 100% | 100%–100% | 90% | 0/2 |
| medium | 2 | 0% | 0%–0% | 0% | 0/2 |
| hard | 2 | 0% | 0%–0% | 0% | 0/2 |

## 条件汇总

| 条件 | 状态 | 有效 rollout | error | task_success | 通过率 |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | completed | 60 | 0 | 20 | 33% |
| sft | completed | 60 | 0 | 18 | 30% |

## 失败检查项计数

| 任务 | baseline 失败检查（10 次中） | SFT 失败检查（10 次中） |
| --- | --- | --- |
| `U-S15V2-01-EXTRACT-E` | 无 | 无 |
| `U-S15V2-05-SELECT-M` | total_correct=10, files_exact=9, selected_count_correct=8, sorted=7, json_valid=6, tools_allowed=5, within_steps=3, required_tools_used=3 | total_correct=10, tools_allowed=4, no_excluded_read=4, files_exact=3, selected_count_correct=3, required_tools_used=2 |
| `U-S15V2-08-SEARCH-M` | ids_exact=10, files_exact=10, count_correct=10, required_tools_used=10 | ids_exact=10, count_correct=10, required_tools_used=10, files_exact=8, sorted=2 |
| `U-S15V2-09-SEARCH-H` | ids_exact=10, files_exact=10, count_correct=10, required_tools_used=10 | ids_exact=10, files_exact=10, count_correct=10, required_tools_used=10 |
| `U-S15V2-10-EDIT-E` | 无 | edit_file_used=2, token_replaced=2, verified_after_edit=2, required_tools_used=2 |
| `U-S15V2-15-AUDIT-H` | resolved_exact=10, conflicts_exact=10, sources_exact=10, conflict_count_correct=10, required_tools_used=9, json_valid=3, tools_allowed=2 | resolved_exact=10, conflicts_exact=10, conflict_count_correct=10, sources_exact=4 |

## 统计口径

- 本 pilot 只用于判断 baseline 是否形成区分度，不作为 1.5d 的正式证据。
- 每个任务 10 次 rollout，stable 门槛为 ≥6/10；staging 的 1 次 rollout 不判 stable。
- 零方差或全 0/全 1 时，McNemar 的单向不一致对、Bootstrap CI 和功效分析都会退化；必须同时报告“不显著”和“不可估计”，不能只报 p 值。
- 进入 1.5c 的门禁是至少 3 个任务 baseline 通过率在 20–80%；不是“SFT <90%”。

## 判定

**任务难度无法形成区分度；统计显著验证不成立，收尾**

- baseline 通过率：`U-S15V2-01-EXTRACT-E` 100%、`U-S15V2-05-SELECT-M` 0%、`U-S15V2-08-SEARCH-M` 0%、`U-S15V2-09-SEARCH-H` 0%、`U-S15V2-10-EDIT-E` 100%、`U-S15V2-15-AUDIT-H` 0%。
- baseline 通过率落在 20–80% 的任务数：0/6。
- SFT 通过率 ≥90% 的任务数：1/6。
- 门禁：第 2 轮仍只有 0/6 个任务落在 20–80%，未达到 3 个任务；在 6 任务预注册门槛下，任务难度无法形成区分度，统计显著验证不成立，收尾。

## 边界

- 本报告只写实际 Kaggle rollout；没有跑成功的条件不写通过。
- stable 只在每个任务 10 次 rollout 内计算，不替代 1.5c 的样本量/功效分析。
- 6 个 pilot 任务覆盖 easy/medium/hard 各 2 个；结果不外推到全部 15 个任务。
