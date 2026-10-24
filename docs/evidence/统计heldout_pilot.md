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

## 判定

**进入 1.5c 功效分析**

- SFT 10/10 的任务数：1/5。
- SFT 通过率 <90% 的任务数：4/5。
- baseline 与 SFT 都 ≥90% 的任务数：0/5。
- baseline 在全部 pilot 任务上为 0/10，存在地板效应；pilot 可以判定“未饱和”，但不能估计总体 baseline 方差或支持广泛训练增益。
- SFT 唯一有通过样本的任务是 `U-S15-10-EDIT-A`；效应不能外推到其他族。
- 用户本轮门禁：至少 3 个任务 SFT <90%，判定不饱和；进入 1.5c。

## 边界

- 本报告只写实际 Kaggle rollout；没有跑成功的条件不写通过。
- stable 只在每个任务 10 次 rollout 内计算，不替代 1.5c 的样本量/功效分析。
- 5 个 pilot 任务覆盖 5 个工作流族；结果不外推到全部 15 个任务。
