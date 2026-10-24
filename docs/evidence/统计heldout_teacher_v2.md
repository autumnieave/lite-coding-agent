# Stage 1.5 教师门禁验证

- 生成时间：`2026-10-24T14:41:56.440204+00:00`
- 教师模型：`deepseek-v4.1-flash`
- base_url：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 协议：`openai_chat_completions_tool_calls_v1`
- 采样：`temperature=0.0`，`top_p=None`，`seed=20261019`
- 保留门槛：门禁通过率 ≥ 40%，策略合规率 ≥ 60%

## 逐任务结果

| 任务 | 族 | 难度 | rollout | 门禁通过 | 门禁率 | 策略合规 | 合规率 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `U-S15V2-01-EXTRACT-E` | policy_extract | easy | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15V2-02-EXTRACT-M` | policy_extract | medium | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15V2-03-EXTRACT-H` | policy_extract | hard | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15V2-04-SELECT-E` | selective_aggregate | easy | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15V2-05-SELECT-M` | selective_aggregate | medium | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15V2-06-SELECT-H` | selective_aggregate | hard | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15V2-07-SEARCH-E` | search_collect | easy | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15V2-08-SEARCH-M` | search_collect | medium | 5/5 | 4 | 80.0% | 5 | 100.0% | 保留 |
| `U-S15V2-09-SEARCH-H` | search_collect | hard | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15V2-10-EDIT-E` | edit_verify | easy | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15V2-11-EDIT-M` | edit_verify | medium | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15V2-12-EDIT-H` | edit_verify | hard | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15V2-13-AUDIT-E` | audit_reconcile | easy | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15V2-14-AUDIT-M` | audit_reconcile | medium | 5/5 | 2 | 40.0% | 4 | 80.0% | 保留 |
| `U-S15V2-15-AUDIT-H` | audit_reconcile | hard | 5/5 | 4 | 80.0% | 5 | 100.0% | 保留 |

## 失败检查项

- `U-S15V2-01-EXTRACT-E`：无
- `U-S15V2-02-EXTRACT-M`：无
- `U-S15V2-03-EXTRACT-H`：无
- `U-S15V2-04-SELECT-E`：无
- `U-S15V2-05-SELECT-M`：无
- `U-S15V2-06-SELECT-H`：无
- `U-S15V2-07-SEARCH-E`：无
- `U-S15V2-08-SEARCH-M`：`count_correct`
- `U-S15V2-09-SEARCH-H`：无
- `U-S15V2-10-EDIT-E`：无
- `U-S15V2-11-EDIT-M`：无
- `U-S15V2-12-EDIT-H`：无
- `U-S15V2-13-AUDIT-E`：无
- `U-S15V2-14-AUDIT-M`：`conflict_count_correct`, `conflicts_exact`, `json_valid`, `resolved_exact`, `sources_exact`
- `U-S15V2-15-AUDIT-H`：`resolved_exact`

## 判定

- 保留任务：15/15
- 需调整或丢弃：0
- 本报告只记录教师可解性，不代表 baseline/SFT 难度结论。
