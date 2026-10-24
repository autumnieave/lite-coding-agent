# Stage 1.5 教师门禁验证

- 生成时间：`2026-10-24T13:20:53.046450+00:00`
- 教师模型：`deepseek-v4.1-flash`
- base_url：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 协议：`openai_chat_completions_tool_calls_v1`
- 采样：`temperature=0.0`，`top_p=None`，`seed=20261019`
- 保留门槛：门禁通过率 ≥ 40%，策略合规率 ≥ 60%

## 逐任务结果

| 任务 | 族 | 难度 | rollout | 门禁通过 | 门禁率 | 策略合规 | 合规率 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `U-S15-01-EXTRACT-A` | policy_extract | weak | 5/5 | 4 | 80.0% | 4 | 80.0% | 保留 |
| `U-S15-02-EXTRACT-B` | policy_extract | medium | 5/5 | 4 | 80.0% | 4 | 80.0% | 保留 |
| `U-S15-03-EXTRACT-C` | policy_extract | medium | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15-04-SELECT-A` | selective_aggregate | medium | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15-05-SELECT-B` | selective_aggregate | medium | 5/5 | 3 | 60.0% | 5 | 100.0% | 保留 |
| `U-S15-06-SELECT-C` | selective_aggregate | strong | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15-07-SEARCH-A` | search_collect | medium | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15-08-SEARCH-B` | search_collect | strong | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15-09-SEARCH-C` | search_collect | strong | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15-10-EDIT-A` | edit_verify | medium | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15-11-EDIT-B` | edit_verify | strong | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15-12-EDIT-C` | edit_verify | strong | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15-13-AUDIT-A` | audit_reconcile | strong | 5/5 | 5 | 100.0% | 5 | 100.0% | 保留 |
| `U-S15-14-AUDIT-B` | audit_reconcile | strong | 5/5 | 2 | 40.0% | 4 | 80.0% | 保留 |
| `U-S15-15-AUDIT-C` | audit_reconcile | strong | 5/5 | 4 | 80.0% | 5 | 100.0% | 保留 |

## 失败检查项

- `U-S15-01-EXTRACT-A`：`exact_keys`, `json_valid`, `no_fence`, `policy_applied`, `value_correct`
- `U-S15-02-EXTRACT-B`：`policy_applied`, `value_correct`
- `U-S15-03-EXTRACT-C`：无
- `U-S15-04-SELECT-A`：无
- `U-S15-05-SELECT-B`：`tools_allowed`
- `U-S15-06-SELECT-C`：无
- `U-S15-07-SEARCH-A`：无
- `U-S15-08-SEARCH-B`：无
- `U-S15-09-SEARCH-C`：无
- `U-S15-10-EDIT-A`：无
- `U-S15-11-EDIT-B`：无
- `U-S15-12-EDIT-C`：无
- `U-S15-13-AUDIT-A`：无
- `U-S15-14-AUDIT-B`：`conflict_count_correct`, `conflicts_exact`, `resolved_exact`
- `U-S15-15-AUDIT-C`：`resolved_exact`

## 判定

- 保留任务：15/15
- 需调整或丢弃：0
- 本报告只记录教师可解性，不代表 baseline/SFT 难度结论。
