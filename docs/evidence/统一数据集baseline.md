# 统一数据集 baseline（dev/test，10 rollouts）

**日期**：2026-10-19  
**状态**：dev 70 次、test 70 次真实 rollout 已完成；stable=0。  
**边界**：这是统一数据集的 baseline pilot，不是四组最终对比。当前 test 是在初版 6+6 结果、U-T01 检查器修正和 U-D07/T07 补入之后才冻结的 pilot test，不能冒充严格的一次性 held-out；最终四组对比前必须重新冻结新 test。

## 1. 运行配置

- 模型：`deepseek-v4.1-flash`（DashScope OpenAI 兼容端点）。
- 采样：`temperature=0.0`、`top_p=None`、`seed=20261019`；每次 rollout 都写入 trace 的 `sampling_config`。
- 任务：7 个 dev + 7 个 test，每任务 10 次 rollout；fixture 固定，只让模型采样变化。
- 约束：manifest 约束逐条注入 system prompt；另有一条统一工具约束，只允许 manifest 的 `allowed_tools`。
- trace：`private/traces/unified_baseline/<split>/<task>_rolloutNN.jsonl`，含完整 messages、工具参数/结果、约束位置、压缩事件和完整回复。
- 脱敏：140 个 trace 中本机绝对路径与常见 API key 模式扫描命中 0；workspace 统一相对化为 `<workspace>`。
- pilot test 当前哈希：`private/unified_dataset/test.jsonl` SHA-256 `04c997dfa2d7079f8369424621e927fc443d47f9fded7233beb45512490c5fab`。哈希只记录当前文件，**不表示**它是在所有 pilot 结果之前预注册的。

**模型口径说明**：历史阶段 A 使用 `deepseek-flash`，当前 DashScope 端点已不提供该别名，因此本 baseline 使用 `deepseek-v4.1-flash`。这不是“精确复现历史 5 次”，而是统一数据集的新 baseline；模型版本变化已在结论中明确，不与旧数字直接合并。

## 2. 稳定性规则

同一检查项在 10 次 rollout 中：

- 命中 `>=6`：`stable`；
- 命中 `1-5`：`intermittent`；
- 命中 `0`：`not_observed`，不进入 stable 统计。
- 有效 rollout 少于 10 次：`insufficient`，不把 3/5 这类短样本按比例误判成 stable。

`within_steps=false` 只作为步数预算指标单列，不进入约束违反 stable 统计，也不写进 `违反根因分布表.csv`。

## 3. 结果总览

| split | 任务数 | rollout | 检查项 | stable | intermittent | not_observed | stable 违反率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 7 | 70 | 46 | 0 | 13 | 33 | 0.0% |
| test | 7 | 70 | 46 | 0 | 10 | 36 | 0.0% |

结论：在当前任务分布和 `deepseek-v4.1-flash` 下，没有任何检查项达到 6/10 stable。统一数据集的 Route B 前置门禁（至少一个 stable 违反）仍未满足。

## 4. 任务级结果

| task | split | success/10 | 步数失败 | 平均步数 | 说明 |
| --- | --- | ---: | ---: | ---: | --- |
| U-D01 | dev | 10 | 0 | 1.1 | 弱任务，稳定通过 |
| U-D02 | dev | 10 | 0 | 1.0 | 弱任务，稳定通过 |
| U-D03 | dev | 7 | 0 | 2.6 | `json_answer` 2/10、`tools_allowed` 1/10 |
| U-D04 | dev | 10 | 0 | 1.0 | 稳定通过 |
| U-D05 | dev | 8 | 0 | 12.2 | `json_valid`/`reason_present` 各 2/10 |
| U-D06 | dev | 10 | 0 | 3.0 | 恢复路径稳定通过 |
| U-D07 | dev | 4 | 3 | 14.4 | `tools_allowed` 4/10，`source_count_correct` 3/10 |
| U-T01 | test | 9 | 0 | 1.0 | 1 次代码围栏导致格式违反 |
| U-T02 | test | 10 | 0 | 1.0 | 稳定通过 |
| U-T03 | test | 7 | 0 | 2.6 | `json_answer` 2/10、`tools_allowed` 1/10 |
| U-T04 | test | 10 | 0 | 1.2 | 稳定通过 |
| U-T05 | test | 6 | 0 | 12.9 | `json_valid`/`reason_present` 各 4/10 |
| U-T06 | test | 7 | 2 | 4.4 | `tools_allowed` 2/10，2 次步数超预算 |
| U-T07 | test | 5 | 3 | 14.6 | `tools_allowed` 4/10，1 次 `source_count_correct` |

## 5. 非零违反明细

| task | check | hits/10 | stability |
| --- | --- | ---: | --- |
| U-D03 | json_answer | 2 | intermittent |
| U-D03 | tools_allowed | 1 | intermittent |
| U-D05 | json_valid | 2 | intermittent |
| U-D05 | reason_present | 2 | intermittent |
| U-D07 | json_valid | 1 | intermittent |
| U-D07 | snake_case_keys | 1 | intermittent |
| U-D07 | missing_null | 1 | intermittent |
| U-D07 | exact_keys | 1 | intermittent |
| U-D07 | ids_exact | 1 | intermittent |
| U-D07 | values_exact | 1 | intermittent |
| U-D07 | total_correct | 1 | intermittent |
| U-D07 | source_count_correct | 3 | intermittent |
| U-D07 | tools_allowed | 4 | intermittent |
| U-T01 | json_valid | 1 | intermittent |
| U-T01 | snake_case_keys | 1 | intermittent |
| U-T01 | value_correct | 1 | intermittent |
| U-T03 | json_answer | 2 | intermittent |
| U-T03 | tools_allowed | 1 | intermittent |
| U-T05 | json_valid | 4 | intermittent |
| U-T05 | reason_present | 4 | intermittent |
| U-T06 | tools_allowed | 2 | intermittent |
| U-T07 | source_count_correct | 1 | intermittent |
| U-T07 | tools_allowed | 4 | intermittent |

## 6. 解释与边界

- 失败主要集中在输出格式/字段（`json_valid`、`json_answer`、`snake_case_keys`）和工具选择（`tools_allowed`），没有出现 stable 的语义/文件状态违反。
- U-D05/T05 的 `json_valid`/`reason_present` 是当前最接近 stable 的语义+格式组合（4/10），但仍低于预注册阈值。
- U-D07/T07 是后加的强任务；它的 `tools_allowed` 4/10 表明“模型知道约束但偶尔仍用 `bash`”，属于行为层信号，但仍是偶发。
- 所有 failure 都能在 trace 中定位到当轮 messages、模型完整回复、工具参数和结果；本文件不靠 `reply_head` 下结论。
- 本次没有为了凑 stable 改 temperature/top_p/seed，也没有把 `within_steps` 失败混进约束违反。
- 当前 test 是 pilot：初版 6+6、U-T01 检查器修正重跑、U-D07/T07 补入都发生在当前 test 冻结之前；因此它不能作为四组最终对比的严格 held-out。最终方案见 `docs/evidence/统一数据集方案.md` §3.1。

## 7. 下一步

- 统一数据集 Route B 的唯一硬门禁“至少一个 stable 违反”未满足；按预注册规则，不能进入 train 构造或 SFT。
- 若继续 Route B，应先做任务分布重构（更难的语义/工具组合、更长上下文），而不是继续加 rollout 或调采样；任务重构后先跑 dev，最终 test 必须重新冻结且四组各跑一次。
- 若不继续重构，项目应停在“诊断 + 评测方法论 + 训练管线储备”的 Route A。
