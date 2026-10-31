# final-test-v2 S0 baseline（Qwen2.5-1.5B-Instruct，离线 vLLM）

**日期**：2026-10-21  
**状态**：S0 已完成；S1 已完成（负结果，见 `docs/evidence/final_test_v2_S1.md`）；训练组与 S2 未启动。  
**结论**：110 条 rollout、0 error；原始 19 个 stable 检查项，剔除 `within_steps=false` 后
15 个 valid stable，分布在 6 个任务。满足“至少 3 个任务出现 stable 违反”的门禁，可以继续
S1，但不得把这些数字直接当作训练必要性结论。

## 1. 运行口径

| 项 | 值 |
| --- | --- |
| Kernel | `autumnsq/lite-agent-small-model-tuning` v23，`COMPLETE` |
| 数据集 | `final-test-v2`，SHA-256 `d8290751424106321c8dad19d66b4cf9d6c3e98a00c38e4b3f7c5bda9dc4458e` |
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct` |
| 运行环境 | Tesla T4 16GB；Python 3.13.15；vLLM 0.12.0；`max_model_len=8192` |
| 采样 | `temperature=0.0`、`top_p=None`、`seed=20261019`、`is_default=false` |
| 系统变体 | S0：现有 harness，无额外 prompt 后缀 |
| 任务 | 11 个任务 × 10 rollout = 110 条 |
| 运行时长 | 410s（`05:40:07Z` → `05:46:57Z`） |
| 产物 | `private/kaggle_pull_s0_v23/`（不入库） |

离线 provider 只记录 `model_version`，没有 request id / system fingerprint；这是 in-process
vLLM 与 OpenAI 兼容 server 的差异，不伪造字段。

## 2. 总体结果

| 指标 | 结果 |
| --- | ---: |
| records / errors | 110 / 0 |
| 检查项数 | 56 |
| 原始 stable 检查项 | 19 |
| valid stable（剔除 `within_steps=false`） | 15 |
| intermittent | 0 |
| 触发 stable 的任务 | 6 / 11 |
| task_success | 40 / 110 = 36.4% |
| within_steps 失败 | 10 / 110（全部集中在 `U-FT01-BASE`） |

## 3. 逐任务结果

| 任务 | 维度/档位 | 检查项 | 原始 stable | valid stable | task_success | within_steps 失败 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `U-FT01-BASE` | constraint_count/base | 7 | 4 | 0 | 0/10 | 10 |
| `U-FT02-HARD` | constraint_count/hard | 7 | 5 | 5 | 0/10 | 0 |
| `U-FT03-BASE` | context_length/base | 4 | 0 | 0 | 10/10 | 0 |
| `U-FT04-HARD` | context_length/hard | 4 | 2 | 2 | 0/10 | 0 |
| `U-FT05-BASE` | tool_parameters/base | 6 | 2 | 2 | 0/10 | 0 |
| `U-FT06-HARD` | tool_parameters/hard | 6 | 1 | 1 | 0/10 | 0 |
| `U-FT07-NONE` | conflict/none | 4 | 0 | 0 | 10/10 | 0 |
| `U-FT08-WEAK` | conflict/weak | 5 | 3 | 3 | 0/10 | 0 |
| `U-FT09-STRONG` | conflict/strong | 4 | 2 | 2 | 0/10 | 0 |
| `U-FT10-BASE` | check_depth/base | 3 | 0 | 0 | 10/10 | 0 |
| `U-FT11-HARD` | check_depth/hard | 6 | 0 | 0 | 10/10 | 0 |

## 4. 步数失败单独记录

`U-FT01-BASE` 的 10 条 rollout 全部触发 `within_steps=false`：模型在读到正确文件内容后，
把同一个 `read_file` 调用重复了 6 轮，直到 `max_turns`，没有进入最终 JSON 输出。对应 4 个
检查项（`json_valid`、`snake_case_keys`、`exact_keys`、`value_correct`）虽然 10/10 命中，
但按预注册口径**不计入约束违反归因**，也不进入 CSV。它应作为“步数预算/工具循环”问题单独
统计，而不是 stable 约束违反。

修正后的归因脚本在 `load_run()` 中直接丢弃 `within_steps=false` 的 run，并跳过 `smoke/`
目录；单测见 `tests/test_attribution.py`。

## 5. 归因分布（valid stable）

`docs/evidence/final_test_v2_S0_违反根因分布表.csv` 共 15 行（原始 trace 与
`private/kaggle_pull_s0_v23/` 保留在本地私有目录）：

CSV 的 `trace_files` 相对 `private/kaggle_pull_s0_v23/traces/`，`evidence_events` 指向
具体 `event_index`；例如 `U-FT02-HARD` 的 5 行均可回溯到 10 个 rollout 的 `run_end`。

| 根因分类 | 行数 |
| --- | ---: |
| 能力问题（多约束/冲突证据） | 14 |
| 行为问题 | 1 |

15 行全部为 `stable`，`rollout_hits=10/10`；本轮没有 intermittent 行。主要违反仍是输出格式
（`json_valid`、`snake_case_keys`、`no_fence`、`exact_keys`）与语义/文件检查
（`value_correct`、`content_exact`），工具参数本身合法、执行时未越界。

## 6. 与门禁的关系

- S0 满足“至少 3 个任务出现 stable 违反”，因此允许继续 S1；不因此自动进入训练。
- `U-FT01-BASE` 的步数失败不计入约束违反，不能用来抬高 stable 数量。
- S1 没有把 valid stable 压到 0–1，反而从 15 升到 20，task_success 从 40/110 降到
  10/110；按预注册规则停止，不进入阶段 B/C，也不启动四组对比。

