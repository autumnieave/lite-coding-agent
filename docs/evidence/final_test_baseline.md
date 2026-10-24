# final-test-v1 baseline（历史证据，已作废）

**日期**：2026-10-20  
**状态**：v1 已作废，仅作历史证据保留；最终对比使用 `final-test-v2`。  
**作废原因**：`U-FT10-BASE` / `U-FT11-HARD` 的 prompt 引用了 dev fixture（`part02.txt`、
第 37 行、`FIX-D03`），而 test fixture 实际为 `part07.txt`、第 53 行、`FIX-T03`。这两个任务
的 6 个 stable 检查项不能归因给模型。

## 1. 运行口径

| 项 | 值 |
| --- | --- |
| Kernel | `autumnsq/lite-agent-small-model-tuning` v9，`COMPLETE` |
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct` |
| 环境 | Tesla T4 16GB；Python 3.13.15；vLLM 0.12.0；`max_model_len=8192` |
| 采样 | `temperature=0.0`、`top_p=None`、`seed=20261019` |
| 任务 | `spec_set=final-test`，11 任务 × 10 rollout = 110 条 |
| 时长 | 198.2s（`2026-10-20T10:32:44Z` → `10:40:54Z`） |
| 产物 | `private/kaggle_pull_ft_184800/`（不入库） |

## 2. 总体结果

| 指标 | 结果 |
| --- | ---: |
| records / errors | 110 / 0 |
| 检查项 | 56 |
| stable 检查项 | 26 |
| intermittent | 0 |
| 任务成功 | 2/11 = 18.2% |
| 约束检查通过率（全部） | 300/560 = 53.6% |
| 约束检查通过率（剔除 U-FT10/11） | 270/470 = 57.4% |
| 工具调用成功 | 110/150 = 73.3% |
| 工具调用成功（剔除 U-FT10/11） | 110/110 = 100% |
| 必需工具成功 | 80/100 = 80.0% |
| 轨迹完整 | 110/110 = 100% |

## 3. 逐任务

| 任务 | 维度/档位 | 检查项 | stable | task_success | 平均步数 |
| --- | --- | ---: | ---: | ---: | ---: |
| `U-FT01-BASE` | constraint_count/base | 7 | 5 | 0/10 | 1.0 |
| `U-FT02-HARD` | constraint_count/hard | 7 | 5 | 0/10 | 1.0 |
| `U-FT03-BASE` | context_length/base | 4 | 0 | 10/10 | 1.0 |
| `U-FT04-HARD` | context_length/hard | 4 | 2 | 0/10 | 2.0 |
| `U-FT05-BASE` | tool_parameters/base | 6 | 2 | 0/10 | 1.0 |
| `U-FT06-HARD` | tool_parameters/hard | 6 | 1 | 0/10 | 2.0 |
| `U-FT07-NONE` | conflict/none | 4 | 0 | 10/10 | 1.0 |
| `U-FT08-WEAK` | conflict/weak | 5 | 3 | 0/10 | 1.0 |
| `U-FT09-STRONG` | conflict/strong | 4 | 2 | 0/10 | 1.0 |
| `U-FT10-BASE` | check_depth/base | 3 | 2（invalid） | 0/10 | 2.0 |
| `U-FT11-HARD` | check_depth/hard | 6 | 4（invalid） | 0/10 | 2.0 |

剔除 `U-FT10/11` 的 6 个缺陷检查后，仍有 20 个 valid stable，分布在 7 个任务；因此 v1 的
门禁在数字上满足，但任务定义不可信，不能直接跑四组对比。

## 4. 失败形态（仅作历史参考）

- `U-FT01/02`：读到正确文件，但回复是 Markdown 围栏 JSON，`json_valid`、`snake_case_keys`、
  `no_fence`、`exact_keys`、`value_correct` 均 10/10 失败。
- `U-FT04`：只读 `part01` 与 `part06`，没有按任务读取 10 个文件，最终答案错误。
- `U-FT05/06`：JSON 输出或数值检查失败；工具参数本身合法。
- `U-FT08`：弱冲突下没有写成约束要求的文件名，说明也未出现。
- `U-FT09`：强冲突下直接写 `ALPHA BETA`，没有声明互斥。

## 5. 后续处理

`final-test-v2` 已重新冻结，修正 `U-FT10/11` 的 prompt 引用并增加冻结前校验；v1 的
manifest 与文件保持原样，仅通过 `private/unified_dataset/FINAL_TEST_V1_INVALIDATED.json`
标记作废。v2 的 S0 结果见 `docs/evidence/final_test_v2_baseline.md`。

