# held-out 难度筛选

## 目的

当前 6 个统计 held-out 任务出现地板效应，无法用于功效分析。本阶段不租卡，先用
Kaggle T4 对现有任务池做 8 次预注册采样，筛出 baseline 通过率在 30%–70% 的中间态
任务。只有中间态任务达到预注册门槛，才构造 `final-test-v3` 并重测 SFT/DPO 增量。

## 口径修正：轨迹行不等于任务

`private/sft_data_v5/train.jsonl` 有 468 行，但同一任务可能保留多条教师轨迹。逐行采样会
把重复轨迹当成独立任务，夸大样本量。本阶段先按 `metadata.task_id` 去重：

| 项目 | 数量 |
| --- | ---: |
| train-v5 轨迹行 | 468 |
| train-v5 唯一任务 | 394 |
| 去掉的重复轨迹行 | 74 |
| Round 1/2 统计任务 | 15 |
| 最终评测任务 | 409 |

唯一任务来源：

| 来源 | 任务数 |
| --- | ---: |
| train-v2 retained（DeepSeek） | 15 |
| train-v4 oracle | 180 |
| train-v5 teacher | 107 |
| xLAM schema-mapped public grep | 92 |
| stats-heldout-v2 | 15 |

后文统一称“483 条任务池”仅表示 468 条训练轨迹 + 15 个统计任务；实际采样按 409 个
唯一任务执行，避免重复计数。

### 训练重叠边界

394 个 train-v5 唯一任务已进入 SFT 训练数据；只有 15 个 `U-S15V2-*` 统计任务未出现在
train-v5 中。因此本阶段首先解决的是“任务池是否有中间态”，不等于“训练无关 held-out
是否成立”：

- 从 train-v5 子集筛出的中间态任务，只能先称为难度校准/诊断集；
- 若用它们构造 `final-test-v3` 并直接比较 SFT，必须标注“与训练任务重叠”，不能把增益
  写成泛化增益；
- 要恢复严格 held-out 语义，需要在重测前把入选任务从 SFT 训练集移除并重训，或改用
  全新任务。报告会单独给出 15 个未进训练的统计任务上的中间态数量。

## 可执行化边界

- `U-TR-*`、`U-TR4-*`、`U-TR5-*` 分别映射回冻结的 train-v2、train-v4 clone、train-v5 spec。
- 92 个 `PUB-XLAM-*` 原本只有单轮 grep 训练轨迹，本阶段新增 `public_grep` 可执行族：
  fixture 重建 `project/public/knowledge.txt`，prompt 与训练样本一致，门禁要求恢复原始
  `matches`（pattern + grep 行）且只调用 `grep`。
- `U-S15V2-*` 是已有 15 个统计 held-out 任务，直接加入池，但不与 train-v5 ID 重叠。

池清单：

- `private/heldout_difficulty_pool/pool_manifest.json`
- `private/heldout_difficulty_pool/pool_specs.jsonl`
- train-v5 train SHA-256：`6ee7e79d87cb7c41f4c1b91964d70abcf5e886c5ac75450318760a805b78f830`

## 预注册

以下参数在采样前固定，结果出来后不调整：

| 项目 | 值 |
| --- | --- |
| 每任务采样次数 | 8 |
| temperature | 0.7 |
| top_p | 0.9 |
| seed | `20261019 + rollout_index` |
| 中间态定义 | 有效 8 次采样的通过率 ∈ [0.30, 0.70] |
| 不完整任务 | 任一 rollout error 或有效采样 <8，排除出中间态计数 |
| ≥10 个中间态 | 构造 `final-test-v3`，冻结 SHA，重测 SFT/DPO/GRPO |
| 5–9 个中间态 | 用第 2 层系统性降难补到 ≥10 |
| <5 个中间态 | 进入第 2 层系统性调参降难 |

## 执行方案

- 模型：`Qwen/Qwen2.5-7B-Instruct`，NF4，单 T4，`device_map={"": 0}`。
- 每任务 8 次 rollout，`max_model_len=8192`、`max_tokens=512`。
- 409 个任务分 3 片（137 / 136 / 136），每片独立追加写 `runs.jsonl`，支持 `--resume`。
- 首片先跑 5 个 staging 任务（每个来源取最短任务）各 1 次；要求 `errors==0` 且至少
  1 条 `task_success=true`，否则停止，不进入全量。
- runner：`training/kaggle/difficulty_screen.py`
- kernel builder：`training/kaggle/build_difficulty_kernel.py`
- 汇总器：`scripts/analyze_heldout_difficulty.py`

## 7B 结果

### 采样完整性

| 项目 | 值 |
| --- | ---: |
| 已筛任务 | 409 |
| 有效样本数=8 且无 error 的任务 | 408 |
| 中间态任务（0.30–0.70） | 45 |
| 被排除的不完整任务 | 1 |
| 判定 | `proceed_final_test_v3` |

### 通过率分布

| 通过率（k/8） | 任务数 |
| --- | ---: |
| `0.0` | 160 |
| `0.125` | 28 |
| `0.25` | 11 |
| `0.375` | 18 |
| `0.5` | 11 |
| `0.625` | 16 |
| `0.75` | 31 |
| `0.875` | 30 |
| `1.0` | 103 |

### 按来源

| 分组 | 任务数 | 中间态任务数 | 中间态占比 |
| --- | ---: | ---: | ---: |
| `stats-heldout-v2` | 15 | 0 | 0.0% |
| `train-v2-retained` | 15 | 4 | 26.7% |
| `train-v4-oracle` | 180 | 12 | 6.7% |
| `train-v5-teacher` | 107 | 8 | 7.5% |
| `xlam-schema-mapped` | 92 | 21 | 22.8% |

### 按类别

| 分组 | 任务数 | 中间态任务数 | 中间态占比 |
| --- | ---: | ---: | ---: |
| `BASE` | 46 | 13 | 28.3% |
| `CONFLICT` | 59 | 0 | 0.0% |
| `CTX` | 57 | 0 | 0.0% |
| `HARD` | 80 | 9 | 11.2% |
| `TOOL` | 152 | 23 | 15.1% |
| `audit_reconcile` | 3 | 0 | 0.0% |
| `edit_verify` | 3 | 0 | 0.0% |
| `policy_extract` | 3 | 0 | 0.0% |
| `search_collect` | 3 | 0 | 0.0% |
| `selective_aggregate` | 3 | 0 | 0.0% |

### 按任务族

| 分组 | 任务数 | 中间态任务数 | 中间态占比 |
| --- | ---: | ---: | ---: |
| `01` | 9 | 2 | 22.2% |
| `02` | 5 | 0 | 0.0% |
| `03` | 61 | 7 | 11.5% |
| `audit_reconcile` | 3 | 0 | 0.0% |
| `conflict` | 59 | 0 | 0.0% |
| `context_recall` | 57 | 0 | 0.0% |
| `edit_verify` | 3 | 0 | 0.0% |
| `json_constraints` | 51 | 13 | 25.5% |
| `path_params` | 60 | 2 | 3.3% |
| `policy_extract` | 3 | 0 | 0.0% |
| `public_grep` | 92 | 21 | 22.8% |
| `search_collect` | 3 | 0 | 0.0% |
| `selective_aggregate` | 3 | 0 | 0.0% |

### 按难度标签

| 分组 | 任务数 | 中间态任务数 | 中间态占比 |
| --- | ---: | ---: | ---: |
| `easy` | 5 | 0 | 0.0% |
| `hard` | 5 | 0 | 0.0% |
| `medium` | 113 | 17 | 15.0% |
| `strong` | 148 | 6 | 4.1% |
| `weak` | 138 | 22 | 15.9% |

### 按训练重叠

| 分组 | 任务数 | 中间态任务数 | 中间态占比 |
| --- | ---: | ---: | ---: |
| `False` | 15 | 0 | 0.0% |
| `True` | 394 | 45 | 11.4% |

### 中间态任务清单

| task_id | 来源 | 类别 | 任务族 | 通过率 | 训练重叠 |
| --- | --- | --- | --- | ---: | --- |
| `PUB-XLAM-00001` | xlam-schema-mapped | TOOL | public_grep | 0.500 | 是 |
| `PUB-XLAM-00002` | xlam-schema-mapped | TOOL | public_grep | 0.625 | 是 |
| `PUB-XLAM-00003` | xlam-schema-mapped | TOOL | public_grep | 0.375 | 是 |
| `PUB-XLAM-00019` | xlam-schema-mapped | TOOL | public_grep | 0.500 | 是 |
| `PUB-XLAM-00023` | xlam-schema-mapped | TOOL | public_grep | 0.625 | 是 |
| `PUB-XLAM-00028` | xlam-schema-mapped | TOOL | public_grep | 0.625 | 是 |
| `PUB-XLAM-00031` | xlam-schema-mapped | TOOL | public_grep | 0.375 | 是 |
| `PUB-XLAM-00039` | xlam-schema-mapped | TOOL | public_grep | 0.375 | 是 |
| `PUB-XLAM-00042` | xlam-schema-mapped | TOOL | public_grep | 0.625 | 是 |
| `PUB-XLAM-00046` | xlam-schema-mapped | TOOL | public_grep | 0.625 | 是 |
| `PUB-XLAM-00051` | xlam-schema-mapped | TOOL | public_grep | 0.500 | 是 |
| `PUB-XLAM-00052` | xlam-schema-mapped | TOOL | public_grep | 0.625 | 是 |
| `PUB-XLAM-00058` | xlam-schema-mapped | TOOL | public_grep | 0.625 | 是 |
| `PUB-XLAM-00071` | xlam-schema-mapped | TOOL | public_grep | 0.625 | 是 |
| `PUB-XLAM-00076` | xlam-schema-mapped | TOOL | public_grep | 0.375 | 是 |
| `PUB-XLAM-00077` | xlam-schema-mapped | TOOL | public_grep | 0.500 | 是 |
| `PUB-XLAM-00078` | xlam-schema-mapped | TOOL | public_grep | 0.375 | 是 |
| `PUB-XLAM-00079` | xlam-schema-mapped | TOOL | public_grep | 0.375 | 是 |
| `PUB-XLAM-00087` | xlam-schema-mapped | TOOL | public_grep | 0.500 | 是 |
| `PUB-XLAM-00098` | xlam-schema-mapped | TOOL | public_grep | 0.625 | 是 |
| `PUB-XLAM-00102` | xlam-schema-mapped | TOOL | public_grep | 0.500 | 是 |
| `U-TR-A-C1-BASE` | train-v2-retained | BASE | json_constraints | 0.375 | 是 |
| `U-TR-A-TOOL-HARD` | train-v2-retained | TOOL | path_params | 0.375 | 是 |
| `U-TR-B-C1-HARD` | train-v2-retained | HARD | json_constraints | 0.375 | 是 |
| `U-TR-B-TOOL-BASE` | train-v2-retained | TOOL | path_params | 0.375 | 是 |
| `U-TR4-00-C1-HARD` | train-v4-oracle | HARD | json_constraints | 0.625 | 是 |
| `U-TR4-00-CHECK-BASE` | train-v4-oracle | HARD | 03 | 0.625 | 是 |
| `U-TR4-02-C1-BASE` | train-v4-oracle | BASE | json_constraints | 0.375 | 是 |
| `U-TR4-03-C1-BASE` | train-v4-oracle | BASE | json_constraints | 0.375 | 是 |
| `U-TR4-04-C1-BASE` | train-v4-oracle | BASE | json_constraints | 0.500 | 是 |
| `U-TR4-05-C1-BASE` | train-v4-oracle | BASE | json_constraints | 0.375 | 是 |
| `U-TR4-06-C1-BASE` | train-v4-oracle | BASE | json_constraints | 0.500 | 是 |
| `U-TR4-07-C1-BASE` | train-v4-oracle | BASE | json_constraints | 0.375 | 是 |
| `U-TR4-08-C1-BASE` | train-v4-oracle | BASE | json_constraints | 0.500 | 是 |
| `U-TR4-16-CHECK-HARD` | train-v4-oracle | HARD | 03 | 0.625 | 是 |
| `U-TR4-17-CHECK-HARD` | train-v4-oracle | HARD | 03 | 0.625 | 是 |
| `U-TR4-19-CHECK-HARD` | train-v4-oracle | HARD | 03 | 0.625 | 是 |
| `U-TR5-00-C1-BASE` | train-v5-teacher | BASE | json_constraints | 0.500 | 是 |
| `U-TR5-05-CHECK-HARD` | train-v5-teacher | HARD | 03 | 0.625 | 是 |
| `U-TR5-10-C1-BASE` | train-v5-teacher | BASE | json_constraints | 0.375 | 是 |
| `U-TR5-15-C1-BASE` | train-v5-teacher | BASE | json_constraints | 0.500 | 是 |
| `U-TR5-18-FILE-BASE` | train-v5-teacher | BASE | 01 | 0.375 | 是 |
| `U-TR5-21-CHECK-HARD` | train-v5-teacher | HARD | 03 | 0.375 | 是 |
| `U-TR5-23-CHECK-HARD` | train-v5-teacher | HARD | 03 | 0.625 | 是 |
| `U-TR5-27-FILE-BASE` | train-v5-teacher | BASE | 01 | 0.375 | 是 |

### 被排除的不完整任务

| task_id | 原因 |
| --- | --- |
| `U-TR5-01-CHECK-HARD` | valid=7,errors=1,expected=8 |


## 14B 结果

### T4 可行性

14B 的 1 任务 preflight 通过：`Qwen/Qwen2.5-14B-Instruct` NF4 在 T4 14.56 GB 上加载成功，
峰值 9.84 GB（reserved 10.23 GB），1/1 `task_success`，staging 178.9 s（含加载约 123 s）。
因此“14B 在 T4 上跑不动”不成立；真正的限制是时间/配额。

### 同任务对照（7B 中间态任务）

在 shard 2 完成前，7B 已筛出 **28 个中间态任务**。对这 28 个任务用同一套 8 次
预注册采样跑 14B（224 条 rollout）。shard 2 完成后官方中间态增至 45 个，因此 14B
只覆盖最终集合的 **28 / 45**，其余 17 个未跑 14B：

| 项目 | 7B | 14B |
| --- | ---: | ---: |
| 任务数 | 28 | 28 |
| rollout 数 | 224 | 224 |
| error | 0 | 0 |
| `task_success` | 102 | 216 |
| 总体通过率 | 45.5% | 96.4% |
| 中间态任务数（30%–70%） | 28 | **0** |
| 与 7B 中间态重合 | — | **0 / 28** |
| 覆盖最终 45 个中间态 | 28 / 45 | 28 / 45 |

逐任务变化：**27/28 个任务升到 8/8（100%）**；1 个任务
`U-TR5-10-C1-BASE` 从 3/8（37.5%）降到 0/8（0%）。该任务是判定器伪影，不是能力退化：
14B 8 条全部输出 `"value": "mode = shared-171010\n"`，与 fixture 文件原文一致，但判定器
把期望值存成 `content.strip()`，因此 `value_correct=false`。同一条 7B 轨迹里也有 3 条
因同样原因失败。详见“判定器伪影”一节。

### 结论

- 7B 的 30%–70% 中间态任务，在 14B 上几乎全部饱和；**中间态是 7B 容量特异性现象**，
  不是任务本身缺少难度梯度。
- 不能把 14B 全量 409 任务 × 8 次跑完：按本次 224 条/4299 s 的速度，3272 条约需
  **17.4 GPU·h**，超过本项目当周剩余 Kaggle 配额；因此没有“14B 全池中间态数”。
- 14B 对照只覆盖 7B 中间态子集，属于条件性子集对照，不外推到全池。
- 具体到最终 45 个中间态：14B 已测 28 个，这 28 个里 14B 中间态为 0；另外 17 个
  在 shard 2 才出现，未进入 14B 对照。报告中的“14B 中间态 0”只适用于已测子集，
  不是 45 个任务的全量结论。


## 判定

- 中间态任务数：**45**（预注册判定 `proceed_final_test_v3`）。
- 训练重叠任务的中间态数：**45** / 394。
- 未进训练的 `stats-heldout-v2` 任务中间态数：**0** / 15。
- 按预注册门槛，任务池已具备构造 `final-test-v3` 的中间态数量；但训练重叠任务只能作为难度校准/诊断集，不能直接支撑泛化增益结论。
- 有 1 个任务因 error 或有效采样不足被排除，不把不完整任务计入中间态。
- 8 次采样只能估计任务级通过率，不能据此宣称 `pass^8` 稳定性。

### 判定器伪影（post-hoc，不改变官方口径）

官方 45 个中间态里有 13 个来自 `json_constraints` 族。逐条检查发现：判定器比较
`payload["value"]` 与 `fixture.main_content`，后者在构造时做了 `content.strip()`；
而 fixture 文件以 CRLF 结尾。模型若忠实复制整行（含尾随 `\r\n`），反而被判
`value_correct=false`。这不是模型能力差异，而是判定器的空白字符伪影。

对 409 个任务的 3272 条 rollout 做一次性重判（只放宽“值等于原文去掉尾随空白”，
不改工具调用、路径、格式等其它门禁）：

| 项目 | 官方口径 | 重判口径 |
| --- | ---: | ---: |
| 中间态任务 | 45 | **35** |
| 受影响 rollout | — | 133 |
| 受影响任务 | — | 31 |
| 移出中间态的任务 | — | 10 |

移出中间态的 10 个任务：`U-TR4-00-C1-HARD`、`U-TR4-02-C1-BASE`、
`U-TR4-03-C1-BASE`、`U-TR4-04-C1-BASE`、`U-TR4-05-C1-BASE`、`U-TR4-06-C1-BASE`、
`U-TR4-07-C1-BASE`、`U-TR4-08-C1-BASE`、`U-TR5-00-C1-BASE`、`U-TR5-15-C1-BASE`。

两套数字同时报告：预注册判定用官方口径，重判只作 post-hoc 敏感性分析。`public_grep`
仍用精确对象相等（`matches` 顺序敏感），没有在重判里放宽，因此 21 个 public_grep
中间态不受影响。

### final-test-v3 冻结

两个口径都冻结，避免事后挑数字：

| 口径 | 文件 | 任务数 | 训练无关任务 | 训练重叠任务 | SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| 官方（预注册） | `private/final_test_v3/tasks.jsonl` | 45 | 0 | 45 | `20e20b3b...c4fa5b` |
| 宽松（重判） | `private/final_test_v3_lenient/tasks.jsonl` | 35 | 0 | 35 | `b9f39aed...a48b77` |

**关键边界**：数量门槛（≥10）满足，但“训练无关”的语义门槛没有满足。45 个中间态
全部来自 train-v5；15 个真正未进训练的 `stats-heldout-v2` 任务中间态为 **0**。因此
不能写“held-out 问题已解决”，只能写“任务池存在足够的中间态，但它们与训练集重叠”。

重测计划（若启动）：

1. 把 `final-test-v3` 明确标为**难度校准诊断集**，不用于泛化增益结论；
2. 如需 clean held-out，两条路二选一：
   - 把入选任务从 SFT 训练集移除后重训，再做对比；或
   - 用同一任务族新造未进训练的任务，重新过教师门禁和 8 次采样；
3. SFT/DPO 对比必须同时报 BASE/HARD 与 `training_overlap` 标记；
4. 在本轮预算内不启动重训；先完成收尾材料，是否重训由用户决定。

## 已知边界

- 采样为 8 次，只能估计任务级通过率，不足以做严格 `pass^8` 稳定性结论。
- `heldout难度筛选.csv` 含全部 409 个任务的逐任务结果，`middle_state` 列标记 45 个
  官方中间态；重判口径见 `json_whitespace_rejudge.json`，不在 CSV 里混列。
- xLAM public grep 是合成单轮任务，与本地六工具任务分布不同；报告按来源分拆。
- 14B 只在 shard 2 完成前的 28 个中间态上跑过（0 个 14B 中间态，重合 0/28）；
  最终 45 个里有 17 个未跑 14B，不能把“14B 中间态 0”外推到全量 45。

## 执行记录（7B）

| 项目 | 值 |
| --- | --- |
| 模型 | `Qwen/Qwen2.5-7B-Instruct`，NF4，`device_map={"": 0}` |
| 硬件 | Kaggle Tesla T4，14.56 GB |
| 分片 | 3 片，137 / 136 / 136 个任务 |
| 每任务采样 | 8 次，`temperature=0.7`、`top_p=0.9`、`seed=20261019+rollout_index` |
| kernel 0 | `autumnsq/lite-agent-held-out-difficulty-screen-0`，含 5 任务 staging |
| kernel 1 | `autumnsq/lite-agent-held-out-difficulty-screen-1` |
| kernel 2 | `autumnsq/lite-agent-held-out-difficulty-screen-2`，14:23–18:21 +08:00，`COMPLETE`，1088/1088 |
| 开始时间 | 2026-10-25 09:53 +08:00 |
| staging | 5/5 任务 `task_success=true`，`errors=0`，耗时 98.1 s，显存峰值 5.62 GB |

staging 通过标准为 `errors==0` 且至少 1 条 `task_success=true`；本次 5 条全部通过，
因此没有把“模型能加载”误当成“任务可执行”。staging 的 5 个任务各只跑 1 次，不参与
中间态判定。

### 成本记录

| 项目 | 值 |
| --- | ---: |
| 并发 GPU session | 最多 2（shard 0 + shard 1 并行；shard 2 与 14B 串行） |
| 单 session 运行上限 | 27000 s（7.5 h） |
| 7B 全量 GPU 时间 | 16270.8 + 15851.6 + 14112.2 s = **12.84 GPU·h** |
| 14B 子集 GPU 时间 | preflight 178.9 s + 28 任务 4309.1 s = **1.25 GPU·h** |
| 失败 session GPU 时间 | v1 103 s + v2 181 s + shard 2 下载失败 26 s ≈ **0.09 GPU·h** |
| 本阶段合计 | **约 14.18 GPU·h** |
| 失败 session | 见 `kaggle_runbook.md` 第 15 节（v1 导入失败、v2 `args.limit` 失败） |

### 判定口径的两个硬边界

1. **训练重叠**：409 个任务里 394 个来自 train-v5，已进入 SFT 训练数据。即使它们
   通过率落在 30%–70%，也只能叫“难度校准/诊断集”；把它们直接当 `final-test-v3`
   比较 SFT，不能写成泛化增益。报告按 `training_overlap` 分开统计。
2. **判定器版本**：运行的 kernel 用 `public_grep` 精确对象相等判定（`matches` 顺序
   敏感）。本地源码已恢复为同一版本，避免用宽松判定重算而污染结果。若后续决定放宽，
   必须整批重跑，不能混用两版结果。
