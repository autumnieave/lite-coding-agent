# Clean held-out 配对统计（train-v6）

> 目的：train-v5 的 45 个中间态任务全部与训练集重叠。train-v6 移除这些任务的
> 全部 67 条轨迹后重训 7B SFT，在 45 个任务上与已有 baseline rollout 做配对比较。

## 1. 版本与数据完整性

- train-v6 train SHA-256：`920a4ee73199b64a81db867deb5bd1e143b40c27420f7d2a5ce4c713a5cdaf70`
- SFT adapter SHA-256：`37928a28d67a5a8350f3b49450fc6945c3ffe05c09da7ea0de36136b4a6a4b96`
- 任务数：45
- baseline：360 行，缺失任务 0，error 行 0，有效 rollout 分布 8 条 × 45 任务
  - 来源：`private\kaggle_difficulty\runs0\difficulty\shard-00-of-03\baseline\runs.jsonl`, `private\kaggle_difficulty\runs1\difficulty\shard-01-of-03\baseline\runs.jsonl`, `private\kaggle_difficulty\runs2\difficulty\shard-02-of-03\baseline\runs.jsonl`
- sft：450 行，缺失任务 0，error 行 0，有效 rollout 分布 10 条 × 45 任务
  - 来源：`private\kaggle_pull_clean_heldout_meta\clean_heldout\full\sft\runs.jsonl`

### 1.1 训练与运行

- 硬件：Tesla T4，3.44 GPU·h，总耗时 12369s。
- 训练：train-v6 401 行 / val 84 行；100 optimizer steps；train loss 0.1644；adapter 645,975,704 bytes。
- tokenizer gate：train 最大 3064 tokens，val 最大 3052 tokens，全部通过 3072 门禁。
- staging：3/3 成功，error 0；full：449/450 成功，error 0。
- 失败 rollout：`PUB-XLAM-00051` rollout 6 failed_checks=['matches_complete']。

### 1.2 Adapter 本地归档

- 路径：`private/kaggle_pull_clean_heldout_v1/clean_heldout/adapter/`
- `adapter_model.safetensors`：645,975,704 bytes，SHA-256 `37928a28d67a5a8350f3b49450fc6945c3ffe05c09da7ea0de36136b4a6a4b96`
- 拉取来源：Kaggle kernel `autumnsq/lite-agent-clean-heldout-v6`
- 结构校验：`safetensors` 可读取 392 个 LoRA tensor；`adapter_config.json` 为 r=64 / peft 0.20.0。
- 加载验证：原 kernel 的 staging 3/3 任务使用同一 adapter，推理峰值显存 6.82 GB；本机仅有 2 GB MX250，未重跑 7B rollout。

## 2. 总体指标

| 指标 | baseline 均值 | SFT 均值 | 效应量 | 95% CI | 任务级 McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| task_success | 49.4% | 99.8% | +50.3pp | [+44.2, +56.4]pp | 5.684e-14 | 1.705e-13 |
| within_steps | 95.3% | 100.0% | +4.7pp | [+1.4, +8.9]pp | 0.01562 | 0.03125 |
| tool_selection_ok | 99.2% | 100.0% | +0.8pp | [+0.0, +3.1]pp | 1 | 1 |

口径：均值为 45 个任务通过率的等权平均；CI 为按任务分层的 Bootstrap 百分位区间
（1000 次，seed=20261025）。Holm 校正覆盖 
task_success、within_steps、tool_selection_ok 三个指标。

### 任务级 McNemar 方向表（task_success）

| SFT 更好 | SFT 更差 | 持平 | 不一致任务数 | p（双侧） |
| ---: | ---: | ---: | ---: | ---: |
| 45 | 0 | 0 | 45 | 5.684e-14 |

口径：任务级 McNemar 以任务为配对单位，不一致对是“SFT 通过率高于 baseline”与“SFT 通过率低于 baseline”的任务；由于 baseline 为 8 rollout、SFT 为 10 rollout，任务级检验是主分析。下面的 rollout 级表只配对共同 rollout_index 0–7，作敏感性分析。

### Rollout 级 McNemar（敏感性分析）

baseline 为 8 rollout、SFT 为 10 rollout，仅按共同 rollout_index 0–7 配对：

| 指标 | 配对数 | baseline 失败 → SFT 成功 | baseline 成功 → SFT 失败 | p（双侧） |
| --- | ---: | ---: | ---: | ---: |
| task_success | 360 | 181 | 0 | 6.525e-55 |
| within_steps | 360 | 17 | 0 | 1.526e-05 |
| tool_selection_ok | 360 | 3 | 0 | 0.25 |

## 3. 模板 / 同族重叠审计

- 45 个 held-out 任务中，train-v6 仍存在同族任务的任务数：45/45。
- 泛化范围判定：`within_family_instance`

| family | held-out 任务数 | train-v6 同族任务数 | train-v6 同族轨迹数 |
| --- | ---: | ---: | ---: |
| `01` | 2 | 7 | 7 |
| `03` | 7 | 54 | 64 |
| `json_constraints` | 13 | 38 | 38 |
| `path_params` | 2 | 58 | 62 |
| `public_grep` | 21 | 71 | 71 |

解释边界：task_id 级去重叠已成立（这些任务的 67 条轨迹已从 train-v6 移除），但同族模板在 train-v6 仍有其他实例。因此本结果证明同族实例泛化，不能证明跨模板泛化；跨模板结论需要整族 held-out 实验。

### 3.1 Prompt 级重叠与分组统计

- 逐字 prompt 与 train-v6 重叠的任务：13/45；归一化模板重叠：23/45。

| 分组 | 任务数 | baseline | SFT | 效应量 | 95% CI | 任务级 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `exact_prompt_overlap` | 13 | 50.0% | 100.0% | +50.0pp | [+39.4, +61.5]pp | 0.0002441 |
| `no_exact_prompt` | 32 | 49.2% | 99.7% | +50.5pp | [+43.6, +57.4]pp | 4.657e-10 |
| `normalized_prompt_overlap` | 23 | 47.8% | 100.0% | +52.2pp | [+43.5, +60.9]pp | 2.384e-07 |
| `no_normalized_prompt` | 22 | 51.1% | 99.5% | +48.4pp | [+39.8, +57.6]pp | 4.768e-07 |

解释：有/无逐字 prompt 重叠两组都提升约 50pp，说明增益不只来自逐字记忆；但 45/45 任务仍有同族模板，仍不能外推为跨模板泛化。

## 4. 中间态是否消失

- baseline 有任务内混合结果的任务数：45/45
- SFT 有任务内混合结果的任务数：1/45
- 判定：SFT 的任务内中间态从 45/45 收缩到 1/45，接近天花板但未完全消失；baseline 仍是完整中间态。

## 5. 逐任务结果

| task_id | baseline | SFT | 差值 |
| --- | ---: | ---: | ---: |
| `PUB-XLAM-00001` | 4/8 | 10/10 | +50.0pp |
| `PUB-XLAM-00002` | 5/8 | 10/10 | +37.5pp |
| `PUB-XLAM-00003` | 3/8 | 10/10 | +62.5pp |
| `PUB-XLAM-00019` | 4/8 | 10/10 | +50.0pp |
| `PUB-XLAM-00023` | 5/8 | 10/10 | +37.5pp |
| `PUB-XLAM-00028` | 5/8 | 10/10 | +37.5pp |
| `PUB-XLAM-00031` | 3/8 | 10/10 | +62.5pp |
| `PUB-XLAM-00039` | 3/8 | 10/10 | +62.5pp |
| `PUB-XLAM-00042` | 5/8 | 10/10 | +37.5pp |
| `PUB-XLAM-00046` | 5/8 | 10/10 | +37.5pp |
| `PUB-XLAM-00051` | 4/8 | 9/10 | +40.0pp |
| `PUB-XLAM-00052` | 5/8 | 10/10 | +37.5pp |
| `PUB-XLAM-00058` | 5/8 | 10/10 | +37.5pp |
| `PUB-XLAM-00071` | 5/8 | 10/10 | +37.5pp |
| `PUB-XLAM-00076` | 3/8 | 10/10 | +62.5pp |
| `PUB-XLAM-00077` | 4/8 | 10/10 | +50.0pp |
| `PUB-XLAM-00078` | 3/8 | 10/10 | +62.5pp |
| `PUB-XLAM-00079` | 3/8 | 10/10 | +62.5pp |
| `PUB-XLAM-00087` | 4/8 | 10/10 | +50.0pp |
| `PUB-XLAM-00098` | 5/8 | 10/10 | +37.5pp |
| `PUB-XLAM-00102` | 4/8 | 10/10 | +50.0pp |
| `U-TR-A-C1-BASE` | 3/8 | 10/10 | +62.5pp |
| `U-TR-A-TOOL-HARD` | 3/8 | 10/10 | +62.5pp |
| `U-TR-B-C1-HARD` | 3/8 | 10/10 | +62.5pp |
| `U-TR-B-TOOL-BASE` | 3/8 | 10/10 | +62.5pp |
| `U-TR4-00-C1-HARD` | 5/8 | 10/10 | +37.5pp |
| `U-TR4-00-CHECK-BASE` | 5/8 | 10/10 | +37.5pp |
| `U-TR4-02-C1-BASE` | 3/8 | 10/10 | +62.5pp |
| `U-TR4-03-C1-BASE` | 3/8 | 10/10 | +62.5pp |
| `U-TR4-04-C1-BASE` | 4/8 | 10/10 | +50.0pp |
| `U-TR4-05-C1-BASE` | 3/8 | 10/10 | +62.5pp |
| `U-TR4-06-C1-BASE` | 4/8 | 10/10 | +50.0pp |
| `U-TR4-07-C1-BASE` | 3/8 | 10/10 | +62.5pp |
| `U-TR4-08-C1-BASE` | 4/8 | 10/10 | +50.0pp |
| `U-TR4-16-CHECK-HARD` | 5/8 | 10/10 | +37.5pp |
| `U-TR4-17-CHECK-HARD` | 5/8 | 10/10 | +37.5pp |
| `U-TR4-19-CHECK-HARD` | 5/8 | 10/10 | +37.5pp |
| `U-TR5-00-C1-BASE` | 4/8 | 10/10 | +50.0pp |
| `U-TR5-05-CHECK-HARD` | 5/8 | 10/10 | +37.5pp |
| `U-TR5-10-C1-BASE` | 3/8 | 10/10 | +62.5pp |
| `U-TR5-15-C1-BASE` | 4/8 | 10/10 | +50.0pp |
| `U-TR5-18-FILE-BASE` | 3/8 | 10/10 | +62.5pp |
| `U-TR5-21-CHECK-HARD` | 3/8 | 10/10 | +62.5pp |
| `U-TR5-23-CHECK-HARD` | 5/8 | 10/10 | +37.5pp |
| `U-TR5-27-FILE-BASE` | 3/8 | 10/10 | +62.5pp |

## 6. 结论

- 分支：`significant_positive`
- 泛化范围：`within_family_instance`
- SFT 在去重叠的中间态任务上出现显著正增量（效应量 50.3pp，Holm p=1.705e-13，95% CI [44.2, 56.4]pp）。 但 45/45 个任务在 train-v6 仍有同族任务，因此只能表述为同族实例泛化，不能写成跨模板泛化。 Prompt 级：13/45 个任务与 train-v6 有逐字重叠；有重叠组 +50.0pp，无重叠组 +50.5pp。

边界：

- baseline 复用 difficulty screen 的已有 8 rollout，不重跑；SFT 为 10 rollout。
- 45 个任务来自 train-v5 的官方中间态口径（baseline 通过率 30%–70%）。
- 只评估 7B + 单一 SFT 配方 + 单一 seed，不推广到其他模型或配方。
- Holm 校正只覆盖本文三个指标；p 值与 CI 均应按 45 个冻结任务解释。
