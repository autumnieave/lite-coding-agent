# Cross-template experiment control variables

这些值是阶段 1 的唯一控制变量源。脚本通过 `CONTROL_VARS.md` 的 JSON 块读取；不要在单个脚本里另设固定值。

```json
{
  "total_train_rows": 401,
  "train_v5_rows": 468,
  "optimizer_steps": 100,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "learning_rate": 0.0002,
  "num_train_epochs": 2,
  "max_seq_length": 3072,
  "seed": 20261019,
  "loss_mode": "assistant_only",
  "lora_r_default": 64,
  "e5b_negatives": 36,
  "probe_items": 32,
  "probe_tasks": 16,
  "evaluation_tasks": 45
}
```

| 变量 | 值 | 说明 |
| --- | ---: | --- |
| `total_train_rows` | 401 | E5-B / E1 / E2 固定训练行数 |
| `train_v5_rows` | 468 | E1 的采样源行数 |
| `optimizer_steps` | 100 | 目标值：401 行 × 2 epoch ÷ 8；实际以 `run_manifest.json` 为准，`STRICT_CONTROL=1` 时不一致即停 |
| `per_device_train_batch_size` | 1 | 单卡 batch |
| `gradient_accumulation_steps` | 8 | 等效 batch 8 |
| `learning_rate` | 2e-4 | 与 train-v6 相同 |
| `num_train_epochs` | 2 | 与 train-v6 相同 |
| `max_seq_length` | 3072 | 与 train-v6 相同 |
| `seed` | 20261019 | 采样、负样本构造和数据顺序统一使用 |
| `loss_mode` | assistant_only | 与 train-v6 相同 |
| `lora_r_default` | 64 | 非 E2 实验使用 |
| `e5b_negatives` | 36 | baseline 中的 executed_violation 条数 |
| `probe_items` | 32 | untagged 单组的探针条数 |
| `probe_tasks` | 16 | 探针任务数 |
| `evaluation_tasks` | 45 | 45 任务通过率评测任务数 |

修改任何值都必须同步修改本文件，并重新跑 dry-run；否则阶段 1 的“只变目标变量”不成立。
