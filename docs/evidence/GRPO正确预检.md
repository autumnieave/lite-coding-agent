# GRPO 正确预检（train-v6 adapter）

**结论先行**：用 train-v6 adapter 对 45 个中间态候选任务各跑 8 次，45 个任务中
44 个为 8/8、1 个为 7/8，落在预注册“2/8–6/8”区间的任务数为 **0**。按预注册
判定，当前 SFT 已饱和，**GRPO 不该做**；没有启动 20 条 × 8 rollout 的方差预检，
因而不存在可报告的 `measured_positive / measured_zero` 组内方差统计。

## 1. 预注册口径

| 项 | 值 |
| --- | --- |
| 候选任务池 | `private/grpo_middle_pool/task_ids.jsonl`，45 个任务 |
| 候选池 task_ids SHA-256 | `1f80a63acc59d03262013e4864474f352e7e6c6c53b90abc894957c855dfffdd` |
| 候选池 prompts SHA-256 | `32453d1f860701e334aed9639870e7193998193dcf997e9b37f2068f0eacc3e1` |
| 评估任务定义 | `private/final_test_v3/tasks.jsonl`，SHA-256 `20e20b3beabaf0c48c00dae56d97c1aea356b1ac89c4cdb0fd9decc799c4fa5b` |
| 训练数据（仅作输入校验） | train-v5，SHA-256 `6ee7e79d87cb7c41f4c1b91964d70abcf5e886c5ac75450318760a805b78f830` |
| Adapter | train-v6 assistant-only，SHA-256 `37928a28d67a5a8350f3b49450fc6945c3ffe05c09da7ea0de36136b4a6a4b96` |
| 每个任务 rollout | 8 |
| 采样 | temperature=0.7，top_p=0.9，base seed=20261019，seed=base_seed+rollout_index |
| 中间态定义 | SFT `task_success` 在 2/8–6/8（含端点），且 `errors=0` |
| 方差阈值 | 中间态任务 ≥10 时，最多选 20 个任务，另用 seed=20262019 再跑 8 次；≥30% 任务为正组内方差才判 `pass` |

预注册冻结在 kernel：
`training/kaggle/build_grpo_preflight_v6_kernel.py`；
选择/方差逻辑：`scripts/grpo_preflight_v6.py`。

## 2. 运行环境

| 项 | 值 |
| --- | --- |
| 平台 | Kaggle 免费 GPU |
| GPU | Tesla T4，14.56 GB |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| 推理 | transformers-offline，4-bit NF4 |
| 最大模型长度 | 8192 |
| 每条最大生成 token | 512 |
| 总耗时 | 4992.8 s（83.2 min，1.39 GPU·h） |

## 3. 45 任务筛选结果

- 任务数：45
- Rollout 数：45 × 8 = **360**
- 有效 rollout：**360/360**
- 错误：**0**
- `within_steps` 失败：**0**
- 总成功：**359/360（99.72%）**
- 每任务成功数分布：

| 每任务成功数 | 任务数 |
| ---: | ---: |
| 8/8 | 44 |
| 7/8 | 1 |
| 2/8–6/8 | **0** |
| 0–1/8 | 0 |

唯一非满分任务：`PUB-XLAM-00051`，rollout 6 返回
`{"matches": []}`，`checks.matches_complete=false`；该任务仍为 7/8。单条失败不足以
构成稳定的能力失败模式。

## 4. 判定

```text
middle_count = 0
middle_task_ids = []
decision = sft_saturated_grpo_should_not_run
```

判定链路：

1. 预注册要求先找到 `2/8–6/8` 的中间态任务；
2. 实测 45 个候选中 0 个落入该区间（44 个满分、1 个 7/8）；
3. 因而 `<5` 分支成立：**SFT 饱和，GRPO 不该做**；
4. 没有启动 `preflight20`，因此没有 20 条 × 8 rollout 的组内方差结果，不能写
   “组内方差为 0”或“组内方差不足”；准确表述是“中间态任务为 0，方差预检未进入”。

## 5. 边界

- 45 个候选任务全部与训练集同族/重叠，**不是 clean held-out**，本结果不能作为
  跨模板泛化或 SFT 有效性证据；它是“是否值得启动 GRPO”的筛选。
- 本预检只回答“有没有组内方差信号”，不回答“GRPO 是否有效”；本轮连方差统计
  都未启动。
- 没有重推失败 kernel，没有伪造 rollout；所有数字来自
  `private/kaggle_pull_grpo_preflight_v6/grpo_preflight_v6/screen45/summary.json`
  与同目录 `sft/runs.jsonl`。
- 唯一一次失败是 `matches_complete=false`，不是格式/工具 schema 问题；不得据此
  推断 GRPO 会改善该任务。
