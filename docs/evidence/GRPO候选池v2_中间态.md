# GRPO 候选池 v2：45 个中间态任务

> **最新口径（2026-10-26）**：45 任务正确预检已经执行：44 个 8/8、1 个 7/8、中间态 0，
> 结论是 SFT 饱和、GRPO 不该做。本文记录的是旧 20 任务子集预检，只能作历史对照；
> 不能替代 45 任务口径，也不写“GRPO 组内方差为 0”。最新结果见
> `docs/evidence/GRPO正确预检.md`。


**日期**：2026-10-26
**目的**：修正“GRPO 组内方差为 0”的口径，并把候选池从 v1 的 train-v5 全量分层抽样
改为 baseline 通过率落在 30%–70% 的中间态任务。
**状态**：池子已冻结；2026-10-26 已完成 20 条 × 8 rollout 的 full-loop 预检，
结论见 `docs/evidence/GRPO预检结果.md`。

## 1. 为什么换池

v1 的 100 条候选 prompt 来自 `train-v5` 的 468 条成功轨迹，按
BASE/HARD/CTX/TOOL/CONFLICT 五类做稳定哈希抽样（每类 20 条）。它**不是**从 45 个
中间态任务中选的，因此不能回答“在 baseline 有区分度的任务上，SFT 策略的 8 次采样
是否存在组内方差”。

v2 直接使用 `private/kaggle_difficulty/final/middle_tasks.jsonl` 中 baseline 通过率
37.5%–62.5% 的 45 个任务；它们全部满足 30%–70% 门槛，且 8 次 rollout 完整、0 error。

## 2. 来源与选择规则

| 项目 | 值 |
| --- | --- |
| 中间态任务文件 | `private/kaggle_difficulty/final/middle_tasks.jsonl` |
| 中间态任务 SHA-256 | `60ad8b719c15d0b91806500c3d3d153e620134e385865128f82c466d9a982986` |
| 任务 spec 文件 | `private/final_test_v3/tasks.jsonl`（官方 45 任务） |
| 任务 spec SHA-256 | `20e20b3beabaf0c48c00dae56d97c1aea356b1ac89c4cdb0fd9decc799c4fa5b` |
| 选择规则 | `0.30 <= pass_rate <= 0.70`；`valid_rollouts == expected_rollouts`；`errors == 0` |
| 选中任务数 | 45 / 45 |

## 3. 分布

| 维度 | 分布 |
| --- | --- |
| 类别 | BASE 13、HARD 9、TOOL 23 |
| family | `public_grep` 21、`json_constraints` 13、`03` 7、`path_params` 2、`01` 2 |
| baseline 通过率 | 37.5% 18 条、50.0% 11 条、62.5% 16 条 |
| 训练重叠 | 45/45 `training_overlap=true` |

**边界**：这 45 个任务与 `final-test-v3` 是同一批诊断任务，且全部与 train-v5 有同族
训练重叠。它们只用于 GRPO reward/组内方差预检，**不能**写成跨模板泛化或 clean
held-out 证据。

## 4. 冻结产物

产物在 `private/grpo_middle_pool/`（本地，不入 Git）：

| 文件 | 行数 | SHA-256 |
| --- | ---: | --- |
| `task_ids.jsonl` | 45 | `1f80a63acc59d03262013e4864474f352e7e6c6c53b90abc894957c855dfffdd` |
| `prompts.jsonl` | 45 | `32453d1f860701e334aed9639870e7193998193dcf997e9b37f2068f0eacc3e1` |
| `manifest.json` | — | `83c786b7337b6c1f3a777e7af46e166a9bf1302559841992656c161148b157a4` |

逐任务表：`docs/evidence/GRPO候选池v2_中间态.csv`（SHA-256
`5d2fd6416c8be81dfa8b2f4e50800f99a7755398f722cf4893e2106cdbc82c6d`）。

复现命令：

```powershell
.\.venv\Scripts\python.exe scripts/build_grpo_middle_pool.py `
  --middle-tasks private/kaggle_difficulty/final/middle_tasks.jsonl `
  --task-specs private/final_test_v3/tasks.jsonl `
  --out-dir private/grpo_middle_pool
```

## 5. 预检结果（2026-10-26）

- **执行**：从 45 条中按类别比例选 20 条（BASE 6 / HARD 4 / TOOL 10），每条
  8 rollout；`temperature=0.7`、`top_p=0.9`、`seed=20261019+rollout`、
  `max_model_len=8192`、`max_tokens=512`，full-loop。
- **staging**：1 prompt × 8 rollout，8/8 成功、0 error，144.7s。
- **full**：20 × 8 = 160 条全部完整，`task_success=160/160`、`errors=0`；
  20 个 prompt 都是 8/8。
- **方差**：`measured_positive=0/20`、`measured_zero=20/20`、
  `not_measured=0/20`，正方差比例 0.0%。按预注册 `>=30%` 门槛，判定
  **“实测组内方差稀疏（0/20），GRPO 不启动”**。
- **解释**：20/20 任务与训练集重叠；baseline 的 37.5%–62.5% 中间态在 train-v5
  assistant-only SFT 后变成 8/8，符合训练覆盖后的拟合，不能写成泛化。20/20 饱和是
  实测结果，不是“未验证”；但本轮没有训练 GRPO，不能写“GRPO 无效”。
- **产物**：`docs/evidence/GRPO预检结果.md`；raw runs 与 summary 在
  `private/kaggle_pull_grpo_preflight_v3/`（本地，不入 Git）。

## 6. 相关文档

- `docs/evidence/GRPO失败诊断.md`
- `docs/evidence/GRPO_prompt池方差预检.md`
- `docs/plan/训练侧执行计划.md`
