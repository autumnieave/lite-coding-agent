# GRPO 候选池 v2：45 个中间态任务

**日期**：2026-10-26
**目的**：修正“GRPO 组内方差为 0”的口径，并把候选池从 v1 的 train-v5 全量分层抽样
改为 baseline 通过率落在 30%–70% 的中间态任务。
**状态**：池子已冻结；**时间限制，只修正口径，预检未做**。

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

## 5. 预检状态与判定口径

- **本轮未跑预检（时间限制，只修正口径，预检未做）**：只修口径、确认 v1 池来源、
  冻结 v2 池；没有 7B SFT rollout。
- **基础设施现状**：Kaggle status API 已恢复（不再 403），但旧 v2 kernel 仍是
  `KernelWorkerStatus.ERROR`；仓库没有 v2 中间态 full-loop 的 kernel builder，
  重跑需要新 builder 和 adapter 挂载，未在本轮时间盒内执行。
- 如果后续跑：从 45 条中选 20–30 条，每条 8 rollout；失败 1 次即停。
- 判定：每个 prompt 的 8 个二值 reward 中，方差 > 0 记为“有组内方差”。
  `>=30%` 的 prompt 有组内方差 → “预检通过，组内方差信号存在”；`<30%` →
  “实测组内方差稀疏（X/N），GRPO 不启动”。
- **没有 rollout 时**：写“组内方差未验证”，不写“方差为 0”，也不写“组内方差不足”。

## 6. 相关文档

- `docs/evidence/GRPO失败诊断.md`
- `docs/evidence/GRPO_prompt池方差预检.md`
- `docs/plan/训练侧执行计划.md`
