# GRPO 组内方差预检结果（v2 中间态池）

> **最新口径（2026-10-26）**：45 任务正确预检已经执行：44 个 8/8、1 个 7/8、中间态 0，
> 结论是 SFT 饱和、GRPO 不该做。本文记录的是旧 20 任务子集预检，只能作历史对照；
> 不能替代 45 任务口径，也不写“GRPO 组内方差为 0”。最新结果见
> `docs/evidence/GRPO正确预检.md`。


**日期**：2026-10-26  
**范围**：只验证 SFT 策略在中间态 prompt 上是否存在组内 reward 方差；不训练 GRPO，
不验证 GRPO 有效性。  
**结论**：20 个 prompt × 8 rollout = 160 条全部有效，`task_success=160/160`，
`measured_positive=0/20`、`measured_zero=20/20`；判定
**“实测组内方差稀疏（0/20），GRPO 不启动”**。

## 1. 运行环境与产物

| 项目 | 值 |
| --- | --- |
| Kernel | `autumnsq/lite-agent-grpo-preflight-v3`（version 1） |
| Kernel 状态 | `COMPLETE`；summary `status=ok`、`stage=done` |
| GPU | Tesla T4，14.56 GB |
| 运行时 | PyTorch `2.11.0+cu128`，CUDA `12.8` |
| 总耗时 | 2509.9 s（41.8 min，0.70 GPU·h） |
| 其中 full 评测 | 2237.3 s |
| 其中 staging | 144.7 s |
| 显存峰值 | 6.84 GB reserved（推理阶段） |
| 成本 | Kaggle 免费 T4，边际成本 0；不按租卡单价折算 |

kernel 输出在 `private/kaggle_pull_grpo_preflight_v3/`（本地，不入 Git）。

## 2. 输入与版本锁定

| 输入 | SHA-256 / 值 |
| --- | --- |
| 学生基座 | `Qwen/Qwen2.5-7B-Instruct`，4-bit NF4 |
| SFT adapter | train-v5 assistant-only，`f5b685d4da795a783fbcb21b15880daebfa08c64b1661792f7e2c6caab29fea7`（645,975,704 bytes） |
| train-v5 | `6ee7e79d87cb7c41f4c1b91964d70abcf5e886c5ac75450318760a805b78f830` |
| final-test-v3 task specs | `20e20b3beabaf0c48c00dae56d97c1aea356b1ac89c4cdb0fd9decc799c4fa5b` |
| 45 中间态池 `task_ids.jsonl` | `1f80a63acc59d03262013e4864474f352e7e6c6c53b90abc894957c855dfffdd` |
| 45 中间态池 `prompts.jsonl` | `32453d1f860701e334aed9639870e7193998193dcf997e9b37f2068f0eacc3e1` |
| 45 中间态池 manifest | `83c786b7337b6c1f3a777e7af46e166a9bf1302559841992656c161148b157a4` |
| 20 条预检池 | `2bd5de68a2adb6eec6a73ade188e6d7b72edf436ed88f297284fc9ebcdbebc7d` |
| `runs.jsonl` | `f4da96098cfd8e3d243d2c2e2cca43cf11efeda80b1cf7439e293bab524523b6` |
| full `summary.json` | `b1971d8cb7916ea89ef81a6511a1e75258d18e6791a97a5504a425dc84659f6a` |
| kernel 总 summary | `2da90d9b158253c97bd13ef562185a0c0ef075c61ffe099a691608af3a9fcb84` |

adapter 使用 train-v5 assistant-only，而不是 train-v6 的 `37928a28...`：后者来自
train-v6 clean held-out 重训，和 v2 中间态池的分布不一致；本预检要测的是 train-v5
策略在中间态 prompt 上的行为。

## 3. 任务选择与采样协议

- 45 条池全部是 baseline 7B 通过率 37.5%–62.5% 的中间态任务，且 45/45
  `training_overlap=true`。
- 从中按类别比例选 20 条：BASE 6、HARD 4、TOOL 10；通过率构成 37.5% 7 条、
  50.0% 6 条、62.5% 7 条。
- 选择规则：按类别配额 + `SHA-256(seed:task_id)` 稳定排序，不做人工挑题。
- staging：1 个 prompt（`U-TR4-02-C1-BASE`）× 8 rollout，先确认 full-loop、
  reward 和 checkpoint 路径可用。
- full：20 prompt × 8 rollout = 160 条；`temperature=0.7`、`top_p=0.9`、
  `seed=20261019+rollout_index`、`max_model_len=8192`、`max_tokens=512`。
- 每个 rollout 都跑完整 agent loop，不是只重采样最后一轮 completion。
- 失败处理写死为一次：staging 或 full 失败即停，不反复重推；本轮 staging 与 full
  都一次通过。

## 4. 结果

### 4.1 staging

| 项目 | 值 |
| --- | ---: |
| prompt | 1 |
| rollout | 8 |
| valid | 8 |
| errors | 0 |
| task_success | 8/8 |
| 结论 | staging 通过，进入 full |

### 4.2 full：逐任务

| task_id | category | family | rollouts | valid | errors | success | pass_rate | variance | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| U-TR4-02-C1-BASE | BASE | json_constraints | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| U-TR4-03-C1-BASE | BASE | json_constraints | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| U-TR5-18-FILE-BASE | BASE | 01 | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| U-TR4-04-C1-BASE | BASE | json_constraints | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| U-TR4-06-C1-BASE | BASE | json_constraints | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| U-TR5-15-C1-BASE | BASE | json_constraints | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| U-TR4-17-CHECK-HARD | HARD | 03 | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| U-TR5-21-CHECK-HARD | HARD | 03 | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| U-TR4-00-CHECK-BASE | HARD | 03 | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| U-TR4-00-C1-HARD | HARD | json_constraints | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| PUB-XLAM-00087 | TOOL | public_grep | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| PUB-XLAM-00031 | TOOL | public_grep | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| PUB-XLAM-00042 | TOOL | public_grep | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| PUB-XLAM-00028 | TOOL | public_grep | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| PUB-XLAM-00002 | TOOL | public_grep | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| PUB-XLAM-00078 | TOOL | public_grep | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| PUB-XLAM-00039 | TOOL | public_grep | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| PUB-XLAM-00051 | TOOL | public_grep | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| PUB-XLAM-00001 | TOOL | public_grep | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |
| PUB-XLAM-00058 | TOOL | public_grep | 8 | 8 | 0 | 8 | 1.0000 | 0.000000 | measured_zero |

方差口径：二值 `task_success` 的样本方差（`ddof=1`），不是 rollout 数。20 个 prompt 的
8 条 rollout 全部成功，因此每个 prompt 的样本方差都是 0.000000；`bernoulli_variance`
同样为 0。

### 4.3 汇总

| 指标 | 值 |
| --- | ---: |
| prompt 数 | 20 |
| rollout 数 | 160 |
| valid rollout | 160 |
| errors | 0 |
| task_success | 160/160（100%） |
| within_steps 失败 | 0 |
| measured_positive | 0/20（0.0%） |
| measured_zero | 20/20 |
| not_measured | 0/20 |
| 预注册门槛 | `>=30%` prompt 有正方差才通过 |
| verdict | **`sparse`** |
| 判定 | **实测组内方差稀疏（0/20），GRPO 不启动** |

补充行为统计：160 条 rollout 的步数分布为 1 步 88 条、2 步 71 条、3 步 1 条；
`within_steps` 160/160 通过，`tools_allowed` 160/160 通过，`stopped_reasons` 全部为
`completed`。没有出现 A-C1-HARD 式的重复读取循环。

## 5. 解释与决策

1. **零方差是实测，不是未验证**。20 个 prompt 各 8 条 rollout 都完整、无 error，
   且全部成功；因此每个 group 的 reward 都是常数 1，20/20 饱和，没有中间态；这不作为 45 任务池的方差结论。
2. **这不是“GRPO 无效”的证据**。本轮只测到“在 train-v5 SFT 策略 + 20 个
   training-overlap 中间态 prompt 上，没有组内方差”，没有训练 GRPO，也没有
   reward hacking 或泛化结论。
3. **为什么 baseline 有中间态、SFT 却全对**：这 20 条的 37.5%–62.5% 来自更早的
   base 7B 8-rollout 难度筛选；SFT 后在同族训练任务上全部 8/8，符合“训练集覆盖后
   拟合”的解释，不能写成跨模板泛化。20/20 `training_overlap=true` 也支持这个解释。
4. **决策**：按预注册门槛，`0/20 < 30%`，GRPO 正式不启动。不调 reward 权重、不换
   prompt 反复试、不跑全量 GRPO；项目训练侧保持“违反归因 + 可复现评测 + 训练能力
   边界验证”的定位。
5. **如果将来要验证 GRPO**：需要一批 SFT 策略通过率落在 20%–80% 的 clean prompt，
   且与训练集不重叠；当前 45 个中间态任务全部训练重叠，clean held-out 上 SFT 又已
   99.8% 饱和，因此现成池子不具备组内方差信号。构造新池属于后续实验，不在本轮结论内。

## 6. 边界

- 预检只验证组内方差信号，不验证 GRPO 有效；
- 20/45 是分层子集，不是 45 条全池；8 rollout/prompt，单轮运行；
- 20/20 与训练集重叠，不能当 clean held-out 或泛化证据；
- 只测 SFT 策略，不比较 baseline vs SFT 的增益；
- 不写“GRPO 无效”，也不把历史上的 `0/800 rollout` 写成“方差为 0”；
- 未修改 `core/loop.py`、`compaction.py`、`constraints.py`。

## 7. 复现

```powershell
# 1) 构建 kernel 包（adapter 从 train-v5 kernel source 挂载）
.\.venv\Scripts\python.exe training/kaggle/build_grpo_preflight_kernel.py `
  --root . `
  --out-dir private/kaggle_kernel_grpo_preflight_v3 `
  --kernel-id autumnsq/lite-agent-grpo-preflight-v3

# 2) 单次推送（失败不重推）
.\.venv\Scripts\python.exe -m kaggle kernels push -p private/kaggle_kernel_grpo_preflight_v3

# 3) 拉取输出后独立复核方差
.\.venv\Scripts\python.exe scripts/analyze_grpo_preflight.py `
  --runs private/kaggle_pull_grpo_preflight_v3/grpo_preflight/full/sft/runs.jsonl `
  --pool private/grpo_middle_pool/preflight_20.jsonl `
  --expected-tasks 20 --rollouts-per-task 8
```

**相关文档**：`docs/evidence/GRPO候选池v2_中间态.md`、
`docs/evidence/GRPO失败诊断.md`、`docs/plan/训练侧执行计划.md`。
