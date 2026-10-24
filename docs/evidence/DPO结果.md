# DPO 结果（租卡执行记录）

**日期**：2026-10-24  
**状态**：**未执行（环境阻塞）**  
**输入**：`private/dpo_v2/`（45 train / 13 val）、train-v5 assistant-only adapter  
**预算与失败诊断**：`docs/evidence/DPO失败诊断.md`、`docs/evidence/DPO偏好对预检.md`

## 0. 结论先行

- 版本对齐和 T4 smoke 已完成：三个 SHA 全部通过，T4 仍在 reference precompute 的
  `logits.log_softmax` 上 OOM（请求 3.21 GiB，GPU 仅剩 122.81 MiB）。
- 显存预算结论：24GB 保守余量 3–7 GB，满足预注册的 ≥3GB 门槛；**首选 24GB 做一次
  `max_steps=1` smoke**，40GB 只作 fallback。
- 当前环境没有 `vastai` / `runpodctl` / `gcloud` / `aws` / `az`，也没有 24GB/40GB
  租卡凭据；本轮 **0 个租卡 session、0 元、0 条 DPO rollout**。
- 因此本文不能写“DPO 有增益”，也不能写“24GB 不支持”；只能写“T4 关闭，24GB smoke
  未执行”。

## 1. 执行状态

| 项目 | 状态 |
| --- | --- |
| dpo_v2 数据 | 已冻结：45 train / 13 val；SHA 见 `DPO失败诊断.md` 6.1 |
| train-v5 adapter | 已校验：645,975,704 bytes；SHA `f5b685d4...29fea7` |
| T4 smoke | 已执行并失败：`OutOfMemoryError`，阶段 `reference precompute 0/45` |
| 24GB smoke | **未执行**：无租卡接口/凭据 |
| 40GB smoke | **未执行**：24GB 尚未失败，不提前租 |
| DPO 全量 45 对 | 未执行 |
| held-out 三组对比 | 未执行（baseline / SFT / SFT+DPO 均无新 DPO 结果） |
| 成本 | 0 个租卡 session，0 元；T4 失败成本已计入 v3 的 227.2s |

## 2. 预注册止损规则（执行前写死）

1. 24GB 只跑一次 `max_steps=1` smoke；目标是确认 `logits.log_softmax` 不再 OOM。
2. 24GB smoke OOM：立即停止，不调 `batch_size`、不降序列长度、不试 1024，最多换
   40GB 一次。
3. 40GB smoke 仍 OOM：写“24GB/40GB 均不支持 3072 + 双 7B DPO”，DPO 关闭，不再租卡。
4. smoke 通过：跑全量 45 对；held-out 固定为 baseline / SFT / SFT+DPO 三组，沿用同一
   6 任务 × 10 rollout、`temperature=0.0`、`top_p=None`、`seed=20261019`。
5. 任何失败都记录卡型、阶段、`error_kind`、GPU 显存、时长和成本；不把 OOM 写成
   “DPO 方法无效”。

## 3. 租到 24GB 后的执行清单

前置：Python 3.11、CUDA 可用、`pip install -e ".[dev]"` 与
`pip install -r training/requirements-kaggle.txt` 完成；模型、数据和 adapter 路径与
v3 manifest 一致。

```powershell
# 1) 单 step smoke（只验证显存路径）
python training/train_dpo.py `
  --config training/configs/dpo_7b_t4.json `
  --model-path <Qwen2.5-7B-Instruct> `
  --train-file private/dpo_v2/dpo_train.jsonl `
  --val-file private/dpo_v2/dpo_val.jsonl `
  --init-adapter <train-v5-assistant-only-adapter> `
  --output-dir <out>/dpo_smoke `
  --max-steps 1 `
  --expected-train-sha256 691ea2667893b483d02004e9f8af800a6978df2b9dba80d19a245f4098b43aed `
  --expected-val-sha256 4072a9fec7616e6df3e9a17ecafe30f0c404a6495da63204ca0f2413ec8466f2 `
  --expected-init-adapter-sha256 f5b685d4da795a783fbcb21b15880daebfa08c64b1661792f7e2c6caab29fea7

# 2) smoke 通过后才跑全量 45 对；去掉 --max-steps
python training/train_dpo.py ... --output-dir <out>/dpo_full
```

smoke 验收：`run_manifest.json` 的 `status` 不是 `error`、`error_kind` 为空、至少完成
1 个 optimizer step；否则按第 2 节止损。

## 4. 结果表（当前无数据）

| 组 | task_success | valid stable | within_steps | 工具结果正确率 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 未执行 | 未执行 | 未执行 | 未执行 | 沿用 v9 历史结果，不冒充本轮 |
| SFT | 未执行 | 未执行 | 未执行 | 未执行 | 沿用 train-v5 assistant-only |
| SFT + DPO | 未执行 | 未执行 | 未执行 | 未执行 | 0 rollout |

本轮不产生新的 held-out 数字；历史 SFT 结果仍以 `docs/evidence/训练管线验证.md` 为准。

## 5. 口径边界

- 58 对偏好数据全部来自 CTX 的 `code_prefix`；即使未来跑通 DPO，结论也只限于该类型，
  不能外推到 BASE/HARD/TOOL/CONFLICT 或“DPO 普遍有效”。
- T4 失败是显存/实现路径问题，不是 DPO 方法失败；24GB 未执行，也不能写成硬件不支持。
- 只有拿到 24GB 或 40GB 的真实 smoke 与 held-out 输出，才能关闭“DPO 结果”这一项。

**相关文档**：`docs/evidence/DPO失败诊断.md`、`docs/evidence/DPO偏好对预检.md`、
`docs/evidence/kaggle_runbook.md`、`docs/plan/训练侧执行计划.md`。
