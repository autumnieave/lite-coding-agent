# DPO 结果（AutoDL RTX PRO 6000 96GB）

**日期**：2026-10-24

**状态**：**已执行完成（smoke 与全量均跑通）**

**输入**：`private/dpo_v2/`（45 train / 13 val）、train-v5 assistant-only adapter
**执行窗口**：2026-10-24 19:30:13 +08:00 至 20:14:30 +08:00

## 0. 结论先行

- 96GB 路径跑通：smoke `status=completed`、1 个 optimizer step；全量
  `status=completed`、5 个 optimizer step。没有 OOM，也不需要改 batch、序列长度或 beta。
- 全量 DPO 的实际更新次数是 **5**，不是 45：45 条样本、`batch=1`、
  `gradient_accumulation_steps=8`，最后 5 条不足以凑成一次完整更新，因此没有进入
  optimizer step。训练集本身仍是 45 对，val 仍是 13 对。
- held-out 三组为同一批 6 任务 × 10 rollout、`temperature=0.0`、`top_p=None`、
  `seed=20261019`、transformers + NF4：
  baseline `40/60`、SFT `60/60`、SFT+DPO `60/60`。
- DPO 在这批已饱和任务上没有可测增益：SFT 与 SFT+DPO 的 task_success、valid stable、
  within_steps 都相同。不能写成“DPO 有效”，也不能写成“DPO 无效”；只能写
  “在这批数据与 held-out 上没有测到额外收益”。
- 58 对偏好数据全部来自 CTX `code_prefix`，结论不能外推到 BASE/HARD/TOOL/CONFLICT
  或“DPO 普遍有效”。
- 实例仍保持运行，未关机；等待用户确认。

## 1. 环境检查

| 项目 | 实际值 | 检查结论 |
| --- | --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition | 目标卡可用 |
| 显存 | 97,887 MiB（约 95.59 GiB） | 96GB 规格 |
| 驱动 | 580.82.09 | 可支持当前 CUDA 12.8 运行时 |
| Python | 3.12.3 | 满足 `>=3.11` |
| PyTorch | 2.8.0+cu128 | `torch.cuda.is_available()=True` |
| CUDA runtime | 12.8 | 与 PyTorch cu128 匹配 |
| compute capability | `(12, 0)` | SM120，NF4 实测可用 |
| transformers | 4.46.3 | 在 `training/requirements-kaggle.txt` 范围内 |
| trl | 0.12.2 | 在 `>=0.9,<0.13` 范围内 |
| peft | **0.20.0** | 与 adapter 保存时的 `peft_version=0.20.0` 对齐 |
| bitsandbytes | 0.50.2 | NF4 前向实测成功 |
| accelerate | 0.34.2 | 可用 |
| datasets | 3.0.1 | 可用 |

环境检查时还做了独立 NF4 前向：构造 `bnb.nn.Linear4bit` 后移到 CUDA，前向成功，
输出 `torch.float16`。本机没有 `nvcc`，但训练只需要 CUDA runtime，不需要本地编译。

### 1.1 第一次 smoke 的依赖错配

第一次 smoke 在加载 adapter 时失败，尚未进入训练：

```text
TypeError: LoraConfig.__init__() got an unexpected keyword argument 'alora_invocation_tokens'
```

根因不是 96GB 显存，也不是训练配方：train-v5 adapter 由 `peft 0.20.0` 保存，
实例初始安装的是 `peft 0.13.2`，旧版 `LoraConfig` 不认识新字段。把 peft 对齐到
`0.20.0` 后，同样命令的 smoke 通过。训练配置、数据、adapter 和序列长度均未改变。

## 2. 96GB 显存预算与实际占用

### 2.1 事前预算

| 项目 | 估算 | 说明 |
| --- | ---: | --- |
| policy 4-bit 7B | 约 4 GB | NF4 权重 + train-v5 LoRA |
| reference | 约 0 GB | 复用同一 4-bit 基座并关闭 adapter |
| optimizer states | 1–2 GB | `paged_adamw_8bit`，只覆盖 LoRA 参数 |
| 激活值 | 4–6 GB | 3072 序列、batch 1、非重入 checkpointing |
| `logits.log_softmax` | 3.2–3.4 GB | 双行、约 3008 token、151,643 词表 |
| CUDA context + 碎片 | 1–2 GB | allocator 与运行时开销 |
| **事前合计** | **14–18 GB** | 96GB 预计余量 78–82 GB |

### 2.2 实测

全程每 5 秒采样一次 `nvidia-smi`，峰值 **29.32 GiB / 97,887 MiB**，
剩余约 66.27 GiB；smoke 阶段约 28.3 GiB。实测高于事前 14–18 GB 的估算，说明
4-bit 模型加载、LoRA、完整词表 logits 与 allocator 的综合开销比静态估算更大，
但 96GB 仍有充足余量，未发生 OOM。

## 3. DPO smoke

配置与 T4 完全一致，只改硬件：

- `max_seq_length=3072`、`max_prompt_length=3072`
- `beta=0.1`、LoRA `r=64`、`all-linear`
- `paged_adamw_8bit`、batch=1、gradient accumulation=8
- `precompute_ref_log_probs=true`、`precompute_ref_batch_size=1`
- 数据 SHA、val SHA、train-v5 adapter SHA 全部通过

| 项目 | 结果 |
| --- | --- |
| 状态 | `completed`，无 `error_kind` |
| optimizer steps | 1 |
| loss | 0.3423 |
| `train_runtime` | 21.5s |
| 总耗时 | 86.1s |
| 采样峰值 | 约 28.3 GiB |
| adapter 权重 | 645,975,704 bytes |
| adapter SHA-256 | `fe33dd9c1561a9a90b2c219ffe5308616620a70c4edd6ccea4144571f3951fd3` |

smoke 同时确认了前向、反向、optimizer step、checkpoint 保存和 adapter 非空校验。

## 4. 全量 DPO

| 项目 | 结果 |
| --- | --- |
| 状态 | `completed`，无 `error_kind` |
| 数据 | 45 train / 13 val；58 对均为 CTX `code_prefix` |
| 实际 optimizer steps | **5**（`45 // 8`） |
| 实际参与更新的样本 | 40 对；最后 5 对因不足一次 gradient accumulation 未更新 |
| epoch | 0.8889 |
| loss 序列 | `[0.3423, 0.0970, 0.0338, 0.0080, 0.0029]` |
| 平均 `train_loss` | 0.0968 |
| `train_runtime` | 107.0s |
| 总耗时 | 169.1s |
| adapter 权重 | 645,975,704 bytes |
| adapter SHA-256 | `83eef684ce5e9c8491bb47f0ace58312bf8eb6cfeee621087808552496cb9a38` |

loss 在 5 个小 step 内快速降到 0.0029，只能说明这小批偏好对被快速拟合；
样本量和更新次数都不支持“DPO 学会通用偏好”的结论。

## 5. held-out 三组对比

固定 6 个任务、每组 10 次 rollout，共 3 组、180 次 rollout。baseline 与 SFT
来自第一轮，SFT+DPO 与作为参照的 SFT 来自第二轮；SFT 在两轮中都是 `60/60`，
说明重复条件在 task_success 上稳定。

### 5.1 总体结果

项目既有门禁把“10 次 rollout 中至少 6 次成功”记为稳定通过；这里同时给出严格
`pass^10`（10/10 全过）。本批 baseline 的失败任务是 0/10、SFT/DPO 是 10/10，
因此两种口径的通过任务数相同，都是 baseline 4/6、SFT 6/6、SFT+DPO 6/6。

| 组 | task_success | BASE | HARD | 任务级 pass^10 | valid stable | within_steps 失败 | 工具结果正确率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 40/60（66.67%） | 20/30 | 20/30 | 4/6 | 4 | 0 | 160/180（88.89%） |
| SFT | 60/60（100%） | 30/30 | 30/30 | 6/6 | 0 | 0 | 100% |
| SFT+DPO | 60/60（100%） | 30/30 | 30/30 | 6/6 | 0 | 0 | 100% |

### 5.2 逐任务结果

| 任务 | 难度 | baseline | SFT | SFT+DPO | baseline stable 项 |
| --- | --- | ---: | ---: | ---: | --- |
| `U-TR-A-C1-HARD` | HARD | 0/10 | 10/10 | 10/10 | `value_correct` 10/10 |
| `U-TR-B-C1-BASE` | BASE | 10/10 | 10/10 | 10/10 | 无 |
| `U-TR-B-CHECK-BASE` | BASE | 10/10 | 10/10 | 10/10 | 无 |
| `U-TR-B-CONFLICT-WEAK` | BASE | 0/10 | 10/10 | 10/10 | `file_created_correct`、`content_exact`、`reason_present` 各 10/10 |
| `U-TR-B-CTX-HARD` | HARD | 10/10 | 10/10 | 10/10 | 无 |
| `U-TR-B-TOOL-HARD` | HARD | 10/10 | 10/10 | 10/10 | 无 |

SFT 把 baseline 的 4 个 stable 违反项全部消掉，同时把两个 0/10 任务拉回
10/10。SFT+DPO 没有再改变任何任务的结果。

### 5.3 如何解释 DPO 的零增量

- **不是 DPO 破坏模型**：SFT+DPO 仍然 60/60，没有新增 stable 违反或步数失败。
- **不是 DPO 有效证据**：SFT 已经 60/60，held-out 没有留下可提升空间。
- **DPO 数据覆盖太窄**：58 对全部来自 CTX `code_prefix`；held-out 的
  `U-TR-B-CTX-HARD` 在 SFT 下已是 10/10，任务类型与已饱和能力重合。
- **训练量太小**：5 个 optimizer step、40 对有效更新，loss 快速塌到 0.0029，
  更像是小样本拟合，而不是可泛化的偏好学习。
- **评测边界**：只有 6 个任务、1 个 seed、已出现明显 ceiling effect；要验证
  DPO 增益，需要更难或更长、且与 `code_prefix` 不完全重合的 held-out。

## 6. 成本与运行状态

| 项目 | 记录 |
| --- | --- |
| 卡型 | AutoDL RTX PRO 6000 96GB |
| 实验窗口 | 2026-10-24 19:30:13–20:14:30 +08:00，约 44 分 17 秒 |
| 费率 | ¥7.08/h（用户提供） |
| 本窗口估算 | 约 **¥5.2** |
| 关机 | **未关机**，等待用户确认 |

费用只按本次实验窗口估算，不包含用户创建实例、上传代码/模型之前的等待时间，
也不代表最终账单。

## 7. 产物与复现

远端工作目录：

```text
/root/autodl-tmp/dpo_work/
  repo/
  sft_adapter/
  out/dpo_smoke_96g/          # 第一次失败：peft 0.13.2
  out/dpo_smoke_96g_v2/       # smoke 通过
  out/dpo_full_96g/           # 全量 DPO adapter + manifest
  out/heldout_baseline_vs_sft/
  out/heldout_sft_vs_dpo/
```

本地归档（`private/` 不入 Git）：

```text
private/dpo_remote/autodl_rtxpro6000_96g_20261024/
  adapter_model.safetensors
  dpo_full_96g/run_manifest.json
  dpo_smoke_96g_v2/run_manifest.json
  heldout_baseline_vs_sft/comparison.json
  heldout_sft_vs_dpo/comparison.json
  dpo_full_96g.log
  heldout_baseline_vs_sft.log
  heldout_sft_vs_dpo.log
  gpu_mem.log
```

训练入口保持为：

```powershell
python training/train_dpo.py `
  --config training/configs/dpo_7b_t4.json `
  --model-path <Qwen2.5-7B-Instruct> `
  --train-file private/dpo_v2/dpo_train.jsonl `
  --val-file private/dpo_v2/dpo_val.jsonl `
  --init-adapter <train-v5-adapter> `
  --output-dir <output>
```

## 8. 边界

- 58 对偏好数据全部来自 CTX `code_prefix`，结论只限该类型。
- 96GB 跑通不证明 24GB 可用；24GB 仍未测试，不能写“24GB 不支持”。
- T4 的 OOM 是 16GB 硬限制下的路径失败，不是 DPO 方法失败。
- 只有 1 个 seed、6 个已饱和任务；不报置信区间。
- 不写“DPO 有效”或“DPO 无效”；本次可写的是“在这批数据和 held-out 上未测到
  相对于 SFT 的额外收益”。

**相关文档**：`docs/evidence/DPO失败诊断.md`、`docs/evidence/DPO偏好对预检.md`、
`docs/evidence/训练管线验证.md`、`docs/evidence/kaggle_runbook.md`。
