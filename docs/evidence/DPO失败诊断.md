# DPO 失败诊断

**日期**：2026-10-24  
**范围**：v1/v2 只读诊断 + v3 单 step T4 smoke；未跑 DPO 全量、未租卡、未执行 held-out。  
**证据**：`private/kaggle_pull_method_dpo_meta/lite-agent-method-dpo.log`、`private/kaggle_pull_method_dpo_v2_meta/lite-agent-method-dpo.log`、`private/kaggle_pull_dpo_v3_smoke/`、三份 `run_manifest.json`、`training/train_dpo.py`、`training/configs/dpo_7b_t4.json`。

## 0. 结论先行

- **reference 不是独立 fp16 全权重模型**。基座以 4-bit NF4 加载，reference 走同一基座的 adapter-disabled 路径。
- **v1 和 v2 不是同一个 OOM 阶段**：
  - v1 在训练 policy forward 的 `log_softmax` OOM，`Tried to allocate 1.84 GiB`；
  - v2 在 reference log-prob 预计算的 `log_softmax` OOM，`Tried to allocate 2.34 GiB`。
- **v2 没有测试 dpo_v2 数据**。v2 的实际训练命令仍读取 `private/dpo_v1/`，`--init-adapter` 也是 train-v4 上的 SFT adapter；日志显示 68 train / 24 val。
- v2 的 OOM 与 `precompute_ref_batch_size=1` 不矛盾：TRL 会把 chosen/rejected 拼成两行，batch 1 仍产生 `2 x seq_len x vocab` 的完整 logits 缓冲。
- **v3 版本对齐后仍在 T4 OOM**：train / val / train-v5 adapter 三个 SHA 全部校验通过，失败点仍是 reference precompute 首个 batch 的 `logits.log_softmax`（`Tried to allocate 3.21 GiB`，GPU 仅剩 122.81 MiB），未产生 optimizer step 或 adapter。
- 因此 DPO 当前是 **基础设施负结果**，不是“DPO 方法无效”，也没有任何 DPO 指标。T4 路径本轮放弃，不重推、不试 1024 序列。

## 1. 环境与配置

v2 的完整配置可从当前 `training/configs/dpo_7b_t4.json` 确认，SHA-256：

```text
4d9649fd7d09c0ea3dcb6679b6944df6ff0b3f98fd80ac982fdb0f96c630fc6c
```

| 配置项 | v2 | v1 |
| --- | --- | --- |
| model | `Qwen/Qwen2.5-7B-Instruct` | 同一入口；未单独持久化 v1 JSON |
| 量化 | 4-bit NF4，double quant，compute dtype float16 | 同一训练入口，代码固定 4-bit NF4；逐字段 JSON 未拉取到 |
| `max_seq_length` | 3072 | 未从 v1 独立快照确认 |
| `max_prompt_length` | 2048 | 未从 v1 独立快照确认 |
| `per_device_train_batch_size` | 1 | 未从 v1 独立快照确认 |
| `gradient_accumulation_steps` | 8 | 未从 v1 独立快照确认 |
| `beta` | 0.1 | 未从 v1 独立快照确认 |
| `learning_rate` | 5e-5 | 未从 v1 独立快照确认 |
| `num_train_epochs` | 1 | 未从 v1 独立快照确认 |
| `lora_r` / `lora_alpha` | 64 / 128 | 未从 v1 独立快照确认 |
| `optim` | `paged_adamw_8bit` | 未从 v1 独立快照确认 |
| `precompute_ref_log_probs` | true | 日志无 `Train dataset reference log probs` 进度；runbook 记录 v1 未启用 |
| `precompute_ref_batch_size` | 1 | 不适用 |
| GPU | Tesla T4，14.56 GiB | 同左 |
| 运行环境 | Python 3.13、torch 2.11.0+cu128、transformers 4.46.3、TRL 0.12.2、bitsandbytes 0.50.2 | 同左 |

**v1 配置的边界**：仓库只保留 v2 的配置快照，v1 没有单独的 `dpo_7b_t4.json` pull 产物。上表中 v1 的“未确认”项不能拿 v2 配置反填；可以确认的是 v1 日志没有 reference precompute 阶段，且 OOM 发生在训练 policy forward。

## 2. reference model 的实际加载方式

`training/train_dpo.py` 的加载路径：

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)
```

使用 `--init-adapter` 时：

```python
model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
```

训练入口没有构造独立 reference model，也没有把 `ref_model` 传给 `DPOTrainer`。TRL 0.12.2 的行为可对应到：

- `private/_trl_inspect/x/trl/trainer/dpo_trainer.py:457-462`：PeftModel 或 `precompute_ref_log_probs=True` 时设 `ref_model=None`；
- 同文件 `930-951`：`null_ref_context()` 对 PeftModel 调用 `disable_adapter()`，然后 `compute_ref_log_probs()` 用同一个 `self.model` 做前向。

**判定**：reference 是“同一 4-bit 基座、关闭 adapter”的路径，不是另起一个 fp16 7B reference。此前关于“reference 以 fp16 全权重加载导致 14GB”的假设不成立。OOM 的直接触发点是完整词表 logits / `log_softmax` 缓冲。

## 3. v1 与 v2 的完整失败位置

### 3.1 v1：训练 policy forward

日志阶段：SFT 完成后，DPO 进入第一个 step，42.3s 后失败。关键调用链：

```text
DPOTrainer.compute_loss
  -> get_batch_loss_metrics
  -> concatenated_forward(model, batch)
  -> logits.log_softmax(-1)
torch.OutOfMemoryError: Tried to allocate 1.84 GiB
GPU 14.56 GiB total, 436.81 MiB free
14.13 GiB in use, 13.77 GiB allocated by PyTorch
```

对应日志行：`private/kaggle_pull_method_dpo_meta/lite-agent-method-dpo.log:757-765`。

这是 policy 的训练前向，不是 reference precompute，也不是 backward。v1 的 SFT 阶段耗时 5070.6s（约 84.5 分钟）。

原始 traceback（从日志 JSON 还原，省略进度条行）：

```text
Traceback (most recent call last):
  File "/kaggle/working/repo/training/kaggle/method_runner.py", line 157, in <module>
    raise SystemExit(main())
  File "/kaggle/working/repo/training/kaggle/method_runner.py", line 108, in main
    run_manifest = run_dpo(
  File "/kaggle/working/repo/training/train_dpo.py", line 187, in run_dpo
    train_result = trainer.train()
  File "/usr/local/lib/python3.13/dist-packages/transformers/trainer.py", line 2123, in train
    return inner_training_loop(
  File "/usr/local/lib/python3.13/dist-packages/transformers/trainer.py", line 2481, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
  File "/usr/local/lib/python3.13/dist-packages/transformers/trainer.py", line 3579, in training_step
    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
  File "/usr/local/lib/python3.13/dist-packages/trl/trainer/dpo_trainer.py", line 1371, in compute_loss
    loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
  File "/usr/local/lib/python3.13/dist-packages/trl/trainer/dpo_trainer.py", line 1323, in get_batch_loss_metrics
    model_output = self.concatenated_forward(model, batch)
  File "/usr/local/lib/python3.13/dist-packages/trl/trainer/dpo_trainer.py", line 1274, in concatenated_forward
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.84 GiB. GPU 0 has a total capacity of 14.56 GiB of which 436.81 MiB is free. Including non-PyTorch memory, this process has 14.13 GiB memory in use. Of the allocated memory 13.77 GiB is allocated by PyTorch, and 233.03 MiB is reserved by PyTorch but unallocated.
```

### 3.2 v2：reference log-prob 预计算

日志阶段：SFT 完成后，在 `Train dataset reference log probs` 预计算到 2/68 左右失败，60.0s。关键调用链：

```text
transformers.Trainer._inner_training_loop
  -> get_train_dataloader
  -> DPOTrainer.compute_ref_log_probs
  -> concatenated_forward(self.model, batch)
  -> logits.log_softmax(-1)
torch.OutOfMemoryError: Tried to allocate 2.34 GiB
GPU 14.56 GiB total, 1.88 GiB free
12.68 GiB in use, 12.49 GiB allocated by PyTorch
```

对应日志行：`private/kaggle_pull_method_dpo_v2_meta/lite-agent-method-dpo.log:735-764`。

v2 的 SFT 阶段耗时 5795.2s（约 96.6 分钟）。v2 没有进入 policy 第一个 optimizer step，因此不能再把 v2 描述成“第一个训练 step OOM”。

原始 traceback（从日志 JSON 还原，省略进度条行）：

```text
Traceback (most recent call last):
  File "/kaggle/working/repo/training/kaggle/method_runner.py", line 157, in <module>
    raise SystemExit(main())
  File "/kaggle/working/repo/training/kaggle/method_runner.py", line 108, in main
    run_manifest = run_dpo(
  File "/kaggle/working/repo/training/train_dpo.py", line 192, in run_dpo
    train_result = trainer.train()
  File "/usr/local/lib/python3.13/dist-packages/transformers/trainer.py", line 2123, in train
    return inner_training_loop(
  File "/usr/local/lib/python3.13/dist-packages/transformers/trainer.py", line 2152, in _inner_training_loop
    train_dataloader = self.get_train_dataloader()
  File "/usr/local/lib/python3.13/dist-packages/trl/trainer/dpo_trainer.py", line 857, in get_train_dataloader
    ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(padded_batch)
  File "/usr/local/lib/python3.13/dist-packages/trl/trainer/dpo_trainer.py", line 948, in compute_ref_log_probs
    ref_model_output = self.concatenated_forward(self.model, batch)
  File "/usr/local/lib/python3.13/dist-packages/trl/trainer/dpo_trainer.py", line 1274, in concatenated_forward
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.34 GiB. GPU 0 has a total capacity of 14.56 GiB of which 1.88 GiB is free. Including non-PyTorch memory, this process has 12.68 GiB memory in use. Of the allocated memory 12.49 GiB is allocated by PyTorch, and 70.66 MiB is reserved by PyTorch but unallocated.
```

## 4. `precompute_ref_batch_size=1` 为什么仍 OOM

TRL 的 `concatenated_inputs()` 会把 chosen 和 rejected 拼成两行。因此即使预计算 dataloader 的 batch size 是 1，reference forward 的 logits 仍是：

```text
2 (chosen + rejected) x effective_seq_len x 151665 (Qwen vocab)
```

本地用 Qwen2.5-7B tokenizer 和当前工具 schema 复算 dpo_v1 的 68 条 train：

- prompt token：min 1476 / p50 1621 / max 6454；
- `max_prompt_length=2048` 截断后，最大有效序列约 2087 token（2048 prompt + 39 completion）；
- 按 2 行完整词表、每元素 4B 估算：

```text
2 x 2086 x 151665 x 4 B ≈ 2.36 GiB
```

与 v2 日志的 `2.34 GiB` 同量级。这个复算使用当前 tokenizer/tool schema，和当时运行可能有少量 token 漂移，因此只把它作为量级解释，不声称精确到单条样本。结论不变：**OOM 来自完整词表 logits/log_softmax 的临时缓冲，不是独立 fp16 reference 权重。**

另外，dpo_v2 的 58 对数据更长：

| 指标 | manifest 记录 | 本次当前 tokenizer 复算 |
| --- | ---: | ---: |
| prompt min | 2168 | 2178 |
| prompt p50 | 2770 | 2793 |
| prompt max | 3001 | 3024 |
| `max_prompt_length=2048` 截断 | 58/58 | 58/58 |
| `max_seq_length=3072` 超限 | 0/58 | 0/58 |

即使把 v2 的数据路径修对，v2 当时的 `max_prompt_length=2048` 仍会截断全部 58 对；该旧配置本身也不兼容 dpo_v2。v3 已改为 3072，见第 6 节。

## 5. 版本对齐审计

### 5.1 数据

| 版本 | 对数 | train / val | train SHA-256 | val SHA-256 |
| --- | ---: | ---: | --- | --- |
| dpo_v1 | 92 | 68 / 24 | `3b588480d59826b5d26f88c8b97c93d3db3ece5f5f06186dc7186aa62a34ec57` | `4f346786f0d7d0b74064120f4edc28009108978885e6ee8d4412a21e6685bff3` |
| dpo_v2 | 58 | 45 / 13 | `691ea2667893b483d02004e9f8af800a6978df2b9dba80d19a245f4098b43aed` | `4072a9fec7616e6df3e9a17ecafe30f0c404a6495da63204ca0f2413ec8466f2` |

**关键事实**：v2 日志的实际 `RUN train_dpo` 命令是：

```text
--train-file /kaggle/working/repo/private/dpo_v1/dpo_train.jsonl
--val-file   /kaggle/working/repo/private/dpo_v1/dpo_val.jsonl
--init-adapter /kaggle/working/method/dpo/sft_adapter
```

日志同时显示 `Generating train split: 68 examples` 和 `24 examples`。同一错误命令在 `CalledProcessError` 里被再次打印。对应行：v2 日志 `723-728`、`766`、`790`。

因此：

- v2 **没有消费** `private/dpo_v2/` 的 45/13 对数据；
- v2 **没有**验证 train-v5 的偏好对；
- v2 只是 v1 数据 + train-v4 SFT adapter 上，加了 reference precompute 后的再次内存失败；
- 不能把 v2 写成“dpo_v2 数据在 T4 上 OOM”。

### 5.2 SFT adapter

v1/v2 的 `method/dpo/sft_adapter/run_manifest.json` 关键字段一致：

| 字段 | 值 |
| --- | --- |
| `train_file` | `/kaggle/working/repo/private/sft_data_v4/train.jsonl` |
| `train_rows` / `val_rows` | 361 / 74 |
| `max_seq_length` | 3072 |
| `loss_mode` | full |
| `optimizer_steps` | 90 |
| `adapter_model.safetensors` | 645,975,704 bytes |
| `adapter_total_bytes` | 661,861,305 bytes |

**adapter SHA 不可得**：当前 pull 结果只有 manifest，没有 DPO 这两个 kernel 产出的 `adapter_model.safetensors` 本体。不能用其他同尺寸 run 的 SHA 反推，也不能伪造。可确认的版本关系只有“train-v4 数据 + 90 optimizer steps + 上述文件大小”。

## 6. v3 版本对齐

### 6.1 数据与 adapter

| 项目 | v3 固定值 | 验证方式 |
| --- | --- | --- |
| 偏好对 | `private/dpo_v2/`，45 train / 13 val | 训练入口实际读取路径 + 邻近 `manifest.json` |
| train SHA-256 | `691ea2667893b483d02004e9f8af800a6978df2b9dba80d19a245f4098b43aed` | 本地逐字节复算 |
| val SHA-256 | `4072a9fec7616e6df3e9a17ecafe30f0c404a6495da63204ca0f2413ec8466f2` | 本地逐字节复算 |
| policy init | train-v5 assistant-only adapter | `autumnsq/lite-agent-method-assistant-train-v5` 输出 |
| adapter SHA-256 | `f5b685d4da795a783fbcb21b15880daebfa08c64b1661792f7e2c6caab29fea7` | 本地重新 pull 后复算 |
| adapter bytes | 645,975,704 | 本地文件大小 + Kaggle `summary.json` 一致 |
| 数据 manifest | `dpo-v2-preflight`；source train-v5 | `private/dpo_v2/manifest.json` |
| 配置 SHA-256 | `82d136adf1f05eb10d2ed3550ef1bd501c7f496cbd080bf85cad1a42c71314ee` | 当前 `dpo_7b_t4.json` |

adapter 下载曾在 `IncompleteRead` 处断流两次；最终通过 Range 分块并拼接，完整文件 SHA 与 Kaggle 输出摘要一致。没有用其他同尺寸 adapter 反推 SHA。

### 6.2 reference 与 precompute 输入 shape

- reference 仍是同一 4-bit NF4 基座关闭 adapter，不另起 fp16 reference；
- v3 把 `max_prompt_length` 从 2048 提到 **3072**，`max_seq_length` 保持 **3072**；
- dpo_v2 manifest 记录 prompt max 3001、chosen 8 token、rejected 10 token，最大总长约 3011；Kaggle smoke 实测 prompt max 2998、rejected+prompt max 3008；本地 tokenizer 复算为 3024 + 10 = 3034。差异来自 tokenizer/tool schema 版本，三种口径均未超过 3072，实测截断 0 对；
- TRL 把 chosen/rejected 拼成两行，因此 `precompute_ref_batch_size=1` 的实际输入是 **2 × 3008 × 151,643 = 912,284,288** 个 logits 元素；v3 `run_manifest.json` 已记录 prompt/total token 分位数、vocab size 和 max logits elements。

### 6.3 代码防线

- `train_dpo.py` 新增 `--expected-train-sha256`、`--expected-val-sha256`、`--expected-init-adapter-sha256`；任一不匹配立即 `ValueError`，不进入模型加载；
- `run_manifest.json` 在 `trainer.train()` 前先落盘，OOM 时也会更新为 `status=error` 并保留 error_kind/traceback；
- `dpo_smoke_v3` 使用独立 kernel，不重训 SFT，数据固定 dpo_v2，`max_steps=1`；
- 因 `precompute_ref_log_probs=true`，smoke 仍会先遍历 45 条 train 做 reference 预计算；`max_steps=1` 约束的是 optimizer step，不是“只读 1 个 batch”。

### 6.4 T4 smoke 结果

已提交并完成 `autumnsq/lite-agent-method-dpo-smoke-v3-train-v5`（version 1），状态 `ERROR`：

| 项目 | 实测 |
| --- | --- |
| 版本校验 | train / val / init adapter 三个 `sha256_verified` 全为 `true` |
| 数据 | 45 train / 13 val；实际读取 `private/dpo_v2/` |
| init adapter | train-v5 assistant-only，645,975,704 bytes |
| 模型加载 | Qwen2.5-7B-Instruct 4-bit NF4，Tesla T4 14.56 GiB |
| 观测 prompt token | min 2168 / p50 2770 / max 2998 |
| 观测 chosen+prompt token | max 3006 |
| 观测 rejected+prompt token | max 3008 |
| `max_prompt_length` / `max_length` | 3072 / 3072；train 超过上限 0 对 |
| reference precompute shape | `2 × 3008 × 151,643` = 912,284,288 logits 元素 |
| 估算 logits 缓冲 | `912,284,288 × 4 B ≈ 3.40 GiB` |
| 失败位置 | TRL `compute_ref_log_probs -> concatenated_forward -> logits.log_softmax` |
| 失败进度 | `Train dataset reference log probs: 0/45` |
| OOM 分配请求 | `Tried to allocate 3.21 GiB`；GPU 仅剩 122.81 MiB |
| 耗时 | DPO 阶段 106.7s；kernel 总计约 227.2s |
| optimizer step / adapter / rollout | 0 / 未产出 / 0 |

**结论**：版本错配已被排除，但 v3 仍在 reference precompute 的第一个 batch OOM。T4 16GB 不支持“3072 序列 + 4-bit 7B + chosen/rejected 双行完整词表 logits”这一路径。注意 reference 与 policy 共用同一个 4-bit 基座，不是两份 7B 权重；显存峰值来自双行完整词表 logits，不是独立 fp16 reference。

因此不再重推 T4，也不再调 `precompute_ref_batch_size=1`（1 已是最小 batch；TRL 仍拼成 2 行）。若继续 DPO，租 24GB 单卡是唯一有意义的硬门槛；本轮不租、不重推。

## 7. 租卡显存预算

### 7.1 逐项估算（24GB 卡）

以 v3 实测的 `2 × 3008 × 151,643` logits 为基准，`logits.log_softmax` 缓冲为
`912,284,288 × 4 B = 3.40 GiB`。其余项目按保守上限估算：

| 项目 | 保守估算 | 说明 |
| --- | ---: | --- |
| policy 4-bit 7B | 4.0 GB | NF4 权重 + train-v5 LoRA adapter |
| reference 4-bit 7B | 0–4.0 GB | 代码复用同一 4-bit 基座并关闭 adapter；只有把 reference 当独立副本时才另计 4GB |
| optimizer states | 1–2 GB | `paged_adamw_8bit`，只覆盖 LoRA 可训练参数 |
| 激活值 | 4–6 GB | 3072 序列、batch 1、非重入 gradient checkpointing |
| `logits.log_softmax` | 3.2–3.4 GB | 实测 shape 2 × 3008 × 151,643，FP32 |
| CUDA context + 碎片 | 1–2 GB | 驱动、通信缓冲、allocator 碎片 |
| **保守合计** | **17–21 GB** | 把 reference 单独计 4GB 的口径 |
| **共享基座口径** | **13–17 GB** | reference 不另起副本；与 T4 实测 `14.44 GiB + 3.21 GiB ≈ 17.65 GiB` 的上界一致 |

### 7.2 决策

- **24GB 余量**：保守口径下剩余 **3–7 GB**，共享基座口径下剩余 **7–11 GB**；
  两种口径都不低于 3GB 的预注册门槛。
- **首选 24GB（RTX 3090，约 ¥1.32/h）**：只做一次 `max_steps=1` smoke，验证
  `logits.log_softmax` 是否还 OOM。
- **40GB（A100，约 ¥3.28/h）仅作 fallback**：24GB smoke OOM 后最多换 40GB 一次；
  40GB 仍 OOM 才写“24GB/40GB 均不支持 3072 + 双 7B DPO”，随后关闭 DPO。
- 不在 24GB 上反复调 batch size 或序列长度，不试 1024 序列。

### 7.3 T4 阶段执行状态（历史）

当前 Windows 环境只配置了 Kaggle 与 DashScope 凭据；`vastai`、`runpodctl`、`gcloud`、
`aws`、`az` 均不存在，也没有 24GB/40GB 租卡接口。因此本轮**没有实际租卡、没有 GPU
smoke、没有 DPO held-out 结果**。预算结论和止损规则已写死，执行记录见
`docs/evidence/DPO结果.md`；不能把“未执行”写成“24GB 不支持”。96GB 实例上的
实际执行见第 10 节。

## 8. 修复决策

| 分支 | 决策 | 理由 |
| --- | --- | --- |
| 当前 T4 DPO | **关闭，不重推** | v3 版本对齐后仍在 reference precompute 0/45 的完整词表 logits 上 OOM；不是数据/adapter 错配 |
| 1024 序列 | **不做** | 2048 下 dpo_v2 已 58/58 截断；降到 1024 只会把任务指令截得更严重，不是有效方法对照 |
| 租 24GB 卡 | **可讨论，未启动** | v3 已排除版本错配并证明 T4 16GB 是硬限制；24GB 是继续 DPO 的合理最低配置，但本轮不租 |
| 结论叙事 | **基础设施负结果** | 0 条 DPO 指标；不能写 DPO 有效，也不能写 DPO 方法无效 |

如果未来恢复 DPO，最低前置条件是：

1. 用 `private/dpo_v2/` 的 45/13 对，并把 SFT adapter 明确对齐到同一 train-v5 版本；
2. `max_prompt_length >= 3001`，`max_seq_length >= 3072`（实际建议 4096，并重新跑截断预检）；
3. 不再让 reference precompute 生成完整 `2 x seq_len x vocab` 的 FP32 logits 缓冲；需要改实现而不是只调 batch size；
4. 先做 3 条 staging，确认 `errors=0`、至少 1 条 `task_success=true`，再谈全量。

## 9. 成本与边界

- v1：SFT 5070.6s，DPO 42.3s 后失败；
- v2：SFT 5795.2s，DPO 60.0s 后失败；
- v3：另起独立 smoke kernel（`max_steps=1`，不重训 SFT），DPO 阶段 106.7s 后在 reference precompute 0/45 失败；kernel 总计约 227.2s；
- 租卡：截至 T4 阶段为 0 个 session、0 元；当前环境没有租卡接口/凭据，24GB smoke 未执行；
- 三轮都未执行 DPO held-out，未产生任何 DPO rollout 或 DPO adapter；
- 本文对 v1/v2 只读；v3 只新增 SHA 校验、`max_steps` 与诊断字段，未改 DPO 配方，未改 `core/loop.py`、`compaction.py`、`constraints.py`。

## 10. 96GB 租卡执行结果（2026-10-24）

用户提供 AutoDL RTX PRO 6000 96GB 实例后，DPO 走同一配置重新执行，未改 batch、
序列长度、beta、LoRA rank 或数据：

- GPU：RTX PRO 6000 Blackwell Server Edition，97,887 MiB；驱动 580.82.09；
  PyTorch 2.8.0+cu128；NF4 前向通过。
- 第一次 smoke 在加载 adapter 时失败，但失败不是 OOM：train-v5 adapter 由
  `peft 0.20.0` 保存，实例初始为 `0.13.2`，报
  `LoraConfig.__init__() got an unexpected keyword argument 'alora_invocation_tokens'`。
  对齐到 `peft 0.20.0` 后，同一 smoke 通过。
- smoke：`status=completed`、1 optimizer step、loss 0.3423、adapter
  645,975,704 bytes、SHA `fe33dd9c...f3951fd3`；全程采样峰值约 28.3 GiB。
- 全量：`status=completed`、5 optimizer step（45 // 8，末 5 对未形成完整更新）、
  loss `0.3423 → 0.0970 → 0.0338 → 0.0080 → 0.0029`、adapter
  645,975,704 bytes、SHA `83eef684...6cb9a38`；全程采样峰值 29.32 GiB。
- held-out：6 任务 × 10 rollout、`temperature=0.0`、`top_p=None`、
  `seed=20261019`；baseline `40/60`、SFT `60/60`、SFT+DPO `60/60`；
  valid stable `4 → 0 → 0`，within_steps 失败均为 0。
- 结论：96GB 上 DPO 能跑通，但 SFT 在 6 个 held-out 任务上已经饱和，DPO 没有
  可测增量；58 对偏好全部来自 CTX `code_prefix`，不外推。
- 成本窗口：19:30:13–20:14:30 +08:00，约 44 分 17 秒，按 ¥7.08/h 约 ¥5.2；
  不含创建实例、上传模型和用户等待时间。实例未关机，等待用户确认。

完整记录见 `docs/evidence/DPO结果.md`。

**相关文档**：`docs/evidence/DPO偏好对预检.md`、`docs/evidence/kaggle_runbook.md`、`docs/plan/训练侧执行计划.md`。
