# Kaggle T4 执行说明

本目录承载 1.5B QLoRA SFT、全链路 held-out 对比与 constrained decoding 的**实际执行入口**，
不属于 core 运行时依赖。本地 MX250 不再是训练环境；本地只跑单测、dry-run 和文档。
7B 只在 1.5B 结果出来后作验证锚点，不重跑全量 110 条。

> 当前状态：1.5B QLoRA SFT 已真实跑完（`autumnsq/lite-agent-sft-v2`，`COMPLETE`）：
> train 89 / val 30、22 optimizer step、adapter 311 MB 校验通过；held-out 6 任务 ×
> 10 rollout × 2 组，task_success 10/60 → 20/60，但 BASE 持平、HARD 提升只来自
> `U-TR-B-TOOL-HARD`，同时 `U-TR-A-C1-HARD` 新增 10/10 步数循环。结论是“管线跑通 +
> 单任务行为变化 + 明显副作用”，不是训练普遍有效。7B 验证锚点尚未执行。
> 小基座 Coder-1.5B 在 v6 暴露的是 Qwen2.5-Coder 系列与 Hermes 工具调用格式不匹配，
> 不是 stable=0，也不是 parser/环境故障；改用 `Qwen/Qwen2.5-1.5B-Instruct` 后探针通过，
> v7 + CTX 重跑 v8 合并 110 条有效 rollout、20 个 stable 违反检查项；最终 test 为
> `final-test-v2`，四组对比仍未跑。constrained decoding 仍为 0 rollout（v1 路径 ERROR、
> v2 依赖 ERROR、v3 取消、v4 超时），不写命中率。详见
> `docs/evidence/模型工具调用协议审计.md`、`docs/evidence/小基座v7结果.md` 与
> `docs/evidence/训练管线验证.md`。GPU 名额与 kernel 队列审计见
> `docs/evidence/Kaggle队列审计.md`。

## 硬件与固定配置

| 项 | 值 |
| --- | --- |
| GPU | Kaggle T4 16GB（单卡） |
| 量化 | `load_in_4bit=true`、`bnb_4bit_quant_type=nf4`、`bnb_4bit_compute_dtype=float16` |
| 显存优化 | `gradient_checkpointing=true`、`per_device_train_batch_size=1`、`gradient_accumulation_steps=8` |
| 序列长度 | `max_seq_length<=4096` |
| 模型（主线） | `Qwen/Qwen2.5-1.5B-Instruct` |
| 模型（验证锚点） | `Qwen/Qwen2.5-Coder-7B-Instruct`，仅 1.5B 结果出来后跑 30 条 |
| 输出 | 只把 LoRA adapter 写 `/kaggle/working/sft_adapter`，不复制基座权重 |

1.5B 配置固定在 `training/configs/qlora_1_5b_t4.json`：`lora_r=64`、`all-linear`、
4-bit NF4、gradient checkpointing、batch 1 × grad accum 8、`max_seq_length=4096`，
并由 `training/config.py` 在 dry-run 时校验。

## 模型与数据挂载

### 模型

把模型作为 Kaggle Dataset 挂载到：

```text
/kaggle/input/qwen2.5-1.5b-instruct
```

7B 验证锚点另挂 `/kaggle/input/qwen2.5-coder-7b-instruct`。**不要**把模型权重下载到
`/kaggle/working`，否则会挤爆 20GB 工作目录。

### 训练/验证数据

本地用 train-v2 教师数据扩采样后，把以下文件作为**私有 Kaggle Dataset** 挂到
`/kaggle/input/lite-agent-training/`：

```text
private/sft_data_v2_deepseek_expanded/train.jsonl
private/sft_data_v2_deepseek_expanded/val.jsonl
private/sft_data_v2_deepseek_expanded/manifest.json
```

train/val 与 `final-test-v2` 严格不重叠，manifest 记录 train-v2 SHA-256、
BASE/HARD 分布、通过率、rejected 原因和 `teacher_api_calls`。不要把历史
`private/sft_data`（train-v1 16/5）或 Qwen3 对比数据混入训练集。

### constrained decoding prompts

同一 Dataset 下放 `constrained_prompts.jsonl`，每行：

```json
{"run_id": 2, "rollout_index": 0, "prompt": "...最终 probe 的完整 prompt..."}
```

stress run 2/10 各 5 条 rollout，共 10 条；不要和 SFT 的 val 输出混在一起。

## 运行

### 1. 安装依赖

Kaggle Notebook 中：

```bash
pip install -q -r /kaggle/input/lite-coding-agent/training/requirements-kaggle.txt
```

### 2. SFT dry-run 与正式训练

```bash
python /kaggle/input/lite-coding-agent/training/kaggle/kaggle_sft_qlora.py \
  --repo /kaggle/input/lite-coding-agent \
  --dry-run

python /kaggle/input/lite-coding-agent/training/kaggle/kaggle_sft_qlora.py \
  --repo /kaggle/input/lite-coding-agent \
  --model-path /kaggle/input/qwen2.5-1.5b-instruct \
  --train-file /kaggle/input/lite-agent-training/train.jsonl \
  --val-file /kaggle/input/lite-agent-training/val.jsonl \
  --output-dir /kaggle/working/sft_adapter \
  --eval-after-train \
  --heldout-rollouts 10 \
  --heldout-out-root /kaggle/working/heldout
```

`run_sft()` 保存 adapter 后立即调用 `validate_saved_adapter()`：权重文件缺失或 0 字节直接
失败，不进入 held-out。`--eval-after-train` 在**同一 kernel** 内先跑 baseline、再加载
adapter 跑 SFT 组，避免跨 kernel 下载 adapter 的 0 字节问题。报告记录
`optimizer_steps` / `optimizer_steps_ok`，少于 20 步只作管线验证，不写增益。

预计运行时间：50-80 条训练样本、序列 4096，T4 上约 30-90 分钟（含 4-bit 加载和
120 条 held-out rollout）；以 Kaggle 实测为准。

### 3. SFT held-out 全链路对比

`--eval-after-train` 调用 `training/kaggle/heldout_compare.py`，在 train-v2 的 val 任务上
分别跑 baseline 与 adapter，每任务 10 次 rollout，输出：

```text
/kaggle/working/heldout/comparison.json
/kaggle/working/heldout/baseline/runs.jsonl
/kaggle/working/heldout/sft/runs.jsonl
```

对比指标按 BASE/HARD 分拆：stable 违反检查项数、task_success、trace 中工具结果成功率。
BASE = `difficulty` 为 weak/medium，HARD = strong，与 SFT 数据口径一致。

旧的 `training/kaggle/kaggle_eval_sft.py` 是 probe-only 评测（只测最终输出格式，
`tool_call_success=N/A`），保留作快速 smoke，**不得**用于报告 SFT 增益。

### 4. constrained decoding baseline

```bash
python /kaggle/input/lite-coding-agent/training/kaggle/kaggle_constrained_decoding.py \
  --prompts-file /kaggle/input/lite-agent-training/constrained_prompts.jsonl \
  --model-path /kaggle/input/qwen2.5-coder-7b-instruct \
  --out /kaggle/working/constrained_decoding_results.jsonl \
  --summary-out /kaggle/working/constrained_decoding_summary.json
```

Notebook 入口见 `kaggle_constrained_decoding.ipynb`。先跑 `--dry-run` 校验路径与 schema。
首次 Kaggle 运行使用的 `qwen2-5-coder-7b-instruct/Qwen2.5-Coder-7B-Instruct` 路径不存在，
已在模型加载阶段失败；重跑前先按 `/kaggle/input/**/config.json` 定位实际目录，找不到就
打印目录清单并失败，不写命中率。

`resolve_model_path()` 已支持 `--preferred-model`。若同时挂载 7B 和 1.5B，用
`--preferred-model Qwen2.5-Coder-1.5B-Instruct` 明确选择小基座；只有 1.5B 挂载时会
自动解析到实际目录。汇总里的 `model_version` 从解析后的路径推导，不再硬编码 7B。

## 小基座复跑（路线 B 门禁）

入口：`training/kaggle/kaggle_small_model_tuning.py`。它不加载 7B，也不改 core；
在 Kaggle T4 上启动 vLLM 的 OpenAI 兼容服务，再复用 `scripts/difficulty_tuning.py`
跑同一套 11 个 dev-only 任务。

- 基座：`Qwen/Qwen2.5-1.5B-Instruct`（协议探针通过；Coder-1.5B 因 Hermes 格式不匹配不适用）。
- vLLM：`0.12.0`（Python 3.10–3.13，T4 compute capability 7.5），
  `--extra-index-url https://download.pytorch.org/whl/cu129`、`--dtype float16`、`--max-model-len 4096`、
  `--gpu-memory-utilization 0.85`、`--enable-auto-tool-choice --tool-call-parser hermes`、
  `--generation-config vllm`（避免 HF generation config 覆盖服务端默认采样）。
- 采样：`temperature=0.0`、`top_p=None`、`seed=20261019`，与 API 模型实验保持一致。
- T4 兼容：`VLLM_ATTENTION_BACKEND=TRITON_ATTN`、`VLLM_USE_FLASHINFER_SAMPLER=0`，
  并给服务加 `--enforce-eager`；避免 FlashInfer 在 sm75 上 JIT 链接 `libcuda`。
- smoke 门禁：先跑 `U-TUNE-C1-BASE` 一次，确认 trace 的 `llm_turn.tool_calls`
  至少出现一次；否则判定为工具调用协议未生效，不继续 110 次实验。
- 协议探针：smoke 前分别以 `tool_choice=auto` / `required` 直连 vLLM，结果写入
  `tool_protocol_probe.json`；`auto` 为空而 `required` 成功时，先归因为模型系列与
  parser 协议不匹配，不记作 `stable=0`，也不写成模型能力不足。trace 的
  `tools_offered` 记录实际下发的工具名。
- 输出：`/kaggle/working/small_model/run_manifest.json`、`difficulty_tuning/` 和
  `/kaggle/working/traces/difficulty_tuning/`。

本地打包与提交：

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe private\kaggle_small_model\build_kernel.py
kaggle kernels push -p private\kaggle_small_model
```

当前状态：kernel `autumnsq/lite-agent-small-model-tuning` 的 v6 为 `ERROR`，但
vLLM、模型加载和 API server 均正常。`tool_protocol_probe.json` 显示：
`tool_choice=auto` 时 Coder-1.5B 把工具调用写成 fenced JSON 文本、`tool_calls=[]`；
`tool_choice=required` 时能返回真实 `read_file` 调用。结合 vLLM PR #47878 / #32931，
根因是 Qwen2.5-Coder 系列不遵循 Hermes 格式，需要专用 `qwen2_5_coder` parser，
不是模型能力或 `stable=0`。因此 v6 未跑 110 次，不写小模型命中率。

已用 `Qwen/Qwen2.5-1.5B-Instruct` 做独立协议探针（kernel
`autumnsq/lite-agent-tool-protocol-audit` v1，`COMPLETE`）：chat template 原生包含
tools 字段，`tool_choice=auto` 返回非空 `read_file` tool call，`finish_reason=tool_calls`，
无 fenced JSON、无协议错误。v7 已 `COMPLETE`：110 次 rollout，90 次有效，20 次为 CTX
任务 4575 > 4096 token 的 400 error（配置错误）。CTX 重跑 v8 通过 25 行/文件 fixture +
`max_model_len=8192` 修复：20/20 有效、0 error。合并 v7 + v8 后为 110 条有效、20 个
stable 违反检查项（全部 10/10）；`CTX-HARD`、`TOOL-HARD`、`CONFLICT-WEAK/STRONG` 有行为层
失败。明细见 `docs/evidence/小基座v7结果.md`；`final-test-v1` 已因 prompt/fixture 引用
问题作废，最终 test 为 `final-test-v2`，四组对比尚未跑。

## 交付记录

每次 Kaggle 运行必须记录：

- `model_version`、`system_fingerprint`、`response_id`：本地 transformers/Outlines 没有
  API request id，写 `N/A` 或 `null`，不伪造；
- 运行时长、GPU 型号、显存；
- 实际 rollout 次数；
- SFT 前后 held-out 的约束遵守率、工具调用成功率（本次已按 trace 实测）、内容判定通过率；
- constrained decoding 的 `json_valid` / `snake_case_keys` 命中率。

这些记录写入 `docs/evidence/训练管线验证.md`，并和阶段 A 的采样非确定性叙事对齐：
temperature=0 仍有翻转时，不把单次差异写成训练收益。

## 提交前 staging 门禁（v10 起）

任何全量 kernel 之前先推 staging：`--preflight-only` 检查模型/数据/依赖/GPU，再跑
`--staging` 的 3 条记录。加载超时固定 180 秒，单条推理/rollout 超时 120 秒；结果 JSONL
追加写并 flush，失败写 `error_kind`，不允许继续推全量。四层检查清单见
`docs/evidence/kaggle提交契约.md`，失败排查见 `docs/evidence/kaggle_runbook.md`。

```bash
python training/kaggle/kaggle_constrained_decoding.py \
  --prompts-file /kaggle/input/lite-agent-training/constrained_prompts.jsonl \
  --model-path /kaggle/input/qwen2-5-1-5b-instruct/qwen2.5-1.5b-instruct \
  --preflight-only

python training/kaggle/kaggle_constrained_decoding.py \
  --prompts-file /kaggle/input/lite-agent-training/constrained_prompts.jsonl \
  --model-path /kaggle/input/qwen2-5-1-5b-instruct/qwen2.5-1.5b-instruct \
  --staging
```

small model tuning 同理：`--preflight-only` 后再 `--staging`；staging 输出目录与全量目录分开，
不覆盖历史 `run_manifest.json`。

### 结果解释（跑之前锁定）

- constrained decoding 若把 `json_valid` / `snake_case_keys` 违反压到 0：写
  “前沿系统侧方案可解决输出格式违反，进一步说明主样本不需要训练”，不据此宣称 SFT 无效。
- constrained decoding 若仍有翻转：写“constrained decoding 在 Agent 多轮场景下仍受采样
  非确定性影响”，不把单次 0 违反写成稳定解决。
- SFT 结果按 full-loop held-out 报告；单任务提升只写单任务，不把“格式提升”外推成
  “Agent 行为提升”，也不把 `within_steps=false` 混入约束违反统计。
- 所有对比只报实际 rollout 次数与命中数，不伪造置信区间；样本不足时明确写“不足以证明”。

## 边界

- 不把训练依赖写进 `pyproject.toml`；
- 不改 `core/loop.py`、`core/compaction.py`、`core/constraints.py`；
- 不把 7B 权重或大 trace 提交进仓库；
- 不在主实验里用 SFT 结果替代阶段 A 的违反归因结论。
