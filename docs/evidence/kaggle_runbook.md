# Kaggle 运行 Runbook

**版本**：v1  
**日期**：2026-10-20  
**适用**：`kaggle_constrained_decoding.py`、`kaggle_small_model_tuning.py` 及其 staging kernel。

## 1. 提交顺序

1. 本地跑 `pytest -q -m "not slow"`、`ruff check .`、`ruff format --check .`。
2. 推 staging kernel：只跑 3 条，加载超时 180 秒，单条推理超时 120 秒。
3. 拉取 `preflight.json`、`run_manifest.json`、`summary.json`、结果 JSONL。
4. 检查 `status`、`error_kind`、记录数、schema。
5. staging 通过后才推全量；staging 失败不推全量。

查看 kernel 状态与输出：

```powershell
kaggle kernels list -m
kaggle kernels status autumnsq/lite-agent-small-model-tuning
kaggle kernels output autumnsq/lite-agent-small-model-tuning -p private/kaggle_pull
```

## 2. 常见失败模式

| 现象 | 优先检查 | 典型 `error_kind` | 修复 |
| --- | --- | --- | --- |
| `FileNotFoundError` / `HFValidationError` | `preflight.json` 的 `model_path`；`/kaggle/input` 是否挂载 | `model_path_missing` / `model_path_unverified` | 改本地挂载路径；staging 不用 Hub id |
| preflight 报 tokenizer 缺失 | 模型 Dataset 是否只挂了权重 | `tokenizer_config_missing` | 重新挂完整模型目录 |
| prompts/data 0 字节 | Dataset 版本、打包脚本、UTF-8 BOM | `input_missing` / `input_empty` | 重新上传并本地 `Get-Item` 校验大小 |
| `pip install` 失败 | Python 3.13、vLLM 0.12.0、torch cu129 | `dependency_missing` / `unknown` | 用脚本内置 `--install-runtime`；不要混装旧 vLLM |
| 日志停在 `Loading weights: 0%` | `vllm_server.log`；180 秒是否触发 | `model_load_timeout` | 先 staging 小模型；确认模型是 1.5B 而非 7B |
| 进程 RUNNING 但无输出 | `summary.json` 是否已写；是否有进度日志 | `inference_timeout` | 加 `--limit 3`、降 `--max-new-tokens` |
| 结果 JSONL 0 字节 | 是否在模型加载前失败；`run_manifest.json` | `output_empty` | 不写指标，先修加载 |
| summary 缺字段 | 脚本版本与产物是否匹配 | `summary_schema_invalid` | 用当前脚本重跑，不手改 summary |
| CUDA OOM | `max_model_len`、batch、4bit | `oom` | 降序列长度/batch；7B 用 4bit |
| `vLLM 服务未就绪` | 端口 8000、依赖、GPU 是否被占 | `model_load_timeout` | 看 server log；不要在旧 kernel 占用时重推 |
| kernel 一直 QUEUED | `kaggle kernels list -m` 的 Running/Queued | `kernel_queued` | 等或取消遗留 kernel；不抢名额 |

## 3. 排查命令

```powershell
# 1. 看 manifest 的阶段与 error_kind
Get-Content private/kaggle_pull/small_model_staging/run_manifest.json -Raw -Encoding UTF8

# 2. 看 preflight 的模型/依赖/GPU
Get-Content private/kaggle_pull/small_model_staging/preflight.json -Raw -Encoding UTF8

# 3. 看 vLLM 启动日志
Get-Content private/kaggle_pull/small_model_staging/vllm_server.log -Tail 120 -Encoding UTF8

# 4. 看结果是否非空
Get-Item private/kaggle_pull/constrained_staging.jsonl | Select-Object Length
Get-Content private/kaggle_pull/constrained_staging_summary.json -Raw -Encoding UTF8
```

如果 `run_manifest.json` 的 `status=error`，先看 `stage`：

- `install`：依赖安装失败，不进入 GPU 加载；
- `preflight`：路径/数据/环境契约失败，不启动 vLLM；
- `run`：模型加载、推理或输出校验失败，保留 `vllm_server.log`。

## 4. 两次失败规则

同一 kernel 连续失败 2 次，停止重推，换方案：

- 路径问题：改本地挂载或改为 staging 专用模型目录；
- 依赖问题：换镜像/降低依赖组合，不在同一 kernel 反复 pip；
- 加载超时：先用 1.5B + 3 条 staging，确认不是 7B/长 fixture；
- 输出 0 字节：先跑 `--preflight-only`，再跑 `--staging`，不直接推全量；
- 每次都记录失败，不把“重推后成功”写成第一次就通过。

## 5. 失败记录模板

每次失败写入 `docs/evidence/kaggle提交契约.md` 的“当前状态”或单独审计文件：

| 字段 | 示例 |
| --- | --- |
| kernel | `autumnsq/lite-agent-small-model-tuning` |
| version | v10 |
| 失败阶段 | `preflight` / `run` / `output` |
| `error_kind` | `model_load_timeout` |
| GPU 占用时长 | 2m18s |
| 是否浪费 session | 是（加载失败，0 条记录） |
| 证据 | `run_manifest.json`、`vllm_server.log` 路径 |
| 下一步 | 改模型路径 / 降 `--limit` / 换方案 |

## 6. staging 通过标准

| 检查 | 通过 |
| --- | --- |
| 退出码 | 0 |
| `preflight.json` | `status=ok` |
| 结果 JSONL | 至少 1 条，且不是错误记录 |
| `summary.json` | `status=completed`，`error_kind=null` |
| schema | 必需字段齐全 |
| 进度日志 | 至少出现一次 `processed` 或 `server ready` |
| GPU 时长 | 记录实际秒数，不估算 |

staging 通过后，把 kernel version、GPU 型号/时长、3 条结果和 `error_kind=null` 写回
`docs/evidence/kaggle提交契约.md`，再推全量。全量仍使用同一采样配置，不因 staging
结果改 temperature/top_p/seed。

## 7. 2026-10-20 真实失败补充

- `ModuleNotFoundError: training`：builder 启动训练脚本时必须设置 `cwd=/kaggle/working/repo` 与 `PYTHONPATH=/kaggle/working/repo`；只把 zip 解压到 repo 不够。
- `/kaggle/input` 冷挂载慢：vLLM 读 3GB safetensors 可能在 staging 窗口内卡在 `Loading safetensors checkpoint shards: 0/1`。优先使用显式 `--allow-hub-model` 的 Hub 镜像；不要把该现象写成模型能力失败。
- readiness 失败但 vLLM 日志已 `Application startup complete`：v15 已排除代理变量（`NO_PROXY` + 无代理 opener）仍失败；下一步用 `ss -ltnp`、`curl -v http://127.0.0.1:8000/v1/models` 记录监听地址/HTTP 状态，确认是否 loopback 或命名空间问题。不要只凭 vLLM 日志判定 ready。
- 预算拆分（v12 历史值；当前 staging 为 `load_timeout=180s`）：`load_timeout=120s`、单条 `rollout_timeout=120s`、staging 外层 `RUN_TIMEOUT=300s`。外层 120s 会在 vLLM 冷启动后立刻杀掉协议探针和 rollout。
- 同一 kernel 连续失败 2 次后必须换方案；v10→v11 修打包路径，v11→v12 换 Hub 镜像，v12→v13 校准外层预算，均已在 `docs/evidence/kaggle提交契约.md` 第 9 节留证。
- 当前仍未通过 staging；在 readiness 根因修复并真实拿到非空 JSONL + `summary.status=completed` 前，不推全量。

## 8. 2026-10-21 readiness 与 staging 结论

### 8.1 v14/v15 `Connection refused` 定位

v19 在同一 kernel 内执行了一次限时 readiness 诊断（`status=ok`，总耗时 125.3s）：

| 检查 | 结果 |
| --- | --- |
| `ss -ltnp` | `LISTEN 127.0.0.1:8000`，进程 pid=100 |
| `curl http://127.0.0.1:8000/v1/models` | HTTP 200，返回 `Qwen/Qwen2.5-1.5B-Instruct` |
| `curl http://localhost:8000/v1/models` | HTTP 200（先试 `::1` refused，再回退 IPv4） |
| `curl http://[::1]:8000/v1/models` | `curl: (7) Connection refused` |
| socket `127.0.0.1:8000` | connected |
| socket `::1:8000` | `ConnectionRefusedError` |

结论：vLLM 只监听 IPv4 loopback，不监听 IPv6。任何先解析到 `::1` 的客户端都会看到
`Connection refused`；旧 kernel 的 `localhost`/代理配置足以解释 v14/v15。当前固定使用
`http://127.0.0.1:8000/v1`，不再用 `localhost`。这不是模型能力问题，也不是 stable=0 证据。

### 8.2 v17 server staging（真实行为观察）

v17 使用 server 模式跑 `U-TUNE-C1-BASE` 3 条 staging：

| 项目 | 结果 |
| --- | --- |
| 运行状态 | 3 条 rollout、0 error |
| 工具 | `read_file`，`params_ok=true`，`within_steps=true` |
| `task_success` | 0/3 |
| 失败原因 | 模型输出被 Markdown 围栏包裹，`json_valid` / `no_fence` 等检查失败 |
| 门禁 | `status=degraded`、`error_kind=staging_degraded` |

这说明“基础设施可达”和“行为验收通过”必须分开记录：v17 不是连接失败，也不是
stable 违反；它只是 3 条 staging 的行为观察，不能写成模型能力结论。

### 8.3 切换决策

- server 模式已验证可达，不再把 v14/v15 当作持续网络故障。
- 仍保留 vLLM 离线 API（`training/kaggle/offline_provider.py`）作为备选；离线 staging 结果见
  `docs/evidence/kaggle提交契约.md` 第 9 节。
- staging smoke 任务改用 `U-FT03-BASE`（上下文召回），避免用已知会触发 JSON 围栏的
  `U-FT01-BASE` 把“格式违反”误判成“基础设施失败”。
- 任何 staging 失败先看 `error_kind` / `staging_check`，不直接重推全量。

### 8.4 v21 离线 staging 根因与一次修复

v21 改用 vLLM 离线 API（`LLM(...)` + `generate()`，不再起 server），仍在 engine warmup 阶段失败：

| 项目 | 结果 |
| --- | --- |
| 失败阶段 | `run`（engine init，未进入 rollout） |
| 关键日志 | `Warming up FlashInfer attention`；`RuntimeError: Ninja build failed`；`collect2: error: ld returned 1 exit status` |
| 根因 | T4（compute capability 7.5）无法 JIT 编译 FlashInfer kernel；vLLM 0.12 在该模型上自动选择了 `FLASHINFER` 后端 |
| 非根因 | 不是模型路径、不是 `max_model_len`、不是网络（离线模式无 server） |

修复：`training/kaggle/offline_provider.py` 在导入 vLLM 前用 `setdefault` 设置
`VLLM_ATTENTION_BACKEND=TRITON_ATTN` 与 `VLLM_USE_FLASHINFER_SAMPLER=0`，与 server 模式保持同一后端；
显式环境变量仍可覆盖，默认值不改变非 T4 环境的显式配置。新增两条单测验证默认值与覆盖行为。

按止损规则只重推一次（v22）。若 v22 仍失败，直接降级 transformers 原生 `generate` + 4-bit，
不再尝试第三种 vLLM 模式，也不在 server 模式继续消耗 session。

### 8.5 v22 离线 staging 通过（2026-10-21）

| 项目 | 结果 |
| --- | --- |
| Kernel | `autumnsq/lite-agent-small-model-tuning` v22，`COMPLETE` |
| 模式 | `provider=offline`、`system_variant=s0`、`U-FT03-BASE` × 3 rollouts |
| 运行时长 | 126s（`05:27:38Z` → `05:29:44Z`） |
| errors | 0 |
| `task_success` | 3/3 |
| `max_rate` | 0.0（3 条不判 stable，`stability_judgement=insufficient`） |
| 工具 | `read_file`，`params_ok=true`，`within_steps=true`，`steps=1` |
| 结论 | `status=ok`、`staging_check.passed=true`；离线 vLLM 首次端到端跑通 |

离线 provider 的 metadata 只有 `model_version`（模型路径），没有 request id / system fingerprint；
这是 in-process vLLM 与 OpenAI 兼容 server 的差异，报告中如实标注，不伪造字段。

### 8.6 v23/v24 主实验（final-test-v2 S0/S1）

| 版本 | 变体 | records / errors | 原始 stable | valid stable | task_success | 时长 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| v23 | S0 | 110 / 0 | 19 | 15 | 40/110 | 410s |
| v24 | S1 | 110 / 0 | 24 | 20 | 10/110 | 481s |

两轮都使用同一 `final-test-v2`、`temperature=0.0`、`top_p=None`、`seed=20261019`。
S1 的 prompt 自检把 `U-FT03/10/11` 从 10/10 成功推到 0/10，并引入 `U-FT02` 的 10/10
步数循环，属于负收益；因此不工程化，也不启动四组对比。明细见
`docs/evidence/final_test_v2_baseline.md` 与 `docs/evidence/final_test_v2_S1.md`。

### 8.7 本轮 GPU session 成本

本次契约工作共 14 次 kernel 运行：v10、v11、v12、v14、v15、v17、v18、v19、v20、v20b、
v21、v22、v23、v24（v13 排队后被停止，不计 GPU；v18/v20/v20b 在 preflight 秒级失败）。其中：

- v11/v12/v14/v15：浪费在模型挂载、外层预算和 server 连接，约 8 分钟；
- v21：离线 vLLM 在 T4 上 JIT 编译 FlashInfer，约 4 分钟后失败；
- v22：一次修复后 staging 通过，126s；
- v23/v24：S0/S1 各 110 条主实验，分别 410s / 481s。

失败模式按“打包路径 → 挂载慢 → 外层预算 → server 可达性 → 参数契约 → IPv4/IPv6 →
离线后端选择”逐层收敛。契约层建成后，staging 失败暴露的是 vLLM 部署问题，不是契约问题。

### 9. Adapter 0 字节与 SFT held-out（2026-10-21）

**已知问题**：早期 Kaggle 运行出现 adapter 下载后 0 字节。根因不在训练脚本，而在跨 kernel
下载产物；本轮改为同 kernel 训练 + 评测，不再跨 kernel 下载 adapter。

**防护**：

- 训练保存后立即 `validate_saved_adapter()`，0 字节直接失败；
- `--eval-after-train` 在同一个 kernel 内先跑 baseline、再加载 adapter 跑 SFT 组；
- 结果落 `/kaggle/working/heldout/comparison.json`，包含 BASE/HARD 分拆；
- 若 adapter 校验失败，`kernel_status.json` 写 `error_kind=adapter_empty`，不进入 held-out。

**排查顺序**：

1. `ls -l /kaggle/working/sft_adapter`，确认 `adapter_model.safetensors` 非 0；
2. 看 `sft_adapter/run_manifest.json` 的 `adapter_files` / `adapter_total_bytes`；
3. 看 `heldout/baseline/runs.jsonl` 与 `heldout/sft/runs.jsonl` 是否非空；
4. 看 `heldout/comparison.json` 的 `conditions.*.errors`，不要只看最终指标。

**成本记录**：`autumnsq/lite-agent-sft-v2` v1 于 2026-10-21 推送，训练到 11/22
optimizer step 后在 per-epoch eval 的全词表 logits 处 OOM；修复为 `eval_strategy="no"`
并加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后 v2 `COMPLETE`。v2 训练
`train_runtime=123.0s`、全流程 `131.3s`，train 89 / val 30、22 optimizer step、
adapter 311,374,543 bytes 且校验通过。held-out 结果见
`private/kaggle_pull_sft_v2_meta/heldout/comparison_valid.json`（原始 `comparison.json`
保留作 raw 口径）；结论是 BASE 持平、HARD 仅 `TOOL-HARD` 提升，`A-C1-HARD` 新增
10/10 步数循环，不能写成训练普遍有效。

### 10. 7B 在 Kaggle T4 x2 上的单卡约束（2026-10-22）

7B NF4 与 AWQ 预检暴露了两个与模型无关的环境问题：

1. **`device_map="auto"` 把 7B 分到两张 T4**：训练 loss 出现
   `Expected all tensors to be on the same device, cuda:1 and cuda:0`。修复为
   `device_map={"": 0}`，7B NF4 权重约 4.5 GiB，单张 16 GB T4 足够。
2. **`SFTTrainer` 仍把模型包成 DataParallel**：报
   `RuntimeError: chunk expects at least a 1-dimensional tensor`。修复为在 Kaggle
   wrapper 的子进程环境里设置 `CUDA_VISIBLE_DEVICES=0`，并在 `run_sft()` 内
   `setdefault` 同一变量，确保训练进程只看到一张卡。

这两条都是运行时配置问题，不是模型能力或 stable 违反证据。7B 预检结论：
AWQ vLLM 推理可用（模型加载 5.1961 GiB、KV 5.71 GiB），但 AWQ + LoRA 反向因
`autoawq`/`transformers` 的 `PytorchGELUTanh` 导入不兼容失败；主对比改用 NF4 对称路径。

### 11. 7B NF4 v4 OOM 与 v5 汇总失败（2026-10-22）

v4 在 `max_seq_length=4096`、LoRA r=64 下训练到第 3 个 optimizer step 时触发
`torch.OutOfMemoryError`（T4 14.56GB，仅剩 880.81MiB）。这不是数据或模型能力问题。

修复：v5 将 `max_seq_length` 降到 2048，并只在 7B 配置启用 `paged_adamw_8bit`，
r=64 与数据不变。按 Qwen2.5-7B tokenizer 统计，train 90 条中只有 1 条超过 2048 token，
4096 时也只丢这 1 条，因此没有额外数据损失。v5 训练完成：89 train / 30 val、22 step、
403.6s、adapter 645,975,704 bytes。

v5 最后 `aggregate()` 在父进程报 `ModuleNotFoundError: No module named 'training'`：
`PYTHONPATH` 只设在子进程 `env` 里，父进程导入 `training` 时没有继承。该错误发生在
baseline 与 SFT 各 60 条 runs 和 120 条 trace 已落盘之后；本地用同一
`summarize_condition` 重算，不能写成 baseline/SFT 失败。后续修复方式：父进程也设置
`sys.path.insert(0, str(REPO))` 或 `os.environ["PYTHONPATH"]`，并把汇总结果写入
`kernel_status.json` 的 `stage=aggregate`。

### 12. 7B 模板修复 v6–v9（2026-10-22）

这一轮只改变训练渲染模板与序列长度，baseline、数据、lr、epoch、rank、step 不变。失败
链条和止损记录如下：

| 版本 | 阶段 | 现象 | 处理 | 是否产出可用指标 |
| --- | --- | --- | --- | --- |
| v6 | 模板探针/导入 | `ModuleNotFoundError: agent` | `default_tool_schemas()` 自行把 `<repo>/src` 加入 `sys.path` | 否 |
| v7 | 3072 训练 | optimizer state 建立后 OOM，`Tried to allocate 1.35 GiB` | 加非重入 gradient checkpointing | 否 |
| v8 | 3072 训练 | 仍 OOM，峰值未下降 | 启用 Liger fused linear cross entropy | 否 |
| v9 | 训练 + held-out | 训练完成，held-out 120 条完成 | 记录结果，不再调参 | 是 |

v9 真实成本与结果：

- 训练时长 1520.1s；train_loss 0.6501 → 0.0988；adapter 总 661,861,305 bytes；
- baseline 60 条 trace 累计 908.8s；SFT 60 条 trace 累计 1243.6s（串行、单卡，不含模型
  加载）；
- baseline 40/60、SFT 60/60；BASE/HARD 均从 20/30 到 30/30；valid stable 4 → 0；
- 训练渲染 1567 token、推理 1570 token，二者都含 `# Tools` / `<tools>`。

v6–v8 的失败会话没有产出指标，不能记成 stable=0 或模型能力失败。v9 之后不重推 Kaggle，
不做参数扫描，也不把 Liger 写成能力变量。

### 13. DPO v1 OOM 与 v2 内存修复（2026-10-22 至 2026-10-23）

DPO 方法对照第一次运行（kernel `autumnsq/lite-agent-method-dpo` version 1）在 SFT
阶段正常完成，然后在 DPO 第一个 step 失败：

| 阶段 | 结果 |
| --- | --- |
| SFT（DPO 初始化） | 完成，361 train / 74 val，90 optimizer step，adapter 645,975,704 bytes |
| DPO | 42.3s 后 `torch.OutOfMemoryError`：Tried to allocate 1.84 GiB，GPU 仅剩 436.81 MiB |
| held-out | 未执行，0 条 DPO 指标 |

失败点不在数据、协议或 adapter，而是 DPO 同时做 policy/reference 两次前向。T4 14.56 GB
在 `max_length=3072`、`max_prompt_length=2048`、batch 1 下不够。按“只改内存路径、
不改配方”的原则做了三处修复：

- `precompute_ref_log_probs=true`：预先计算 reference log probs，训练 step 不再跑
  reference 前向；
- `precompute_ref_batch_size=1`：限制预计算时的 batch；
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：减少碎片。

`max_seq_length=3072`、`beta=0.1`、`lr=5e-5`、1 epoch、数据、rank 均不变。version 2
仍失败：在 `Train dataset reference log probs` 预计算阶段 OOM（`Tried to allocate
2.34 GiB`）。进一步审计发现 v2 实际仍读取 `private/dpo_v1/`，`--init-adapter` 仍是
train-v4 SFT adapter；因此 v2 只验证了“加 precompute 后仍 OOM”，没有测试 dpo_v2
数据。详见 `docs/evidence/DPO失败诊断.md`。v1/v2 都不产生 DPO 指标，不能把 OOM
写成“DPO 无效”；v1 SFT 5070.6s、v2 SFT 5795.2s，是必须记录的成本。

### 14. DPO v3 版本对齐与单 step smoke（2026-10-24）

DPO v3 不再让 kernel 重新训练 SFT，而是通过 `kernel_sources` 挂载
`autumnsq/lite-agent-method-assistant-train-v5`。提交前固定并复算：

- dpo_v2 数据：train 45 / val 13，train SHA `691ea266...43aed`，val SHA
  `4072a9fe...466f2`；
- train-v5 assistant-only adapter：645,975,704 bytes，SHA
  `f5b685d4...29fea7`；
- `dpo_7b_t4.json`：`max_prompt_length=3072`、`max_seq_length=3072`；
- smoke kernel：`autumnsq/lite-agent-method-dpo-smoke-v3-train-v5` version 1，
  `max_steps=1`。

`train_dpo.py` 在加载模型前校验三个 SHA，不匹配直接失败；`run_manifest.json` 在
`trainer.train()` 前落盘，OOM 也会保留版本与 error_kind。注意 `precompute_ref_log_probs=true`
会先遍历 45 条 train 做 reference 预计算，因此 `max_steps=1` 不是“只读一个 batch”。

**实测结果**（`autumnsq/lite-agent-method-dpo-smoke-v3-train-v5` version 1）：

| 项目 | 实测 |
| --- | --- |
| 状态 | `ERROR` |
| 版本校验 | train / val / init adapter 三个 `sha256_verified` 全为 `true` |
| 数据 | 45 train / 13 val；实际读取 `private/dpo_v2/` |
| init adapter | train-v5 assistant-only，645,975,704 bytes |
| 观测 token | prompt max 2998；chosen+prompt max 3006；rejected+prompt max 3008；截断 0 对 |
| reference precompute shape | `2 × 3008 × 151,643` = 912,284,288 logits 元素 |
| 失败位置 | `compute_ref_log_probs -> concatenated_forward -> logits.log_softmax`，进度 `0/45` |
| OOM 分配请求 | `Tried to allocate 3.21 GiB`；GPU 仅剩 122.81 MiB |
| 产出 | optimizer step 0、DPO adapter 未产出、held-out/rollout 0 |
| 成本 | DPO 阶段 106.7s；kernel 总计约 227.2s（1 个 T4 session，无可用指标） |

**结论**：版本错配已被排除；T4 16GB 不支持“3072 序列 + 4-bit 7B + chosen/rejected
双行完整词表 logits”。不重推 T4，不降到 1024，不再调 `precompute_ref_batch_size`
（已是最小 batch）。若继续 DPO，24GB 单卡是唯一有意义的硬门槛；本轮未租。
