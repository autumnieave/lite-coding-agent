# Kaggle 提交前契约

**版本**：v1  
**日期**：2026-10-20  
**目的**：在消耗 GPU session 之前，用可机器校验的契约拦住“路径不存在、依赖缺失、
加载超时、输出 0 字节”四类已知失败。  
**门禁**：`preflight` 通过 → `staging`（3 条短跑）通过 → 才允许推全量。

## 1. 输入契约

提交前必须能回答：模型在哪里、数据在哪里、文件是否非空、模型和 tokenizer 是哪个版本。

| 检查项 | 通过条件 | 失败 `error_kind` | 实现位置 |
| --- | --- | --- | --- |
| 模型目录 | 本地目录存在且 `config.json` 非空 | `model_path_missing` / `model_config_empty` | `constraints.prepare_run`、`preflight_small_model` |
| tokenizer | `tokenizer_config.json` 非空 | `tokenizer_config_missing` / `tokenizer_config_empty` | 同上 |
| 模型来源 | staging 只接受本地挂载；Hub id 必须显式 `--allow-hub-model` | `model_path_unverified` | `preflight_small_model` |
| prompts | 文件存在、非空、每行含 `run_id` / `rollout_index` / `prompt` | `input_missing` / `input_empty` | `load_prompts` |
| SFT 数据 | `train.jsonl` / `val.jsonl` 非空，train/val 不重叠 | `input_empty` | `training/data.py` 的 split 校验 |
| 版本记录 | `preflight.json` 写入模型路径、config 字节数、tokenizer 字节数、runtime 版本 | `preflight` 不通过 | 两个入口的 `preflight` |

本地/远端预检命令（不加载模型）：

```bash
python training/kaggle/kaggle_constrained_decoding.py \
  --prompts-file /kaggle/input/lite-agent-training/constrained_prompts.jsonl \
  --model-path /kaggle/input/qwen2-5-1-5b-instruct/qwen2.5-1.5b-instruct \
  --preflight-only \
  --preflight-out /kaggle/working/constrained_preflight.json

python training/kaggle/kaggle_small_model_tuning.py \
  --repo /kaggle/working/repo \
  --model /kaggle/input/qwen2-5-1-5b-instruct/qwen2.5-1.5b-instruct \
  --preflight-only \
  --out-dir /kaggle/working/small_model_staging
```

模型 commit / tokenizer 版本：本地目录优先读 `config.json` 的 `_name_or_path`、
`transformers_version`、`torch_dtype`；Hub 快照额外记录 `huggingface_hub` 的 snapshot
commit。没有就写 `null`，不伪造 API request id。

## 2. 环境契约

| 检查项 | T4 约束 | 说明 |
| --- | --- | --- |
| Python | 3.11+；Kaggle 当前 3.13 | 不在本地 MX250 上跑训练 |
| GPU | Tesla T4 16GB，单卡 | `nvidia-smi` 写入 `preflight.json` |
| 1.5B constrained decoding | fp16 加载，`max_new_tokens<=256`，`max_model_len` 按 fixture 需要 | 不默认 4bit，减少 bitsandbytes 变量 |
| 7B QLoRA | `load_in_4bit=true`、`bnb_4bit_quant_type=nf4`、`gradient_checkpointing=true`、`per_device_train_batch_size=1`、`gradient_accumulation_steps=8`、`max_seq_length<=4096` | 配置由 `training/config.py` 校验 |
| vLLM | `0.12.0`；T4 使用 `TRITON_ATTN`、关闭 FlashInfer sampler、`--enforce-eager` | 避免 sm75 JIT 链接 `libcuda` |
| Outlines | `outlines>=0.1.11`；schema 在启动时打印 | 版本写入 summary |

环境检查失败时不要继续加载模型；把 `runtime` 和 `gpu` 原样写入 `preflight.json`，
方便下一次审计。Kaggle 的 `requirements-kaggle.txt` 只属于训练目录，不进入 core 依赖。

## 3. 运行契约

| 检查项 | 固定要求 | 实现 |
| --- | --- | --- |
| 加载超时 | 180 秒；超时写 `model_load_timeout` | `time_limit(..., "model_load_timeout")`、`_wait_for_server(timeout_s=180)` |
| 推理超时 | constrained decoding 每条 120 秒；Agent rollout 用 `--rollout-timeout` | `time_limit(..., "inference_timeout")`、`asyncio.wait_for` |
| 进度日志 | constrained decoding 每 N 条打印并 flush；vLLM 每 10 秒打印等待时长 | `--progress-every`、`_wait_for_server` |
| 先小后大 | 先 `--preflight-only`，再 `--staging`（3 条），最后全量 | `apply_staging_defaults` |
| 追加写 | 结果 JSONL 以 append 打开，每条 flush；按 `(run_id, rollout_index)` 或 `(task_id, rollout_index)` 续跑 | `append_jsonl`、`--resume` |
| 失败可见 | 每条失败记录 `error_kind` 与 `error`，不静默跳过 | `classify_exception` |

staging 命令：

```bash
# constrained decoding：3 条，256 token 上限，180 秒加载/120 秒单条推理超时
python training/kaggle/kaggle_constrained_decoding.py \
  --prompts-file /kaggle/input/lite-agent-training/constrained_prompts.jsonl \
  --model-path /kaggle/input/qwen2-5-1-5b-instruct/qwen2.5-1.5b-instruct \
  --staging \
  --out /kaggle/working/constrained_staging.jsonl \
  --summary-out /kaggle/working/constrained_staging_summary.json

# small model tuning：3 条 rollout，只跑 smoke task
python training/kaggle/kaggle_small_model_tuning.py \
  --repo /kaggle/working/repo \
  --model /kaggle/input/qwen2-5-1-5b-instruct/qwen2.5-1.5b-instruct \
  --spec-set final-test \
  --staging \
  --out-dir /kaggle/working/small_model_staging
```

staging 通过条件：进程退出码 0、`runs.jsonl` / 结果 JSONL 至少 1 条有效记录、
`summary.json` 通过 schema 校验、`error_kind` 为空。任何一项不满足，不推全量。

## 4. 输出契约

### constrained decoding

`summary.json` 必须包含：

| 字段 | 含义 |
| --- | --- |
| `status` | `completed` / `partial` / `error` |
| `error_kind` | 首个失败类别；成功为 `null` |
| `prompts_requested` / `prompts_completed` | 计划与实际条数 |
| `records` | 实际写入的记录数 |
| `overall` / `by_run` | `json_valid`、`snake_case_keys`、`no_code_fence`、`schema_accepts` |
| `metadata` | 模型、GPU、时长、rollout 数、超时配置 |
| `staging` | 是否 staging 短跑 |

### small model tuning

`summary.json` 必须包含 `records`、`errors`、`checks`、`stable_checks`、`summary`；
`runs.jsonl` 必须非空。失败时 `run_manifest.json` 写 `status=error`、`stage`、
`error_kind`、`error`，并保留 `preflight.json` 与 `vllm_server.log`。

**禁止**把 0 字节结果写成“0 违反”；`output_empty` 是运行失败，不是模型能力结论。

## 5. `error_kind` 词表

| `error_kind` | 含义 | 下一步 |
| --- | --- | --- |
| `model_path_missing` | 模型目录或 config 不存在 | 检查 `/kaggle/input` 挂载 |
| `model_path_unverified` | 传了 Hub id，staging 无法确认本地权重 | 改本地路径或显式 `--allow-hub-model` |
| `tokenizer_config_missing` | tokenizer 文件缺失 | 重新挂载完整模型 |
| `input_missing` / `input_empty` | prompts/data 不存在或为空 | 检查 Dataset 与打包脚本 |
| `dependency_missing` | vLLM/Outlines/transformers 未安装 | 先跑 `--install-runtime` 或安装 requirements |
| `model_load_timeout` | 180 秒内模型/服务未就绪 | 看 `vllm_server.log`；不要直接重推全量 |
| `inference_timeout` | 单条推理或 rollout 超时 | 降低 `--limit`，检查 fixture 长度 |
| `output_empty` | 进程结束但 JSONL 无记录 | 看 worker 日志，不写指标 |
| `summary_schema_invalid` | summary 缺必需字段 | 修脚本，不手工补 summary |
| `oom` | 显存不足 | 降 `max_model_len` / 4bit / batch |
| `unknown` | 未分类异常 | 记录 traceback，按 runbook 处理 |

## 6. 代码落点

- `training/kaggle/contracts.py`：`ContractError`、`classify_exception`、`time_limit`、
  `append_jsonl`、`completed_keys`、`validate_summary`。
- `training/kaggle/kaggle_constrained_decoding.py`：`--preflight-only`、`--staging`、
  `--limit`、`--load-timeout`、`--inference-timeout`、`--progress-every`、append/resume。
- `training/kaggle/kaggle_small_model_tuning.py`：`--preflight-only`、`--staging`、
  `--limit`、`--load-timeout`、`--rollout-timeout`、`--progress-every`、`--resume`。
- `scripts/difficulty_tuning.py`：`--limit`、`--rollout-timeout`、`--progress-every`、
  `--resume`；`runs.jsonl` 合并写回。
- 单测：`tests/test_kaggle_contracts.py`、`tests/test_kaggle_scripts.py`、
  `tests/test_kaggle_small_model.py`、`tests/test_difficulty_tuning.py`。

## 7. 当前状态

- constrained decoding v1–v4：路径/依赖/取消/加载超时，均没有可用指标；不写成 0 违反。
- 本契约的 staging 短跑需在 Kaggle 上真实执行后，把 kernel version、GPU 时长、
  `error_kind` 和输出条数写回本文件；在真实结果出来前，状态只能写“待跑”。
- final-test-v2 与 S0/S1 重跑不受本契约阻塞；staging 只作为提交门禁。

## 8. 本次实现补充

- 续跑语义：`completed_keys` 只把无 `error_kind` 的记录视为完成；超时/失败行会在下一次运行重试，`latest_records` 保证同一 `(run_id, rollout_index)` 以最后一次结果为准，不丢历史行。
- 单条失败：`generate_records` 捕获非 `ContractError` 异常，统一写 `error_kind=unknown` 和 `error`，后续 prompt 继续执行，不静默中断。
- 打包契约：`private/kaggle_build.py` 的 Dataset zip 已包含 `training/kaggle/contracts.py`；`private/kaggle_small_model/build_kernel.py` 的 self-contained zip 同样包含该文件，避免 Kaggle 侧 `ModuleNotFoundError`。
- staging 门禁：两个 builder 均先执行 `--preflight-only`，再执行 3 条 staging；small-model staging 使用独立 `*_staging` 输出目录，默认 `RUN_TIMEOUT=300/420` 秒、`load_timeout=180` 秒并跳过额外 smoke，避免 3 条门禁变成 4 条。
- 失败可见：builder 写 `kernel_status.json`，包含 `stage`、`error_kind`、超时和部分输出字节数；训练入口自身继续写 `run_manifest.json` / `preflight.json`。
- 当前状态：代码与本地语法编译已通过；真实 Kaggle staging 已尝试，但尚未通过。失败记录见第 9 节，不写“staging 通过”，也不填任何模型指标。

## 9. 真实 staging 记录（2026-10-20 / 2026-10-21）

| kernel | version | 失败阶段 | `error_kind` | 关键证据 | 是否浪费 session |
| --- | --- | --- | --- | --- | --- |
| constrained decoding | v5 | run | `run_timeout` (120s) | preflight 通过；`/kaggle/input` 模型加载未完成；`partial_output_bytes=0` | 是，约 2 分钟 |
| small model | v10 | preflight | `ModuleNotFoundError: training` | builder 未设置 `cwd`/`PYTHONPATH`，脚本未进入 GPU 加载 | 否，立即失败 |
| small model | v11 | run | `run_timeout` (120s) | vLLM 从 `/kaggle/input` 读取 safetensors 卡在 0/1 | 是，约 2 分钟 |
| small model | v12 | run | `run_timeout` (120s) | vLLM 110.1s 才 ready，剩余时间不足协议探针+3 条 rollout | 是，约 2 分钟 |
| small model | v13 | queue | 无 | 排队约 15 分钟后被用户停止，未产生结果 | 否 |
| small model | v14 | run | `model_load_timeout` (120s) | vLLM 日志显示 startup complete，但 `/v1/models` 探针未观察到 200；0 条 rollout | 是，约 2 分钟 |
| small model | v15 | run | `model_load_timeout` (120s) | 已加 `NO_PROXY` 与无代理 opener；vLLM `Application startup complete`，但父进程仍持续 `Connection refused`，0 条 rollout | 是，约 2 分钟 |
| small model | v17 | run | `staging_degraded` | server 模式实际启动成功；3 条 rollout、0 error、工具调用与参数校验均通过；但模型用 Markdown 围栏包 JSON，`task_success=0`，按新门禁记 degraded | 否，产出 3 条可分析 trace |
| small model | v18 | preflight | `unrecognized arguments: --system-variant` | builder 传了新参数但 wrapper 未声明；2 秒内失败，未进 GPU 加载 | 否 |
| small model | v19 | readiness | 无（`status=ok`） | `ss` 显示 `127.0.0.1:8000`；IPv4 200、`localhost` 200、IPv6 `::1` refused；server_ready=true，耗时 125.3s | 否，一次诊断到位 |
| small model | v20 | preflight | `ModuleNotFoundError: training.system_variants` | 离线 provider 已加入，但打包清单漏掉 `training/system_variants.py`；秒级失败 | 否 |
| small model | v20b | preflight | `ModuleNotFoundError: training.system_variants` | 同上，4 秒内失败；未进入模型加载 | 否 |
| small model | v21 | run | `child_process_failed` | 离线 vLLM 在 T4 选择 FlashInfer 后端，JIT 编译 `collect2: error: ld returned 1 exit status`；`EngineCore` 初始化失败，0 条 rollout | 是，约 4 分钟加载后失败 |
| small model | v22 | run | 无（`status=ok`） | 离线 provider 固定 `TRITON_ATTN` 后通过：`U-FT03-BASE` × 3、0 error、3/3 `task_success`、`max_rate=0.0`、126s | 否，产出 3 条 trace |
| small model | v23 | run | 无（`status=ok`） | `final-test-v2` S0 全量：110 条、0 error、19 原始 stable / 15 valid stable、task_success 40/110、410s | 否，主实验产出 |
| small model | v24 | run | 无（`status=ok`） | `final-test-v2` S1 全量：110 条、0 error、24 原始 stable / 20 valid stable、task_success 10/110、481s | 否，负结果产出 |

结论：
- 契约按预期阻止了全量推送；v10 的打包路径、v11 的挂载慢、v12 的外层预算问题均已修复或校准。
- v14/v15 的 `Connection refused` 在 v17 未复现：v17 的真实 server staging 拿到 3 条 rollout、0 error，`tools_used=["read_file"]`、`params_ok=true`，说明 vLLM 服务可达，前两次更可能是旧配置或父进程超时窗口问题。v17 仍按新门禁记 `degraded`，原因是模型输出格式（Markdown 围栏）导致 `task_success=0`，不是部署失败。
- 因此 staging 门禁现在区分两件事：`error_kind` 为运行/部署失败；`staging_degraded` 为“基础设施跑通但行为验收未过”。后者不应写成“连接失败”或“模型能力 stable”，只能写成 3 条 staging 的行为观察。
- v19 readiness 进一步定位：vLLM 只监听 IPv4 loopback `127.0.0.1:8000`，不监听 `::1`；`curl http://localhost:8000/v1/models` 会先试 IPv6 并收到 refused，再回退 IPv4 成功。`getaddrinfo` 行为或旧 kernel 的 `localhost`/代理配置足以解释 v14/v15；当前基线使用 `http://127.0.0.1:8000/v1`，不再依赖 `localhost`。
- GPU session 成本（本次契约工作，已知记录）：v10/v11/v12/v14/v15/v17/v18/v19/v20/v20b/v21/v22/v23/v24，共 14 次；其中 v10/v18/v20/v20b 在 preflight 内快速失败，v13 排队后被停止，v11/v12/v14/v15 浪费在加载/预算/连接，v17 产出 3 条 trace，v19 完成 readiness，v21 暴露 T4 FlashInfer JIT 限制，v22 用离线 API 跑通 staging，v23/v24 各自完成 110 条主实验。失败模式按“打包路径 → 挂载慢 → 外层预算 → server 可达性 → 参数契约 → IPv4/IPv6 → 离线后端选择”逐层收敛。
- v21 的修复把离线 provider 的 attention 后端固定为 `TRITON_ATTN`，与 server 模式一致；按止损规则只重推一次 v22，结果为 `status=ok`、3/3 `task_success`、0 error。该结果只说明 staging 基础设施与行为验收通过，不构成 stable 违反结论。
- 本表只记录真实失败，不把 0 条 rollout 写成“0 违反”，也不把配置/加载问题写成模型能力结论。
