# Constrained decoding 对照（Kaggle T4）

**日期**：2026-10-19（首次失败），2026-10-20（1.5B 路径重跑与 v4）  
**状态**：v1 因 7B 模型路径不存在 `ERROR`；v2 因缺少 `bitsandbytes>=0.46.1` `ERROR`；
v3 在 `RUNNING` 期间被手动取消；v4 改为 limit=5、max_new_tokens=256、900 秒外层超时，
最终仍为 `ERROR`，超时发生在模型权重加载阶段（日志停在 `Loading weights: 0%`）。
四次运行都**没有完成任何 rollout，也没有生成 `json_valid` / `snake_case_keys` 指标**。  
**结论状态**：本地代码路径与 1.5B 路径解析可用，但 Kaggle v4 未跑通；没有可用指标，
不可用于“前沿系统侧方案是否足够”的结论，也不阻塞四组对比。

## 1. 运行对象与真实失败

- Kernel：`autumnsq/lite-agent-constrained-decoding-baseline`
- 入口：`training/kaggle/kaggle_constrained_decoding.py`，由 `private/kaggle_kernel/run.py`
  在 Kaggle 内启动
- 模型：目标为 `Qwen/Qwen2.5-Coder-7B-Instruct`
- 实际传给脚本的路径：
  `/kaggle/input/qwen2-5-coder-7b-instruct/Qwen2.5-Coder-7B-Instruct`
- Kernel 配置：T4 16GB；实际分配的 GPU 型号/显存未在失败前写入日志，不能补造
- 日志：`private/kaggle_outputs/lite-agent-constrained-decoding-baseline.log`

真实异常：

```text
huggingface_hub.errors.HFValidationError: Repo id must be in the form
'repo_name' or 'namespace/repo_name':
'/kaggle/input/qwen2-5-coder-7b-instruct/Qwen2.5-Coder-7B-Instruct'.
```

这不是 constrained decoding 的输出翻转，也不是模型拒绝生成，而是本地模型目录不存在后
Transformers 把路径当成 Hugging Face repo id。失败发生在约 33 秒，notebook 约 39 秒后
以非零状态退出。

## 2. 实际状态表

| 项 | 值 |
| --- | --- |
| Kernel 状态 | `KernelWorkerStatus.ERROR` |
| 失败阶段 | 模型/分词器加载，生成前 |
| 失败原因 | 模型路径不存在，触发 `HFValidationError` |
| 已完成 rollouts | 0 |
| 输出文件 | 无 `constrained_decoding_results.jsonl` / `constrained_decoding_summary.json` |
| 实际 GPU 型号/显存 | 未记录，N/A |
| 运行时长 | 约 33 秒失败，notebook 约 39 秒退出 |
| `json_valid` | N/A |
| `snake_case_keys` | N/A |
| `no_code_fence` | N/A |
| `schema_accepts` | N/A |

不能把本地 schema 单测、dry-run 或这次 ERROR 当成真实 baseline。

## 3. 修复动作（已本地验证并重跑）

1. 在 Kaggle 内用 `Path("/kaggle/input").rglob("config.json")` 定位实际模型目录，而不是
   写死 `qwen2-5-coder-7b-instruct/...`。
2. 找不到 `config.json` 时打印 `/kaggle/input` 的目录清单并立即失败，禁止继续假装模型已加载。
3. 构建脚本的 kernel id 改为实际的
   `autumnsq/lite-agent-constrained-decoding-baseline`。
4. `resolve_model_path()` 支持 `preferred_model` / `--preferred-model`：同时挂载 7B 与
   1.5B 时可明确选择目标；汇总文件的 `model_version` 从实际解析目录推导，不再硬编码
   7B。若 Kaggle 只挂载 1.5B，就如实记录 1.5B，不能写成 7B。
5. 等小基座 kernel 释放 GPU 队列后重新推送；重跑前不写任何命中率。

### 3.1 1.5B 路径重跑记录（2026-10-20）

| 版本 | 模型路径 | 结果 | 失败/运行阶段 |
| --- | --- | --- | --- |
| v1 | 7B 路径 `qwen2-5-coder-7b-instruct/...` | `ERROR` | 模型路径不存在，`HFValidationError`，0 rollout |
| v2 | 1.5B 路径 `qwen2-5-1-5b-instruct/qwen2.5-1.5b-instruct` | `ERROR` | 路径解析成功；加载 4-bit 时缺少 `bitsandbytes>=0.46.1`，0 rollout |
| v3 | 同上 | `CANCEL_ACKNOWLEDGED` | 改为 fp16 加载（1.5B 在 T4 16GB 不需要 4-bit），长时间无输出后被用户手动取消 |
| v4 | 同上 | `ERROR` | limit=5、max_new_tokens=256、外层 900s 超时；权重加载停在 0%，0 条结果 |

v1/v2/v4 的失败都发生在模型加载阶段，不是 constrained decoding 生成失败，也不是模型能力
或 `json_valid` 翻转；v3 在长时间无输出后被手动取消。`resolve_model_path()` 已在日志中确认
`resolved_model_path=/kaggle/input/qwen2-5-1-5b-instruct/qwen2.5-1.5b-instruct`。
因此本文不给任何 `json_valid` / `snake_case_keys` 数字。代码审查发现三处 API 风险：
旧 `_generate_one` 先把 `schema` 作为第二个位置参数传入，某些 Outlines 版本会把它吞进
inference kwargs；Outlines 0.1 的生成器使用 `max_tokens` 而不是 `max_new_tokens`；
`outlines.from_transformers()` 返回的是模型适配器，不能直接当绑定 schema 的生成器用。
已改为：先构造 `outlines.generate.json(model, schema)`，按生成器签名选择
`max_tokens` / `max_new_tokens`，并仅在 `TypeError` 时回退到显式 schema。新增 3 个本地
单测覆盖这两条 API。v4 已带上这些修复重推，但仍在权重加载阶段 900 秒超时；因此不能据此
声称 Outlines 生成路径已被验证，也不能把超时写成模型能力失败。

本地已用临时目录模拟 `/kaggle/input`：只挂载
`Qwen2.5-Coder-1.5B-Instruct/config.json` 时 resolver 能找到该目录；没有任何
`config.json` 时抛 `FileNotFoundError`。该 smoke 只验证路径，不代表模型加载或
constrained decoding 已有结果。

### 3.2 v4 真实结果（2026-10-20）

| 项 | 值 |
| --- | --- |
| Kernel | `autumnsq/lite-agent-constrained-decoding-baseline-1-5b` v4 |
| 状态 | `KernelWorkerStatus.ERROR` |
| 模型路径 | `/kaggle/input/qwen2-5-1-5b-instruct/qwen2.5-1.5b-instruct`，日志确认 `resolved_model_path` 正确 |
| 运行参数 | `--limit 5`、`--max-new-tokens 256`、wrapper `TIMEOUT_S 900` |
| 失败阶段 | 模型权重加载，`Loading weights: 0%`；生成前 |
| 运行时长 | 约 913 秒；`TIMEOUT 900` 在 909.6 秒触发 |
| 结果文件 | `constrained_decoding_results.jsonl` 为 0 字节 |
| summary | `{"status": "timeout", "timeout_s": 900, "output": "...", "note": "partial results may exist in the output JSONL"}` |
| 已完成 rollout | 0 / 5 条限制；prompts 文件共 20 条 |
| 实际 GPU 型号/显存 | 日志未写入；kernel 配置为 T4，不能把配置当成运行实测 |
| `json_valid` | N/A |
| `snake_case_keys` | N/A |
| `no_code_fence` | N/A |
| `schema_accepts` | N/A |

这次没有出现 `HFValidationError`，也没有出现 `bitsandbytes` 缺失；路径解析成功，失败点是
Outlines/Transformers 加载 1.5B 权重在 900 秒内没有完成。不能写成“模型在 constrained
decoding 下仍然违反”，也不能写成“压到 0 违反”。若未来重试，优先预加载并缓存模型权重、
使用更长的加载超时，或改用 vLLM guided decoding；不要在 GPU 队列紧张时反复重推。

## 4. 跑完后的预注册比较口径

与阶段 A / 方向 C 对照时，只比较同一批 run 2/10 的 `json_valid` 与
`snake_case_keys`，并单独报告 `no_code_fence`、`schema_accepts`、实际 rollout 数和耗时。

- 基线：阶段 A 固定采样配置下的 20 次 rollout；按 run 2/10 汇总。
- 方向 C 手工自检：20 次 rollout；同样只取 run 2/10。
- constrained decoding：20 条 prompt 各生成一次；这是单轮 probe 约束解码，不是完整多轮
  Agent loop。

两种预设结论：

1. 若 `json_valid` / `snake_case_keys` 违反被压到 0：写“前沿系统侧方案可解决输出格式违反，
   主样本不需要训练”，但只限定在输出格式层，不外推到工具行为或任务完成。
2. 若仍有翻转：写“多轮 Agent 上下文下的 token 级约束仍受采样/上下文非确定性影响”，
   不把单次 0 违反写成稳定解决。

## 5. 交付要求

若 Kaggle 完成，必须回填：GPU 型号与显存、运行时长、实际 rollout 数、`model_version`、
`system_fingerprint`、`response_id`（本地 transformers/Outlines 没有 API 响应时写 `N/A`，
不伪造）、逐 run 命中数和总体命中率。若配额不足只跑了子集，按实际条数报告，不补造未运行部分。

当前 v4 未完成：只有“0 条结果、900 秒加载超时”这一条事实；在拿到真实 rollout 前，本文
不提供任何命中率，也不把 constrained decoding 加入四组对比的 `+系统侧` 组。
