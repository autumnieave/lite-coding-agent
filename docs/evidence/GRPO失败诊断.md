# GRPO 失败诊断

> **最新口径（2026-10-26）**：45 任务正确预检已经执行：44 个 8/8、1 个 7/8、中间态 0，
> 结论是 SFT 饱和、GRPO 不该做。本文记录的是旧 20 任务子集预检，只能作历史对照；
> 不能替代 45 任务口径，也不写“GRPO 组内方差为 0”。最新结果见
> `docs/evidence/GRPO正确预检.md`。


**日期**：2026-10-24（2026-10-26 修订口径）
**范围**：只读诊断 GRPO prompt-pool preflight v1/v2；不重推 kernel，不修改 reward 或训练代码。  
**证据**：`private/kaggle_pull_grpo_preflight_v1/`、`private/kaggle_pull_grpo_preflight_v2/`、`docs/evidence/GRPO_prompt池方差预检.csv`、`scripts/grpo_prompt_pool.py`、`tests/test_training_reward.py`、`tests/test_reward_penalties.py`、`tests/test_reward_validation.py`。

> **口径修正（2026-10-26）**：此前“组内方差为 0 / 组内方差不足”的表述不成立。
> 计划是 100 个 prompt × 8 次采样 = **800 条 rollout**，实际产出是 **0/800**。
> `0/800` 是 rollout 数，不是方差；0 条 rollout 不提供任何方差证据。
> 正确口径是：**GRPO 基础设施预检未产出 rollout（v1 `ModuleNotFoundError: agent`；
> v2 首次 403、复诊 `ERROR` 且 `failure_message` 为空），组内方差未验证**。
> 不写“GRPO 无效”，也不写“组内方差不足”。
>
> **候选池来源**：v1 的 100 条来自 `train-v5` 的 468 条成功轨迹，按
> BASE/HARD/CTX/TOOL/CONFLICT 五类做稳定哈希抽样（每类 20 条），**不是**从 45 个
> 中间态任务中选的。中间态池 v2 见 `docs/evidence/GRPO候选池v2_中间态.md`。
>
> **2026-10-26 补充**：v2 中间态池的 full-loop 预检已执行完成（20 prompt × 8
> rollout），实测 160/160 成功、0 个 prompt 有正方差，判定 20/20 饱和、GRPO 不
> 启动；详见第 9 节和 `docs/evidence/GRPO预检结果.md`。此前“组内方差未验证”只适用于
> 旧的 v1/v2 基础设施失败阶段，不再是当前状态。

## 0. 结论先行

- v1 是 **打包/路径问题**：`ModuleNotFoundError: No module named 'agent'`，缺的是 `src` 进入 import path，不是 reward bug，也不是模型能力或 GRPO 方法问题。
- v2 首次状态查询确实返回过 Kaggle API `403 Permission 'kernels.get' was denied`；但 2026-10-24 复诊时 status 接口已恢复，v2 当前为 **`KernelWorkerStatus.ERROR`**，`failure_message` 为空。v2 的具体内核错误仍未知，不能写成“v1 的 import 错误在 v2 复现”。
- 账号下可读取状态的 kernel 中没有 `RUNNING`/`QUEUED`；v1/v2 都是 `ERROR`，没有证据显示 GRPO 被其他 kernel 占用 GPU 配额。
- reward 本地回归 11/11 通过，没有发现 reward 代码 bug 证据。用户提到的 `tests/test_reward.py` 不存在。
- 候选池有 100 条（5 类各 20），但 `grpo_prompts_filtered.jsonl` 不存在，实际 rollout 为 **0/800**。**0/800 是 rollout 数，不是方差**；0 条 rollout 不提供方差证据，因此不能计算 20%–80% 过滤后数量、通过率分布或组内方差。
- 决策：上一轮不修、不重推，GRPO 保持可选分支；2026-10-26 修订口径后，预检是否重跑取决于基础设施是否修通，失败 1 次即停。恢复时先修打包并重新获取有 `kernels.get` 权限的 token，最多重试一次。

## 1. v1：`agent` import 路径错误

**kernel**：`autumnsq/lite-agent-grpo-prompt-pool-preflight`  
**状态**：`ERROR`  
**耗时**：83.4s；其中 model download 73.4s，`grpo_prompt_pool` 阶段 0.2s。

完整关键堆栈：

```text
File "/kaggle/working/repo/scripts/grpo_prompt_pool.py", line 376, in <module>
  raise SystemExit(main())
File "/kaggle/working/repo/scripts/grpo_prompt_pool.py", line 301, in main
  from training.kaggle.transformers_provider import TransformersOfflineProvider
File "/kaggle/working/repo/training/kaggle/transformers_provider.py", line 14, in <module>
  from agent.core.llm import BaseProvider, LLMResponse
ModuleNotFoundError: No module named 'agent'
```

对应日志：`private/kaggle_pull_grpo_preflight_v1/lite-agent-grpo-prompt-pool-preflight.log:8-18`。

诊断：

- v1 打包进去的 `private/kaggle_pull_grpo_preflight_v1/repo/scripts/grpo_prompt_pool.py` **没有** `sys.path.insert(...)`；
- 当前仓库的 `scripts/grpo_prompt_pool.py:24-25` 已有：

```python
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
```

- 失败发生在 import 阶段，模型下载完成后只跑了 0.2s，尚未进入 GPU rollout，因此不是 reward 或组内方差问题；
- 这与 small model v10 的 `ModuleNotFoundError: training` 不是同一错误：这里缺的是 `agent`，属于 `src` 未进入 import path 的打包/路径问题。

## 2. v2：历史 403 与当前 ERROR

**kernel**：`autumnsq/lite-agent-grpo-prompt-pool-preflight-v2`  
**`kaggle kernels list -m` 的 lastRunTime**：`2026-10-24 08:50:32.777`

### 2.1 首次状态查询的 403 原文

当次通过 Kaggle SDK 请求 `GetKernelSessionStatus`，原始响应体为：

```json
{"error":{"code":403,"message":"Permission 'kernels.get' was denied","status":"PERMISSION_DENIED"}}
```

同一时间 `kaggle kernels status autumnsq/lite-agent-grpo-prompt-pool-preflight-v2` 和 `kaggle kernels output ...` 也返回 403。这个 403 是 **Kaggle API 权限拒绝**，不是 kernel 内部训练代码返回的 403。

### 2.2 2026-10-24 复诊

status 接口已恢复，CLI 返回：

```text
autumnsq/lite-agent-grpo-prompt-pool-preflight-v2 has status "KernelWorkerStatus.ERROR"
```

同一响应中 `failure_message` 为空。`kaggle kernels output` 重试 60s 无返回后终止，本地 `private/kaggle_pull_grpo_preflight_v2/` 为空，没有 `summary.json`、`runs.jsonl` 或训练日志。

因此：

- v2 的最终状态可确认是 **ERROR**；
- v2 的具体失败原因 **未取得**，不能断言仍是 `agent` import 错误；
- 也不能把 403 写成“kernel 训练失败”或“GRPO 方法失败”；
- 本次未进行浏览器 UI 双确认，只有 Kaggle CLI/API 状态与 `kernels list -m` 的交叉记录。

## 3. GPU 名额审计

2026-10-24 只读遍历账号下 31 个 kernel 的 session status：

| 状态 | 数量 | 说明 |
| --- | ---: | --- |
| `COMPLETE` | 16 | 已完成 |
| `ERROR` | 6 | 含 GRPO v1/v2、DPO、7B AWQ、constrained decoding 等 |
| `CANCEL_ACKNOWLEDGED` | 1 | 已取消 |
| `RUNNING` | 0 | 无 |
| `QUEUED` | 0 | 无 |
| status 查询 HTTPError | 8 | 旧 kernel，无法读取状态 |

结论：

- 当前可读取状态的 23 个 kernel 中，没有 `RUNNING`/`QUEUED`；
- v1/v2 都是 `ERROR`，不占用 GPU；
- 8 个旧 kernel 的 status 查询失败，不能据此推断它们的 GPU 占用，但没有任何证据显示 GRPO 被它们挤占；
- 403 更符合 token 权限问题，而不是 GPU 配额问题。

## 4. reward 回归与覆盖映射

用户要求运行的 `tests/test_reward.py` **不存在**。实际执行的是以下三个文件，共 11 个测试：

| 测试文件 | 测试数 | 覆盖内容 |
| --- | ---: | --- |
| `tests/test_training_reward.py` | 5 | JSON 可解析、snake_case、路径越界、真实文件观察、文件任务硬门禁 |
| `tests/test_reward_penalties.py` | 4 | 重复调用、`within_steps`、答案后调用、`tools_allowed` |
| `tests/test_reward_validation.py` | 2 | reward 检查项与 trace 检查项对齐 |

用户要求的四个行为信号均有独立测试：

| reward 项 | 对应测试 |
| --- | --- |
| `tools_allowed` | `test_disallowed_tool_is_an_independent_penalty` |
| 重复调用 | `test_repeated_identical_calls_are_penalized` |
| 答案后调用 | `test_tool_call_after_answer_is_penalized` |
| `within_steps` | `test_within_steps_failure_sets_zero_reward` |

运行结果：

```text
11 passed
```

没有发现 reward 代码 bug 的证据。该结果只证明本地 reward 回归通过，不代表 Kaggle 上的 GRPO 训练已经可用。

## 5. prompt 池、过滤后池与 rollout 状态

候选池：`docs/evidence/GRPO_prompt池方差预检.csv`

| 项目 | 值 |
| --- | --- |
| 行数 | 100 |
| BASE / HARD / CTX / TOOL / CONFLICT | 20 / 20 / 20 / 20 / 20 |
| SHA-256 | `d73daddbc69f8c9745ae03365d5710c595e0fc64686ecac7c03a2e7404c10674` |
| 来源 | `train-v5` 468 条按类别稳定哈希抽样；**不是** 45 个中间态任务 |
| `run_status` | 全部 `not_run_kaggle_403` |
| 实际 rollout | 0/800（100 prompt × 8；这是 rollout 数，不是方差） |

仓库中不存在 `grpo_prompts_filtered.jsonl`。本轮**没有生成过滤后池**，原因是：

1. 候选池需要 train-v5 assistant-only SFT 模型对 100 条 prompt 各采样 8 次，但 v1/v2 均未产出任何 rollout（实际 0/800）；
2. 用户明确要求本轮不修 GRPO 打包、不修 token 权限、不重推 kernel；
3. 本地只有 MX250 2GB，不能运行 7B 采样，无法用本地结果替代 Kaggle rollout。

因此本节的结论是“**过滤后池未生成、通过率分布未验证、组内方差未验证**”，而不是“过滤后池不足”或“组内方差为 0”：

| 项目 | 结果 |
| --- | --- |
| 候选池 | 100 条，5 类各 20 |
| 实际 rollout | 0/800（rollout 数，不是方差） |
| 过滤后 20%–80% prompt 数 | 未验证 |
| 通过率分布 / reward variance | 未验证 |
| `grpo_prompts_filtered.jsonl` | 未生成 |

如果未来恢复运行并得到 ≥50 条 20%–80% prompt，GRPO 仍只保持“可选”，不自动启动；如果 <50 条，才写“池不足、组内方差信号稀疏”。本轮两种数字都不能伪造。

## 6. 修复决策

| 分支 | 决策 | 理由 |
| --- | --- | --- |
| 本轮 GRPO | **不修、不重推** | 0 rollout，且 v2 具体错误未取得；继续重推只会消耗 GPU session |
| GRPO 定位 | **可选分支，不启动** | reward 回归通过；当时过滤后池和组内方差未验证。最新 v2 预检为 0/20 正方差，仍不启动（第 9 节） |
| 未来恢复 | 最多重试一次 | 先确认打包包含 `src` 的 `sys.path`，再获取有 `kernels.get` 权限的 Kaggle token；仍失败就停止 |
| 结果口径 | **旧状态未验证；最新已实测** | 旧 0/800 是 rollout 数，不是方差；最新 v2 预检 160 条实测 20/20 饱和（第 9 节）；不写“GRPO 无效” |

恢复前的检查清单：

1. 重新生成 kernel bundle，并在远端确认 `scripts/grpo_prompt_pool.py` 含 `sys.path.insert(.../src)`；
2. 用最小 `staging` 模式先跑 3 条，确认 `errors=0` 且至少 1 条 `task_success=true`；
3. 重新生成/更换 Kaggle API token，确认 `kernels status` 和 `kernels output` 都有权限；
4. 只有拿到 ≥50 条 20%–80% prompt，才进入 GRPO 训练分支；
5. 若再失败，保持“GRPO 可选、未验证”，不继续消耗 session。

## 7. 2026-10-26 口径修正与后续

- **修正内容**：删除“组内方差为 0 / 组内方差不足”的错误口径。实际是计划 800 条
  rollout、产出 0 条；`0/800` 是 rollout 数，不是方差。0 条 rollout 不能用来推断
  方差为 0，也不能用来判断 GRPO 是否有效。
- **基础设施状态**：v1 是 `ModuleNotFoundError: agent` 的打包/路径问题；v2 首次状态
  查询 403，复诊为 `ERROR` 且 `failure_message` 为空，输出目录没有 `summary.json`
  或 `runs.jsonl`。因此正确表述是“基础设施预检未产出 rollout，组内方差未验证”。
- **候选池 v1 审计**：100 条来自 train-v5 468 条的分层稳定哈希，不是 45 个中间态
  任务。v1 池不能回答“中间态任务上是否有组内方差”。
- **候选池 v2**：改用 45 个 baseline 通过率 37.5%–62.5% 的中间态任务，冻结
  `task_ids.jsonl` / `prompts.jsonl` / manifest，SHA 见
  `docs/evidence/GRPO候选池v2_中间态.md`。
- **预检决策**：时间限制，只修正口径，预检未做；不跑全量 GRPO。Kaggle status API
  已恢复（不再 403），但旧 v2 kernel 仍是 ERROR，且仓库没有 v2 中间态 full-loop
  kernel builder。若后续重跑，选 20–30 条中间态任务 ×8 rollout；失败 1 次即停；
  ≥30% 的 prompt 有组内方差才写“预检通过”，否则写“实测组内方差稀疏（X/N），
  GRPO 不启动”。没有实测数字时写“未验证”，不写“方差为 0”。

## 8. 成本与边界

- v1：83.4s，其中模型下载 73.4s，脚本 0.2s；没有 rollout；
- v2：至少一次推送并产生 `lastRunTime`；是否实际占用 GPU、占用多久，无法从输出确认；不伪造耗时或 rollout；
- 本文只读诊断，没有重推 kernel，没有修改 `core/loop.py`、`compaction.py`、`constraints.py`，没有修改 reward 逻辑。

## 9. 2026-10-26 v2 中间态池预检实测

**Kernel**：`autumnsq/lite-agent-grpo-preflight-v3`（version 1，`COMPLETE`）  
**硬件**：Tesla T4 14.56 GB；总耗时 2509.9 s（0.70 GPU·h，Kaggle 免费额度）。

修复与执行：

- 新 builder 把 `src/` 和仓库根目录同时加入 `PYTHONPATH`，`ModuleNotFoundError:
  agent` 消失；
- adapter 从 `autumnsq/lite-agent-method-assistant-train-v5` 挂载，SHA-256
  `f5b685d4...9fea7` 校验通过；
- 之前 PATH 上的 Kaggle CLI 1.7.4.5 不认识 `KGAT` token，状态查询 403；改用仓库
  `.venv` 的 Kaggle CLI 2.2.4 后鉴权正常。该 403 是 CLI/凭据格式问题，不是 kernel
  权限或 GPU 配额问题。

结果：

| 项目 | 值 |
| --- | ---: |
| prompt | 20（BASE 6 / HARD 4 / TOOL 10） |
| rollout | 160（20 × 8） |
| valid / errors | 160 / 0 |
| task_success | 160/160 |
| measured_positive | 0/20 |
| measured_zero | 20/20 |
| not_measured | 0/20 |
| verdict | `sparse` |

20 个 prompt 全部 8/8，20/20 饱和、没有中间态；staging（1 × 8）也是 8/8、0 error。按预注册
`>=30%` 门槛，结论是 **“实测组内方差稀疏（0/20），GRPO 不启动”**。这与此前
`0/800 rollout` 不是同一件事：`0/800` 是 rollout 数，不能当方差；本次是 160 条真实
rollout 测到的 20/20 饱和；不作为 45 任务池的方差结论。

边界：20 个任务全部 `training_overlap=true`，SFT 后 8/8 更符合训练覆盖后的拟合；
不是 clean held-out，也不是跨模板泛化。未训练 GRPO，不能写“GRPO 无效”。详见
`docs/evidence/GRPO预检结果.md`。

**相关文档**：`docs/evidence/GRPO_prompt池方差预检.md`、`docs/evidence/GRPO_prompt池方差预检.csv`、`docs/plan/训练侧执行计划.md`。
