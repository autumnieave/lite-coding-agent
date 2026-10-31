# GRPO prompt 池方差预检

**日期**：2026-10-24（2026-10-26 修订口径）
**输入**：`train-v5`（468 train；SHA-256 `6ee7e79d87cb7c41f4c1b91964d70abcf5e886c5ac75450318760a805b78f830`）  
**学生模型（预注册）**：Qwen2.5-7B-Instruct + train-v5 assistant-only adapter  
**候选池**：100 条（BASE/HARD/CTX/TOOL/CONFLICT 各 20，按 task id + seed 稳定抽样）  
**实现**：`scripts/grpo_prompt_pool.py`  
**候选池清单**：`docs/evidence/GRPO_prompt池方差预检.csv`

> **口径修正（2026-10-26）**：计划 100 prompt × 8 = 800 条 rollout，实际 **0/800**。
> `0/800` 是 rollout 数，不是方差；0 条 rollout 不提供方差证据。正确口径是
> **“GRPO 基础设施预检未产出 rollout（v1 `ModuleNotFoundError: agent`；v2 首次 403、
> 复诊 ERROR），组内方差未验证”**。不写“组内方差为 0”，不写“组内方差不足”，
> 也不写“GRPO 无效”。
>
> **后续（2026-10-26）**：v2 中间态池的 full-loop 预检已完成：20 prompt × 8
> rollout = 160 条，160/160 成功、0 error，0/20 prompt 有正方差，判定“实测组内方差
> 稀疏，GRPO 不启动”。本文件保留 v1 的协议与失败历史；最新结果见
> `docs/evidence/GRPO预检结果.md`。

## 1. 预检协议

- 每条候选 prompt 重采样 **8 次**，`temperature=0.8`、`top_p=None`、`seed=20261019+rollout`、`max_new_tokens=256`。
- 评分使用 `training.reward.behavior_reward` 的行为信号，并叠加最终 completion 的严格匹配；答错时 `pool_reward=0`，避免“格式合法但答案错误”获得正奖励。
- 预注册过滤口径：只保留通过率在 **20%–80%** 的 prompt；过滤后 **≥50 条**才认为 GRPO 有可用组内方差，否则写“池不足，GRPO 降级为可选”。

**协议边界（必须保留）**：本预检只重采样冻结轨迹里的最后一轮 assistant completion，历史 messages、工具选择和文件副作用均直接复用 train-v5，不重跑 Agent loop。因此它只能测“输出级采样方差”，**不能覆盖工具选择方差**；不能把结果当作完整 Agent 的 GRPO 上界。

## 2. 候选池构成

| 类别 | 候选数 |
| --- | ---: |
| BASE | 20 |
| HARD | 20 |
| CTX | 20 |
| TOOL | 20 |
| CONFLICT | 20 |
| **合计** | **100** |

候选池来自 train-v5 的 468 条成功轨迹，按 messages 的 SHA-256 去重后，再按五类做稳定哈希抽样（每类 20 条）；CSV 记录了候选 task/source id，但 rollout 结果列为空，未用空值冒充未执行的结果。**v1 池不是从 45 个中间态任务中选的**；中间态池 v2 见 `docs/evidence/GRPO候选池v2_中间态.md`。

## 3. 执行结果

| 尝试 | 状态 | 证据与处理 |
| --- | --- | --- |
| v1 `autumnsq/lite-agent-grpo-pool-preflight` | ERROR | `ModuleNotFoundError: No module named 'agent'`；已修 `sys.path`，未在原 kernel 上无限重推 |
| v2 `autumnsq/lite-agent-grpo-prompt-pool-preflight-v2` | ERROR | 首次状态查询返回 403；2026-10-24 复诊 `kernels status` 返回 `ERROR`，`failure_message` 为空；`kernels output` 60s 无返回，仍无 `summary.json`/`runs.jsonl`；详见 `docs/evidence/GRPO失败诊断.md` |

**实际 rollout 数：0/800（100 prompt × 8）**。这是 rollout 数，不是方差。由于没有可读取的输出，不能计算过滤后池数量、通过率分布或 reward variance；这些字段保持未验证，不伪造数字。

## 4. 判定与边界

- 预检状态：**未完成（运行侧未验证）**。原因不是协议或任务设计结论，而是 Kaggle kernel 的模块错误后，新 kernel 状态接口持续 403；v2 复诊为 `ERROR` 且 `failure_message` 为空。
- 按预注册止损：不重复推送、不更换采样配置、不以本地推断替代 GPU 结果；**GRPO 降级为可选，当前不启动**。
- 不能写“组内方差为 0”，也不能写“组内方差不足”，因为 0 条 rollout 不提供方差证据；只能写“组内方差未验证，GRPO 暂缓”。
- 若后续恢复 Kaggle，先修打包/权限，再最多重试一次；只有拿到 ≥50 条 20%–80% prompt，才进入 GRPO 训练分支。本轮不重推。

## 5. 中间态候选池 v2（2026-10-26）

v1 池是 train-v5 全量分层抽样，不能回答“中间态任务上是否有组内方差”。v2 改用
45 个 baseline 通过率 37.5%–62.5% 的中间态任务，冻结 `task_ids.jsonl` /
`prompts.jsonl` / `manifest.json`。池来源、类别分布和 SHA 见
`docs/evidence/GRPO候选池v2_中间态.md`。

本轮只修口径和候选池；尚未用 v2 池重跑 Kaggle 预检。若重跑，按 20–30 条 ×8 rollout
执行，失败 1 次即停；没有实测数字时，组内方差仍记为“未验证”。
