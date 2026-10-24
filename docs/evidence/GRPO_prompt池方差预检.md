# GRPO prompt 池方差预检

**日期**：2026-10-24  
**输入**：`train-v5`（468 train；SHA-256 `6ee7e79d87cb7c41f4c1b91964d70abcf5e886c5ac75450318760a805b78f830`）  
**学生模型（预注册）**：Qwen2.5-7B-Instruct + train-v5 assistant-only adapter  
**候选池**：100 条（BASE/HARD/CTX/TOOL/CONFLICT 各 20，按 task id + seed 稳定抽样）  
**实现**：`scripts/grpo_prompt_pool.py`  
**候选池清单**：`docs/evidence/GRPO_prompt池方差预检.csv`

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

候选池来自 train-v5 的 468 条成功轨迹，按 messages 的 SHA-256 去重后抽取；CSV 记录了候选 task/source id，但 rollout 结果列为空，未用空值冒充未执行的结果。

## 3. 执行结果

| 尝试 | 状态 | 证据与处理 |
| --- | --- | --- |
| v1 `autumnsq/lite-agent-grpo-pool-preflight` | ERROR | `ModuleNotFoundError: No module named 'agent'`；已修 `sys.path`，未在原 kernel 上无限重推 |
| v2 `autumnsq/lite-agent-grpo-pool-preflight-v2` | 状态查询 403 Forbidden | 推送成功后查询 `/kernels/status` 被 Kaggle 拒绝；没有可见的 `summary.json`、`runs.jsonl` 或完成日志 |

**实际 rollout 数：0/800（100 prompt × 8）**。由于没有可读取的输出，不能计算过滤后池数量、通过率分布或 reward variance；这些字段保持未验证，不伪造数字。

## 4. 判定与边界

- 预检状态：**完成，但运行侧未验证**。原因不是协议或任务设计结论，而是 Kaggle kernel 的模块错误后，新 kernel 状态接口持续 403。
- 按预注册止损：不重复推送、不更换采样配置、不以本地推断替代 GPU 结果；**GRPO 降级为可选，当前不启动**。
- 不能写“组内方差不足”，因为 0 条 rollout 不提供方差证据；只能写“组内方差未验证，GRPO 暂缓”。
- 若后续恢复 Kaggle，直接使用现有 v2 kernel 配置重试一次；只有拿到 ≥50 条 20%–80% prompt，才进入 GRPO 训练分支。
