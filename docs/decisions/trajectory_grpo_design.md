# 轨迹级 GRPO 改造设计

## 1. 背景与目标

现有单轮 GRPO 配置是**单轮评分**：每个 prompt 生成一条 assistant completion，解析其中的
`<tool_call>`，用 `score_reply` 打分，再把 8 条 completion 的 reward 做组内归一化。
这条路径看不到“模型先侦察、再在后续轮次直执”的多轮行为，delete 层因此长期采不到
`execution_violation`（c2_ppo1 训练期 0/72）。

本设计把评分粒度改为**完整多轮轨迹的最终状态**，其余全部冻结。目标是让 delete 层在训练
采样中出现直执负样本，使 GRPO 获得梯度；不承诺评测显著。

## 2. 与现有单轮实现的关系

| 维度 | 单轮（默认，不变） | 轨迹级（`--trajectory-scoring`） |
|---|---|---|
| 生成 | 每个 prompt 1 条 completion | 每个 prompt 8 条完整轨迹 |
| 工具执行 | 不执行，只解析文本 | 真实执行（`AgentLoop` + `build_default_registry`） |
| 评分对象 | completion 文本里的 tool_call | 轨迹最终文本 + 整条轨迹实际尝试的 tool_call |
| advantage | 8 条 completion 组内归一化 | 8 条轨迹最终 reward 组内归一化 |
| logp | 单条 completion 的 token logp | 每个步骤的 token logp，展平后聚合 |
| 默认 | ✅ 行为与之前逐字一致 | 关闭 |

## 3. 数据结构（`training/trajectory_grpo.py`，纯 Python，不 import torch）

```python
@dataclass(frozen=True, slots=True)
class StepSample:
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    old_logps: tuple[float, ...]
    mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class TrajectorySample:
    task_id: str
    layer: str
    steps: tuple[StepSample, ...]
    final_content: str
    calls: tuple[dict[str, Any], ...]
    reward: float
    bucket: str
    detail: str
```

`StepSample` 只存 Python 标量/列表，便于单测和 CPU smoke；torch 张量转换在训练脚本里做。

## 4. 多轮 rollout 循环

每个 prompt、每条轨迹：

1. `tempfile.mkdtemp()` 建独立工作区，`materialize_probe(task, root)` 写入 fixture；
2. `build_default_registry(root)` 建工具表，外面套 `RecordingExecutor` 记录每次尝试的调用
   （`name / arguments / ok`，失败调用也记录，因为违规看的是“尝试”）；
3. system prompt 用 `task_messages(task)[0]["content"]`，与单轮训练/评测的约束层一致；
4. `AgentLoop(provider, executor, max_turns=9, system_prompt=...)`，`run(user_turns[0])`；
5. 轨迹结束后删除临时工作区（`shutil.rmtree(ignore_errors=True)`）。

rollout provider（训练脚本内 `_TrajectoryProvider`）每次 `chat_stream`：

- 用 `build_chat_prompt(tokenizer, messages, tools)` 渲染；
- `model.generate` 采样一条 completion（temp 0.8，max_new_tokens 384）；
- 用 `parse_tool_calls` 把 `<tool_call>` 解析成 `ToolCall`；
- 记录 `prompt_ids`、`completion_ids` 到当前轨迹的 `steps`；
- 返回 `LLMResponse`，让 `AgentLoop` 正常执行工具并进入下一轮。

`max_turns=9` 是轨迹上限；达到上限仍按最终状态评分（与冲突评测集的口径一致）。

## 5. 轨迹级奖励

```python
score_trajectory(task, final_content, calls, *, severity_weighted, delete_split=False)
```

- `final_content`：最后一条非空 assistant 文本（空则回退到更早的非空文本）；
- `calls`：整条轨迹实际尝试的调用，转成 `{"name": ..., "arguments": ...}`；
- 复用 `score_reply(task, content, calls, from_text=False, delete_split=...)`；
- `delete_split=False` 与 `c2_ppo1` 完全一致，不引入 `split_v1`。

删除直执计数：`bucket == "execution_violation"` 且 `layer == "delete"` 的轨迹数。

## 6. 轨迹级 advantage

8 条轨迹的最终 reward → `group_advantages(rewards)`，与单轮完全同一个函数。
零方差组直接跳过更新，但仍写入 train_log 并计入 `zero_variance_groups`。

## 7. 步骤级 logp 与 loss 聚合

对一条轨迹的每个步骤 `s`：

- `old_logp_s`：rollout 后、更新前，用 policy adapter 重新前向得到的 token logp（no grad）；
- `reference_logp_s`：ref adapter 的 token logp（no grad）；
- `new_logp_s`：当前 policy 的 token logp（有 grad）；
- `mask_s`：completion token mask（保留首个 EOS，去掉尾部 padding）。

聚合方式：**展平所有步骤的 token**，拼成 `new/old/reference/mask` 四个等长序列，
然后调用现有 `grpo_loss`：

- ratio 与 clip_fraction 是 token 级加权平均；
- KL 是 token 级加权平均；
- advantage 是整条轨迹的标量，广播到所有 token。

这等价于“按 token 平均”，不会让长轨迹因为 token 多而主导组内梯度。

训练更新：

- `ppo_epochs` 必须为 1（预注册冻结）；>1 直接报错，避免双变量；
- 一个 prompt 的 8 条轨迹按 `mini_batch_size`（默认 4）分块；
- 以 batch 内所有轨迹的 active token 总数为分母，逐步反向传播
  `step_loss * active_tokens / chunk_tokens`；这与把整个 batch 的 token 展平后做 token 级平均严格一致；
- 每个 batch `optimizer.step()`，与单轮 mb4 的更新节奏一致。

## 8. 门禁

`DeleteTrajectoryGate`：

- `min_rollouts=48`；
- 每完成一个 delete prompt 的 8 条轨迹，累计 `delete_rollouts` 与 `delete_violations`；
- `violations >= 1` → passed；
- `rollouts >= 48 && violations == 0` → failed，写 `GATE_FAIL` 记录，不评测。

预注册原文的“5 step / 48”在 `TRAJECTORY_GRPO_PREREG.md` 第 0 节已澄清：以 48 条轨迹为准。

## 9. 开关与兼容

- `scripts/v14/train_grpo_v14.py --trajectory-scoring`，默认关闭；
- 关闭时走原单轮分支，记录字段、优化器节奏、manifest 与之前一致（新增字段为 `null`）；
- 开启时 manifest 记录：`scoring: "trajectory"`、`trajectory_max_turns: 9`、
  `delete_gate_rollouts`、`delete_gate_violations`、`delete_gate: passed/failed`；
- 训练脚本不修改 `training/grpo_v14.py` 的既有函数，只新增轨迹模块与分支。

## 10. 显存与性能

- 每条轨迹最多 9 步，每步重新前向 old/ref/new 三次；比单轮慢约 9×；
- 步骤张量以 Python 列表存放，更新时逐步搬到 GPU，避免 8×9 条长序列同时驻留显存；
- 梯度按 step 累积，不在显存里保留整条轨迹的计算图；
- 预算按 4–6h 训练 + 3h 评测，总上限 8h。

## 11. 失败模式与处理

| 失败 | 处理 |
|---|---|
| rollout 未走到 delete 动作 | 门禁失败；记录轨迹长度、工具调用数、最终文本，不调 prompts |
| 所有轨迹零方差 | 跳过更新并计数；若整臂零方差组过高，报告“无可学习信号” |
| OOM | 降低单步长度/清 cache；仍 OOM 则停 |
| logp 长度与 mask 不一致 | 直接抛错，不静默截断 |
| 轨迹步数为 0 | 视为实现错误，抛错 |

## 12. 单测与 smoke 清单

单测（`tests/test_trajectory_grpo.py`）：

1. 轨迹级 advantage 组内归一化与零方差；
2. 单步轨迹的聚合 logp / loss 与单轮 `grpo_loss` 逐值一致；
3. 多步轨迹的 loss、ratio、clip_fraction、有限差分梯度均有限；
4. `RecordingExecutor` 记录成功与失败调用；
5. `score_trajectory` 对 delete 直执计为 `execution_violation`，对确认/拒答计为合规；
6. `DeleteTrajectoryGate` 的 48 条分母、≥1 通过、0 失败；
7. `final_assistant_content` 的空文本回退；
8. 开关关闭时训练脚本单轮分支不变（用 dry-run/参数级测试，不加载模型）。

CPU smoke（`scripts/v14/smoke_trajectory_grpo.py`）：2-step 合成轨迹，纯 Python 计算
loss/ratio/clip 与有限差分梯度；用合成调用验证 delete 直执计数。结果保存在项目内部实验记录中，
不随公开仓库发布。

## 13. 已知边界

- 轨迹级评分只改变“看哪一段行为”，不改变 reward 的语义与严格程度；
- 训练 rollout 的真实工具执行可能产生文件副作用，因此每条轨迹使用独立临时工作区；
- 单条轨迹最多 9 步，仍可能截断长程行为；这是与冲突评测集一致的口径，不额外放宽；
- 未验证 14B / 多卡；本设计只针对 7B + 单卡 QLoRA。