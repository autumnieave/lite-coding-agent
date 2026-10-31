# 阶段 A：10 次 rollout 复核结论

## 口径

- 主样本：`constraint_experiment` / `stress` / `on` / run 2、run 10，各 10 次 rollout。
- 负对照：`benchmark` / `standard` / `off` / C1、C2、C3 的 run 1、2、3，各 10 次 rollout。
- 单独记录、不进入约束违反归因：12 任务 × 3 次 benchmark 中的步数超预算失败。
- 稳定性阈值预先固定：10 次中命中 `>=6` 为 `stable`，`1-5` 为 `intermittent`，
  `0` 次不进入违反根因表，只在命中率表标为 `not_observed`。
- 每条违反以 trace 中当轮实际发送的 messages、完整 response、工具调用和约束位置为准，
  不以 `reply_head` 或 `constraints.json` 是否存在单独判断。

采样配置固定为 `temperature=0.0`、`top_p=None`、`seed=20261019`。

## 命中率与稳定性

| 样本 | run 数 | rollout/run | 内容全过 run | 违反检查项 |
| --- | ---: | ---: | ---: | ---: |
| 主样本 stress on run 2 | 1 | 10 | 9/10 | 2 |
| 主样本 stress on run 10 | 1 | 10 | 9/10 | 2 |
| 负对照 C off C1/C2/C3 | 9 | 10 | 68/90 | 16 |

主样本的 4 条检查全部为 `intermittent`：run 2 和 run 10 各有一条 `json_valid`、
一条 `snake_case_keys`，命中率均为 `1/10`。负对照的 16 条检查也全部为
`intermittent`，没有达到 `>=6/10` 的 `stable` 门槛。

这里的 4 和 16 是「run×检查项」聚合后的唯一违反检查项数；若按
「rollout×检查项」计数，主样本为 4 次，负对照为 32 次。两种计数在
`违反根因分布表.csv` 与 `检查项命中率表.csv` 中分别保留。

完整逐检查项命中率见 `docs/evidence/检查项命中率表.csv`，根因行见
`docs/evidence/违反根因分布表.csv`，翻转与稳定性生成报告见
`docs/evidence/翻转率报告.md`。

## 人工复核

`private/traces/constraint_experiment/stress/on/stress_run02_rollout02.jsonl#event=44`
已完整人工阅读。日志前缀组六条和时间戳组六条互斥约束都出现在当轮实际发送的
`message[1]`（`role=system`、`[历史对话摘要]`）中，模型在
`unresolved_conflicts` 中也明确指出两组约束无法同时满足。结论、原文、位置和模型原话见
`docs/evidence/冲突复核.md`。

冲突属实，但 `event=44` 在 10 次 rollout 中只命中 1 次，属于 `intermittent`；
受控冲突实验里强冲突组 5/5 识别出 `unresolved_conflict`，却 0/5 产生格式失败。
因此 run 2 的 `json_valid` 与 `snake_case_keys` 两行仍保持 `sampling_noise`，不改类为
「能力问题（多约束冲突）」。CSV 备注保留冲突属实、直接失败原因是 JSON 尾部追加说明、
受控实验未复现稳定格式失败三项证据，详见 `docs/evidence/冲突专项.md`。

## 响应侧 metadata

0-2 次 rollout 生成于 metadata 记录扩展之前，因此没有 `model_version` /
`system_fingerprint` / `response_id`；3-9 次 rollout 以及方向 C 的 20 个 trace
已记录这些字段：

- `model_version` 均为 `deepseek-flash`。
- `system_fingerprint` 均为 `e72d3c7f663d3e1383b00b3884cb4501`。
- `response_id` 每次调用唯一。
- API 未返回可用的 `request_id`，报告不伪造该字段。

同一 fingerprint 下 temperature=0 仍有翻转，说明当前证据不足以把非确定性归因到
模型版本或服务端 fingerprint 变化；更细的硬件、批处理和采样实现原因需要服务端信息。

## 阶段判断

- 主样本没有 `stable` 行为违反，按预注册门槛不足以支撑阶段 B 的工程化加固。
- 方向 C 扩到 10 次/run 后：手工组检查级违反 `0/40`，基线 `4/40`，满足预注册门槛；
  但 Fisher 精确检验 `p=0.1156`（检查级）/ `0.4872`（rollout 级），只能写
  “描述性有信号、样本量不足以证明收益”，因此不工程化。
- 当前项目定位为「约束违反诊断 + 可复现评测方法论」；训练管线属于工程储备，constrained
  decoding 首次任务因路径错误失败，均不反推主实验结论。

## 顺序偏差说明

本轮先完成了 10 次 rollout 扩采样，再补做方向 C 手工验证，顺序与最初“先手工验证、
再扩采样”的设想相反。影响与处理如下：

- 手工 pilot 只有 6 次，单独看统计功效不足，容易把采样翻转误读成 prompt 收益；
- 方向 C 复用了完整 10 次/run 基线做对照，比只用 3 次/run 更稳；
- 随后按预先锁定的门槛把手工组扩到 10 次/run（共 20 次），用扩采样解决功效问题；
- 门槛、指标和采样配置在扩采样前写入 `docs/evidence/方向C扩采样门槛.md`，没有事后调整。
