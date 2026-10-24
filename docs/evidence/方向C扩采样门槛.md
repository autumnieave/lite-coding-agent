# 方向 C 扩采样门槛（先锁定，后执行）

## 锁定时间与范围

- 日期：2026-10-19。
- 范围：stress on run 2/10，各 10 次基线 + 10 次手工自检变体，共 40 次 rollout。
- 采样配置保持 `temperature=0.0`、`top_p=None`、`seed=20261019`；不为了凑 stable 改配置。
- 已有 6 次手工 pilot 和 20 次固定配置基线按同一口径并入，不调门槛、不删失败样本。

## 指标定义

主指标只看两项输出格式检查：

`violations = json_valid 失败数 + snake_case_keys 失败数`

按 run 聚合时，基线每 run 的 `violations / 10` 为：

| run | json_valid | snake_case_keys | 合计 |
| ---: | ---: | ---: | ---: |
| 2 | 1/10 | 1/10 | 2/10 |
| 10 | 1/10 | 1/10 | 2/10 |
| 合并 | 2/20 | 2/20 | 4/20 |

## 预先锁定的有收益条件

手工组必须同时满足：

1. 每个 run 的检查级违反数 `<=1/10`（不超过基线 2/10 的一半）；
2. 两个 run 合并的检查级违反数 `<=2/20`（不超过基线 4/20 的一半）。

只有同时满足才写“有收益信号”。否则写“未证明收益”，不工程化。

## 统计检验

- 若手工组达到 `0/20` 或接近门槛，使用 Fisher 精确检验报告 p 值；
- 主检验按“检查级违反事件 vs 非违反事件”的 2×2 表报告，同时补充“rollout 任一格式检查失败”的保守二元表；
- `p >= 0.05` 时写“样本量不足以证明收益”，不能把描述性下降写成显著结论。

## 执行记录

- 基线 20 次来自既有固定配置 trace；
- 已有手工 pilot 6 次来自 `private/traces/manual_self_check/`；
- 本次补跑手工 rollout 3-9，每个 run 7 次，共 14 次；
- 所有新 trace 仍落 `private/traces/manual_self_check/`，结果记录追加到
  `private/traces/manual_self_check_runs.jsonl`。
