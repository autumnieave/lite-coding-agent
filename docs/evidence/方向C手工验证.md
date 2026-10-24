# 方向 C 扩采样：输出前约束自检

## 目的与边界

- 目的：在工程化之前，只验证「输出前自检」prompt 能否降低 `json_valid` 与
  `snake_case_keys` 的违反率。
- 不改 `core/loop.py`、`core/compaction.py`、`core/constraints.py`，不进入阶段 B/C，
  不写 SFT/GRPO 代码。
- 复用 `scripts/constraint_experiment.py` 的 `run_once`、约束夹具、判定逻辑与 trace 结构；
  只在 `private/direction_c_expand.py` 中 monkeypatch `PROBE_JSON`。
- trace 落 `private/traces/manual_self_check/`，逐次记录在
  `private/traces/manual_self_check_runs.jsonl`，不提交大文件。

## 固定采样配置

| 参数 | 值 |
| --- | --- |
| temperature | `0.0` |
| top_p | `None` |
| seed | `20261019` |
| 夹具 seed | run 2 = `20262914`；run 10 = `20270914` |
| run | stress / on / run 2、run 10 |
| rollout | 每个 run 10 次，共 20 次手工组；基线每个 run 10 次，共 20 次 |
| 判定 | 原 `judge()`；不按人工观感改判 |

## 手工 prompt 变体

原 prompt：

> 输出一个 JSON 对象总结本次会话，键名遵循你被要求遵守的约束。

手工变体：

> 输出一个 JSON 对象总结本次会话，键名遵循你被要求遵守的约束。输出前先自检：
> 整个回复必须恰好是一个 JSON 对象，键名全部 snake_case，禁止在 JSON 前后追加任何说明；
> 自检后只输出 JSON。

已核对 trace：`probe_json` 轮次的最后一条 user 内容确实是上述变体；每轮
`sampling_config` 记录 `model_version=deepseek-flash`、
`system_fingerprint=e72d3c7f663d3e1383b00b3884cb4501` 和唯一 `response_id`。

## 结果

### 基线（正式 constraint_experiment，每个 run 10 次）

| run | json_valid 失败 | snake_case_keys 失败 | no_code_fence 失败 | 任一格式失败 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1/10 | 1/10 | 0/10 | 1/10 |
| 10 | 1/10 | 1/10 | 0/10 | 1/10 |
| 合计 | 2/20 | 2/20 | 0/20 | 2/20 |

按检查项计，基线违反事件为 `2 + 2 = 4`，检查事件总数为 `2 项 × 20 rollout = 40`。

### 手工自检组（每个 run 10 次）

| run | json_valid 失败 | snake_case_keys 失败 | no_code_fence 失败 | 任一格式失败 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0/10 | 0/10 | 0/10 | 0/10 |
| 10 | 0/10 | 0/10 | 0/10 | 0/10 |
| 合计 | 0/20 | 0/20 | 0/20 | 0/20 |

完整逐 run × 检查项命中率见 `docs/evidence/方向C命中率对比.csv`。

### 预先锁定的门槛与执行记录

门槛写在跑之前：`docs/evidence/方向C扩采样门槛.md`。

| 条件 | 要求 | 实测 | 是否满足 |
| --- | --- | --- | --- |
| run 2 检查级违反 | `<=1/10` | `0/10` | 是 |
| run 10 检查级违反 | `<=1/10` | `0/10` | 是 |
| 合并检查级违反 | `<=2/20` | `0/20` | 是 |

因此按预注册口径，手工组**达到“有收益信号”的门槛**：违反率从基线 `4/40`
降到 `0/40`，每个 run 和合并口径都满足。

## 统计检验

为判断这个下降是否只是小样本波动，补做 Fisher 精确检验（双尾）。

### 主表：检查级违反事件

| 组 | 违反 | 非违反 |
| --- | ---: | ---: |
| 基线 | 4 | 36 |
| 手工 | 0 | 40 |

Fisher 精确检验 `p = 0.1156`。

### 保守表：rollout 任一格式检查失败

| 组 | 失败 rollout | 通过 rollout |
| --- | ---: | ---: |
| 基线 | 2 | 18 |
| 手工 | 0 | 20 |

Fisher 精确检验 `p = 0.4872`。

**复核更正（2026-10-19）**：此前的保守表把基线误写成 `4/16`，与上面“任一格式失败
2/20”的逐 run 表矛盾。回读 `private/traces/constraint_experiment/stress/on/` 后确认：
真正失败的是 run 2 rollout 02 与 run 10 rollout 00 两次，均为 `json_valid` 与
`snake_case_keys` 同时失败；因此 rollout 级应为 `2/20`，Fisher 重算为 `p=0.4872`。
检查级 `4/40` 与 `p=0.1156` 不变，结论仍是不显著、不工程化。

两个口径都 `p >= 0.05`。因此只能写：

> 手工自检有描述性收益信号，但当前样本量不足以证明收益。

不能把 `0/20` 写成显著改善，也不能据此进入阶段 B/C 工程化。

## 结论

1. 方向 C 在当前样本上有**初步收益信号**：`json_valid` / `snake_case_keys` 从基线
   各 `2/20` 降到 `0/20`，满足预先锁定的门槛。
2. 统计功效不足：检查级 `p=0.1156`，rollout 级 `p=0.4872`，不能宣称显著。
3. 按用户口径，本次**不工程化方向 C**，不进入阶段 B/C；只保留“若后续扩到更大样本，
   可用同一门槛复验”的方案。
4. 20 次手工 trace 均记录 `model_version`、`system_fingerprint`、`response_id`；
   基线 0-2 次 rollout 生成于 metadata 扩展之前，缺这三个字段，已在阶段 A 报告中标注，
   不追溯补造。

## 若后续批准的工程化前置方案（本次未实施）

1. 只在实验脚本层增加自检 prompt 开关，不改 `core`；默认关闭，保持旧 payload 与旧行为。
2. 把每个 run 扩到更大样本（例如 30-50 次），仍固定 temperature/top_p/seed，记录
   `prompt_variant` 与 `rollout_index`。
3. 预先锁定同样的门槛与 Fisher 检验口径，达标才实现；若自检把违反压到 0-1 次，
   停止，不进入阶段 C。

## 顺序偏差说明

本轮先完成 10 次 rollout 扩采样，再补做方向 C 手工验证，顺序与最初设想相反。
好处是基线直接复用完整 10 次/run，对照更稳；代价是手工 pilot 只有 6 次时统计功效不足。
该问题已通过把手工组扩到 10 次/run 解决，且门槛在扩采样前锁定，没有事后调整阈值。
