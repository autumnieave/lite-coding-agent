# SFT 前后会话约束合规对比（train-v6，方向性证据）

**日期**：2026-10-28  
**范围**：8 个中间态任务 × 2 类会话约束 × 2 种注入方式 × 2 个条件 = 64 条观测；
其中 baseline 32 条、SFT 32 条，按 `task_id + mode + constraint_id` 一一配对。  
**定位**：方向性证据，不做 McNemar / Bootstrap，不写“统计显著”。  
**一句话结论**：同一批探针下，SFT 合规从 `12/32` 升至 `28/32`，
违反从 `17` 降至 `0`；这是行为层的大幅方向性改善，不是泛化结论。

## 0. 先修正旧口径

旧报告中的 `23/64` 不是 baseline 单组合规数。`会话约束三处查.csv` 的
baseline 单组为 `12/32` 合规，SFT（train-v5）为 `11/32` 合规；
两者合计 `23/64`，旧报告进一步以语义保留的 `59` 条为分母，写成 `23/59`。  
三组数字的实际口径如下：

| 组 | adapter | 合规 | 违反 | 只读/未分类 | 合计 |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | 禁用 adapter 的 Qwen2.5-7B | 12 | 17 | 3 | 32 |
| SFT（旧） | train-v5 `f5b685d4...` | 11 | 1 | 20 | 32 |
| SFT（新） | train-v6 `37928a28...` | 28 | 0 | 4 | 32 |

`23/64 = 12/32 + 11/32` 是 baseline 与 train-v5 SFT 合并的旧口径，已废弃；
不能和新 train-v6 的 `28/32` 混写。两个 SFT 是同一批探针、同一判定规则下的
两次 adapter 实验，不是同一个 adapter。

本次新跑仍为 32 对配对，但 baseline 与 SFT 都使用 train-v6 adapter：
baseline 加载 adapter 后调用 `disable_adapter()`（基座模型），SFT 启用 adapter。

## 1. 模型与复现条件

| 项目 | 配置 |
| --- | --- |
| 基座 | `Qwen/Qwen2.5-7B-Instruct`，4-bit NF4 |
| baseline | 加载 train-v6 adapter 后 `disable_adapter()`，即基座模型 |
| SFT | 启用 train-v6 assistant-only adapter |
| adapter SHA-256 | `37928a28d67a5a8350f3b49450fc6945c3ffe05c09da7ea0de36136b4a6a4b96` |
| adapter 校验 | 645,975,704 bytes；392 个 LoRA tensor 可读 |
| 任务 | 8 个中间态任务，tasks SHA-256 `f0a8067a...40716` |
| 运行环境 | Kaggle Tesla T4，greedy decoding，seed `20261026` |
| 原始数据 | `private/kaggle_pull_session_constraint_v6/session_constraint/{baseline,sft}.jsonl` |

探针代码、压缩器、注入方式、判定规则均未改动，仅替换 adapter。每个条件有
`8 任务 × 2 注入方式 × 2 约束类型 = 32` 条约束探针；两条件合计 64 条观测。

## 2. 总体行为合规

“只读/未分类”包含两类：baseline 的 3 条 `ambiguous` 仅给出文字计划、
没有 tool call；SFT 的 4 条 `deferred_tool` 只调用 `read_file`，未执行写入。
它们不等于违反，也不计入严格合规。

| 指标 | baseline | SFT | 变化 |
| --- | ---: | ---: | ---: |
| 合规 | 12/32（37.5%） | 28/32（87.5%） | +16 条，+50.0pp |
| 违反 | 17/32（53.1%） | 0/32（0.0%） | -17 条，-53.1pp |
| 只读/未分类 | 3/32（9.4%） | 4/32（12.5%） | +1 条，+3.1pp |
| 语义保留 | 32/32（100%） | 31/32（96.9%） | -1 条 |

## 3. 按约束类型拆分

| 类型 | 条件 | 合规 | 违反 | 只读/未分类 |
| --- | --- | ---: | ---: | ---: |
| 行为规则 | baseline | 12/16（75.0%） | 4/16（25.0%） | 0/16（0.0%） |
| 行为规则 | SFT | 15/16（93.8%） | 0/16（0.0%） | 1/16（6.3%） |
| 信息约束 | baseline | 0/16（0.0%） | 13/16（81.3%） | 3/16（18.8%） |
| 信息约束 | SFT | 13/16（81.3%） | 0/16（0.0%） | 3/16（18.8%） |

信息约束是主要变化来源：baseline 的 16 条中 13 条直接写入了另一个文件，
SFT 后 13 条合规、0 条违反，剩余 3 条只读取了文件但未执行写入。

## 4. 配对转移

以 baseline 的 32 条为基准，对应到同一 `task_id + mode + constraint_id` 的 SFT 结果：

| baseline → SFT | 条数 |
| --- | ---: |
| 合规 → 合规 | 11 |
| 违反 → 合规 | 15 |
| 违反 → 只读/未分类 | 2 |
| 未分类 → 合规 | 2 |
| 未分类 → 只读/未分类 | 1 |
| 合规 → 只读/未分类 | 1 |
| **合计** | **32** |

没有出现“合规 → 违反”。SFT 后唯一的语义丢失是 `PUB-XLAM-00071` untagged
行为约束 `BVQ4HP`：正文丢失，探针只调用了 `read_file`，因此记为只读/未分类，
不是违反。

## 5. 判定

**SFT 部分解决行为层约束遵守：合规 `12/32 → 28/32`（+50.0pp），
违反 `17 → 0`，仍有 4 条只读/未分类。**  
这支持“SFT 能改善约束遵守行为”，但不支持“SFT 普遍有效”：本轮任务仍来自
训练重叠的中间态池，且只有 8 个任务、32 对配对，属于方向性证据，不能外推到
未见任务族或 clean held-out。

原始逐条记录见 `private/kaggle_pull_session_constraint_v6/session_constraint/`；
独立重算结果见同目录 `analysis_recomputed.json`。
