# API vs 本地 7B：45 个受限任务 A/B 对照

> 目的：在同一套 45 个冻结任务、同一 fixture、同一 6 工具 schema、同一确定性 judge、同一 `temperature=0.7/top_p=0.9` 下，比较 DashScope API baseline 与本地 Qwen2.5-7B baseline/SFT。

## 1. 前置检查

- 任务口径：官方 45 任务，SHA-256 `20e20b3beabaf0c48c00dae56d97c1aea356b1ac89c4cdb0fd9decc799c4fa5b`；不使用重判 35 口径。
- Judge：`scripts/unified_eval.py` 的同一 `build_fixture/judge/run_once`，S0。
- base_url：`https://dashscope.aliyuncs.com/compatible-mode/v1`。
- 采样：`temperature=0.7`、`top_p=0.9`；未向 API 发送 seed，存在 API 侧额外方差。
- 模型现场探测（2026-10-26）：`deepseek-v3` 精确别名返回 **404 model_not_found**；`vanchin/deepseek-v3` 返回 **400 product_not_activated**；`deepseek-v4.1-flash` 与 `qwen-max` 均返回 200。
- 选择 A 组 `deepseek-v4.1-flash`：它是当前可用的同系列教师 API，端到端预检返回原生 `tool_calls`，不是 fenced JSON；`qwen-max` 也通过预检但未作为主组。
- 单任务预检：`PUB-XLAM-00001`，1 rollout，task_success=1.0, API 请求 2 次，原生 tool-call 请求 1 次，fenced JSON 0 次，token=3578。
- 全量协议审计：1052 次 API 请求，532 次返回原生 tool-call，40 个最终答案使用 fenced JSON；error_rows=0，within_steps=100.0%，tool_selection=100.0%。

## 2. 三组结果

| 组 | 模型 | valid rollout | task_success（等权） | pooled | stable 任务 | within_steps | tool_selection |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A API baseline | `deepseek-v4.1-flash` | 450 | 90.9% | 90.9% | 44/45 | 100.0% | 100.0% |
| B 本地 baseline | Qwen2.5-7B 4-bit | 360 | 49.4% | 49.4% | N/A（8–9 rollout） | 95.3% | 99.2% |
| C 本地 SFT | Qwen2.5-7B + QLoRA | 450 | 99.8% | 99.8% | 45/45 | 100.0% | 100.0% |

口径：等权均值是 45 个任务通过率的平均；pooled 是全部有效 rollout 合并；stable 为 10 次中≥6 次通过。

## 3. API vs 本地 SFT 配对统计

- 配对单位：同一 `task_id + rollout_index`，共 450 对。
- 本地 SFT − API：+8.9pp；paired task-cluster bootstrap 95% CI [+4.7, +13.3]pp，1000 次。
- 精确 McNemar：API 失败→SFT 成功 41 对；API 成功→SFT 失败 1 对；p=1.955e-11，Holm p=5.866e-11。
- 任务级 stable 敏感性：SFT-only stable 1 个，API-only stable 0 个，p=1，Holm p=1。

| 指标 | API（等权） | 本地 SFT（等权） | 本地−API | 95% CI | p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `task_success` | 90.9% | 99.8% | +8.9pp | [+4.7, +13.3]pp | 1.955e-11 | 5.866e-11 |
| `within_steps` | 100.0% | 100.0% | +0.0pp | [+0.0, +0.0]pp | 1 | 1 |
| `tool_selection_ok` | 100.0% | 100.0% | +0.0pp | [+0.0, +0.0]pp | 1 | 1 |

敏感性：
- API vs 本地 baseline（共同 rollout）：360 对，baseline 失败→API 成功 169，baseline 成功→API 失败 18，p=5.977e-32。
- 本地 SFT vs 本地 baseline（共同 rollout）：360 对，baseline 失败→SFT 成功 181，baseline 成功→SFT 失败 0，p=6.525e-55。

## 4. 输出契约归因（post-hoc，非确认性）

- 官方严格 judge 下：API 409/450 成功，本地 SFT 449/450 成功。
- API 的 41 条失败中，40 条是最终答案被 fenced JSON 包裹；其余 1 条为非格式内容错误。
- 40/40 条 fenced 回复可直接抽出合法 JSON；在临时 fixture 中重放已执行工具调用后，40/40 通过同一 judge。
- 去 fence 后 API 内容级成功 449/450（等权 99.8%），SFT 449/450（等权 99.8%）；post-hoc 配对 McNemar p=1。
- 因此官方 +8.9pp 差距应解释为“严格固定输出契约下的可靠性差异”，不能解释为本地模型任务推理更强。若 API 侧做确定的 fence 剥离或结构化输出约束，本任务集上的内容级结果与本地 SFT 相同。

## 5. 延迟与成本

- API：单请求中位数 1.97s，P95 4.68s；单 rollout 中位数 4.60s。含网络与工具执行。
- 本地 SFT：单 rollout 中位数 10.50s；450 次推理约 1.63 GPU·h（Kaggle T4）。
- API token：prompt 1,824,329，completion 104,603，total 1,928,932。
- API 费用：API 响应不返回账单金额；需按上表 token 用量 × 控制台账单单价计算。
- 本地成本：Kaggle 免费 T4 增量账单为 0；租卡口径未给定，不伪造金额。

## 6. 结论

严格固定输出契约下，本地 7B SFT 在 45 个受限任务上显著超过 API baseline（效应量 +8.9pp，Holm p=5.866e-11，95% CI [+4.7, +13.3]pp）。失败归因显示 40/41 次 API 失败是最终答案被 fenced JSON 包裹；离线去除 fence 后 40/40 通过同一 judge，内容级 API≈99.8% vs SFT≈99.8%（post-hoc McNemar p=1）。因此这是输出契约可靠性的优势，不能写成任务内容推理优势。

## 7. 边界

- 45 个任务是诊断集，不是跨模板 held-out；clean-heldout 报告已确认 45/45 仍有同族训练任务。
- 结论只覆盖受限 6 工具、固定输出契约的子智能体场景，不外推到开放域 coding。
- 本地 baseline 为 8–9 rollout，B 组只作三组描述与共同 index 敏感性，不与 A/C 的 450 对混算。
- API model id 与行为可能漂移；本报告固定记录 2026-10-26 的 model/version。
- 未发送 seed，API 侧存在额外采样方差；未对语义失败做重试或挑样本。
- 输出契约归因是读取既有回复的事后诊断，不是新的 API rollout；官方通过率仍以严格 judge 为准，反事实数字不进入确认性统计。

## 8. 逐任务结果

| 任务 | family | difficulty | API | 本地 baseline | 本地 SFT | SFT−API |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `PUB-XLAM-00001` | `public_grep` | medium | 9/10 | 4/8 | 10/10 | +10.0pp |
| `PUB-XLAM-00002` | `public_grep` | weak | 10/10 | 5/8 | 10/10 | +0.0pp |
| `PUB-XLAM-00003` | `public_grep` | medium | 9/10 | 3/8 | 10/10 | +10.0pp |
| `PUB-XLAM-00019` | `public_grep` | medium | 10/10 | 4/8 | 10/10 | +0.0pp |
| `PUB-XLAM-00023` | `public_grep` | medium | 9/10 | 5/8 | 10/10 | +10.0pp |
| `PUB-XLAM-00028` | `public_grep` | weak | 10/10 | 5/8 | 10/10 | +0.0pp |
| `PUB-XLAM-00031` | `public_grep` | medium | 7/10 | 3/8 | 10/10 | +30.0pp |
| `PUB-XLAM-00039` | `public_grep` | medium | 10/10 | 3/8 | 10/10 | +0.0pp |
| `PUB-XLAM-00042` | `public_grep` | weak | 9/10 | 5/8 | 10/10 | +10.0pp |
| `PUB-XLAM-00046` | `public_grep` | medium | 7/10 | 5/8 | 10/10 | +30.0pp |
| `PUB-XLAM-00051` | `public_grep` | weak | 10/10 | 4/8 | 9/10 | -10.0pp |
| `PUB-XLAM-00052` | `public_grep` | weak | 10/10 | 5/8 | 10/10 | +0.0pp |
| `PUB-XLAM-00058` | `public_grep` | medium | 8/10 | 5/8 | 10/10 | +20.0pp |
| `PUB-XLAM-00071` | `public_grep` | weak | 10/10 | 5/8 | 10/10 | +0.0pp |
| `PUB-XLAM-00076` | `public_grep` | weak | 10/10 | 3/8 | 10/10 | +0.0pp |
| `PUB-XLAM-00077` | `public_grep` | medium | 10/10 | 4/8 | 10/10 | +0.0pp |
| `PUB-XLAM-00078` | `public_grep` | medium | 7/10 | 3/8 | 10/10 | +30.0pp |
| `PUB-XLAM-00079` | `public_grep` | medium | 10/10 | 3/8 | 10/10 | +0.0pp |
| `PUB-XLAM-00087` | `public_grep` | medium | 9/10 | 4/8 | 10/10 | +10.0pp |
| `PUB-XLAM-00098` | `public_grep` | weak | 9/10 | 5/8 | 10/10 | +10.0pp |
| `PUB-XLAM-00102` | `public_grep` | medium | 8/10 | 4/8 | 10/10 | +20.0pp |
| `U-TR-A-C1-BASE` | `json_constraints` | weak | 10/10 | 3/8 | 10/10 | +0.0pp |
| `U-TR-A-TOOL-HARD` | `path_params` | strong | 5/10 | 3/8 | 10/10 | +50.0pp |
| `U-TR-B-C1-HARD` | `json_constraints` | strong | 10/10 | 3/8 | 10/10 | +0.0pp |
| `U-TR-B-TOOL-BASE` | `path_params` | weak | 10/10 | 3/8 | 10/10 | +0.0pp |
| `U-TR4-00-C1-HARD` | `json_constraints` | strong | 10/10 | 5/8 | 10/10 | +0.0pp |
| `U-TR4-00-CHECK-BASE` | `03` | medium | 8/10 | 5/8 | 10/10 | +20.0pp |
| `U-TR4-02-C1-BASE` | `json_constraints` | weak | 10/10 | 3/8 | 10/10 | +0.0pp |
| `U-TR4-03-C1-BASE` | `json_constraints` | weak | 10/10 | 3/8 | 10/10 | +0.0pp |
| `U-TR4-04-C1-BASE` | `json_constraints` | weak | 9/10 | 4/8 | 10/10 | +10.0pp |
| `U-TR4-05-C1-BASE` | `json_constraints` | weak | 10/10 | 3/8 | 10/10 | +0.0pp |
| `U-TR4-06-C1-BASE` | `json_constraints` | weak | 10/10 | 4/8 | 10/10 | +0.0pp |
| `U-TR4-07-C1-BASE` | `json_constraints` | weak | 10/10 | 3/8 | 10/10 | +0.0pp |
| `U-TR4-08-C1-BASE` | `json_constraints` | weak | 10/10 | 4/8 | 10/10 | +0.0pp |
| `U-TR4-16-CHECK-HARD` | `03` | medium | 6/10 | 5/8 | 10/10 | +40.0pp |
| `U-TR4-17-CHECK-HARD` | `03` | medium | 9/10 | 5/8 | 10/10 | +10.0pp |
| `U-TR4-19-CHECK-HARD` | `03` | medium | 10/10 | 5/8 | 10/10 | +0.0pp |
| `U-TR5-00-C1-BASE` | `json_constraints` | weak | 10/10 | 4/8 | 10/10 | +0.0pp |
| `U-TR5-05-CHECK-HARD` | `03` | strong | 6/10 | 5/8 | 10/10 | +40.0pp |
| `U-TR5-10-C1-BASE` | `json_constraints` | weak | 9/10 | 3/8 | 10/10 | +10.0pp |
| `U-TR5-15-C1-BASE` | `json_constraints` | weak | 10/10 | 4/8 | 10/10 | +0.0pp |
| `U-TR5-18-FILE-BASE` | `01` | weak | 10/10 | 3/8 | 10/10 | +0.0pp |
| `U-TR5-21-CHECK-HARD` | `03` | strong | 8/10 | 3/8 | 10/10 | +20.0pp |
| `U-TR5-23-CHECK-HARD` | `03` | strong | 8/10 | 5/8 | 10/10 | +20.0pp |
| `U-TR5-27-FILE-BASE` | `01` | weak | 10/10 | 3/8 | 10/10 | +0.0pp |

## 9. 可追溯性

- API runs SHA-256：`80268b058e258f0af87f5be0be863cc28f1e58086aa7b3313a34f46fd3a9572c`
- API summary SHA-256：`b2fe168dfa03b4ebfa82336fad4b092e10747c47c98cc434d322cbf952a022f1`
- 本地 SFT runs SHA-256：`81aad73c6bf35cb54bf6699f5d5893774b051c5655cf19884eafacfaf2aa7517`
- 本地 SFT adapter SHA-256：`37928a28d67a5a8350f3b49450fc6945c3ffe05c09da7ea0de36136b4a6a4b96`
- 任务 manifest SHA-256：`20e20b3beabaf0c48c00dae56d97c1aea356b1ac89c4cdb0fd9decc799c4fa5b`
- 原始 JSONL 与 trace 位于 `private/`，不进入 Git。

