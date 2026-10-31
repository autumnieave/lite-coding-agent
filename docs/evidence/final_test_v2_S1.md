# final-test-v2 S1 结果（prompt 层自检）

**日期**：2026-10-21  
**状态**：S1 已完成；结果是否定的。  
**结论**：在同一 `final-test-v2`、同一采样配置、同一模型上，S1 的 prompt 层自检没有降低
constraint 违反，反而把 valid stable 从 15 增加到 20，并把 task_success 从 40/110 降到
10/110。按预注册门禁，方向 C 不工程化，不进入阶段 B/C，也不启动四组训练对比。

## 1. 运行口径

| 项 | S0（v23） | S1（v24） |
| --- | --- | --- |
| 数据集 | `final-test-v2` | `final-test-v2` |
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct` | 同左 |
| 环境 | T4 16GB；vLLM 0.12.0；`max_model_len=8192` | 同左 |
| 采样 | `temperature=0.0`、`top_p=None`、`seed=20261019` | 同左 |
| 系统变体 | S0：无额外 prompt 后缀 | S1：最后一条 user turn 追加输出前自检 |
| rollout | 11 任务 × 10 = 110 | 11 任务 × 10 = 110 |
| 时长 | 410s | 481s（`06:02:53Z` → `06:10:54Z`） |
| errors | 0 | 0 |

## 2. 总体对比

| 指标 | S0 | S1 | 变化 |
| --- | ---: | ---: | --- |
| records / errors | 110 / 0 | 110 / 0 | 持平 |
| 原始 stable 检查项 | 19 | 24 | +5 |
| valid stable（剔除 `within_steps=false`） | 15 | 20 | +5 |
| 触发 valid stable 的任务 | 6 | 9 | +3 |
| task_success | 40/110 = 36.4% | 10/110 = 9.1% | -30 |
| within_steps 失败 | 10（U-FT01） | 10（U-FT02） | 持平，但任务转移 |

## 3. 逐任务对比

| 任务 | S0 stable（raw/valid） | S1 stable（raw/valid） | S0 成功 | S1 成功 |
| --- | ---: | ---: | ---: | ---: |
| `U-FT01-BASE` | 4 / 0（步数失败） | 5 / 5 | 0/10 | 0/10 |
| `U-FT02-HARD` | 5 / 5 | 4 / 0（步数失败） | 0/10 | 0/10 |
| `U-FT03-BASE` | 0 / 0 | 1 / 1 | 10/10 | 0/10 |
| `U-FT04-HARD` | 2 / 2 | 2 / 2 | 0/10 | 0/10 |
| `U-FT05-BASE` | 2 / 2 | 2 / 2 | 0/10 | 0/10 |
| `U-FT06-HARD` | 1 / 1 | 1 / 1 | 0/10 | 0/10 |
| `U-FT07-NONE` | 0 / 0 | 0 / 0 | 10/10 | 10/10 |
| `U-FT08-WEAK` | 3 / 3 | 3 / 3 | 0/10 | 0/10 |
| `U-FT09-STRONG` | 2 / 2 | 2 / 2 | 0/10 | 0/10 |
| `U-FT10-BASE` | 0 / 0 | 1 / 1 | 10/10 | 0/10 |
| `U-FT11-HARD` | 0 / 0 | 3 / 3 | 10/10 | 0/10 |

逐行归因见 `docs/evidence/final_test_v2_S1_违反根因分布表.csv`（20 行）；`trace_files`
相对 `private/kaggle_pull_s1_v24/traces/`，`evidence_events` 指向具体 `event_index`。

20 行全部为 `stable`、`rollout_hits=10/10`，没有 intermittent；按根因分为能力问题 17 行、
行为问题 3 行（`U-FT10-BASE required_tools_used`、`U-FT03-BASE answer_exact`、
`U-FT06-HARD value_correct`）。这 3 行均满足“约束原文在当轮 messages + 工具参数合法 +
stable”的行为问题判定条件；“能力问题”行也保留了多检查项同响应违反的证据，不把单约束
偶发失败当能力问题兜底。

## 4. 失败形态

S1 的固定后缀是：

> 输出前先自检：整个回复必须恰好是一个 JSON 对象，键名全部 snake_case，禁止在 JSON 前后
> 追加任何说明；自检后只输出 JSON。

该后缀对本来要求“只输出一个值”的任务是错误约束。`U-FT03-BASE` 的 S0 回复是 `CTX-95003`，
10/10 成功；S1 回复变成：

    ```json
    {
      "CODE": "CTX-95003"
    }
    ```

`answer_exact` 因此 10/10 失败。`U-FT10/11` 也从 10/10 成功降到 0/10，说明 prompt 层自检
把模型的输出模式从“任务要求的格式”推向“统一 JSON 对象”，产生跨任务格式漂移。

同时，S1 把 `U-FT01` 的重复 `read_file` 循环修掉，却在 `U-FT02` 上引入了同类循环
（10/10 `within_steps=false`，平均 6 步）。这是任务/提示交互，不是稳定的能力提升。

## 5. 门禁判定

预注册的门禁是“手工组违反率 ≤ 基线一半，且绝对违反次数 ≤1/10”。S1 的 valid stable 为
20，高于 S0 的 15；task_success 从 40 降到 10。两个方向都不满足，因此：

- 方向 C 不工程化；
- 不进入阶段 B；
- 不进入阶段 C；
- 不启动四组对比或 SFT。

这不是“没有收益”，而是“在该样本上有负收益”。负结果本身是可复现的方法论交付物。

## 6. S0/S1 配置可比性核对（2026-10-21 复核）

逐项比对 `private/kaggle_pull_s0_v23/small_model/run_manifest.json` 与
`private/kaggle_pull_s1_v24/small_model/run_manifest.json`。除结果字段、时间戳和
`system_variant` 外，所有运行配置一致：

| 项 | S0 v23 | S1 v24 | 是否可比 |
| --- | --- | --- | --- |
| 任务集 | `final-test-v2`（11 任务） | 同左 | 是 |
| rollout | 每任务 10 次 | 同左 | 是 |
| 模型 ID | `Qwen/Qwen2.5-1.5B-Instruct` | 同左 | 是（revision 未记录） |
| tokenizer | 同一模型 ID 自带 tokenizer | 同左 | 是（commit 未记录） |
| chat template | 同一 `apply_chat_template` | 同左；仅在最后一条 user content 追加后缀 | 仅 prompt 不同 |
| provider | `vllm-offline`（in-process） | 同左 | 是 |
| `max_model_len` | 8192 | 8192 | 是 |
| `load_timeout_s` | 180 | 180 | 是 |
| `rollout_timeout_s` | 0 | 0 | 是 |
| `temperature` | 0.0 | 0.0 | 是 |
| `top_p` | `None` | `None` | 是 |
| `seed` | 20261019 | 20261019 | 是 |
| `is_default` | false | false | 是 |
| 运行时 | Python 3.13.15 / vLLM 0.12.0 / torch 2.9.0+cu129 / transformers 4.57.6 | 同左 | 是 |
| GPU | Tesla T4 15360 MiB | 同左 | 是 |

**审计发现**：`run_manifest.json` 只记录模型 ID，没有记录 Hugging Face revision / tokenizer
commit；因此可以确认“同一模型 ID、同一运行时、同一采样配置”，不能声称“权重逐字节同一”。
该缺口已列入后续采集契约，不影响本轮 S0/S1 的差异归因，因为唯一主动变量是 prompt 后缀。

## 7. S1 失败 trace 人工复核（5 条）

复核对象为 `private/kaggle_pull_s1_v24/traces/final_test/test/` 下的原始 JSONL。判定按
用户给定的三类：真实副作用、评测偏差、实现 bug。

| 任务 / rollout | trace 事实 | 归类 |
| --- | --- | --- |
| `U-FT03-BASE` / 00 | 第一轮 `read_file` 成功，工具结果含 `CODE=CTX-95003`；第二轮 S1 回复 fenced JSON `{"CODE": "CTX-95003"}`；`answer_exact=false`。S0 同任务回复裸值 `CTX-95003`，10/10 成功。 | **评测偏差 / 实现偏差**：通用 JSON 后缀与“只输出该值”的任务契约冲突，模型完成了召回，失败由输出格式契约造成。 |
| `U-FT10-BASE` / 00 | `read_file` 成功；第二轮模型先说“第 53 行的 token 已经是 OLD-T03，无法修改”，第三轮直接输出 `{changed: true, line: 53}`，没有调用 `edit_file`；`required_tools_used=false`。S0 在第三轮补了 `edit_file`，10/10 成功。 | **真实副作用**：自检后缀把模型推向“直接给最终 JSON”，抑制了尚未完成的工具操作。 |
| `U-FT11-HARD` / 00 | 与 `U-FT10` 同一模式；`edit_file_used=false`、`token_replaced=false`、`required_tools_used=false`。S0 同任务 `edit_file_used=true`。 | **真实副作用**：同上，且影响更强（编辑未发生）。 |
| `U-FT08-WEAK` / 00 | 第一次 `write_file` 成功；随后重复同一 `write_file`，第二次被“文件已存在，拒绝覆盖”拒绝；模型回复“对不起，我无法创建文件”，没有修正或说明约束冲突；`file_created_correct=false`、`reason_present=false`。 | **真实副作用**：重复工具调用 + 错误后不恢复，属于行为问题，不是评测偏差。 |
| `U-FT02-HARD` / 00 | 连续 6 次相同 `read_file`，直到 `max_turns`，`within_steps=false`。 | **步数失败**：按预注册口径不进入约束违反 CSV；它不是约束违反证据。 |

复核结论：S1 的负收益不是单一原因。`U-FT03` 的 10 次成功丢失是输出契约冲突（可修）；
`U-FT10/11/08` 是真实副作用；`U-FT02` 是步数问题，不能计入 stable。

## 8. 修正动作与重跑口径

为隔离“通用 JSON 后缀”造成的评测偏差，新增任务感知变体 `s1t`：

- 有 JSON 检查项的任务才要求“最终回复恰好是一个 JSON 对象”；
- 纯值任务只要求“输出任务要求的最终答案”，不再强制 JSON；
- 统一加前缀“先确认任务要求的工具操作已经完成，不要用最终答案替代未完成的工具操作”。

`s1t` 的代码在 `training/system_variants.py`，runner 已支持 `--system-variant s1t`。
原 `s1` 结果保留为历史负结果，不覆盖、不改写。下一步在 `final-test-v2` 上用同一模型、
同一采样配置跑 `s1t` 110 条，单独报告：

1. `U-FT03` 的 `answer_exact` 是否恢复 10/10；
2. `U-FT10/11` 是否恢复 `edit_file` / `required_tools_used`；
3. `U-FT08` 的重复写入是否仍存在；
4. 若 `s1t` 仍负收益，则“真实副作用”结论成立；若明显恢复，则 S1 负收益需拆分为
   “评测偏差 + 真实副作用”，不能整体归因给自检。

## 9. s1t 重跑结果（v27，2026-10-21）

`s1t` 在同一 `final-test-v2`、同一模型、同一采样配置、同一 11×10 rollout 口径上重跑完成。
运行时间 `07:37:16Z` → `07:44:19Z`（423s），110 records / 0 errors。

| 指标 | S0 v23 | S1 v24 | s1t v27 |
| --- | ---: | ---: | ---: |
| records / errors | 110 / 0 | 110 / 0 | 110 / 0 |
| 原始 stable 检查项 | 19 | 24 | 22 |
| valid stable（剔除 `within_steps=false`） | 15 | 20 | **14** |
| task_success | 40/110 | 10/110 | **20/110** |
| `within_steps=false` | 10 | 10 | **20** |
| `U-FT03-BASE` 成功 | 10/10 | 0/10 | **10/10** |
| `U-FT10-BASE` 成功 | 10/10 | 0/10 | 0/10 |
| `U-FT11-HARD` 成功 | 10/10 | 0/10 | 0/10 |
| `U-FT08-WEAK` 成功 | 0/10 | 0/10 | 0/10 |

逐项结论：

1. `U-FT03` 恢复 10/10：`s1t` 的纯值分支只要求“输出任务要求的最终答案”，不再强制 JSON。
   这证明原 S1 的 `U-FT03` 失败确属输出契约冲突，属于评测/实现偏差。
2. `U-FT10/11` 仍是 0/10：trace 中模型第二轮仍说“已经是 OLD-T03，无法修改”，第三轮直接
   输出 `{"changed": true, "line": 53}`，没有调用 `edit_file`。`s1t` 的“先完成工具操作”前缀
   没有阻止模型把最终 JSON 当作任务完成。这是真实副作用，不是评测偏差。
3. `U-FT08` 仍是 0/10：重复 `write_file` + 错误后不恢复，真实行为问题。
4. `within_steps=false` 从 10 增到 20（`U-FT01`、`U-FT02` 各 10）：`s1t` 在后缀里加入
   “先完成工具操作”后，模型在两个长上下文任务上更倾向于重复读取，步数预算被耗尽。按口径
   这些不进入约束违反 CSV，但它是系统侧 prompt 加固的负作用，必须单独记录。

s1t 的 valid stable CSV 为 `docs/evidence/final_test_v2_s1t_违反根因分布表.csv`，共 14 行，
全部 stable（10/10），无 intermittent；根因分布为能力问题 12 行、行为问题 2 行。

**最终判定**：S1 的负收益可以拆成两部分：

- 约 10 次 `U-FT03` 成功丢失是输出契约冲突（可修，s1t 已修复）；
- `U-FT10/11/08` 的真实副作用和 `U-FT01/02` 的步数恶化没有修复，s1t 仍使 task_success
  从 40 降到 20，valid stable 从 15 降到 14 但绝对成功数更少。

因此“S1 在该样本上有负收益”仍然成立；不能把负收益全部归因于“模型不会做”，也不能把
s1t 当作解决方案。真实副作用是有 trace 证据的行为问题，后续只能通过工具调用约束/训练侧
改进，不应继续堆 prompt 后缀。

