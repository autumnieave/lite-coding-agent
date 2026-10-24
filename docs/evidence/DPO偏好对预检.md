# DPO 偏好对预检

**日期**：2026-10-24  
**输入**：`train-v5`（train SHA-256 `6ee7e79d87cb7c41f4c1b91964d70abcf5e886c5ac75450318760a805b78f830`）  
**构造器**：`training/build_dpo_v2.py`  
**产物**：`private/dpo_v2/`（58 对；train 45 / val 13；train SHA-256 `691ea2667893b483d02004e9f8af800a6978df2b9dba80d19a245f4098b43aed`，val SHA-256 `4072a9fec7616e6df3e9a17ecafe30f0c404a6495da63204ca0f2413ec8466f2`）

## 1. 偏好对构造

train-v5 的 86 条 rejected 按原因分布如下：

| reject 原因 | 数量 | 是否进入 DPO | 原因 |
| --- | ---: | --- | --- |
| `code_prefix_output` | 58 | 是 | 最终回答为 `CODE=CTX-...`，剥离 `CODE=` 后与 fixture 的 `final_token` 精确一致，可派生无歧义 chosen |
| `checks_not_all_true` | 25 | 否 | 有检查项失败，没有同 prompt 的干净正样本；不能把错误轨迹当 chosen |
| `tool_result_not_ok` | 2 | 否 | 工具行为有缺陷，不能作为正样本 |
| `tools_not_allowed` | 1 | 否 | 工具白名单违规，不能作为正样本 |
| **合计** | **86** | **58** | 其余 28 条只作为失败行为证据保留 |

58 条 `code_prefix_output` 均满足：剥离 `CODE=` 后的文本 58/58 精确等于对应 fixture 的 `final_token`；不是用 judge 猜测的答案。

## 2. 数据隔离

| 检查项 | 结果 |
| --- | ---: |
| 与 SFT train prompt 完全匹配 | 0 |
| 与 SFT val prompt 完全匹配 | 0 |
| 与 6 个 held-out task ID 匹配 | 0 |
| 偏好对总数 | 58（train 45 / val 13） |

隔离按“规范化 messages 的 SHA-256”计算，不是仅按 task ID；58 对全部来自 `train-v5` 的原始 rejected prompt，没有复用 SFT train 正样本。

## 3. Qwen chat template token 预检

使用 `Qwen/Qwen2.5-7B-Instruct` tokenizer、`default_tool_schemas()` 和与训练一致的 `add_generation_prompt=True` 渲染：

| 指标 | min | p50 | max |
| --- | ---: | ---: | ---: |
| prompt tokens | 2,168 | 2,770 | 3,001 |
| chosen completion tokens | 8 | 8 | 8 |
| rejected completion tokens | 10 | 10 | 10 |

**截断数**：

| 限制 | prompt 被截断 | chosen 总长被截断 | rejected 总长被截断 |
| ---: | ---: | ---: | ---: |
| 1,536 | 58/58 | 58/58 | 58/58 |
| 2,048 | 58/58 | 58/58 | 58/58 |
| 3,072 | 0/58 | 0/58 | 0/58 |

## 4. 结论

- 偏好对数量为 **58**，但 **58/58 的 prompt 已超过 DPO 配置的 `max_prompt_length=2048`**；在 2,048 和 1,536 下，chosen/rejected 都会发生尾部截断。若按当前 `training/configs/dpo_7b_t4.json` 启动，所有样本的最后的任务指令都可能被截掉，DPO 输入不合法。
- 在 `max_seq_length=3072` 下，prompt + completion 总长全部可容纳（0/58 超限），但 `max_prompt_length=2048` 仍会先行截断 prompt；不能据此写“DPO 数据已就绪”。
- 因此本轮 DPO 预检判定为：**偏好对不足/长度不满足，DPO 不启动**。没有用 28 条行为缺陷轨迹硬凑正样本，也没有为凑数生成 synthetic negative。
- 若后续继续 DPO，必须先二选一并在运行前记录：(a) 构造更短的 CTX 偏好对；(b) 把 `max_prompt_length` 提升到 2,048 以上且 `max_seq_length` 提升到 4,096，并重新跑本预检。本轮不执行该配置变更。

**边界**：58 对只覆盖 CTX 的 `code_prefix` 修复，不覆盖 BASE/HARD/TOOL/CONFLICT；即使长度问题解决，也不能把本预检写成“DPO 数据覆盖完整”。
