# Token 三段归因

**口径**：对 7B v9 SFT held-out 的 60 条 trace 做只读统计。  
**结论**：本地 `runs.jsonl` 没有 `input_tokens` / `output_tokens` / `prompt_tokens`
分段字段；trace 的 `llm_turn` 里有每轮 `prompt_tokens` / `completion_tokens`。按
Qwen2.5 tokenizer 对 final-turn `messages` 做 role 拆分，正文 token 占比约为：
上下文注入 19.1%、assistant 推理 11.0%、工具返回 69.9%。

## 1. 字段检查

检索：

- `private/kaggle_pull_7b_nf4_v9/7b_nf4/heldout/sft/runs.jsonl`：0 个
  `input_tokens` / `output_tokens` / `prompt_tokens` 命中；
- `private/kaggle_pull_clean_heldout_meta/clean_heldout/full/sft/runs.jsonl`：0 命中；
- `docs/evidence/benchmark_runs.jsonl`：0 命中；
- `private/api_ab/runs.jsonl`：有 `api_audit.prompt_tokens` /
  `completion_tokens`，但那是 API 组，不是本地 7B run 字段。

本地 7B trace 的 `llm_turn.sampling_config` 有逐轮
`prompt_tokens` / `completion_tokens`，可作为总 prompt 规模校验；没有把 token
按“上下文 / 推理 / 工具”预先分段。

## 2. 方法

- 语料：`private/kaggle_pull_7b_nf4_v9/7b_nf4/heldout_traces/sft/` 下 60 个 trace，
  共 170 个 `llm_turn`。
- Tokenizer：本地缓存的 `Qwen/Qwen2.5-7B-Instruct` tokenizer。
- 对每条 trace 取 **最后一个 `llm_turn` 的完整 messages**，按 role 归类：
  - `system` + `user` → 上下文注入；
  - `assistant` → 推理/工具调用生成；
  - `tool` → 工具返回；
  - assistant 消息若带 `tool_calls`，把工具调用 JSON 一并计入推理段。
- 每个 role 的 `content` 与 `tool_calls` 分别用 Qwen tokenizer 编码后计数；
  **不重复计算同一 trace 中跨轮重复出现的同一条消息**。
- 这是消息正文口径的粗归因：没有把 chat template 的 role 标记、special token
  和 tools schema 均摊到三段。

## 3. 结果（final-turn 消息正文）

| 段 | token | 占三段正文比例 | 每条 trace 均值 |
| --- | ---: | ---: | ---: |
| 上下文注入（system + user） | 13,950 | 19.06% | 232.5 |
| 推理（assistant，含 tool_calls JSON） | 8,080 | 11.04% | 134.7 |
| 工具返回（tool） | 51,160 | 69.90% | 852.7 |
| 合计 | 73,190 | 100.00% | 1,219.8 |

同一批 trace 的 final-turn `prompt_tokens` 合计为 **149,340**；上述 role 正文合计为
73,190，差额 76,150（51.0%）来自 chat template、special token、tools schema 等未分摊
开销。因此：

- 三段占比只能用于比较“哪一类消息占正文大头”，不能把 19.1% / 11.0% / 69.9%
  当作完整 prompt 的精确预算分布；
- 工具返回是最大头，和 CTX-HARD 的 4–6 千 token prompt 主要由长 tool response
  构成一致；
- 上下文注入在最后回答轮不是最大头，但这不等于压缩没有代价，压缩摘要与 system
  prompt 仍属于上下文段。

## 4. 总调用量校验

60 条 trace 的 170 个 `llm_turn` 中，逐轮 `prompt_tokens` 合计 **421,300**，
`completion_tokens` 合计 **7,730**。这两个字段是模型侧 usage，不是三段归因；
它们只用于说明：同一批 rollout 的实际累计输入远大于单条 final-turn 正文，因为每轮
都会重新携带历史。

## 5. 边界

- 只统计 7B v9 SFT held-out 的 60 条 trace，不外推到 1.5B、14B 或 API 组；
- 工具调用的 JSON 计入 assistant 推理段，不单独拆“决策 token”与“参数 token”；
- chat template/tools schema 开销未分摊，已在差额中单列；
- 无伪造或手工补数；所有数字可由上述 trace 与 Qwen tokenizer 重算。
