# 阶段 A trace schema

## 文件布局

每次 run 一个 JSONL，事件顺序固定：

```text
run_start
llm_turn   # 每轮 LLM 调用一条
...
run_end
```

本地路径：`private/traces/<experiment>/<task>/<group>/<run>_rollout<NN>.jsonl`。
`private/` 已加入 `.gitignore`，不提交大文件；公开仓库只保留聚合 CSV 与文档。

## run_start

| 字段 | 含义 |
| --- | --- |
| `run_id` / `task_id` / `group` / `profile` | 第几次 / 任务 / on·off / standard·stress |
| `seed` | 夹具随机种子；不是 LLM seed |
| `fixture_seed` | 部分实验显式记录夹具 seed |
| `rollout_index` | 同一 run 的第几次 rollout |
| `sampling_config` | LLM 采样：`temperature` / `top_p` / `seed` / `is_default` |
| `constraints` | 本次 run 的 `{id, content}` 清单；来自约束存储或任务定义 |
| `workspace` | 脱敏后的 `<workspace>` |

## llm_turn

| 字段 | 含义 |
| --- | --- |
| `step_index` / `step_label` / `turn_index` | 轮次与探针标签（如 `probe_json`） |
| `sampling_config` | 当轮实际采样参数；合并 `model_version` / `system_fingerprint` / `response_id` / `request_id`（API 返回时） |
| `messages` | **当轮实际发送**的完整 messages，含 system prompt 与历史摘要 |
| `estimated_tokens` | 当轮上下文估算 token |
| `tools_offered` | 当轮实际下发给模型的工具名列表；区分“工具未发送”和“模型没选择工具” |
| `constraints` | 逐条约束的可见性证据，见下表 |
| `response.content` | 模型完整回复，不截断成 120 字 `reply_head` |
| `response.tool_calls` | 原始工具调用（名称 + 完整 arguments 字符串） |
| `tool_calls` | 结构化工具证据：`arguments_json_valid` / `pydantic_valid` / `path_escape` / `blocked_at` / `executed` |
| `tool_results` | `ok` / `is_error` / `protocol_error` / `error_kind` / `validation_stage` / `content` |
| `compression_events` | 压缩前后 token、被压缩掉的约束 id、补录 id、压缩前后 messages |
| `judgment` | 该轮探针的判定结果（若适用） |

### 约束可见性字段

| 字段 | 含义 |
| --- | --- |
| `constraint_id` / `content` | 约束 id 与原文 |
| `id_in_messages` | id 是否出现在当轮 messages |
| `content_in_messages` | 原文是否逐字出现在当轮 messages |
| `same_message_verbatim` | id 与原文是否在同一条 message |
| `locations` | `system_prompt` / `summary` / `user_declaration` / `replenish` / `tool_result` / `assistant_message` |
| `declaration_message_present` | 是否有原始声明消息 |
| `in_summary` / `in_replenish` | 是否在历史摘要 / 压缩后补录 |
| `compressed_out_of_history` | 是否已被压缩出历史 |
| `model_saw` | **以原文是否出现在当轮 messages 为准**，不是只看 `constraints.json` |

## run_end

| 字段 | 含义 |
| --- | --- |
| `checks` | 脚本化检查结果，如 `json_valid` / `snake_case_keys` / `no_code_fence` / `constraint_kept` / `token_recalled` / `within_steps` |
| `violation_types` | 失败的约束/行为检查；`within_steps` 永不进入违反根因表 |
| `violated` | 违反检查数 |
| `within_steps` | 步数是否达标；步数超预算不是约束违反 |
| `non_constraint_failures` | 非约束类失败，单独记录 |
| `evidence` | 失败检查对应的 `step_label` / `constraint_ids` / `event_index` |
| `step_replies` | 各探针的完整回复（用于审计） |
| `duration_s` / `sampling_config` | 运行时长与采样配置 |

## 脱敏

落盘前统一替换：

- 工作区绝对路径 → `<workspace>`
- 其它盘符绝对路径 → `<abs_path>`
- `sk-...` / Bearer token / `api_key=` / `token=` / `password=` → `<redacted>`

对应实现与单测：`scripts/trace_utils.py`、`tests/test_trace_utils.py`。

## 为什么不能只看 reply_head

`reply_head` 只能证明“模型说了什么开头”，不能证明：

- 约束原文是否真的在当轮 messages 中；
- 工具参数是否通过 Pydantic、是否路径越界、拦截发生在生成后还是执行时；
- 压缩掉了哪些约束、补录了哪些；
- 失败是 JSON 主体非法，还是 JSON 后面追加了自然语言。

因此归因脚本只使用 `messages`、`constraints[].model_saw`、`tool_calls`、`tool_results`、
`compression_events` 与 `run_end.evidence`，`reply_head` 不参与根因判断。
