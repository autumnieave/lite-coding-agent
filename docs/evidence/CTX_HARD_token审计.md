# CTX-HARD 超长 token 审计

> 脚本：`scripts/audit_ctx_hard_tokens.py`
> 输入 train：`private\sft_data_v4\train.jsonl`
> tokenizer：`$env:HF_HOME\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\<revision>`
> 最大长度：3072
> 性质：只读；不修改任务定义、fixture 或冻结数据。

## 逐样本分解

| ID | 总 token | fixture | fixture 占比 | tools 模板 | tools 占比 | system/user | 工具调用 JSON | 最终回答 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train-v4:deepseek:train-v1:U-TR-A-CTX-HARD:rollout7 | 6217 | 4260 | 68.52% | 1260 | 20.27% | 284 | 607 | 7 |
| train-v4:oracle:U-TR4-00-CTX-HARD:rollout00 | 6444 | 4511 | 70.00% | 1260 | 19.55% | 259 | 540 | 8 |
| train-v4:oracle:U-TR4-01-CTX-HARD:rollout00 | 6444 | 4511 | 70.00% | 1260 | 19.55% | 259 | 540 | 8 |
| train-v4:oracle:U-TR4-02-CTX-HARD:rollout00 | 6444 | 4511 | 70.00% | 1260 | 19.55% | 259 | 540 | 8 |
| train-v4:oracle:U-TR4-03-CTX-HARD:rollout00 | 6444 | 4511 | 70.00% | 1260 | 19.55% | 259 | 540 | 8 |
| train-v4:deepseek:train-v1:U-TR-A-C1-BASE:rollout0 | 1567 | 10 | 0.64% | 1260 | 80.41% | 214 | 61 | 26 |

## 判定

- 主因：**fixture 太长**。
- 证据：CTX-HARD 样本的 fixture 平均占 69.70%；`tools=` 模板平均占 19.69%。
- 训练路径与直接 Qwen `apply_chat_template(..., tools=...)` 一致：True。
- 修复边界：缩短新增 train-v5 CTX fixture 的每文件行数，保留 10 文件、多轮读取结构；不修改 final-test-v2 的 fixture，也不靠提高全局 `max_seq_length` 掩盖数据问题。
