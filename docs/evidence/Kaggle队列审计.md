# Kaggle GPU 队列审计

**检查时间**：2026-10-19T18:18:24+08:00  
**目的**：确认小基座 `QUEUED` 是否被遗留 kernel 占用第二个 GPU 名额，决定是否需要取消/等待。

> **后续状态（2026-10-20）**：小基座随后进入运行并在安装 `vllm==0.7.3` 时因 Python 3.13
> 依赖不兼容转为 `ERROR`，不是仍在排队。现已本地改为 `vllm==0.12.0`，第二次运行又因 T4
> FlashInfer JIT 缺 `libcuda` 失败；改为 `TRITON_ATTN` 后 v5 已解决环境问题，v6 进入
> smoke 门禁但因模型 `auto` tool choice 未产生真实 tool_call 而停止。随后协议审计
> kernel 与改用 `Qwen2.5-1.5B-Instruct` 的 v7 均 `COMPLETE`，CTX 重跑 v8 也已 `COMPLETE`；
> 合并后为 110 条有效 rollout、0 error、20 个 stable 检查项。`final-test-v1` baseline v9
> 也已 `COMPLETE`（110 条、26 个 stable，其中 6 个为 U-FT10/11 任务定义缺陷）；constrained
> decoding v4 为 `ERROR`（1.5B 权重加载 900 秒超时，0 条结果）。当前没有需要清理的
> `RUNNING/QUEUED` kernel。
> 本文第 2–4 节的 `QUEUED` 结论只反映 2026-10-19 当时的队列状态。

## 1. 账号 kernel 列表

执行：

```powershell
kaggle kernels list -m
```

账号下共列出 15 个 kernel；`kernels list` 本身不返回状态，因此随后对每个 ref 执行
`kaggle kernels status <ref>`。

## 2. 状态汇总

| 状态 | kernel | 是否占 GPU |
| --- | --- | --- |
| `QUEUED` | `autumnsq/lite-agent-small-model-tuning` | 是（等待调度） |
| `ERROR` | `autumnsq/lite-agent-constrained-decoding-baseline` | 否（已退出） |
| `ERROR` | `autumnsq/aaaaa` | 否（已退出） |
| `COMPLETE` | `autumnsq/save-b`、`autumnsq/test-for-cv`、`autumnsq/csiro` | 否 |
| `CANCEL_ACKNOWLEDGED` | `autumnsq/dinov2` | 否 |
| 404 / Not Found | `connect`、`plusa`、`ab-qwen`、`notebook4a5b229bf0`、`notebook0f206da758`、`notebook7206115aa1`、`notebookdc2a38a502`、`notebook0a25657ed2` | 否 |
| `RUNNING` | 无 | — |

## 3. 结论

- 账号下唯一处于 `QUEUED` 的是小基座 kernel。
- 没有其他 `RUNNING` kernel，也没有第二个遗留的 `QUEUED` GPU 任务。
- `constrained decoding` 是 `ERROR`，不占名额；其余为 `COMPLETE` / `ERROR` /
  `CANCEL_ACKNOWLEDGED` 或旧 404 引用。
- 因此**没有需要取消的遗留 kernel**；小基座属于正常排队，继续等待。
- 小基座 `QUEUED` 期间不重推 constrained decoding，避免挤占 GPU 名额。

## 4. 清理尝试与边界

- 当前安装的 Kaggle CLI 的 `kernels` 子命令只有 `list` / `files` / `init` / `push` /
  `pull` / `output` / `status`，没有 `delete`；Python API 也没有 kernel 删除方法。
- 因此没有执行破坏性清理，也没有调用未经验证的删除接口。旧 `COMPLETE` / `ERROR` /
  404 引用不处于 `RUNNING` / `QUEUED`，不占 GPU，不会阻塞小基座排队。
- 下次审计仍以 `kaggle kernels status <ref>` 区分活动任务；不因列表里存在旧条目就
  取消或重推目标 kernel。
