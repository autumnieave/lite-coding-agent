# 终端 Coding Agent：长任务约束遵循与后训练可靠性验证

Agent 约束遵守实验平台——违反归因、可复现评测与训练侧能力边界验证。
github.com/autumnieave/lite-coding-agent

## 核心发现

- 约束层：压力档（deepseek-flash）19→5；补录只验 ID 不验正文，
  7B untagged 行为规则 3/8 vs 信息 8/8（Qwen2.5-7B，train-v5）。

- 工具层：从零实现 Tool-Calling Agent Loop + 6 工具
  （Schema 校验 + 路径沙箱）+ 幂等键 + checkpoint 恢复，
  单进程 stdio-only 边界。

- 评测层：temperature=0 下 3.57% 翻转率（560 配对），
  45 任务去重叠 SFT +50.3pp。

- 训练层：full-loss 致 CTX-HARD 0/10，
  assistant-only Loss Masking 恢复 10/10；
  API vs 本地 99.8% vs 90.9%，优势在格式。

- 决策层：GRPO 预检 45 任务 44/45 为 8/8 饱和、
  中间态 0，故不做 RL。

## 为什么做

长任务中，压缩会静默丢失约束，模型也未必遵守。
从零构建终端 Coding Agent，按“约束→工具→评测→训练→决策”验证。
每层只证明本层能证明的事。

## 核心特性

- Agent 运行时：Agent Loop + 6 工具（Schema 校验 + 路径沙箱）+
  幂等键 + checkpoint 恢复 + 流式输出。

- Context Engineering：四层压缩 + 约束独立存储 +
  双通道注入 + 压缩后 ID 补录。

- 评测体系：约束保留实验 + 行为 benchmark +
  trace 归因 + 逐任务稳定性。

## 快速开始

Python 3.11+，需要支持工具调用的 LLM API（Claude / GPT-4o / DeepSeek）。

```bash
git clone git@github.com:autumnieave/lite-coding-agent-private.git
cd lite-coding-agent-private && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

export LLM_API_KEY="..." LLM_BASE_URL="..." LLM_MODEL="..."

lite-agent chat "列出当前目录"           # 执行一次任务
lite-agent chat "列出当前目录" --verbose  # 打印工具调用完整参数与结果
```

配置优先级：已存在的环境变量 > .env 文件。退出码：0 成功 / 1 任务失败 / 2 配置错误。
完整用法见 `lite-agent --help`，更多示例见 `docs/usage.md`。

## 项目结构

```text
lite-coding-agent/
├── src/agent/
│   ├── core/
│   ├── tools/
│   ├── memory/
│   └── cli/
├── scripts/
├── training/
├── tests/
├── docs/evidence/
├── AGENTS.md
├── pyproject.toml
├── LICENSE
└── README.md
```

架构图见 `docs/architecture.md`。

## 评测

**约束保留（40 条/25 轮×20 次）**

| 组 | 逐字保留 | 上下文仍在 | 违反 |
|---|---:|---:|---:|
| off | 552/800（69.0%） | 640/800 | 19 |
| on | 760/800（95.0%） | 800/800 | 5 |

**final-test-v2（1.5B，10次）**

| 指标 | S0 | S1 |
|---|---:|---:|
| valid stable | 15 | 20 |
| task_success | 40/110 | 10/110 |

**训练侧方法对照（train-v4，7B QLoRA）**

| 条件 | 总体 | BASE | HARD |
|---|---:|---:|---:|
| baseline | 40/60 | 20/30 | 20/30 |
| full-loss SFT | 50/60 | 30/30 | 20/30 |
| assistant-only SFT | 60/60 | 30/30 | 30/30 |
| +15% Dolly | 50/60 | 30/30 | 20/30 |
| SFT + DPO | 60/60 | 30/30 | 30/30 |

**clean held-out（train-v6，45 任务 × 10 rollout，7B SFT）**

| 条件 | 通过率 |
|---|---:|
| baseline | 49.4% |
| SFT | 99.8%（+50.3pp，McNemar p<1e-13） |

45/45 任务一致；无逐字重叠组 +50.5pp 与有重叠组 +50.0pp 几乎相同，排除逐字记忆。边界：同族泛化，非跨模板。

**复现**：train-v6 adapter 在
`private/kaggle_pull_clean_heldout_v1/clean_heldout/adapter/`，
SHA-256 `37928a28...a4b96` 校验通过，392 个 LoRA tensor 可读；
本机 MX250 2GB 不足，需 GPU ≥16GB。

## 边界与未覆盖

- 主实验四组对比未启动：S1 prompt 自检负收益
  （valid stable 15→20、task_success 40/110→10/110），
  预注册门禁未满足

- GRPO 未启动：45 任务 44 个 8/8 饱和、中间态 0，SFT 已饱和，不做 RL。

- constrained decoding 0 条结果：v1 路径、v2 bitsandbytes ERROR、v3 取消、v4 加载超时。

- Token 三段归因（工具返回 69.9%）属运行时优化，不在后训练主线，未做截断对照；
  证据见 `docs/evidence/Token三段归因.md`。

- 同族泛化非跨模板：45 个 clean 任务仍属同族。

## 参考、Related Work 与 License

### Related Work

主样本可脚本判定格式违反，故选 constrained decoding；Outlines / JSON Schema 为同层基线，暂不接 Guardrails / τ-bench。ADR 见 `docs/decisions.md`。

### 参考

- [claude-code-from-scratch](https://github.com/Windy3f3f3f3f/claude-code-from-scratch)
- [How Claude Code Works](https://github.com/Windy3f3f3f3f/how-claude-code-works)
- [Outlines](https://github.com/dottxt-ai/outlines)
- [JSON Schema](https://json-schema.org/)

### License

MIT，见 [LICENSE](LICENSE)。
