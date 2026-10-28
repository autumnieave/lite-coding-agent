# 终端 Coding Agent：长任务约束遵循与后训练可靠性验证

Agent 约束遵守实验平台——违反归因、可复现评测与训练侧能力边界验证。

长任务中，约束在上下文里不等于模型会遵守。
项目按“约束→工具→评测→训练→决策”五层验证，每层只证明本层能证明的事。
最强信号：45 个去重叠任务上，SFT 单次通过率 49.4% → 99.8%，更严格的 pass^10（10 次全过）SFT 达 44/45；增益不来自逐字记忆（McNemar p<1e-13）。

## 核心发现

- 约束层：压力档（deepseek-flash）违规 19→5；补录只验 ID 不验正文——59 条语义保留的约束中只有 23 条被真正遵守（39.0%），7B untagged 行为规则 3/8 vs 信息 8/8（Qwen2.5-7B，train-v5）。证据见 docs/evidence/。

- 工具层：Tool-Calling Agent Loop + 6 工具（Pydantic 参数校验 + 路径越界拦截），edit_file 三道防线（read-before-edit / mtime 防护 / old_string 唯一性）；幂等键 + checkpoint 恢复，单进程 stdio-only。代码见 src/agent/，测试见 tests/。

- 评测层：同一任务跑 10 次，3.57% 判定翻转（560 配对）→ 单次不可信。45 个去重叠任务上，SFT 单次通过率 49.4% → 99.8%（+50.3pp，McNemar p<1e-13），pass^10（10 次全过）从 0/45（推断）到 44/45。证据见 docs/evidence/。

- 训练层：7B SFT 首跑 BASE 0/30 全灭 → Trace 定位训练/推理模板不一致（307→1567 token）→ 修复后 30/30；full-loss 致 CTX-HARD 0/10，assistant-only Loss Masking 恢复 10/10。API/本地 99.8%/90.9%，优势在格式。证据见 docs/evidence/。

- 决策层：能启动 ≠ 该启动。GRPO 正确预检 360 条 rollout（45 任务 × 8 次）：44/45 已全对、中间态 0——按预注册判据（中间态为 0 则不启动），不做全量 RL。证据见 docs/evidence/。

```mermaid
flowchart LR
A[约束层<br/>原文在≠语义在≠行为生效]-->B[工具层<br/>能调用≠调用可靠]
B-->C[评测层<br/>模型能跑≠结果可信]
C-->D[训练层<br/>能训≠训对了]
D-->E[决策层<br/>能启动≠该启动]
```

五层各自独立验证，不跳层外推。

## 为什么做

约束会丢，模型未必遵守；须逐层归因。

## 核心特性

- 运行时：Agent Loop、6 工具、幂等键、checkpoint/流式输出。
- 上下文：四层压缩、独立约束存储、双通道注入、ID 补录。
- 评测：约束保留、行为 benchmark、trace 归因、稳定性。

**从零实现**：Agent Loop、6 工具、约束独立存储、评测 harness；
**依赖**：vLLM（1.5B 推理）、Pydantic（参数校验）、transformers（14B 对照）。

## 快速开始

Python 3.11+，需支持工具调用的 LLM（Claude/GPT-4o/DeepSeek）。

```bash
git clone git@github.com:autumnieave/lite-coding-agent.git lite-coding-agent
cd lite-coding-agent && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

export LLM_API_KEY="..." LLM_BASE_URL="..." LLM_MODEL="..."

lite-agent chat "列出当前目录"
lite-agent chat "列出当前目录" --verbose
```

> 完整地址：github.com/autumnieave/lite-coding-agent

配置：环境变量 > .env；退出码 0/1/2 = 成功/失败/配置错误。示例见 `docs/usage.md`，完整参数见 `lite-agent --help`。

**3 分钟检查清单**：
1. `lite-agent chat "列出当前目录"` — 验证 Agent Loop 能跑
2. `pytest -q` — 857 条测试（约 2 分钟）
3. 打开 `docs/evidence/会话约束分层验证.md` — 看一条 trace 归因

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

架构见 `docs/architecture.md`。

## 评测

**约束保留（40 条/25 轮×20 次）**

| 组 | 逐字保留 | 上下文仍在 | 违反 |
|---|---:|---:|---:|
| off | 552/800（69.0%） | 640/800 | 19 |
| on | 760/800（95.0%） | 800/800 | 5 |

**final-test-v2（1.5B，10 次）**：S0 valid stable 15 / task_success 40/110；S1 valid stable 20 / task_success 10/110——S1 prompt 自检负收益。

**训练侧方法对照（train-v4，7B QLoRA）**

| 条件 | 总体 | BASE | HARD |
|---|---:|---:|---:|
| baseline | 40/60 | 20/30 | 20/30 |
| full-loss SFT | 50/60 | 30/30 | 20/30 |
| assistant-only SFT | 60/60 | 30/30 | 30/30 |
| +15% Dolly | 50/60 | 30/30 | 20/30 |
| SFT + DPO | 60/60 | 30/30 | 30/30 |

**clean held-out（train-v6，45 任务，7B）**

| 指标 | baseline | SFT |
|---|---:|---:|
| pass@1（单次通过率） | 49.4% | 99.8%（+50.3pp，McNemar p<1e-13） |
| pass^10（10 次全过） | 0/45（推断） | 44/45 |

> 项目门禁另设 ≥6/10 多数阈值（10 次中至少 6 次通过记稳定通过），对应 SFT 45/45；baseline 仅 8 次观测，无法计算该口径。pass^10 是更严格的可靠性下界，生产环境通常看这个。

45/45 一致；增益接近，排除记忆。

**复现**：train-v6 adapter 见 `private/kaggle_pull_clean_heldout_v1/clean_heldout/adapter/`（SHA-256 `37928a28...a4b96` 校验通过）；需 GPU ≥16GB。

## 边界与未覆盖

- 主实验四组对比未启动：S1 prompt 自检负收益（valid stable 15→20、task_success 40/110→10/110），预注册门禁未满足。

- GRPO 未启动：预检判定 SFT 已饱和，无梯度信号。

- constrained decoding 未跑通：4 次尝试分别卡在模型路径、依赖、取消、加载超时，0 条 rollout。

- Token 三段归因（69.9% 来自工具返回）属运行时优化，未做截断对照；见 `docs/evidence/Token三段归因.md`。

- 同族非跨模板：45 个 clean 任务仍与 train 同模板，+50.3pp 不可外推到“没见过的任务族”。

下一步：验证跨模板泛化与 constrained decoding 多后端。

## 参考与 License

### Related Work

格式违反可脚本判定，故选 constrained decoding；基线为 Outlines / JSON Schema。ADR: `docs/decisions.md`。

### 参考

- [claude-code-from-scratch](https://github.com/Windy3f3f3f3f/claude-code-from-scratch)
- [How Claude Code Works](https://github.com/Windy3f3f3f3f/how-claude-code-works)
- [Outlines](https://github.com/dottxt-ai/outlines)
- [JSON Schema](https://json-schema.org/)

License：MIT，见 LICENSE。
