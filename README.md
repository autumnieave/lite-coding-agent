# 终端 Coding Agent：长任务约束遵循与后训练可靠性验证

Agent 约束遵守实验平台——违反归因、可复现评测与训练侧能力边界验证。

针对长任务中压缩静默丢失约束、且“约束在上下文≠模型遵守”的痛点，独立构建终端 Coding Agent 实验平台，诊断约束遵守的失败模式并归因到训练信号质量——聚合指标会掩盖问题，拆到行为模式才能诊断、再控制变量定位根因；方法在 64 条探针上验证稳定性；补录只校验 ID 的缺陷修复后，tagged 故障注入探针上行为层约束遵守完全恢复（语义 64/64、合规 64/64）。验证规模：409 唯一任务池、45 任务配对评测。
最强信号：45 个去重叠任务上，SFT 单次通过率 49.4% → 99.8%，更严格的 pass^10（10 次全过）SFT 达 44/45；增益不来自逐字记忆（McNemar p<1e-13）。

## 核心发现

- 约束层：原文在 ≠ 行为生效。自研约束独立存储 + 双路注入 + 约束补录，压力档（deepseek-flash）违规 19→5；但 Qwen2.5-7B 基座在原始 8 任务 32 条 tagged+untagged 合计中仅 12/32 合规（untagged 单组 8/16），train-v5 SFT 11/32（untagged 3/16）、train-v6 SFT 28/32（untagged 15/16）（小样本、方向性）。补录只校验 ID 导致 tagged 故障注入（强制 ID-only 摘要）下语义保留 0/64（ID 全在、正文全丢）；加正文双重校验后语义 64/64、合规 64/64（修复前 17/64；仅限该故障注入探针），该探针上行为合规完全恢复；untagged 不创建 ConstraintStore、补录路径不可达，历史三组数字保留。证据见 docs/evidence/。

- 工具层：Tool-Calling Agent Loop + 6 工具（Pydantic 参数校验 + 路径越界拦截），edit_file 三道防线（read-before-edit / mtime 防护 / old_string 唯一性）；幂等键 + checkpoint 恢复，单进程 stdio-only；benchmark 48 次运行（on 36/off 12）的 466 次工具调用参数 JSON 全合法、路径越界 0 次。代码见 src/agent/，测试见 tests/。

- 评测层：同一任务跑 10 次，3.57% 判定翻转（560 配对）→ 单次不可信。45 个去重叠任务上，SFT 单次通过率 49.4% → 99.8%（+50.3pp，McNemar p<1e-13），pass^10（10 次全过）SFT 达 44/45。证据见 docs/evidence/。

- 训练层：能训 ≠ 训对了。Loss Masking 单变量对照：full-loss 长上下文任务 0/10、assistant-only 恢复 10/10——算法是必要条件。诊断方法：聚合指标 → 拆失败模式 → 唯一变量对照 → 定位根因——原始 8 任务 32 条 tagged+untagged 合计中，train-v5 SFT 合规 11/32 < baseline 12/32，聚合指标看是负收益；拆失败模式发现违反仅 1、只读 20——不是“没学会遵守”，是“学会了只读最安全”；同算法同模型、唯一变量是 train-v6 移除 67 条中间态轨迹，合计合规升至 28/32——定位到训练信号确定性（歧义样本导致退化解），而非算法选择。稳定性验证：扩到 16 任务后，untagged 单组 baseline 17/32、train-v5 5/32、train-v6 28/32；旧 8 子集 untagged 单组为 8/16、4/16、15/16（原始实验 8/16、3/16、15/16），新增 8 子集为 9/16、1/16、13/16——全量上升来自新增任务，旧任务 baseline/train-v5 各 1 条差异来自任务重排；原始 12/32、11/32、28/32 是合计口径，不能与故障注入并列。边界：补录修复在 tagged 故障注入探针上验证了行为层改善；untagged 是有意对照、不创建 ConstraintStore，其自然摘要结果属于训练侧效果——两条路径各自有效，不能合并或外推（64 条探针、训练重叠池、未做显著性检验）。

- 决策层：能启动 ≠ 该启动。GRPO 预检 360 条 rollout（45 任务 × 8 次）：44/45 已全对、存在部分成功/中间态信号的任务为 0——按预注册判据（中间态为 0 则不启动），不做全量 RL。证据见 docs/evidence/。

```mermaid
flowchart LR
A[约束层<br/>原文在≠行为生效]-->B[工具层<br/>能调用≠调用可靠]
B-->C[评测层<br/>模型能跑≠结果可信]
C-->D[训练层<br/>能训≠训对了]
D-->E[决策层<br/>能启动≠该启动]
```

五层各自独立验证，不跳层外推。

## 为什么做

约束会丢，模型未必遵守；须逐层归因。

## 核心特性

- 运行时：Agent Loop、6 工具、幂等键、checkpoint/流式输出。
- 上下文：四层压缩、独立约束存储、双路注入、约束补录。
- 评测：约束保留、行为 benchmark、trace 归因、稳定性。

**从零实现**：Agent Loop、6 工具、约束独立存储、评测 harness；
**依赖**：vLLM（1.5B 推理）、Pydantic（参数校验）、transformers（14B 对照）。

## 快速开始

Python 3.11+，需支持工具调用的 LLM（Claude/GPT-4o/DeepSeek）。

```bash
git clone https://github.com/autumnieave/lite-coding-agent.git lite-coding-agent
cd lite-coding-agent && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

export LLM_API_KEY="..." LLM_BASE_URL="..." LLM_MODEL="..."

lite-agent chat "列出当前目录"
lite-agent chat "列出当前目录" --verbose
```

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
| pass^10（10 次全过） | 0/45（外推） | 44/45 |

>  baseline 仅 8 次观测：pass^10 的 0/45 是外推值（每任务最多成功 5/8，10 次全过逻辑上不可能）。pass^10 是更严格的可靠性下界，生产环境通常看这个。

重叠组 / 非重叠组增益一致（+50.5pp vs +50.0pp），排除逐字记忆。

**复现**：train-v6 需自行重训（SHA 920a4ee7，100 steps，配置见 docs/evidence/训练管线验证.md）；adapter 本体不入库，需 GPU ≥16GB。

## 边界与未覆盖

- 主实验四组对比未启动：S1 prompt 自检负收益（valid stable 15→20、task_success 40/110→10/110），按预设门槛不启动。

- GRPO 未启动：预检判定 SFT 已饱和，无梯度信号。

- constrained decoding 未跑通：4 次尝试分别卡在模型路径、依赖、取消、加载超时，0 条 rollout。

- Token 三段归因（69.9% 来自工具返回）属运行时优化，未做截断对照；见 `docs/evidence/Token三段归因.md`。

- 同族非跨模板：任务池按审计口径有 6 个模板族（public_grep、conflict、path_params、json_constraints、constraint_edit、context_recall）；45 个 clean 任务覆盖其中 4 个（public_grep、json_constraints、constraint_edit、path_params），另有 2 条 json_file，+50.3pp 不可外推到“没见过的任务族”；409 任务池中 394 个来自 train-v5，用训练集做难度筛选是为了找 SFT 中间态。

- 会话约束 adapter 对比为 16 任务 × 64 条探针、1 rollout，未做显著性检验，任务仍在训练重叠池，不能当 clean held-out 泛化证据。

- 补录修复验证限于 tagged 故障注入（强制 ID-only 摘要，64 条、1 rollout、未做显著性检验）：语义保留 0/64→64/64、合规 17/64→64/64；untagged 是有意对照、不创建 ConstraintStore，补录路径不可达；自然摘要的 tagged+untagged 合计 53/64（untagged 28/32）不能与本次 17/64 直接比较；自然摘要未重测。

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
