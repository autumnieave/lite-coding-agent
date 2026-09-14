# LiteCoding Agent

> 从零实现的终端 Coding Agent，不依赖 LangChain 等 Agent 框架。核心目标：在长任务中**不丢失关键约束**。

**当前状态**：v0.1.0 脚手架阶段 —— CLI 入口可运行，Agent Loop 开发中。各功能实现进度见「核心特性」与「路线图」。

## 为什么做

现有 Coding Agent 在长任务中会因上下文压缩丢失早期设定的关键约束（如「必须兼容 Python 3.9」「禁止修改数据库迁移文件」），导致后续步骤违反规则。本项目在复现 Claude Code 核心机制的基础上，重点解决**上下文压缩中的约束静默丢失**问题。

## 核心特性

**已实现：**

- ✅ **项目脚手架**：src-layout 打包，`lite-agent` 入口可用
- ✅ **CLI 骨架**：基于 argparse 的命令行入口与参数解析

**开发中：**

- 🚧 **Agent Loop**：`while(true)` 主循环，模型返回 tool_call 后执行工具并回填结果
- 🚧 **工具系统**：`bash` / `read_file` / `write_file` / `edit_file` / `grep` / `list_dir`，JSON Schema 参数校验 + 危险命令确认
- 🚧 **上下文压缩**：四层策略（预算截断 / 裁剪重复 / 微压缩 / 全量摘要）
- 🚧 **关键约束保留（独有）**：压缩前提取约束、压缩后校验并自愈

**计划中：**

- 📋 **项目记忆**：加载 `AGENTS.md`，按目录层级注入项目规则
- 📋 **会话持久化**：checkpoint 保存会话状态，支持中断恢复
- 📋 **MCP 客户端**：手写 JSON-RPC over stdio，接入外部工具
- 📋 **子 Agent 隔离**：独立上下文和工具白名单

## 架构图

```mermaid
flowchart TD
    U[用户输入 / 任务] --> L[Agent Loop]
    L --> P[上下文压缩<br/>+ 约束保留校验]
    P --> M[LLM 调用]
    M --> Q{返回 tool_call?}
    Q -- 是 --> T[工具执行<br/>参数校验 / 确认]
    T --> R[结果回填到消息历史]
    R --> L
    Q -- 否 --> F[结束并输出结果]
```

> 待补充：模块依赖图（`core` / `tools` / `memory` / `cli` 之间关系），见 `docs/architecture.md`。

## 快速开始

### 环境要求

- Python 3.11+
- 一个支持工具调用的 LLM API（如 Claude、GPT-4o、DeepSeek）

### 安装

```bash
git clone https://github.com/autumnieave/lite-coding-agent.git
# 网络受限时可用 SSH：git clone git@github.com:autumnieave/lite-coding-agent.git
cd lite-coding-agent
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 配置

> LLM 配置方式（API Key / Base URL / Model）将在 Agent Loop 完成后补充。

### 运行

```bash
lite-agent --help    # 查看用法与参数
```

> 当前进度：脚手架与 CLI 入口已完成，Agent Loop 开发中；交互模式见「路线图」。

## 项目结构

```
lite-coding-agent/
├── src/agent/
│   ├── core/          # Agent Loop、LLM 抽象、上下文管理
│   ├── tools/         # 工具注册表与具体工具
│   ├── memory/        # 项目记忆与会话持久化
│   └── cli/           # 命令行入口
├── tests/             # 单元测试
├── docs/              # 架构与决策文档
├── AGENTS.md          # 项目规则（Agent 启动时加载）
├── pyproject.toml
├── LICENSE
└── README.md
```

## 核心设计

### Agent Loop

主循环只有一条规则：**模型返回 tool_call 就执行，否则结束**。工具执行失败时，错误信息回填给模型，由模型自行修正。

### 上下文压缩与约束保留（独有）

| 层级     | 策略   | 触发条件               |
| -------- | ------ | ---------------------- |
| Tier 1   | 预算截断 | 工具输出超过阈值       |
| Tier 2   | 裁剪重复 | 同文件重复读取、旧搜索结果 |
| Tier 3   | 微压缩   | 空闲后缓存失效         |
| Tier 4   | 全量摘要 | 上下文接近窗口上限     |

**关键约束保留机制**：

- 约束来源：用户显式声明、`AGENTS.md`、Agent 自行识别。
- 存储：独立 `constraints.json`，不参与压缩。
- 保护：摘要 Prompt 强制要求逐条保留约束。
- 校验：压缩后检查约束 ID 是否完整，丢失则重新注入。
- 验证：对比实验，约束违反次数从 X 降至 Y（待补充）。

### 设计取舍

- **为什么不用 LangChain**：AgentExecutor 是黑盒，面试无法解释底层细节；自研可完全掌控循环、压缩和错误恢复。
- **为什么压缩阈值选 60%**：预留缓冲，避免压缩后立即再次触发；60% 是经验值，后续会用实验校准。
- **为什么记忆不用向量数据库**：项目规则是结构化文本，Markdown + 目录层级加载足够，引入向量库增加复杂度和不确定性。

## 路线图

- [x] 项目脚手架与 CLI 入口
- [ ] 交互模式（`lite-agent chat`）
- [ ] Agent Loop + 1 个工具
- [ ] 6 个核心工具 + 流式输出
- [ ] 四层上下文压缩
- [ ] 关键约束保留机制 + 对比实验
- [ ] 项目记忆 + checkpoint
- [ ] 单元测试 + GitHub Actions
- [ ] MCP 客户端
- [ ] 子 Agent 隔离

## 评测

> 待补充：Terminal-Bench 公开任务通过率；约束保留机制开启 / 关闭对比。

## 参考

- claude-code-from-scratch —— 工具系统与 MCP / 多 Agent 部分作为架构对照
- How Claude Code Works —— 源码级解析，用于理解 Agent Loop 与上下文压缩

## License

MIT，见 [LICENSE](LICENSE)。