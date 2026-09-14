# lite-coding-agent

一个轻量级 coding agent 的最小实现骨架。

## 环境

- Python 3.11
- 虚拟环境：`.venv`（已创建）

## 快速开始

```powershell
# 激活虚拟环境（PowerShell）
.\.venv\Scripts\Activate.ps1
# 或直接调用解释器
.\.venv\Scripts\python.exe -m pip install -e .
```

## 目录结构

```
src/agent/
  core/     # 主循环、LLM 客户端、消息与状态管理
  tools/    # 工具定义与执行器（读写文件、执行命令等）
  memory/   # 会话历史与上下文压缩
  cli/      # 命令行入口与交互渲染
tests/      # 测试
docs/       # 设计文档
```