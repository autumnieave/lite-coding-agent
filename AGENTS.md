# AGENTS.md

本文件为在此仓库中工作的 coding agent 提供约定。

## 环境

- 目标 Python 版本：3.11
- 虚拟环境位于仓库根目录的 `.venv`，使用其中的解释器：
  - PowerShell：`.\.venv\Scripts\python.exe`
  - 不要使用系统全局的 `python`（当前指向 3.7.9）

## Shell

- 本项目默认在 Windows + PowerShell 5.1 下开发
- 写脚本时使用 PowerShell 语法，不要用 bash 专有语法（`&&`、`source`、`mkdir -p` 等）
- 写文件时注意编码：PowerShell 5.1 的 `Set-Content -Encoding utf8` 会写入 BOM，
  需要无 BOM 时使用 `[System.IO.File]::WriteAllText(path, text, New-Object System.Text.UTF8Encoding(\False))`

## 代码约定

- 源码放在 `src/agent/` 下，采用 src-layout，通过 `pyproject.toml` 配置打包
- 新增依赖写入 `pyproject.toml` 的 `dependencies`，并同步 `requirements.txt`
- 测试放在 `tests/` 下，使用 pytest
- 提交信息使用中文或英文均可，建议遵循 Conventional Commits 前缀