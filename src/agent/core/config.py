"""配置加载：把 .env 里的键值注入进程环境变量。

运行时不引入 python-dotenv（见 AGENTS.md 约束 C2：只允许标准库与 LLM 官方 SDK），
这里用标准库实现最小解析，容忍三种写法：

    KEY=VALUE
    export KEY=VALUE
    $env:KEY=VALUE      # PowerShell 习惯写法

已存在于环境中的变量不会被 .env 覆盖，这样可以用命令行临时覆盖配置。
"""

from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from pathlib import Path

DEFAULT_ENV_FILENAME = ".env"
ENV_SEARCH_MAX_DEPTH = 5

_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PREFIXES = ("export ", "$env:")


def find_env_file(start: Path, *, max_depth: int = ENV_SEARCH_MAX_DEPTH) -> Path | None:
    """从 start 目录起向上逐级查找 .env，最多上溯 max_depth 层。"""
    current = start.resolve()
    for _ in range(max_depth + 1):
        candidate = current / DEFAULT_ENV_FILENAME
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def _clean_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    for marker in (" #", "\t#"):
        index = value.find(marker)
        if index != -1:
            value = value[:index]
    return value.strip()


def parse_env_text(text: str) -> dict[str, str]:
    """解析 .env 文本。无法识别的行会被忽略，不抛异常。"""
    parsed: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        for prefix in _PREFIXES:
            if lowered.startswith(prefix):
                line = line[len(prefix) :].strip()
                break
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not _KEY_PATTERN.match(key):
            continue
        parsed[key] = _clean_value(value)
    return parsed


def load_env_file(path: Path, *, environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """把 .env 注入 environ（默认 os.environ），返回实际注入的键值。

    已存在的键不会被覆盖。读取失败时抛 OSError，由调用方决定如何提示。
    """
    target = os.environ if environ is None else environ
    text = path.read_text(encoding="utf-8")

    injected: dict[str, str] = {}
    for key, value in parse_env_text(text).items():
        if key in target:
            continue
        target[key] = value
        injected[key] = value
    return injected
