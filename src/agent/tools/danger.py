"""危险命令识别：给 bash 工具做执行前拦截。

规则对照 claude-code-from-scratch 的 is_dangerous，并补齐 Windows 侧的
删除与进程终止类命令。命中不代表禁止，而是要求显式确认。
"""

from __future__ import annotations

import re

DANGEROUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\s", re.IGNORECASE),
    re.compile(r"\bgit\s+(push|reset|clean|checkout\s+\.)", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\s", re.IGNORECASE),
    re.compile(r">\s*/dev/"),
    re.compile(r"\bkill\b", re.IGNORECASE),
    re.compile(r"\bpkill\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\bdel\s", re.IGNORECASE),
    re.compile(r"\brmdir\s", re.IGNORECASE),
    re.compile(r"\bformat\s", re.IGNORECASE),
    re.compile(r"\btaskkill\s", re.IGNORECASE),
    re.compile(r"\bRemove-Item\s", re.IGNORECASE),
    re.compile(r"\bStop-Process\s", re.IGNORECASE),
)


def is_dangerous(command: str) -> bool:
    """命令是否命中危险模式。"""
    return any(pattern.search(command) for pattern in DANGEROUS_PATTERNS)
