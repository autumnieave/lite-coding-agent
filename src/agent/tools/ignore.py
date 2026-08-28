"""遍历目录时需要跳过的目录名。

`.git`、`.venv`、`__pycache__` 这类目录既不属于项目内容，体量又大，
一旦被列目录或搜索命中就会白白吃掉上下文窗口。list_dir 与 grep 共用这份清单。
"""

from __future__ import annotations

IGNORED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".idea",
        "node_modules",
    }
)
