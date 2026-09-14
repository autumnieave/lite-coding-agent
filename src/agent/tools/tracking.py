"""文件读取追踪：为 `edit_file` 的 read-before-edit 与 mtime 防护提供依据。

`read_file` 每次成功读取后记录文件快照；`edit_file` 写入前核对快照，
避免模型基于已过期的内容做替换。两者必须共享同一个 `FileTracker`
实例才有意义，装配工作在 `build_default_registry` 里完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FileState:
    """一次读取时文件的状态快照。"""

    mtime_ns: int
    size: int

    @classmethod
    def from_path(cls, path: Path) -> FileState:
        """把磁盘上文件的当前状态取成快照。"""
        stat = path.stat()
        return cls(mtime_ns=stat.st_mtime_ns, size=stat.st_size)


class FileTracker:
    """记录本次会话中被读取过的文件，供 `edit_file` 判断是否可写。"""

    def __init__(self) -> None:
        self._states: dict[Path, FileState] = {}

    def record(self, path: Path) -> None:
        """记录刚读取或刚写入的文件状态。"""
        self._states[path] = FileState.from_path(path)

    def last_read(self, path: Path) -> FileState | None:
        """返回上次读取时的快照；从未读过则为 `None`。"""
        return self._states.get(path)
