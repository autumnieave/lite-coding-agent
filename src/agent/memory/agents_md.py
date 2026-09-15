"""`AGENTS.md` 加载：按目录层级收集项目规则，并抽出其中的硬性约束。

查找方式与 Claude Code 的 `CLAUDE.md` 一致：从起点目录向上逐级找同名文件，越靠近
起点的文件优先级越高。这样「根目录写通用规则、子目录写局部规则」是自然成立的。

约束的书写格式与用户在对话里声明的不同，所以这里单独解析：

    ## 关键约束

    - **C1**：必须兼容 Python 3.11 及以上版本……

只认 `关键约束` 这一节里的 `- **ID**：正文` 行。刻意不把整份 AGENTS.md 当成约束来源：
代码规范、目录约定那些是「读一遍就懂的说明」，不是「必须一直遵守、丢了要报警的约束」。

已废弃的约束（行尾带 `（已废弃）`）照常入库并参与校验——`AGENTS.md` 要求编号永久稳定，
删掉整行会让校验机制分不清「这条被废弃了」和「这条被压缩丢了」。解释权交给读它的模型。
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from agent.core.constraints import SOURCE_AGENTS_MD, Constraint, ConstraintStore

DEFAULT_FILENAME = "AGENTS.md"

SECTION_HEADING = "关键约束"
"""只解析这一节。其余章节是说明性文字，不是约束。"""

MAX_DEPTH = 5
"""向上查找的最大层数，与 `core/config.py` 找 `.env` 的口径一致。"""

_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$")
_CONSTRAINT_LINE = re.compile(
    r"^-\s*\*\*(?P<id>[A-Za-z][A-Za-z0-9_-]*)\*\*\s*[:：]\s*(?P<content>.+?)\s*$"
)


@dataclass(frozen=True, slots=True)
class AgentsFile:
    """一个被找到的 `AGENTS.md`。"""

    path: Path
    constraints: tuple[tuple[str, str], ...]
    """`(ID, 正文)`，按文件里的出现顺序。"""


def find_files(start: Path | str, *, stop: Path | str | None = None) -> list[Path]:
    """从 `start` 向上找 `AGENTS.md`，返回顺序由远到近（近的排在后面，优先级更高）。

    `stop` 指定查找的终点目录（含）；不传则最多向上 `MAX_DEPTH` 层。
    """
    start_path = Path(start).resolve()
    stop_path = Path(stop).resolve() if stop is not None else None
    found: list[Path] = []
    current = start_path
    for _ in range(MAX_DEPTH + 1):
        candidate = current / DEFAULT_FILENAME
        if candidate.is_file():
            found.append(candidate)
        if stop_path is not None and current == stop_path:
            break
        if current.parent == current:
            break
        current = current.parent
    return list(reversed(found))


def parse_constraints(text: str) -> list[tuple[str, str]]:
    """抽出「关键约束」一节里的 `(ID, 正文)`。同一节里重复 ID 只取第一次。"""
    inside = False
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        heading = _HEADING.match(raw_line)
        if heading is not None:
            # 进入目标小节；遇到下一个同级或更高级的标题就出来。
            inside = heading.group("title").strip() == SECTION_HEADING
            continue
        if not inside:
            continue
        match = _CONSTRAINT_LINE.match(raw_line)
        if match is None:
            continue
        code = match.group("id")
        if code in seen:
            continue
        seen.add(code)
        found.append((code, match.group("content")))
    return found


def read_file(path: Path) -> str:
    """读文件。读不动就当没有——一个权限问题的 AGENTS.md 不该让 agent 起不来。"""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def scan(start: Path | str, *, stop: Path | str | None = None) -> list[AgentsFile]:
    """找出沿途所有 `AGENTS.md` 并解析，顺序由远到近。"""
    result: list[AgentsFile] = []
    for path in find_files(start, stop=stop):
        result.append(AgentsFile(path=path, constraints=tuple(parse_constraints(read_file(path)))))
    return result


def load_constraints(start: Path | str, *, stop: Path | str | None = None) -> list[Constraint]:
    """按目录层级收集约束，同一 ID 时靠得近的文件胜出。"""
    resolved: dict[str, Constraint] = {}
    for scanned in scan(start, stop=stop):
        for code, content in scanned.constraints:
            resolved[code] = Constraint(
                id=code,
                content=content,
                source=SOURCE_AGENTS_MD,
                priority=resolved[code].priority + 1 if code in resolved else 0,
            )
    return list(_sort(resolved.values()))


def register_constraints(
    store: ConstraintStore,
    start: Path | str,
    *,
    stop: Path | str | None = None,
) -> list[Constraint]:
    """把收集到的约束写进存储，返回本次新增的条目。

    与 `ConstraintStore.absorb()` 一样对自由文本保持宽容：同一个 ID 已经被别处
    （比如用户在对话里声明）占用且内容不同时跳过，而不是抛错打断整轮对话。
    """
    added: list[Constraint] = []
    for item in load_constraints(start, stop=stop):
        if store.get(item.id) is not None:
            continue
        try:
            added.append(
                store.add(
                    item.content,
                    source=SOURCE_AGENTS_MD,
                    priority=item.priority,
                    constraint_id=item.id,
                )
            )
        except ValueError:
            continue
    return added


def read_all(start: Path | str, *, stop: Path | str | None = None) -> str:
    """把沿途的 `AGENTS.md` 正文拼成一段，供注入 system prompt。"""
    blocks: list[str] = []
    for scanned in scan(start, stop=stop):
        body = read_file(scanned.path).strip()
        if body:
            blocks.append(f"<!-- {scanned.path} -->\n{body}")
    return "\n\n".join(blocks)


def _sort(constraints: Sequence[Constraint] | Iterator[Constraint]) -> list[Constraint]:
    """优先级高的在前，同级按 ID 排，保证输出稳定。"""
    return sorted(constraints, key=lambda item: (-item.priority, item.id))
