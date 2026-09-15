"""约束存储：把「必须一直遵守」的规则从对话历史里拎出来单独存放。

约束是长会话里最容易静默消失的东西：一旦声明它的那几轮被摘要掉，模型就再也
不知道有这条规则（ADR-004）。所以约束不放在消息里，而是独立存盘，并在压缩
前后各做一次处理——压缩前交给摘要器，压缩后校验摘要有没有漏，漏了就补回去。

存储格式（`constraints.json`）：:

    {"constraints": [
      {"id": "R3MJUD", "content": "输出必须是合法 JSON",
       "source": "user", "priority": 0, "created_at": "2026-09-14T12:00:00+00:00"}
    ]}

`id` 对用户声明的约束直接采用声明里的「代号」，这样校验摘要时人也能看懂。
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOURCE_USER = "user"
SOURCE_AGENTS_MD = "agents_md"
SOURCE_AGENT = "agent"

SOURCES = (SOURCE_USER, SOURCE_AGENTS_MD, SOURCE_AGENT)
"""约束来源：用户显式声明 / `AGENTS.md` / 模型自行识别。"""

SOURCE_LABELS = {
    SOURCE_USER: "用户声明",
    SOURCE_AGENTS_MD: "AGENTS.md",
    SOURCE_AGENT: "模型识别",
}

CONSTRAINT_MARKER = "[CONSTRAINT]"
"""用户声明约束时使用的标记，例如 `[CONSTRAINT] 代号 R3MJUD：输出必须是 JSON`。"""

DEFAULT_FILENAME = "constraints.json"

SYSTEM_PROMPT_HEADING = (
    "# 必须遵守的约束\n"
    "以下条目来自项目规则（AGENTS.md）与用户显式声明，任何时候都不得违反；"
    "动手之前先对照检查一遍。"
)
"""注入 system prompt 时用的标题，与压缩摘要里的清单区分开（那是给摘要器的）。"""

# 分隔符一律用「水平空白」：`\s` 会把换行也吃掉，于是「[CONSTRAINT] 代号 B2：」
# 这种内容为空的行会顺手吞掉下一行，凭空多出一条约束。
_DECLARATION = re.compile(
    r"\[CONSTRAINT\][ \t]*代号[ \t]*([A-Za-z0-9_-]+)[ \t]*[：:][ \t]*([^\n]+)",
)


@dataclass(frozen=True, slots=True)
class Constraint:
    """一条约束。`id` 是对外校验用的稳定标识。"""

    id: str
    content: str
    source: str = SOURCE_USER
    priority: int = 0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "priority": self.priority,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Constraint:
        return cls(
            id=str(payload["id"]),
            content=str(payload["content"]),
            source=str(payload.get("source") or SOURCE_USER),
            priority=int(payload.get("priority") or 0),
            created_at=str(payload.get("created_at") or ""),
        )

    def render(self) -> str:
        """渲染成清单里的一行，供注入 prompt 使用。"""
        return f"- [{self.id}] {self.content}"


@dataclass(frozen=True, slots=True)
class Verification:
    """一次「摘要里还剩下哪些约束」的校验结果。"""

    present: tuple[Constraint, ...]
    missing: tuple[Constraint, ...]

    @property
    def ok(self) -> bool:
        return not self.missing


def extract_declarations(text: str) -> list[tuple[str, str]]:
    """从文本里抽取 `[CONSTRAINT] 代号 X：内容` 形式的声明。

    返回 `(代号, 内容)` 列表，按出现顺序去重（同一代号只取第一次）。
    """
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _DECLARATION.finditer(text):
        code = match.group(1)
        content = match.group(2).strip()
        if not code or not content or code in seen:
            continue
        seen.add(code)
        found.append((code, content))
    return found


def message_text(message: Any) -> str:
    """取出消息里的可读文本。只处理字符串 content，其它一律忽略。"""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


class ConstraintStore:
    """约束的加载、写入、校验。

    `path` 为 None 时只在内存里工作（测试与临时实验用）。`clock` 与 `id_factory`
    可注入，方便断言 `created_at` 与自动生成的 id。
    """

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._clock = clock or (lambda: datetime.now(UTC))
        self._id_factory = id_factory or (lambda: f"c-{secrets.token_hex(3)}")
        self._items: dict[str, Constraint] = {}

    # ---------- 读 ----------

    @property
    def path(self) -> Path | None:
        return self._path

    def load(self) -> None:
        """从磁盘读入。文件不存在、为空、内容损坏时都视为「还没有约束」。"""
        if self._path is None or not self._path.is_file():
            return
        raw = self._path.read_text(encoding="utf-8").strip()
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        items = payload.get("constraints") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return
        for entry in items:
            if not isinstance(entry, dict) or "id" not in entry or "content" not in entry:
                continue
            item = Constraint.from_dict(entry)
            self._items[item.id] = item

    def save(self) -> None:
        """写回磁盘。没有 path 时什么都不做。"""
        if self._path is None:
            return
        payload = {"constraints": [item.to_dict() for item in self.get_all()]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # ---------- 写 ----------

    def add(
        self,
        content: str,
        *,
        source: str = SOURCE_USER,
        priority: int = 0,
        constraint_id: str | None = None,
    ) -> Constraint:
        """新增一条约束。

        同一个 id 重复添加时：内容一致视为幂等（返回已有那条），内容不一致直接报错——
        静默覆盖会让人以为约束改了，实际两条都在用。
        """
        if source not in SOURCES:
            raise ValueError(f"未知的来源：{source}")
        text = content.strip()
        if not text:
            raise ValueError("约束内容不能为空")

        cid = constraint_id or self._id_factory()
        existing = self._items.get(cid)
        if existing is not None:
            if existing.content == text:
                return existing
            raise ValueError(f"约束 id 重复且内容不同：{cid}")

        item = Constraint(
            id=cid,
            content=text,
            source=source,
            priority=priority,
            created_at=self._clock().isoformat(),
        )
        self._items[cid] = item
        return item

    def set(self, item: Constraint) -> Constraint:
        """按 id 直接写入，已存在则整条覆盖。返回写入的条目。

        与 `add()` 不同：这里不做幂等或冲突判断，调用方已经知道要什么。
        只用于「外部文件是唯一真源」的场景——`AGENTS.md` 的 C1–C5 改了就得以文件为准，
        否则库里的旧正文会把新正文永远挡在外面（见 `memory/agents_md.py`）。
        """
        self._items[item.id] = item
        return item

    def absorb(self, messages: Iterable[Any], *, source: str = SOURCE_USER) -> list[Constraint]:
        """扫描消息，把其中的 `[CONSTRAINT]` 声明收进来。返回本次新增的约束。

        只认用户消息：助手消息里的同款文本多半是复述或举例，算成约束会误伤。
        同一 id 已经存在时按「先到先得」跳过——扫的是自由文本，重申同一条约束是常态，
        为此抛错会打断整轮对话。`add()` 保持严格，那里是显式写入。
        """
        added: list[Constraint] = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") != "user":
                continue
            for code, content in extract_declarations(message_text(message)):
                if code in self._items:
                    continue
                added.append(self.add(content, source=source, constraint_id=code))
        return added

    # ---------- 查询与校验 ----------

    def get_all(self) -> tuple[Constraint, ...]:
        """按优先级从高到低、同级按 id 排序，保证输出稳定。"""
        return tuple(sorted(self._items.values(), key=lambda item: (-item.priority, item.id)))

    def get(self, constraint_id: str) -> Constraint | None:
        return self._items.get(constraint_id)

    def verify(self, text: str) -> Verification:
        """检查文本（通常是摘要）里是否还包含每条约束的 id。"""
        present: list[Constraint] = []
        missing: list[Constraint] = []
        for item in self.get_all():
            (present if item.id in text else missing).append(item)
        return Verification(present=tuple(present), missing=tuple(missing))

    def render(self) -> str:
        """渲染成注入用的清单文本。没有约束时返回空串。"""
        return "\n".join(item.render() for item in self.get_all())

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Constraint]:
        return iter(self.get_all())
