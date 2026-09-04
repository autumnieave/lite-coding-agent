"""每轮约束注入：从 `ConstraintStore` 检索相关约束，注入独立的 System Context 层。

与压缩链路**分开**：

- 摘要只压任务历史，不再抄录约束；
- 约束不进摘要，所以摘要丢约束不会触发 `ensure_constraints` 补录；
- 模型每一轮看到的约束来自本模块生成的独立 system 层，按当前任务上下文检索。

token 预算按「中文按 1 字 1 token、其余按 4 字符 1 token」估算——仓库通用的
`estimate_tokens` 是英文口径（4 字符 1 token），直接用来卡中文约束会放大约 4 倍。
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from agent.core.constraints import Constraint, ConstraintStore
from agent.core.llm import system_message

CONSTRAINT_LAYER_HEADING = "[SYSTEM CONTEXT · 约束层]"

_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_WHITESPACE = re.compile(r"\s+")


def estimate_injection_tokens(text: str) -> int:
    """中文按 1 字 1 token、其余按 4 字符 1 token 估算。"""

    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    rest = len(_CJK.sub("", text))
    return cjk + math.ceil(rest / 4)


@dataclass(frozen=True, slots=True)
class InjectionConfig:
    """注入预算。默认 200 token / 最多 10 条。"""

    max_tokens: int = 200
    max_constraints: int = 10
    context_messages: int = 6


@dataclass(frozen=True, slots=True)
class Selection:
    """一次检索的结果。`tokens` 是渲染后整层的估算 token。"""

    items: tuple[Constraint, ...]
    tokens: int
    dropped: int


def _bigrams(text: str) -> set[str]:
    cleaned = _WHITESPACE.sub("", text)
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[index : index + 2] for index in range(len(cleaned) - 1)}


class ConstraintInjector:
    """按上下文相关性挑选约束，并渲染成独立的 system 层。

    `is_behavior` 标记行为类约束：它们排在最前，保证「不确定时最该看的那几条」
    不会被格式类约束挤掉预算。不传时所有约束同权。
    """

    def __init__(
        self,
        store: ConstraintStore,
        *,
        config: InjectionConfig | None = None,
        is_behavior: Callable[[Constraint], bool] | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self._store = store
        self._config = config or InjectionConfig()
        if self._config.max_tokens <= 0:
            raise ValueError("max_tokens 必须为正数")
        if self._config.max_constraints <= 0:
            raise ValueError("max_constraints 必须为正数")
        self._is_behavior = is_behavior or (lambda item: False)
        self._tokens = token_counter or estimate_injection_tokens

    @property
    def config(self) -> InjectionConfig:
        return self._config

    def render(self, items: Sequence[Constraint]) -> str:
        if not items:
            return ""
        lines = [CONSTRAINT_LAYER_HEADING]
        lines.extend(f"- [{item.id}] {item.content}" for item in items)
        return "\n".join(lines)

    def select(self, context: Sequence[Mapping[str, object]]) -> Selection:
        items = self._store.get_all()
        if not items:
            return Selection(items=(), tokens=0, dropped=0)
        text = self._context_text(context)
        scores = {item.id: self._score(item, text) for item in items}
        ordered = sorted(
            items,
            key=lambda item: (
                0 if self._is_behavior(item) else 1,
                -scores[item.id],
                -item.priority,
                item.id,
            ),
        )
        chosen: list[Constraint] = []
        for item in ordered:
            if len(chosen) >= self._config.max_constraints:
                break
            candidate = [*chosen, item]
            if self._tokens(self.render(candidate)) > self._config.max_tokens:
                continue
            chosen.append(item)
        rendered = self.render(chosen)
        return Selection(
            items=tuple(chosen),
            tokens=self._tokens(rendered),
            dropped=len(items) - len(chosen),
        )

    def inject(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        merge_first_system: bool = False,
    ) -> list[dict[str, object]]:
        """返回带约束层的新消息列表；没有可注入内容时只去掉旧层。

        ``merge_first_system`` 把约束层合并进第一条 system 消息，而不是新插一条
        system 消息。部分 chat template 只把首条 system 当作真正的系统提示，
        这种模式用于验证注入位置的影响。
        """

        stripped = [dict(message) for message in messages if not self._is_layer(message)]
        selection = self.select(stripped)
        layer = self.render(selection.items)
        if not layer:
            return stripped
        if merge_first_system and stripped and stripped[0].get("role") == "system":
            first = dict(stripped[0])
            base = str(first.get("content") or "")
            marker = f"\n\n{CONSTRAINT_LAYER_HEADING}"
            if marker in base:
                base = base.split(marker, 1)[0]
            first["content"] = f"{base}\n\n{layer}" if base else layer
            return [first, *stripped[1:]]
        index = 0
        while index < len(stripped) and stripped[index].get("role") == "system":
            index += 1
        return [*stripped[:index], system_message(layer), *stripped[index:]]

    def _context_text(self, context: Sequence[Mapping[str, object]]) -> str:
        window = list(context)[-self._config.context_messages :]
        parts: list[str] = []
        for message in window:
            content = message.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
        return "\n".join(parts)

    def _score(self, item: Constraint, context_text: str) -> float:
        if not context_text:
            return 0.0
        mine = _bigrams(item.content)
        if not mine:
            return 0.0
        return len(mine & _bigrams(context_text)) / len(mine)

    @staticmethod
    def _is_layer(message: Mapping[str, object]) -> bool:
        return message.get("role") == "system" and str(message.get("content") or "").startswith(
            CONSTRAINT_LAYER_HEADING
        )
