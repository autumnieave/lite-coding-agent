"""工具基类。

统一负责三件事：把模型的原始参数解析成 Pydantic 模型、执行工具、
把任何失败转成 `ToolResult.failure` 而不是抛异常——工具失败必须回填给模型，
不能打断 Agent Loop。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel, ValidationError


class ToolError(Exception):
    """工具可预期的失败（文件不存在、参数越界等），会被转成错误结果回填。"""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """工具执行结果。`ok=False` 时 `content` 是给模型看的错误说明。"""

    ok: bool
    content: str

    @classmethod
    def success(cls, content: str) -> ToolResult:
        return cls(ok=True, content=content)

    @classmethod
    def failure(cls, message: str) -> ToolResult:
        return cls(ok=False, content=message)


def resolve_path(root: Path, raw: str) -> Path:
    """解析路径并确保不逃出工作区根目录，越界抛 `ToolError`。"""
    if not raw or not raw.strip():
        raise ToolError("路径不能为空")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ToolError(f"路径越界，只能访问工作区内的文件：{raw}")
    return resolved


def _describe_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ())) or "参数"
        parts.append(f"{location}: {error.get('msg', '校验失败')}")
    return "；".join(parts)


class Tool(ABC):
    """工具接口。子类只需声明元信息并实现 `execute`。"""

    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]

    def spec(self) -> dict[str, Any]:
        """OpenAI 兼容的 function 定义，供 Provider 传给模型。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }

    async def run(self, raw_arguments: str) -> ToolResult:
        """解析参数并执行。任何失败都返回 `ToolResult.failure`，不抛异常。"""
        try:
            payload = json.loads(raw_arguments) if raw_arguments and raw_arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return ToolResult.failure(f"参数不是合法 JSON：{exc}")

        if not isinstance(payload, dict):
            return ToolResult.failure("参数必须是 JSON 对象")

        try:
            args = self.args_model.model_validate(payload)
        except ValidationError as exc:
            return ToolResult.failure(f"参数校验失败：{_describe_validation_error(exc)}")

        try:
            return await self.execute(args)
        except ToolError as exc:
            return ToolResult.failure(str(exc))
        except Exception as exc:  # noqa: BLE001 - 兜底，避免单个工具异常打断整个循环
            return ToolResult.failure(f"工具执行异常：{type(exc).__name__}: {exc}")

    @abstractmethod
    async def execute(self, args: Any) -> ToolResult:
        """执行工具。失败时抛 `ToolError` 或直接返回 failure。"""


class ToolLike(Protocol):
    """工具的最小接口。

    `ToolRegistry` 只依赖这三个成员，所以外部工具（如 MCP）不必继承 `Tool`、
    也不必提供 Pydantic 参数模型就能注册进来——它们的参数由远端自己校验。
    """

    name: str

    def spec(self) -> dict[str, Any]:
        """返回传给模型的工具定义。"""

    async def run(self, raw_arguments: str) -> ToolResult:
        """执行工具。失败返回 `ToolResult.failure`，不抛异常。"""
