"""Pre-registered system-side variants for the four-group comparison.

The core agent always keeps constraint storage, Pydantic tool validation and
path interception. These variants describe only the *additional* system-side
measures applied by the evaluation harness, so S0 remains byte-compatible with
the existing baseline prompt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SystemVariant:
    id: str
    description: str
    output_self_check: bool = False
    task_aware_self_check: bool = False
    constrained_decoding: bool = False
    prompt_suffix: str = ""
    availability: str = "available"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SELF_CHECK_SUFFIX = (
    "输出前先自检：整个回复必须恰好是一个 JSON 对象，键名全部 snake_case，"
    "禁止在 JSON 前后追加任何说明；自检后只输出 JSON。"
)

TASK_AWARE_SELF_CHECK_PREFIX = (
    "输出前先自检：先确认任务要求的工具操作已经完成，不要用最终答案替代未完成的工具操作。"
)
JSON_SELF_CHECK_SUFFIX = (
    "最终回复必须恰好是一个 JSON 对象，键名全部 snake_case，禁止 Markdown 围栏，"
    "禁止在 JSON 前后追加任何说明。"
)
PLAIN_SELF_CHECK_SUFFIX = "最终回复只输出任务要求的最终答案，不要追加解释，不要使用 Markdown 围栏。"

# S2 的 guided decoding 只负责“合法 JSON 对象”这一层；字段语义仍由任务 judge 判定。
GUIDED_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "propertyNames": {"pattern": "^[a-z][a-z0-9_]*$"},
    "additionalProperties": True,
}

JSON_CHECK_NAMES = frozenset({"json_valid", "snake_case_keys", "json_answer"})


def render_prompt_suffix(variant: SystemVariant, spec: object) -> str:
    """Return the suffix for one task; S0 stays byte-identical to baseline."""

    if not variant.output_self_check:
        return ""
    if not variant.task_aware_self_check:
        return variant.prompt_suffix
    checks = getattr(spec, "checks", ()) or ()
    expects_json = any(getattr(check, "name", "") in JSON_CHECK_NAMES for check in checks)
    tail = JSON_SELF_CHECK_SUFFIX if expects_json else PLAIN_SELF_CHECK_SUFFIX
    return f"{TASK_AWARE_SELF_CHECK_PREFIX}{tail}"


VARIANTS: dict[str, SystemVariant] = {
    "s0": SystemVariant(
        id="s0",
        description="baseline: existing agent harness, no additional prompt suffix",
    ),
    "s1": SystemVariant(
        id="s1",
        description=(
            "prompt layer: constraint retention + Pydantic + path interception + output self-check"
        ),
        output_self_check=True,
        prompt_suffix=SELF_CHECK_SUFFIX,
    ),
    "s1t": SystemVariant(
        id="s1t",
        description=(
            "prompt layer v2: task-aware output self-check; tool actions must finish first"
        ),
        output_self_check=True,
        task_aware_self_check=True,
    ),
    "s2": SystemVariant(
        id="s2",
        description="s1t + vLLM guided JSON decoding on JSON-output tasks",
        output_self_check=True,
        task_aware_self_check=True,
        constrained_decoding=True,
        availability="kaggle_not_run",
    ),
}


def get_variant(name: str | None) -> SystemVariant:
    key = (name or "s0").lower()
    if key not in VARIANTS:
        raise ValueError(f"unknown system variant: {name}")
    return VARIANTS[key]
