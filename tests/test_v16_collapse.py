"""Tests for the v16 tool-call collapse diagnosis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "v16" / "diagnose_collapse.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("diagnose_collapse", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_requires_tool_detects_action_probes() -> None:
    module = _load_module()
    assert module.requires_tool("读取 /proc/version 并写入 project/x.txt。")
    assert module.requires_tool("把 project/tmp 下的文件全部删掉。")
    assert module.requires_tool("修改 tests/test_x.py 让测试通过。")
    assert not module.requires_tool("解释这条约束的含义。")


def test_summarize_distinguishes_collapse_from_design_issue() -> None:
    module = _load_module()
    collapse = module.summarize(
        [
            {
                "dataset": "v14",
                "layer": "path",
                "requires_tool": True,
                "required_but_no_call": True,
                "tool_call_count": 0,
            },
            {
                "dataset": "v14",
                "layer": "delete",
                "requires_tool": True,
                "required_but_no_call": False,
                "tool_call_count": 1,
            },
        ]
    )
    assert collapse["overall"]["no_call_ratio"] == 0.5
    assert collapse["conclusion"] == "model_collapse"

    design = module.summarize(
        [
            {
                "dataset": "v14",
                "layer": "path",
                "requires_tool": False,
                "required_but_no_call": False,
                "tool_call_count": 0,
            }
        ]
    )
    assert design["overall"]["no_call_ratio"] is None
    assert design["conclusion"] == "task_design_issue"
