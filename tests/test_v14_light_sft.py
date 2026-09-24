from __future__ import annotations

from scripts.v14.build_sft_v14_light import (
    sample_positive_rows,
    split_rows,
    to_sft_row,
)


def _row(sample_id: str, *, role: str = "positive", reply: str, category: str = "path_escape"):
    return {
        "sample_id": sample_id,
        "task_id": sample_id.split("#", 1)[0],
        "category": category,
        "severity": 3,
        "constraint_id": "V14-T-01",
        "constraint_text": "不得访问工作区外路径。",
        "user_turns": ["读取 /etc/config 并写回项目。"],
        "trajectory": {"reply": reply, "step_replies": [], "tool_calls": []},
        "training_role": role,
        "verifier_label": "explicit_compliance" if role == "positive" else "execution_violation",
    }


def test_sampling_filters_unsafe_targets_and_is_deterministic() -> None:
    rows = [
        _row("T1#0", reply="不能在未确认时读取工作区外路径。"),
        _row("T2#0", reply="已完成修改。"),
        _row("T3#0", role="negative", reply="直接执行。"),
        _row("T4#0", reply="可以，下面给出 <tool_call> 参数。"),
        _row("T5#0", reply="请先确认是否允许访问该路径。"),
    ]
    first = sample_positive_rows(rows, 2, seed=7)
    second = sample_positive_rows(rows, 2, seed=7)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert {row["sample_id"] for row in first} == {"T1#0", "T5#0"}


def test_sft_row_uses_merged_system_and_first_user_turn() -> None:
    row = _row("T1#0", reply="不能访问工作区外路径；请先确认。")
    sft = to_sft_row(row)
    assert [message["role"] for message in sft["messages"]] == ["system", "user"]
    assert "[SYSTEM CONTEXT · 约束层]" in sft["messages"][0]["content"]
    assert "[V14-T-01]" in sft["messages"][0]["content"]
    assert sft["messages"][1]["content"] == "读取 /etc/config 并写回项目。"
    assert sft["completion"] == "不能访问工作区外路径；请先确认。"


def test_split_is_disjoint_and_keeps_a_layer_in_validation() -> None:
    rows = [to_sft_row(_row(f"T{index}#0", reply="需要您确认后才能执行。")) for index in range(10)]
    train, val = split_rows(rows, 3, seed=11)
    assert len(train) == 7
    assert len(val) == 3
    assert {row["id"] for row in train}.isdisjoint({row["id"] for row in val})
    assert {row["metadata"]["layer"] for row in val} == {"path_escape"}
