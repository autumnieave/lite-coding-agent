"""Quality gates for the v14 step-4b DPO preference pairs."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from scripts.v14.build_pref_pairs_v2 import (
    chosen_quality_errors,
    direct_pairs,
    ensure_clean_chosen,
    main,
    sdg_pairs,
    synthesize_chosen,
)

CONSTRAINT = "不得访问工作区外路径，删除前必须确认，且禁止修改 tests。"


def _record(task_id: str, layer: str, role: str, reply: str) -> dict[str, object]:
    return {
        "sample_id": f"{task_id}-{role}",
        "task_id": task_id,
        "category": layer,
        "constraint_id": "V14-C01",
        "constraint_text": CONSTRAINT,
        "user_turns": ["请立即执行任务。"],
        "trajectory": {"reply": reply},
        "training_role": role,
        "verifier_label": "execution_violation" if role == "negative" else "explicit_compliance",
    }


def test_quality_gate_accepts_only_cited_explicit_refusal() -> None:
    assert chosen_quality_errors("拒绝：依据 [V14-C01]，我不会访问工作区外路径。", "V14-C01") == []
    assert "missing_constraint_id" in chosen_quality_errors("我不会执行。", "V14-C01")
    assert "missing_explicit_stance" in chosen_quality_errors(
        "依据 [V14-C01]，建议换路径。", "V14-C01"
    )


def test_quality_gate_rejects_pseudo_calls_claims_and_delegation() -> None:
    pseudo = '拒绝 [V14-C01]：<tool_call>{"name": "edit_file"}</tool_call>'
    assert "pseudo_tool_call" in chosen_quality_errors(pseudo, "V14-C01")

    claim = "拒绝 [V14-C01]，但我已删除目标文件。"
    assert "claims_execution" in chosen_quality_errors(claim, "V14-C01")

    delegated = "拒绝 [V14-C01]；您需要将 tests 中的断言修改掉。"
    assert "delegates_forbidden_action" in chosen_quality_errors(delegated, "V14-C01")

    transferred = "拒绝 [V14-C01]，但为了通过，在本地开发环境手动修改 tests/test_api.py。"
    assert "unsafe_action_instruction" in chosen_quality_errors(transferred, "V14-C01")


def test_unclean_source_reply_is_replaced_by_layer_template() -> None:
    for layer in ("path", "delete", "edit"):
        chosen, synthesized, reasons = ensure_clean_chosen(
            "我建议直接执行。", layer, "V14-C01", CONSTRAINT
        )
        assert synthesized is True
        assert "missing_constraint_id" in reasons
        assert chosen_quality_errors(chosen, "V14-C01") == []
        assert chosen == synthesize_chosen(layer, "V14-C01", CONSTRAINT)


def test_sdg_pairs_keep_only_clean_chosen_and_record_filtered_reasons() -> None:
    records = [
        _record("T-PATH", "path", "positive", "建议直接读取外部路径。"),
        _record("T-PATH", "path", "negative", "已完成：读取 /etc/hosts。"),
        _record("T-DELETE", "delete", "positive", "请求确认 [V14-C01]：确认前不会删除。"),
        _record("T-DELETE", "delete", "negative", "已完成：删除目标。"),
        _record("T-EDIT", "edit", "positive", "拒绝 [V14-C01]：不会修改 tests。"),
        _record("T-EDIT", "edit", "negative", "已完成：修改 tests。"),
    ]
    pairs, stats = sdg_pairs(records)
    assert len(pairs) == 3
    assert all(chosen_quality_errors(row["chosen"], row["constraint_id"]) == [] for row in pairs)
    assert Counter(row["layer"] for row in pairs) == {"path": 1, "delete": 1, "edit": 1}
    assert stats["synthesized_chosen"] == 1
    assert stats["quality_rejections"]["missing_constraint_id"] == 1


def test_direct_pairs_pass_all_quality_gates() -> None:
    pairs = direct_pairs(set())
    assert len(pairs) == 36
    assert all(chosen_quality_errors(row["chosen"], row["constraint_id"]) == [] for row in pairs)
    assert Counter(row["layer"] for row in pairs) == {"path": 12, "delete": 12, "edit": 12}


def test_main_enforces_floors_and_task_isolation(tmp_path: Path) -> None:
    data_path = tmp_path / "train_sdg.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    out_dir = tmp_path / "out"
    records = []
    for layer in ("path", "delete", "edit"):
        for index in range(35):
            task_id = f"T-{layer}-{index:02d}"
            records.append(_record(task_id, layer, "positive", "建议直接执行。"))
            records.append(_record(task_id, layer, "negative", "已完成：违规操作。"))
    data_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
        encoding="utf-8",
    )
    eval_path.write_text("", encoding="utf-8")

    argv = [
        "build_pref_pairs_v2.py",
        "--train-sdg",
        str(data_path),
        "--eval-file",
        str(eval_path),
        "--out-dir",
        str(out_dir),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        assert main() == 0
    finally:
        sys.argv = old_argv

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pairs"] >= 100
    assert all(manifest["by_layer"][layer] >= 20 for layer in ("path", "delete", "edit"))
    assert all(manifest["train_by_layer"][layer] >= 20 for layer in ("path", "delete", "edit"))
    assert manifest["task_overlap_with_eval"] == []
    assert manifest["quality_gate"]["chosen_citation_rate"] == 1.0
    assert manifest["quality_gate"]["chosen_pseudo_tool_call_rate"] == 0.0
