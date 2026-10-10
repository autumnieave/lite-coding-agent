"""Apply the v15 2-6/8 middle-state gate to seed precheck rollouts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(payload: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    rows_by_id = {str(row["task_id"]): row for row in summary.get("task_rows", [])}
    candidates: list[dict[str, Any]] = []
    for candidate in payload.get("candidates", []):
        task_id = str(candidate["task_id"])
        row = rows_by_id.get(task_id)
        if row is None:
            candidates.append(
                {**candidate, "precheck": {"status": "missing_summary", "rollouts": 0}}
            )
            continue
        classes = row.get("classifications") or {}
        successes = int(classes.get("explicit_compliance", 0))
        rollouts = int(row.get("valid_rollouts") or 0)
        middle = rollouts >= 8 and 2 <= successes <= 6
        candidates.append(
            {
                **candidate,
                "precheck": {
                    "status": "completed",
                    "rollouts": rollouts,
                    "explicit_compliance": successes,
                    "middle_state_2_6": middle,
                    "retained": middle,
                    "classifications": classes,
                },
            }
        )
    retained = [row for row in candidates if row.get("precheck", {}).get("retained")]
    by_layer = Counter(str(row["layer"]) for row in retained)
    return {
        **payload,
        "status": "precheck_completed",
        "validated_middle_state_count": len(retained),
        "pending_gpu_precheck": 0,
        "retained_by_layer": dict(sorted(by_layer.items())),
        "candidates": candidates,
        "retained_task_ids": [row["task_id"] for row in retained],
        "run_status": {
            "status": summary.get("status"),
            "rollouts": summary.get("rollouts"),
            "valid_rollouts": summary.get("valid_rollouts"),
            "errors": summary.get("errors"),
        },
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# v15 中间态种子 8-rollout 预检",
        "",
        f"- 有效中间态：{payload['validated_middle_state_count']}/"
        f"{payload['candidate_count']}（2–6/8 explicit_compliance）",
        f"- 保留层分布：{payload['retained_by_layer']}",
        "",
        "| 任务 | 层 | explicit_compliance | 保留 | 判定分布 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in payload["candidates"]:
        pc = row.get("precheck", {})
        lines.append(
            f"| {row['task_id']} | {row['layer']} | "
            f"{pc.get('explicit_compliance', 'NA')}/{pc.get('rollouts', 0)} | "
            f"{'是' if pc.get('retained') else '否'} | {pc.get('classifications', {})} |"
        )
    lines += [
        "",
        "未保留任务不进入 v15 SDG 增量或第二轮 GRPO；不能把候选数写成已验证中间态数。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="analyze v15 seed precheck")
    parser.add_argument(
        "--seeds", type=Path, default=Path("private/v15/middle_state_seeds_v15.json")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("private/v15/seed_precheck/summary.json")
    )
    parser.add_argument(
        "--out-json", type=Path, default=Path("private/v15/seed_precheck_analysis.json")
    )
    parser.add_argument(
        "--out-md", type=Path, default=Path("private/v15/seed_precheck_analysis.md")
    )
    args = parser.parse_args()

    payload = analyze(load_json(args.seeds), load_json(args.summary))
    args.out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(payload, args.out_md)
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in {"candidates", "retained_task_ids"}
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
