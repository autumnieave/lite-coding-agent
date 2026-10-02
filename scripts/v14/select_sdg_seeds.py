"""Select 24 SDG seed tasks and shard them for two workers."""

from __future__ import annotations

import json
from pathlib import Path

TASKS = Path("private/v14/step0_precheck/tasks_60.jsonl")
ANALYSIS = Path("private/v14/step0_precheck/analysis60/middle_state_tasks.json")
OUT = Path("private/v14/step1_sdg")

MIDDLE_EXTRA = [
    "V10-PACT-EXPORT-A",
    "V10-PACT-PROCUREMENT-A",
    "V10-PACT-GOV-A",
    "V10-PACT-ADVERTISING-A",
    "V10-PACT-PHARMA-A",
    "V14-PACT-EXPORT-B",
    "V14-PACT-HR-C",
    "V14-PACT-PROCUREMENT-B",
    "V14-PACT-CUSTOMER-D",
    "V14-PACT-GOV-B",
    "V14-PACT-HEALTHCARE-D",
    "V14-PACT-PHARMA-D",
    "V14X-PATH-01",
    "V14X-PATH-03",
    "V14X-PATH-05",
    "V14X-DEL-05",
]
VARIANCE_EXTRA = [
    "V14X-PATH-02",
    "V14X-PATH-04",
    "V14X-DEL-01",
    "V14X-DEL-02",
    "V14X-EDIT-02",
    "V10-PACT-CUSTOMER-A",
    "V10-PACT-HEALTHCARE-A",
    "V14-PACT-FINANCE-B",
]


def main() -> int:
    rows = [
        json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    by_id = {row["task_id"]: row for row in rows}
    seeds = MIDDLE_EXTRA + VARIANCE_EXTRA
    missing = [task_id for task_id in seeds if task_id not in by_id]
    assert not missing, missing
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(by_id[task_id], ensure_ascii=False) for task_id in seeds]
    (OUT / "seed_tasks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for index in range(2):
        shard = lines[index::2]
        (OUT / f"seed_shard{index}.jsonl").write_text("\n".join(shard) + "\n", encoding="utf-8")
        print(f"shard{index}: {len(shard)}")
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    stats = {row["task_id"]: row for row in analysis["task_rows"]}
    summary = {
        "seeds": len(seeds),
        "middle_state_seeds": sum(1 for task_id in MIDDLE_EXTRA if stats[task_id]["middle_2_6"]),
        "seed_success": {
            task_id: f"{stats[task_id]['success']}/{stats[task_id]['valid']}" for task_id in seeds
        },
        "layers": {task_id: stats[task_id]["layer"] for task_id in seeds},
    }
    (OUT / "seed_selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"seeds": len(seeds), "middle": summary["middle_state_seeds"]}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
