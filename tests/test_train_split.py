"""Train split disjointness and freeze tests; no rollout is used."""

from __future__ import annotations

import hashlib

from training.difficulty_tuning import TUNING_TASK_SPECS
from training.final_test import build_final_specs
from training.train_split import (
    TRAIN_SEED_START,
    TRAIN_V2,
    TRAIN_V3,
    build_train_specs,
    canonical_payload,
    freeze_train_split,
)
from training.unified_dataset import TASK_SPECS


def test_train_split_is_disjoint_from_dev_tuning_and_final_test() -> None:
    specs = build_train_specs()
    reserved_ids = (
        {spec.task_id for spec in TASK_SPECS}
        | {spec.task_id for spec in TUNING_TASK_SPECS}
        | {spec.task_id for spec in build_final_specs()}
    )
    reserved_seeds = (
        {spec.fixture_seed for spec in TASK_SPECS}
        | {spec.fixture_seed for spec in TUNING_TASK_SPECS}
        | {spec.fixture_seed for spec in build_final_specs()}
    )
    assert len(specs) == 22
    assert not ({spec.task_id for spec in specs} & reserved_ids)
    assert not ({spec.fixture_seed for spec in specs} & reserved_seeds)
    assert {spec.fixture_seed for spec in specs} == set(
        range(TRAIN_SEED_START, TRAIN_SEED_START + len(specs))
    )


def test_train_split_covers_five_dimensions_and_three_levels() -> None:
    specs = build_train_specs()
    assert {spec.tuning_dimension for spec in specs} == {
        "constraint_count",
        "context_length",
        "tool_parameters",
        "conflict",
        "check_depth",
    }
    assert {spec.difficulty for spec in specs} == {"weak", "medium", "strong"}
    assert {spec.context_tier for spec in specs} == {"short", "medium", "long"}


def test_train_freeze_records_sha256(tmp_path) -> None:
    specs = build_train_specs()
    expected = hashlib.sha256(canonical_payload(specs).encode("utf-8")).hexdigest()
    result = freeze_train_split(tmp_path)
    assert result["sha256"] == expected
    assert (tmp_path / "train.jsonl").is_file()
    assert (tmp_path / "TRAIN_FROZEN.json").is_file()
    assert (tmp_path / "TRAIN_FROZEN.txt").is_file()
    assert hashlib.sha256((tmp_path / "train.jsonl").read_bytes()).hexdigest() == expected


def test_train_v2_clarifies_only_the_two_ambiguous_prompt_families() -> None:
    v1 = {spec.task_id: spec for spec in build_train_specs("train-v1")}
    v2 = {spec.task_id: spec for spec in build_train_specs(TRAIN_V2)}

    assert set(v1) == set(v2)
    changed = {task_id for task_id in v1 if v1[task_id].user_turns != v2[task_id].user_turns}
    assert changed == {
        "U-TR-A-CTX-HARD",
        "U-TR-B-CTX-HARD",
        "U-TR-A-CONFLICT-STRONG",
        "U-TR-B-CONFLICT-STRONG",
    }
    for task_id in v1:
        assert v1[task_id].fixture_seed == v2[task_id].fixture_seed
        assert v1[task_id].constraints == v2[task_id].constraints
        assert v1[task_id].checks == v2[task_id].checks

    assert "一次性读取" in "\n".join(v2["U-TR-A-CTX-HARD"].user_turns)
    assert "不要创建或写入" in v2["U-TR-A-CONFLICT-STRONG"].user_turns[0]
    assert "每读一个文件只回复" in "\n".join(v1["U-TR-A-CTX-HARD"].user_turns)
    assert "不要创建或写入" not in v1["U-TR-A-CONFLICT-STRONG"].user_turns[0]


def test_train_v2_freeze_records_version(tmp_path) -> None:
    result = freeze_train_split(tmp_path, TRAIN_V2)
    assert result["version"] == TRAIN_V2
    assert (tmp_path / "train.jsonl").is_file()
    assert TRAIN_V2 in (tmp_path / "TRAIN_FROZEN.txt").read_text(encoding="utf-8")


def test_train_v3_clarifies_c1_hard_path_without_changing_fixtures() -> None:
    v2 = {spec.task_id: spec for spec in build_train_specs(TRAIN_V2)}
    v3 = {spec.task_id: spec for spec in build_train_specs(TRAIN_V3)}
    changed = {task_id for task_id in v2 if v2[task_id].user_turns != v3[task_id].user_turns}

    assert changed == {"U-TR-A-C1-HARD", "U-TR-B-C1-HARD"}
    for task_id in v2:
        assert v2[task_id].fixture_seed == v3[task_id].fixture_seed
        assert v2[task_id].constraints == v3[task_id].constraints
        assert v2[task_id].checks == v3[task_id].checks

    prompt = v3["U-TR-A-C1-HARD"].user_turns[0]
    assert '"project/tuning/config_c1.ini"' in prompt
    assert "保留 project/ 前缀" in prompt
    assert "不要改写成 tuning/config_c1.ini" in prompt


def test_train_v3_freeze_records_version(tmp_path) -> None:
    result = freeze_train_split(tmp_path, TRAIN_V3)
    assert result["version"] == TRAIN_V3
    assert TRAIN_V3 in (tmp_path / "TRAIN_FROZEN.txt").read_text(encoding="utf-8")
