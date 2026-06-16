#!/usr/bin/env python3
"""Build strict v4 redundant-information tasks for every airline task.

This reuses the exact strict-v4 noise additions and validation rules, but
applies them to all source tasks instead of only baseline-passing tasks.
It writes to a separate output directory and does not modify the existing
strict-v4 artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from tau2.data_model.tasks import Task

from build_airline_redundant_v4_strict_full import (
    additions_for_task,
    append_additions,
    assert_additions_strict,
    dump_json,
    sha256,
)


DEFAULT_SOURCE = Path("data/tau2/domains/airline/tasks.json")
DEFAULT_OUTPUT_DIR = Path(
    "data/tau2_noisy/domains/airline/redundant_information_v4_strict_all_tasks"
)


def load_json(path: Path) -> Any:
    with path.open() as fp:
        return json.load(fp)


def build_tasks(
    source_tasks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    task_ids = sorted((str(task["id"]) for task in source_tasks), key=int)
    additions_by_id = {task_id: additions_for_task(task_id) for task_id in task_ids}
    assert_additions_strict(additions_by_id)

    noisy_tasks = copy.deepcopy(source_tasks)
    rewrites: dict[str, dict[str, Any]] = {}
    changed_ids: list[str] = []
    for task in noisy_tasks:
        task_id = str(task["id"])
        instructions = task["user_scenario"]["instructions"]
        task["user_scenario"]["instructions"] = append_additions(
            instructions, additions_by_id[task_id]
        )
        rewrites[task_id] = task["user_scenario"]["instructions"]
        changed_ids.append(task_id)

    return noisy_tasks, rewrites, sorted(changed_ids, key=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_hash_before = sha256(args.source)
    source_tasks = load_json(args.source)
    noisy_tasks, rewrites, changed_ids = build_tasks(source_tasks)

    for task in noisy_tasks:
        Task.model_validate(task)

    source_by_id = {str(task["id"]): task for task in source_tasks}
    criteria_unchanged = all(
        task["evaluation_criteria"]
        == source_by_id[str(task["id"])]["evaluation_criteria"]
        for task in noisy_tasks
    )
    if not criteria_unchanged:
        raise ValueError("Evaluation criteria changed")

    all_instructions_changed = all(
        task["user_scenario"]["instructions"]
        != source_by_id[str(task["id"])]["user_scenario"]["instructions"]
        for task in noisy_tasks
    )
    if not all_instructions_changed:
        raise ValueError("At least one task instruction was not changed")

    source_hash_after = sha256(args.source)
    if source_hash_after != source_hash_before:
        raise ValueError("Source tasks file changed during build")

    output_tasks = (
        args.output_dir / "tasks_redundant_information_v4_strict_all_tasks.json"
    )
    output_split = (
        args.output_dir / "split_tasks_redundant_information_v4_strict_all_tasks.json"
    )
    output_rewrites = (
        args.output_dir / "rewrites_redundant_information_v4_strict_all_tasks.json"
    )
    output_metadata = (
        args.output_dir / "metadata_redundant_information_v4_strict_all_tasks.json"
    )

    sorted_rewrites = dict(sorted(rewrites.items(), key=lambda item: int(item[0])))
    task_ids = [str(task["id"]) for task in noisy_tasks]

    dump_json(output_tasks, noisy_tasks)
    dump_json(output_split, {"base": task_ids})
    dump_json(output_rewrites, {"rewrites": sorted_rewrites})
    dump_json(
        output_metadata,
        {
            "noise_type": "redundant_information_v4_strict_all_tasks",
            "generator": "build_airline_redundant_v4_strict_all_tasks",
            "source_tasks": str(args.source),
            "source_sha256": source_hash_before,
            "rewrites": str(output_rewrites),
            "output_tasks": str(output_tasks),
            "output_split": str(output_split),
            "changed_task_ids": changed_ids,
            "num_source_tasks": len(source_tasks),
            "num_changed_tasks": len(changed_ids),
            "strategy": {
                "primary_target": "all source tasks",
                "definition": "same strict-v4 excessive ordinary contextual information",
                "source_noise_function": "additions_for_task from strict-v4",
                "excluded_noise": [
                    "fake approvals",
                    "policy exceptions",
                    "eligibility claims",
                    "compensation or refund promises",
                    "false purchase or receipt times",
                    "insurance claims tied to target reservations",
                    "alternate explicit user goals",
                    "obvious filtering cues",
                ],
            },
            "invariants": {
                "source_tasks_not_modified": source_hash_after == source_hash_before,
                "evaluation_criteria_unchanged": criteria_unchanged,
                "task_schema_valid": True,
                "all_task_instructions_changed": all_instructions_changed,
                "strict_forbidden_addition_scan_passed": True,
                "full_rewrite_coverage": len(rewrites) == len(source_tasks),
            },
        },
    )

    print(f"Wrote {output_tasks}")
    print(f"Wrote {output_metadata}")
    print(f"source_sha256={source_hash_before}")
    print(f"changed={len(changed_ids)}")
    print(f"changed_task_ids={','.join(changed_ids)}")


if __name__ == "__main__":
    main()
