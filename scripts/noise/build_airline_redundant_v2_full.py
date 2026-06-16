#!/usr/bin/env python3
"""Build full airline redundant-information v2 tasks from batch rewrites."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from tau2.data_model.tasks import Task


DEFAULT_SOURCE = Path("data/tau2/domains/airline/tasks.json")
DEFAULT_BATCH_DIR = Path(
    "data/tau2_noisy/domains/airline/redundant_information_v2/batches"
)
DEFAULT_OUTPUT_DIR = Path("data/tau2_noisy/domains/airline/redundant_information_v2")

INSTRUCTION_FIELDS = {
    "domain",
    "reason_for_call",
    "known_info",
    "unknown_info",
    "task_instructions",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as fp:
        return json.load(fp)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_rewrites(batch_dir: Path) -> dict[str, dict[str, Any]]:
    rewrites: dict[str, dict[str, Any]] = {}
    batch_files = sorted(batch_dir.glob("rewrites_redundant_information_v2_tasks_*.json"))
    if not batch_files:
        raise FileNotFoundError(f"No batch rewrite files found in {batch_dir}")
    for path in batch_files:
        payload = load_json(path)
        batch_rewrites = payload.get("rewrites")
        if not isinstance(batch_rewrites, dict):
            raise ValueError(f"{path} does not contain a rewrites object")
        for task_id, instructions in batch_rewrites.items():
            task_id = str(task_id)
            if task_id in rewrites:
                raise ValueError(f"Duplicate rewrite for task {task_id}")
            if not isinstance(instructions, dict):
                raise ValueError(f"Rewrite for task {task_id} in {path} is not an object")
            missing = INSTRUCTION_FIELDS - set(instructions)
            extra = set(instructions) - INSTRUCTION_FIELDS
            if missing or extra:
                raise ValueError(
                    f"Rewrite for task {task_id} in {path} has missing={missing}, extra={extra}"
                )
            rewrites[task_id] = instructions
    return rewrites


def build_full_tasks(
    source_tasks: list[dict[str, Any]], rewrites: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    source_by_id = {str(task["id"]): task for task in source_tasks}
    expected_ids = set(source_by_id)
    rewrite_ids = set(rewrites)
    missing = sorted(expected_ids - rewrite_ids, key=int)
    extra = sorted(rewrite_ids - expected_ids, key=int)
    if missing or extra:
        raise ValueError(f"Rewrite coverage mismatch: missing={missing}, extra={extra}")

    noisy_tasks = copy.deepcopy(source_tasks)
    for task in noisy_tasks:
        task["user_scenario"]["instructions"] = rewrites[str(task["id"])]
    return noisy_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_hash_before = sha256(args.source)
    source_tasks = load_json(args.source)
    rewrites = load_rewrites(args.batch_dir)
    noisy_tasks = build_full_tasks(source_tasks, rewrites)

    for task in noisy_tasks:
        Task.model_validate(task)

    source_by_id = {str(task["id"]): task for task in source_tasks}
    criteria_unchanged = all(
        task["evaluation_criteria"] == source_by_id[str(task["id"])]["evaluation_criteria"]
        for task in noisy_tasks
    )
    if not criteria_unchanged:
        raise ValueError("Evaluation criteria changed")

    unchanged_instruction_ids = [
        str(task["id"])
        for task in noisy_tasks
        if task["user_scenario"]["instructions"]
        == source_by_id[str(task["id"])]["user_scenario"]["instructions"]
    ]
    if unchanged_instruction_ids:
        raise ValueError(f"Tasks not rewritten: {unchanged_instruction_ids}")

    source_hash_after = sha256(args.source)
    if source_hash_after != source_hash_before:
        raise ValueError("Source tasks file changed during build")

    output_tasks = args.output_dir / "tasks_redundant_information_v2_full.json"
    output_split = args.output_dir / "split_tasks_redundant_information_v2_full.json"
    output_rewrites = args.output_dir / "rewrites_redundant_information_v2_full.json"
    output_metadata = args.output_dir / "metadata_redundant_information_v2_full.json"

    sorted_rewrites = dict(sorted(rewrites.items(), key=lambda item: int(item[0])))
    task_ids = [str(task["id"]) for task in noisy_tasks]

    dump_json(output_tasks, noisy_tasks)
    dump_json(output_split, {"base": task_ids})
    dump_json(output_rewrites, {"rewrites": sorted_rewrites})
    dump_json(
        output_metadata,
        {
            "noise_type": "redundant_information_v2",
            "generator": "codex_subagents_batched_full",
            "source_tasks": str(args.source),
            "source_sha256": source_hash_before,
            "batch_dir": str(args.batch_dir),
            "rewrites": str(output_rewrites),
            "output_tasks": str(output_tasks),
            "output_split": str(output_split),
            "changed_task_ids": task_ids,
            "num_source_tasks": len(source_tasks),
            "num_changed_tasks": len(rewrites),
            "invariants": {
                "source_tasks_not_modified": source_hash_after == source_hash_before,
                "evaluation_criteria_unchanged": criteria_unchanged,
                "task_schema_valid": True,
                "full_rewrite_coverage": len(rewrites) == len(source_tasks),
                "all_instructions_changed": len(unchanged_instruction_ids) == 0,
            },
        },
    )

    print(f"Wrote {output_tasks}")
    print(f"Wrote {output_metadata}")
    print(f"source_sha256={source_hash_before}")
    print(f"rewrites={len(rewrites)} tasks={len(noisy_tasks)}")


if __name__ == "__main__":
    main()
