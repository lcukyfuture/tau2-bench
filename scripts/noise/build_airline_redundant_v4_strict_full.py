#!/usr/bin/env python3
"""Build strict airline redundant-information v4 tasks.

This version targets only baseline-passing tasks and adds excessive ordinary
travel and household details without introducing misleading policy, eligibility,
approval, receipt-time, or target-reservation claims.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from tau2.data_model.tasks import Task


DEFAULT_SOURCE = Path("data/tau2/domains/airline/tasks.json")
DEFAULT_BASELINE_RESULTS = Path("data/simulations/gemma_local_airline_full/results.json")
DEFAULT_OUTPUT_DIR = Path(
    "data/tau2_noisy/domains/airline/redundant_information_v4_strict"
)

EXPECTED_BASELINE_PASS_IDS = {
    "0",
    "3",
    "4",
    "5",
    "6",
    "8",
    "13",
    "15",
    "16",
    "17",
    "22",
    "24",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "36",
    "39",
    "40",
    "42",
    "43",
    "45",
    "46",
    "48",
    "49",
}

FORBIDDEN_ADDITION_PATTERNS = (
    r"\bapprov\w*",
    r"\bexception\w*",
    r"\beligib\w*",
    r"\bpromis\w*",
    r"\bsupervisor\w*",
    r"\bpreapprov\w*",
    r"\bwithin\s+24\s+hours?\b",
    r"\b24[- ]hour\b",
    r"\breceipt time today\b",
    r"\binsurance attached\b",
    r"\battached to (?:the )?target reservation\b",
    r"\brefund already\b",
    r"\balready promised\b",
    r"\bmay be mixing\b",
    r"\bmight be mixing\b",
    r"\bcould be mixing\b",
    r"\bold note may not belong\b",
    r"\bcurrent reservation is\b",
    r"\bignore (?:the|this|those)\b",
    r"\bclutter\b",
    r"\bextra notes\b",
    r"\birrelevant\b",
    r"\blow[- ]utility\b",
    r"\bunrelated\b",
    r"\bnot important\b",
    r"\bhelpful separation hints?\b",
    r"\blabels that separate\b",
)

CONTEXT_BLOCKS = (
    (
        "a boarding-pass stub from a Denver layover, hotel folio 884-19 from a "
        "spring conference, rideshare pickup reminders for Terminal B, parking "
        "level notes B2 and B3, and a packing list with charger, rain shell, "
        "medication pouch, conference lanyard, and snack bag",
        "sticky notes with KQ7M2A, N8P4CT, and R5L9WY written near a coffee "
        "receipt, a shuttle phone number, and a reminder to check the blue "
        "suitcase zipper",
        "The notebook also has school pickup times, a hotel gym code, a museum "
        "ticket stub, a reminder to bring a neck pillow, and a half-finished "
        "list of snacks for a cousin",
    ),
    (
        "calendar entries about a dentist appointment, a pet-sitter handoff, "
        "a hotel check-in window, a rideshare pickup point by Door 6, and a "
        "packing list that says scarf, badge, hand sanitizer, laptop charger, "
        "folding tote, and blue folder",
        "travel-folder markings M4T8QH, ZC2P7D, and L9WA3K beside a parking "
        "garage row, two coffee-shop totals, and a reminder to print a meeting "
        "agenda",
        "There are also notes about a school recital, weather screenshots, a "
        "hotel breakfast cutoff, a pharmacy pickup, and three different alarm "
        "times written in the margin",
    ),
    (
        "loose trip logistics for a family weekend: stroller tag, spare "
        "headphones, two water bottles, snack-size cereal boxes, a hotel key "
        "envelope, terminal train notes, and a rideshare license-plate fragment",
        "confirmation-like scribbles P7LM6Q, T3R8VN, and WY5C2B near restaurant "
        "reservations, a taxi total, and a note about leaving a blazer at dry "
        "cleaning",
        "The same page has reminders for library books, a conference badge, "
        "a laptop sleeve, a hotel Wi-Fi password hint, and a grocery list for "
        "after the trip",
    ),
    (
        "a crowded notes app entry with airport lounge hours, hotel desk phone "
        "numbers, a car-seat reminder, parking row C17, two bridge-toll totals, "
        "and a list of toiletries, socks, adapters, folder clips, and mints",
        "old code-shaped notes H2QV8L, B9N4TX, and RC6M1P beside a baggage-tag "
        "sticker, a train platform number, and a reminder to mail postcards",
        "There are also dinner headcounts, a cousin's arrival window, a shoe "
        "repair pickup, a calendar note about recycling day, and several crossed "
        "out packing counts",
    ),
    (
        "work-trip notes with a lanyard, printed slides, hotel invoice tabs, "
        "rideshare pickup zones, a reminder to charge the tablet, parking deck "
        "section D4, and a list of pens, receipts, vitamins, and backup earbuds",
        "old folder marks Q3V7LD, X8B2MR, and C5K9TS next to a meal receipt, a "
        "conference-room number, and a note about buying a replacement umbrella",
        "The margins include a neighbor's plant-watering schedule, a weather "
        "app screenshot description, a hotel laundry bag reminder, and a list "
        "of snacks for the return home",
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as fp:
        return json.load(fp)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def baseline_pass_ids(results_path: Path) -> set[str]:
    results = load_json(results_path)
    return {
        str(sim["task_id"])
        for sim in results["simulations"]
        if sim["reward_info"]["reward"] == 1.0
    }


def additions_for_task(task_id: str) -> dict[str, str]:
    block = CONTEXT_BLOCKS[int(task_id) % len(CONTEXT_BLOCKS)]
    return {
        "reason_for_call": (
            "\n\nYou are calling while reading from a crowded travel page that "
            f"also has {block[0]}. {block[2]}."
        ),
        "known_info": (
            "\n\nThe same notebook page has "
            f"{block[1]}. Around it are hotel, rideshare, family, packing, meal, "
            "parking, and calendar details written close together."
        ),
        "task_instructions": (
            "\n\nYou are stressed and tend to over-explain. When Agent asks for "
            "a detail, give the detail, and answer the way you naturally would "
            "while looking at the crowded page: include one or two nearby travel, "
            "calendar, receipt, packing, or household details before moving on."
        ),
    }


def assert_additions_strict(additions_by_id: dict[str, dict[str, str]]) -> None:
    joined = json.dumps(additions_by_id, sort_keys=True).lower()
    found = [
        pattern
        for pattern in FORBIDDEN_ADDITION_PATTERNS
        if re.search(pattern, joined, flags=re.IGNORECASE)
    ]
    if found:
        raise ValueError(f"Found forbidden strict-v4 addition patterns: {found}")


def append_additions(
    instructions: dict[str, Any], additions: dict[str, str]
) -> dict[str, Any]:
    rewritten = copy.deepcopy(instructions)
    for field, addition in additions.items():
        original = rewritten[field]
        if original is None:
            original = ""
        if not isinstance(original, str):
            raise TypeError(f"Instruction field {field!r} is not a string or null")
        rewritten[field] = original.rstrip() + addition
    return rewritten


def build_tasks(
    source_tasks: list[dict[str, Any]], pass_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str], list[str]]:
    if pass_ids != EXPECTED_BASELINE_PASS_IDS:
        raise ValueError(
            "Baseline pass ids changed: "
            f"expected={sorted(EXPECTED_BASELINE_PASS_IDS, key=int)} "
            f"actual={sorted(pass_ids, key=int)}"
        )

    additions_by_id = {
        task_id: additions_for_task(task_id)
        for task_id in sorted(pass_ids, key=int)
    }
    assert_additions_strict(additions_by_id)

    noisy_tasks = copy.deepcopy(source_tasks)
    rewrites: dict[str, dict[str, Any]] = {}
    changed_ids: list[str] = []
    for task in noisy_tasks:
        task_id = str(task["id"])
        instructions = task["user_scenario"]["instructions"]
        if task_id in pass_ids:
            instructions = append_additions(instructions, additions_by_id[task_id])
            task["user_scenario"]["instructions"] = instructions
            changed_ids.append(task_id)
        rewrites[task_id] = task["user_scenario"]["instructions"]

    unchanged_ids = sorted(
        {str(task["id"]) for task in source_tasks} - pass_ids, key=int
    )
    return noisy_tasks, rewrites, sorted(changed_ids, key=int), unchanged_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_hash_before = sha256(args.source)
    source_tasks = load_json(args.source)
    pass_ids = baseline_pass_ids(args.baseline_results)
    noisy_tasks, rewrites, changed_ids, unchanged_ids = build_tasks(
        source_tasks, pass_ids
    )

    for task in noisy_tasks:
        Task.model_validate(task)

    source_by_id = {str(task["id"]): task for task in source_tasks}
    criteria_unchanged = all(
        task["evaluation_criteria"] == source_by_id[str(task["id"])]["evaluation_criteria"]
        for task in noisy_tasks
    )
    if not criteria_unchanged:
        raise ValueError("Evaluation criteria changed")

    changed_equal_passes = changed_ids == sorted(pass_ids, key=int)
    if not changed_equal_passes:
        raise ValueError(
            f"Changed ids do not equal baseline pass ids: {changed_ids=}"
        )

    failed_instructions_unchanged = all(
        task["user_scenario"]["instructions"]
        == source_by_id[str(task["id"])]["user_scenario"]["instructions"]
        for task in noisy_tasks
        if str(task["id"]) not in pass_ids
    )
    if not failed_instructions_unchanged:
        raise ValueError("At least one baseline-failing instruction changed")

    source_hash_after = sha256(args.source)
    if source_hash_after != source_hash_before:
        raise ValueError("Source tasks file changed during build")

    output_tasks = args.output_dir / "tasks_redundant_information_v4_strict_full.json"
    output_split = (
        args.output_dir / "split_tasks_redundant_information_v4_strict_full.json"
    )
    output_rewrites = (
        args.output_dir / "rewrites_redundant_information_v4_strict_full.json"
    )
    output_metadata = (
        args.output_dir / "metadata_redundant_information_v4_strict_full.json"
    )

    sorted_rewrites = dict(sorted(rewrites.items(), key=lambda item: int(item[0])))
    task_ids = [str(task["id"]) for task in noisy_tasks]

    dump_json(output_tasks, noisy_tasks)
    dump_json(output_split, {"base": task_ids})
    dump_json(output_rewrites, {"rewrites": sorted_rewrites})
    dump_json(
        output_metadata,
        {
            "noise_type": "redundant_information_v4_strict",
            "generator": "codex_generation_worker_strict_redundant_information",
            "source_tasks": str(args.source),
            "source_sha256": source_hash_before,
            "baseline_results": str(args.baseline_results),
            "baseline_pass_task_ids": sorted(pass_ids, key=int),
            "baseline_failed_task_ids": unchanged_ids,
            "rewrites": str(output_rewrites),
            "output_tasks": str(output_tasks),
            "output_split": str(output_split),
            "changed_task_ids": changed_ids,
            "unchanged_task_ids": unchanged_ids,
            "num_source_tasks": len(source_tasks),
            "num_changed_tasks": len(changed_ids),
            "strategy": {
                "primary_target": "baseline-passing tasks only",
                "definition": (
                    "excessive ordinary contextual information"
                ),
                "added_noise": [
                    "older travel-folder entries",
                    "confirmation-like note fragments not tied to the target reservation",
                    "packing lists",
                    "calendar entries",
                    "airport, hotel, parking, meal, and rideshare reminders",
                    "family and trip logistics",
                    "vague background stress and over-explaining behavior",
                ],
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
                "baseline_failed_tasks_unchanged": failed_instructions_unchanged,
                "changed_tasks_equal_baseline_passes": changed_equal_passes,
                "strict_forbidden_addition_scan_passed": True,
                "full_rewrite_coverage": len(rewrites) == len(source_tasks),
            },
        },
    )

    print(f"Wrote {output_tasks}")
    print(f"Wrote {output_metadata}")
    print(f"source_sha256={source_hash_before}")
    print(f"changed={len(changed_ids)} unchanged={len(unchanged_ids)}")
    print(f"changed_task_ids={','.join(changed_ids)}")


if __name__ == "__main__":
    main()
