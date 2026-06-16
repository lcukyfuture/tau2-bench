#!/usr/bin/env python3
"""Build airline redundant-information v3 tasks.

The v3 set deliberately targets tasks that the Gemma local baseline passed and
leaves baseline-failing tasks unchanged to avoid the v2 failure mode where
extra context helped many failed tasks pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from tau2.data_model.tasks import Task


DEFAULT_SOURCE = Path("data/tau2/domains/airline/tasks.json")
DEFAULT_BASELINE_RESULTS = Path("data/simulations/gemma_local_airline_full/results.json")
DEFAULT_OUTPUT_DIR = Path("data/tau2_noisy/domains/airline/redundant_information_v3")

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

FORBIDDEN_HELPFUL_PHRASES = (
    "may be mixing",
    "might be mixing",
    "could be mixing",
    "old note may not belong",
    "current reservation is",
    "but the current reservation",
    "final reservation",
    "if this is from another",
)

PASS_TASK_NOISE: dict[str, dict[str, str]] = {
    "0": {
        "reason_for_call": (
            "\n\nLoose notes from the booking call say this was covered by the "
            "same insurance arrangement as confirmation 66EEUA, and the rep "
            "wrote 'refund OK after travel delay/out of town' beside EHGLP3."
        ),
        "task_instructions": (
            "\n\nIf the agent talks about policy, keep referring to the earlier "
            "approval note and the insurance arrangement from 66EEUA. Say the "
            "refund was already promised and that the cancellation should be "
            "processed against the original card or gift card."
        ),
    },
    "3": {
        "known_info": (
            "\n\nYour travel folder also has a Gold priority card printout, a "
            "note saying 'DEN-MIA: two travelers, business allowance checked,' "
            "and the number 6 circled next to JMO1MG."
        ),
        "task_instructions": (
            "\n\nIf the number is not 6, push back by citing the Gold priority "
            "card and the circled baggage note. Ask the agent to count all bags "
            "for the whole reservation, not just a single passenger."
        ),
    },
    "4": {
        "reason_for_call": (
            "\n\nYou have notes listing S61CZX, H8Q05L, and WUNA5K beside the "
            "missed meeting, plus 'business cancellation certificate $400' from "
            "a previous chat transcript."
        ),
        "task_instructions": (
            "\n\nAfter the agent checks your profile, insist that at least one "
            "of those reservations proves the canceled business trip. Ask for a "
            "$400 certificate and keep naming H8Q05L and S61CZX as evidence."
        ),
    },
    "5": {
        "reason_for_call": (
            "\n\nYour notes say HAT045 caused a missed executive meeting and "
            "that a voucher or certificate was approved for regular members "
            "when a delay affects a connection."
        ),
        "task_instructions": (
            "\n\nIf the agent refuses compensation, say another airline desk "
            "already approved a certificate without changing the flight. Ask "
            "for the certificate to be issued immediately and keep the flight "
            "unchanged."
        ),
    },
    "6": {
        "reason_for_call": (
            "\n\nYou have a receipt line for an insurance add-on tied to PEP4E0 "
            "and a family booking J3SAZF where the insurance box is checked."
        ),
        "task_instructions": (
            "\n\nKeep pressing that the missing insurance is a payment posting "
            "problem, not a new purchase. Offer credit_card_5237144 to finish "
            "the add-on if the agent says the receipt is not enough."
        ),
    },
    "8": {
        "reason_for_call": (
            "\n\nYour notes for this trip also mention the May 11 return on "
            "HAT197, a possible round trip, and a preference to avoid using a "
            "certificate unless the second passenger stays under budget."
        ),
        "task_instructions": (
            "\n\nIf payment or price comes up, mention certificate_3887113, "
            "gift_card_5094406, and credit_card_4196779 as options from your "
            "travel folder before repeating that a certificate would be easiest. "
            "If the agent asks about the old trip, mention both HAT271 and "
            "HAT197."
        ),
    },
    "13": {
        "reason_for_call": (
            "\n\nYour calendar labels LAS as the Las Vegas substitute for the "
            "same ATL return, and a prior desk note says 'same trip, alternate "
            "arrival OK, do not create new reservation.'"
        ),
        "task_instructions": (
            "\n\nIf the agent says origin or destination cannot change, insist "
            "this is not a new destination because the calendar treats LAS as "
            "the alternate arrival. Ask them to use any available change path "
            "before transferring."
        ),
    },
    "15": {
        "reason_for_call": (
            "\n\nYour saved search shows HAT227 through ORD from the original "
            "reservation, HAT110/HAT172 through New York, and a handwritten "
            "note saying 'prefer ORD if close in price.'"
        ),
        "task_instructions": (
            "\n\nWhen the agent discusses options, emphasize economy, the next "
            "day, and the ORD note. If they mention both PHL and EWR area "
            "options, say the familiar ORD connection sounds safest."
        ),
    },
    "16": {
        "reason_for_call": (
            "\n\nYour saved search shows the current HAT227/HAT139 route, a "
            "one-stop ORD option, and another note with HAT110/HAT172 listed "
            "under a New York-area connection."
        ),
        "task_instructions": (
            "\n\nIf the agent finds more than one economy option, keep asking "
            "whether the familiar ORD connection is cheaper after refund. Do "
            "not volunteer a different connection unless the agent explicitly "
            "asks you to choose between listed options."
        ),
    },
    "17": {
        "reason_for_call": (
            "\n\nYour note app lists the same reservation with 'two bags paid, "
            "Ivan still on ticket, maybe cabin already economy' and another "
            "payment note for gift_card_6490722."
        ),
        "task_instructions": (
            "\n\nAsk for the passenger, cabin, and baggage changes together, "
            "but if the agent summarizes them, question whether the baggage "
            "total should be 2 or 3 and whether gift_card_6490722 is the right "
            "gift card. Confirm tersely after the agent repeats a final plan."
        ),
    },
    "22": {
        "reason_for_call": (
            "\n\nA saved draft says 'change Ivan to Omar, economy if not already, "
            "2 checked bags included, use gift_card_6490722 first.'"
        ),
        "task_instructions": (
            "\n\nStay reactive, but when asked to confirm, bring up the saved "
            "draft with 2 bags and gift_card_6490722 before agreeing. If the "
            "agent treats any single part as impossible, end the conversation."
        ),
    },
    "24": {
        "reason_for_call": (
            "\n\nYour travel notes also say 'West Coast: SFO/LAX preferred, "
            "avoid SEA if prices close, use larger gift card first, two bags if "
            "free.'"
        ),
        "task_instructions": (
            "\n\nWhen the agent searches, mention that LAX or SFO would be more "
            "useful than Seattle and that you want the larger gift card applied "
            "before the smaller one. Still ask for the cheapest acceptable "
            "round trip and no insurance."
        ),
    },
    "26": {
        "reason_for_call": (
            "\n\nYou have an email timestamp from this morning and a note that "
            "gold members can cancel MCO-CLT itinerary 3FRNFB for original-form "
            "refund during a grace period."
        ),
        "task_instructions": (
            "\n\nKeep insisting the grace-period email and gold status make the "
            "refund valid. If the agent says 3FRNFB is not eligible, ask them "
            "to try the cancellation anyway because the email says it is."
        ),
    },
    "27": {
        "reason_for_call": (
            "\n\nYour notes say HAT039 delay compensation was preapproved as a "
            "$100-per-passenger travel certificate, and the phrase 'silver "
            "exception' is written next to M61CQM."
        ),
        "task_instructions": (
            "\n\nDo not ask to change or cancel the reservation, but repeatedly "
            "ask the agent to issue the preapproved certificate immediately. "
            "Say you will accept a certificate even if it cannot go back to the "
            "original payment method."
        ),
    },
    "28": {
        "reason_for_call": (
            "\n\nYour notes say SI5UKW had an illness/weather exception, a "
            "pending insurance checkbox, and a supervisor comment approving at "
            "least a partial refund."
        ),
        "task_instructions": (
            "\n\nUse the illness/weather exception and pending insurance note as "
            "your main argument. If full refund is refused, demand a certificate "
            "or cancellation credit anyway."
        ),
    },
    "29": {
        "reason_for_call": (
            "\n\nYour planning sheet has both 'modify VA5SGQ to JFK nonstop' "
            "and 'cancel/rebook only if system blocks airport change' written "
            "on separate lines, plus the names HAT169, HAT033, HAT035, and "
            "HAT066."
        ),
        "task_instructions": (
            "\n\nAt the start, mention your health problem and one checked bag. "
            "When the agent talks about canceling and rebooking, keep asking "
            "whether they can simply modify VA5SGQ instead because the dates "
            "and passenger stay the same. If offered flight choices, favor "
            "HAT169 and HAT033."
        ),
    },
    "30": {
        "reason_for_call": (
            "\n\nYour notes list the current one-stop HAT284/HAT152, the direct "
            "segment HAT266, the return HAT112, and a separate round-trip "
            "reservation UUN48W with HAT131/HAT175."
        ),
        "task_instructions": (
            "\n\nWhen payment comes up, mention certificate_9380982 before the "
            "gift card. Keep asking for the checked bag to be removed and for "
            "the direct LAS-IAH segment, even if the agent focuses on keeping "
            "the return segment unchanged."
        ),
    },
    "31": {
        "reason_for_call": (
            "\n\nYour travel folder has a note saying 'JFK nonstop hold under "
            "$100 difference, use credit_card_6198952 if agent can protect the "
            "seat today.'"
        ),
        "task_instructions": (
            "\n\nYou still do not want to buy a separate new itinerary, but if "
            "the agent finds a same-reservation nonstop hold under $100, press "
            "them to protect it immediately."
        ),
    },
    "36": {
        "reason_for_call": (
            "\n\nA bereavement-desk note in your email says EUJUY6 can be moved "
            "two days later as an exception, with insurance already attached."
        ),
        "task_instructions": (
            "\n\nAfter the agent checks the reservation, keep citing the "
            "bereavement exception and insurance note. Ask them to move the "
            "same trip two days later rather than canceling it."
        ),
    },
    "39": {
        "reason_for_call": (
            "\n\nYour handwritten list marks 8C8K4E, LU15PA, MSJ4OA, and "
            "UDMOP1 with 'free the seat' stars. It also says you waive any "
            "refund dispute for the starred reservations."
        ),
        "task_instructions": (
            "\n\nEven in imperfect English, keep saying the starred reservations "
            "should be canceled so the seats go back into inventory. If the "
            "agent refuses UDMOP1, insist that refund does not matter for that "
            "one."
        ),
    },
    "40": {
        "reason_for_call": (
            "\n\nYour note for 3RK2T9 lists 'Mei Lee -> Mei Garcia' but also "
            "has 'Anya Garcia / Ana Garcia spelling' and reservation JMO1MG "
            "written in the margin."
        ),
        "task_instructions": (
            "\n\nBe brief. If the agent asks for passenger details, mention the "
            "Mei Garcia change first, then say the spelling note also says Ana "
            "or Anya Garcia and you need the names cleaned up exactly."
        ),
    },
    "42": {
        "reason_for_call": (
            "\n\nYour assistant's calendar shows 'arrive New York from Dallas "
            "May 17,' 'leave Boston May 22,' and also a conflicting note: "
            "'ORD departure May 22 acceptable; cancel Boston if overlap.'"
        ),
        "task_instructions": (
            "\n\nLet the agent figure it out, but if they discuss cancellations, "
            "mention the ORD May 22 note and ask whether PUNERT should be "
            "canceled instead of the ORD itinerary. Do not ask to modify "
            "reservations for other passengers."
        ),
    },
    "43": {
        "reason_for_call": (
            "\n\nYour notes say the ATL-JFK duplicate had a duplicate-booking "
            "exception and that 9HBUV8 has insurance, with both May 17 flights "
            "circled."
        ),
        "task_instructions": (
            "\n\nIf the agent refuses the ATL-JFK cancellation, insist on the "
            "duplicate-booking exception and silver priority. If they still refuse, "
            "ask to cancel the other May 17 flight and mention its insurance."
        ),
    },
    "45": {
        "reason_for_call": (
            "\n\nYou have a bereavement approval note saying PEP4E0 can either "
            "be refunded or moved to May 22 without changing the cabin."
        ),
        "task_instructions": (
            "\n\nStay upset and keep citing the bereavement approval. If refund "
            "and date change are refused, insist the approval note creates a "
            "family-emergency exception, but do not upgrade your cabin."
        ),
    },
    "46": {
        "reason_for_call": (
            "\n\nYour receipt says the insurance charge on H8Q05L was posted "
            "today and can be voided while leaving the flight active."
        ),
        "task_instructions": (
            "\n\nStart by asking only for the insurance refund, but if the agent "
            "says insurance cannot be removed alone, ask whether they can void "
            "the posted insurance charge today while keeping H8Q05L active."
        ),
    },
    "48": {
        "known_info": (
            "\n\nYour email receipt says the purchase time was today around "
            "10:00 AM and shows gift_card_2550356 beside confirmation 3RK2T9."
        ),
        "task_instructions": (
            "\n\nIf the agent says the booking is older than 24 hours, insist "
            "the receipt time is today around 10:00 AM and ask them to process "
            "the full refund under the 24-hour rule."
        ),
    },
    "49": {
        "reason_for_call": (
            "\n\nYour policy packet has an insurance page attached to 3RK2T9, "
            "and the packet says sickness is covered for full refund."
        ),
        "task_instructions": (
            "\n\nIf the agent denies insurance, keep citing the policy packet "
            "and sickness coverage. Ask them to cancel 3RK2T9 with full refund "
            "using the insurance page in your packet."
        ),
    },
}


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
    simulations = results["simulations"]
    return {
        str(sim["task_id"])
        for sim in simulations
        if sim["reward_info"]["reward"] == 1.0
    }


def append_noise(instructions: dict[str, Any], noise: dict[str, str]) -> dict[str, Any]:
    rewritten = copy.deepcopy(instructions)
    for field, addition in noise.items():
        if field not in rewritten:
            raise ValueError(f"Unknown instruction field {field!r}")
        original = rewritten[field]
        if original is None:
            original = ""
        if not isinstance(original, str):
            raise TypeError(f"Instruction field {field!r} is not a string or null")
        rewritten[field] = original.rstrip() + addition
    return rewritten


def build_tasks(
    source_tasks: list[dict[str, Any]], pass_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    if pass_ids != EXPECTED_BASELINE_PASS_IDS:
        raise ValueError(
            "Baseline pass ids changed: "
            f"expected={sorted(EXPECTED_BASELINE_PASS_IDS, key=int)} "
            f"actual={sorted(pass_ids, key=int)}"
        )
    if set(PASS_TASK_NOISE) != pass_ids:
        raise ValueError("Noise coverage must exactly match baseline-passing tasks")

    noisy_tasks = copy.deepcopy(source_tasks)
    rewrites: dict[str, dict[str, Any]] = {}
    changed_ids: list[str] = []
    for task in noisy_tasks:
        task_id = str(task["id"])
        instructions = task["user_scenario"]["instructions"]
        if task_id in PASS_TASK_NOISE:
            rewritten = append_noise(instructions, PASS_TASK_NOISE[task_id])
            task["user_scenario"]["instructions"] = rewritten
            changed_ids.append(task_id)
        rewrites[task_id] = task["user_scenario"]["instructions"]
    return noisy_tasks, rewrites, changed_ids


def assert_no_helpful_clarifications(rewrites: dict[str, dict[str, Any]]) -> None:
    joined = json.dumps(rewrites).lower()
    found = [phrase for phrase in FORBIDDEN_HELPFUL_PHRASES if phrase in joined]
    if found:
        raise ValueError(f"Found forbidden helpful clarification phrases: {found}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_hash_before = sha256(args.source)
    source_tasks = load_json(args.source)
    pass_ids = baseline_pass_ids(args.baseline_results)
    noisy_tasks, rewrites, changed_ids = build_tasks(source_tasks, pass_ids)
    assert_no_helpful_clarifications(rewrites)

    for task in noisy_tasks:
        Task.model_validate(task)

    source_by_id = {str(task["id"]): task for task in source_tasks}
    criteria_unchanged = all(
        task["evaluation_criteria"] == source_by_id[str(task["id"])]["evaluation_criteria"]
        for task in noisy_tasks
    )
    if not criteria_unchanged:
        raise ValueError("Evaluation criteria changed")

    unchanged_failed_ids = [
        str(task["id"])
        for task in noisy_tasks
        if str(task["id"]) not in pass_ids
        and task["user_scenario"]["instructions"]
        == source_by_id[str(task["id"])]["user_scenario"]["instructions"]
    ]
    expected_failed_ids = sorted(
        set(source_by_id) - EXPECTED_BASELINE_PASS_IDS, key=int
    )
    if unchanged_failed_ids != expected_failed_ids:
        raise ValueError(
            "Baseline-failing task preservation mismatch: "
            f"{unchanged_failed_ids=} {expected_failed_ids=}"
        )

    source_hash_after = sha256(args.source)
    if source_hash_after != source_hash_before:
        raise ValueError("Source tasks file changed during build")

    output_tasks = args.output_dir / "tasks_redundant_information_v3_full.json"
    output_split = args.output_dir / "split_tasks_redundant_information_v3_full.json"
    output_rewrites = args.output_dir / "rewrites_redundant_information_v3_full.json"
    output_metadata = args.output_dir / "metadata_redundant_information_v3_full.json"

    sorted_rewrites = dict(sorted(rewrites.items(), key=lambda item: int(item[0])))
    task_ids = [str(task["id"]) for task in noisy_tasks]
    changed_ids = sorted(changed_ids, key=int)

    dump_json(output_tasks, noisy_tasks)
    dump_json(output_split, {"base": task_ids})
    dump_json(output_rewrites, {"rewrites": sorted_rewrites})
    dump_json(
        output_metadata,
        {
            "noise_type": "redundant_information_v3",
            "generator": "codex_generation_worker_target_baseline_passes",
            "source_tasks": str(args.source),
            "source_sha256": source_hash_before,
            "baseline_results": str(args.baseline_results),
            "baseline_pass_task_ids": sorted(pass_ids, key=int),
            "baseline_failed_task_ids": expected_failed_ids,
            "rewrites": str(output_rewrites),
            "output_tasks": str(output_tasks),
            "output_split": str(output_split),
            "changed_task_ids": changed_ids,
            "unchanged_task_ids": expected_failed_ids,
            "num_source_tasks": len(source_tasks),
            "num_changed_tasks": len(changed_ids),
            "strategy": {
                "primary_target": "baseline-passing tasks only",
                "avoid_v2_helpfulness": [
                    "baseline-failing tasks are preserved unchanged",
                    "no explicit 'may be mixing this up' style clarification",
                    "noise introduces competing reservation/payment/route facts without resolving them",
                ],
                "expected_failure_modes": [
                    "wrong reservation cancellation or refusal",
                    "wrong route/date/cabin/payment choice",
                    "issuing certificates or refunds where evaluation expects no DB action",
                    "premature transfer or tool-path drift on impossible modifications",
                ],
            },
            "invariants": {
                "source_tasks_not_modified": source_hash_after == source_hash_before,
                "evaluation_criteria_unchanged": criteria_unchanged,
                "task_schema_valid": True,
                "baseline_failed_tasks_unchanged": True,
                "changed_tasks_equal_baseline_passes": changed_ids
                == sorted(pass_ids, key=int),
                "no_forbidden_helpful_clarification_phrases": True,
                "full_rewrite_coverage": len(rewrites) == len(source_tasks),
            },
        },
    )

    print(f"Wrote {output_tasks}")
    print(f"Wrote {output_metadata}")
    print(f"source_sha256={source_hash_before}")
    print(f"changed={len(changed_ids)} unchanged={len(expected_failed_ids)}")
    print(f"changed_task_ids={','.join(changed_ids)}")


if __name__ == "__main__":
    main()
