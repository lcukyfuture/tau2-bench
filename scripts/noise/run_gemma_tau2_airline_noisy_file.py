#!/usr/bin/env python3
"""Run airline tasks from a derived noisy task JSON file with local Gemma.

This intentionally reads from data/tau2_noisy instead of modifying or registering
the original tau2 airline task data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tau2.config import DEFAULT_INPUT_RECOVERY_LLM, DEFAULT_INPUT_RECOVERY_LLM_ARGS
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.metrics.agent_metrics import compute_metrics
from tau2.runner import run_tasks
from tau2.utils.display import ConsoleDisplay
from tau2.utils.utils import DATA_DIR

DEFAULT_TASKS_FILE = Path(
    "data/tau2_noisy/domains/airline/redundant_information/"
    "tasks_redundant_information_full.json"
)


def _parse_json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must decode to a JSON object")
    return parsed


def _load_tasks(tasks_file: Path) -> list[Task]:
    with tasks_file.open() as fp:
        raw_tasks = json.load(fp)
    if not isinstance(raw_tasks, list):
        raise ValueError(f"{tasks_file} must contain a JSON list of tasks")
    return [Task.model_validate(task) for task in raw_tasks]


def _select_tasks(
    tasks: list[Task],
    task_ids: list[str] | None,
    num_tasks: int | None,
) -> list[Task]:
    selected = tasks
    if task_ids:
        wanted = set(task_ids)
        selected = [task for task in selected if task.id in wanted]
        found = {task.id for task in selected}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"task id(s) not found in noisy task file: {missing}")
    if num_tasks is not None:
        selected = selected[:num_tasks]
    if not selected:
        raise ValueError("no tasks selected")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run derived noisy airline tasks with local Gemma via tau2 runner."
    )
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS_FILE)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--save-to", default="gemma_local_airline_redundant_pilot_0_4")
    parser.add_argument("--served-model-name", default="gemma-local")
    parser.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--agent-llm", default=None)
    parser.add_argument("--user-llm", default=None)
    parser.add_argument("--agent-llm-args", type=_parse_json_object, default=None)
    parser.add_argument("--user-llm-args", type=_parse_json_object, default=None)
    parser.add_argument("--input-recovery", action="store_true", default=False)
    parser.add_argument("--input-recovery-llm", default=DEFAULT_INPUT_RECOVERY_LLM)
    parser.add_argument(
        "--input-recovery-llm-args",
        type=_parse_json_object,
        default=DEFAULT_INPUT_RECOVERY_LLM_ARGS,
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tasks_file = args.tasks_file
    if not tasks_file.is_absolute():
        tasks_file = Path.cwd() / tasks_file

    tasks = _select_tasks(_load_tasks(tasks_file), args.task_ids, args.num_tasks)

    local_llm = f"openai/{args.served_model_name}"
    local_llm_args = {
        "api_base": args.base_url,
        "api_key": "EMPTY",
        "temperature": 0.0,
    }

    config = TextRunConfig(
        domain="airline",
        agent="llm_agent",
        user="user_simulator",
        llm_agent=args.agent_llm or local_llm,
        llm_args_agent=args.agent_llm_args or local_llm_args,
        llm_user=args.user_llm or local_llm,
        llm_args_user=args.user_llm_args or local_llm_args,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        timeout=args.timeout,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        log_level=args.log_level,
        enforce_communication_protocol=False,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        auto_resume=not args.no_auto_resume,
        auto_review=False,
        verbose_logs=not args.quiet,
        input_recovery_enabled=args.input_recovery,
        input_recovery_llm=args.input_recovery_llm,
        input_recovery_llm_args=args.input_recovery_llm_args,
    )

    save_dir = DATA_DIR / "simulations" / args.save_to
    save_path = save_dir / "results.json"

    print(f"Loaded {len(tasks)} task(s) from {tasks_file}")
    print(f"Saving results to {save_path}")
    print(f"Agent/User LLM: {config.llm_agent} / {config.llm_user}")
    print(f"Input recovery: {config.input_recovery_enabled} ({config.input_recovery_llm})")

    results = run_tasks(
        config,
        tasks,
        save_path=save_path,
        save_dir=save_dir,
        evaluation_type=EvaluationType.ALL,
        console_display=not args.quiet,
    )
    metrics = compute_metrics(results)
    ConsoleDisplay.display_agent_metrics(metrics)


if __name__ == "__main__":
    main()
