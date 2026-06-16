#!/usr/bin/env python3
"""Run the full airline redundant-information v2 task file with local Gemma."""

from __future__ import annotations

import sys

from run_gemma_tau2_airline_noisy_file import main


if __name__ == "__main__":
    default_tasks_file = (
        "data/tau2_noisy/domains/airline/redundant_information_v2/"
        "tasks_redundant_information_v2_full.json"
    )
    if "--tasks-file" not in sys.argv:
        sys.argv[1:1] = ["--tasks-file", default_tasks_file]
    main()
