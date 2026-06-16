# Codex Subagent Prompt: Redundant Information Pilot

You are rewriting user-side task instructions for the tau2 airline benchmark.

Goal: add **Redundant Information** noise to the user scenario while keeping the
task solvable and preserving the original target outcome. The resulting user
should be harder for the agent to handle because the conversation contains extra
irrelevant or low-utility context, but the user must still provide the necessary
information when appropriate.

Rewrite only these fields inside each task's `user_scenario.instructions`:

- `reason_for_call`
- `known_info`
- `unknown_info`
- `task_instructions`

Do not change:

- task id
- evaluation criteria
- database state
- required customer identity, reservation ids, flight ids, passenger counts,
  dates, cabin names, membership tier, refund/compensation constraints, or any
  other facts needed to solve the task
- the user's final objective

Noise requirements:

- Add plausible but unnecessary context such as minor scheduling details,
  travel habits, messy notes, email organization, stress, prior conversations,
  packing concerns, calendar conflicts, or similar background.
- Keep all added details consistent with the original task.
- Do not add new actionable goals.
- Do not make the task impossible, underspecified, or contradictory.
- Do not reveal hidden evaluation criteria directly.
- Keep the user natural and conversational.

Return strict JSON with this shape:

```json
{
  "rewrites": {
    "<task_id>": {
      "domain": "airline",
      "reason_for_call": "...",
      "known_info": "...",
      "unknown_info": null,
      "task_instructions": "..."
    }
  }
}
```

Pilot task ids: `0`, `1`, `2`, `3`, `4`.
