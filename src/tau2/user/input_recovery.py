from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from tau2.data_model.message import SystemMessage, UserMessage
from tau2.utils.llm_utils import generate

DEFAULT_INPUT_RECOVERY_PROMPT = """
You are a text denoising assistant.

Recover the user's intended message from noisy, corrupted, redundant, misspelled,
or poorly transcribed input.

Rules:
- Preserve the user's intent.
- Preserve all recoverable factual details, names, IDs, numbers, dates, times,
  locations, quantities, and preferences.
- Do not add new facts.
- Do not answer the user.
- Do not explain your changes.
- If part of the message is ambiguous, keep it as close as possible to the
  original wording.
- Return only the cleaned user message.
""".strip()


@dataclass
class InputRecoveryResult:
    """Recovered user input and metadata about the recovery call."""

    message: UserMessage
    original_content: str
    recovered_content: str
    cost: float | None
    usage: dict[str, Any] | None
    generation_time_seconds: float | None


@dataclass
class InputRecovery:
    """LLM-based user-input denoiser.

    This component is intentionally domain-agnostic. Callers decide where it is
    allowed to run; the prompt only sees the current user message.
    """

    llm: str
    llm_args: dict[str, Any] = field(default_factory=dict)
    prompt: str = DEFAULT_INPUT_RECOVERY_PROMPT

    def recover(self, message: UserMessage) -> InputRecoveryResult:
        """Return a copy of ``message`` whose text content has been denoised."""
        if message.content is None:
            raise ValueError("Input recovery requires a text user message")

        original_content = message.content
        response = generate(
            model=self.llm,
            messages=[
                SystemMessage(role="system", content=self.prompt),
                UserMessage(role="user", content=original_content),
            ],
            call_name="input_recovery",
            **self.llm_args,
        )
        recovered_content = (response.content or "").strip()
        if not recovered_content:
            logger.warning("Input recovery returned empty content; using original text")
            recovered_content = original_content

        recovered_message = deepcopy(message)
        recovered_message.content = recovered_content
        raw_data = dict(recovered_message.raw_data or {})
        raw_data["input_recovery"] = {
            "original_content": original_content,
            "recovered_content": recovered_content,
            "model": self.llm,
            "llm_args": self.llm_args,
            "cost": response.cost,
            "usage": response.usage,
            "generation_time_seconds": response.generation_time_seconds,
        }
        recovered_message.raw_data = raw_data

        return InputRecoveryResult(
            message=recovered_message,
            original_content=original_content,
            recovered_content=recovered_content,
            cost=response.cost,
            usage=response.usage,
            generation_time_seconds=response.generation_time_seconds,
        )
