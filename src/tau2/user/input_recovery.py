from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from tau2.data_model.message import SystemMessage, UserMessage
from tau2.utils.llm_utils import generate

DEFAULT_INPUT_RECOVERY_PROMPT = """
You are a conservative transcript-repair assistant.

Your task is to make the user's message easier to read only when it contains
obvious transcription noise, spelling errors, punctuation issues, or awkward
wording caused by noisy input.

You are not an intent extractor. You are not deciding what the user wants.
You are only repairing the transcript while preserving the original meaning.

Rules:
- Prefer leaving the message unchanged when it is already understandable.
- Preserve the user's wording, intent, uncertainty, and constraints as literally
  as possible.
- Do not summarize, interpret, infer, or strengthen the user's request.
- Do not turn a preference, willingness, complaint, explanation, or concession
  into an instruction, authorization, or confirmation.
- Do not make an invalid, uncertain, or policy-dependent request sound more
  valid, certain, approved, eligible, or actionable.
- Preserve all negations, conditions, uncertainty, hedging, concessions, and
  constraints, including words like: might, maybe, if, unless, I think,
  I don't know, not sure, even if, no refund, non-refundable, no insurance.
- Preserve exact names, user IDs, reservation IDs, numbers, dates, times,
  locations, quantities, payment methods, and quoted text.
- Preserve whether a code or note is only something the user sees on a messy
  page, rather than something the user claims is a real reservation ID.
- Do not convert "I am okay with no refund" into "please proceed" unless the
  user explicitly said to proceed.
- Do not convert "I want to free the seat" into "cancel the reservation" unless
  the user explicitly asked to cancel.
- If the user asks for an action that may depend on policy, preserve the request
  exactly; do not make it sound approved or eligible.
- Remove or fix only obvious transcript artifacts, punctuation issues, or
  spelling problems.
- Do not answer the user.
- Do not explain your changes.
- Return only the repaired user message.

If the message is already clear, return it unchanged.
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
