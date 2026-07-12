from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class TransitionRepairOutcome:
    action: str
    input_messages: list[dict[str, Any]]
    validation: dict[str, Any] | None
    blocked: bool
    repair_attempted: bool


def enforce_required_transition(
    *,
    memory: Any,
    agent: Any,
    prompt_source: str,
    action: str,
    input_messages: list[dict[str, Any]],
    normalize_message: Callable[[str, str], dict[str, Any]],
) -> TransitionRepairOutcome:
    """Validate one proposed transition and allow at most one answer-agent repair."""
    validator = getattr(memory, "validate_transition", None) if memory is not None else None
    if not callable(validator):
        return TransitionRepairOutcome(action, input_messages, None, False, False)

    initial = validator(prompt_source, action)
    if initial.get("allow") is not False:
        return TransitionRepairOutcome(action, input_messages, initial, False, False)

    instruction = (
        initial.get("repair_instruction")
        or "Choose an action that satisfies the current required evidence obligation."
    )
    repair_messages = [
        *input_messages,
        normalize_message("assistant", action),
        normalize_message(
            "user",
            "Your proposed action was rejected by the required-evidence transition "
            f"validator. {instruction} Return exactly one executable action and no explanation.",
        ),
    ]
    setattr(agent, "_dart_csl_transition_repair", True)
    try:
        repaired_action = agent.act_with_messages(repair_messages)
    finally:
        setattr(agent, "_dart_csl_transition_repair", False)
    repaired = validator(prompt_source, repaired_action)
    validation = {"initial": initial, "repair": repaired}
    return TransitionRepairOutcome(
        action=repaired_action,
        input_messages=repair_messages,
        validation=validation,
        blocked=repaired.get("allow") is False,
        repair_attempted=True,
    )


def blocked_transition_result(
    *,
    prompt_source: str,
    outcome: TransitionRepairOutcome,
) -> dict[str, Any]:
    validation = outcome.validation or {}
    final = validation.get("repair") if outcome.repair_attempted else validation
    final = final if isinstance(final, dict) else {}
    instruction = (
        final.get("repair_instruction")
        or "Choose an action that satisfies the required evidence obligation."
    )
    marker = "\n\nCurrent environment remains unchanged:\n"
    current_page = prompt_source.rsplit(marker, 1)[-1]
    return {
        "observation": {
            "state": (
                "DART-CSL transition validator blocked the proposed action. "
                + instruction
                + marker
                + current_page
            )
        },
        "reward": 0.0,
        "done": False,
        "info": {
            "dart_csl_transition": outcome.validation,
            "transition_blocked": True,
        },
    }
