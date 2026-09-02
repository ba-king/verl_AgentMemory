"""Explicit budgets for one Answer Agent subtask run."""

from __future__ import annotations

from dataclasses import dataclass

from ..interaction_mode import AnswerAgentInteractionMode


@dataclass(frozen=True)
class AnswerAgentSubtaskRunnerConfig:
    """Runner-owned safety limits, supplied explicitly by the composition root."""

    max_inference_steps: int
    max_tool_calls_per_inference: int
    max_consecutive_tool_failures: int
    interaction_mode: AnswerAgentInteractionMode = AnswerAgentInteractionMode.TOOL_OR_FINAL

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_inference_steps", self.max_inference_steps),
            (
                "max_tool_calls_per_inference",
                self.max_tool_calls_per_inference,
            ),
            (
                "max_consecutive_tool_failures",
                self.max_consecutive_tool_failures,
            ),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if not isinstance(self.interaction_mode, AnswerAgentInteractionMode):
            raise ValueError("interaction_mode must be an AnswerAgentInteractionMode.")
        if (
            self.interaction_mode is AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL
            and self.max_tool_calls_per_inference != 1
        ):
            raise ValueError("ENVIRONMENT_TERMINATED_SINGLE_TOOL requires max_tool_calls_per_inference=1.")

    def to_dict(self) -> dict[str, int | str]:
        serialized: dict[str, int | str] = {
            "max_inference_steps": self.max_inference_steps,
            "max_tool_calls_per_inference": (self.max_tool_calls_per_inference),
            "max_consecutive_tool_failures": (self.max_consecutive_tool_failures),
        }
        if self.interaction_mode is not AnswerAgentInteractionMode.TOOL_OR_FINAL:
            serialized["interaction_mode"] = self.interaction_mode.value
        return serialized


__all__ = ["AnswerAgentSubtaskRunnerConfig"]
